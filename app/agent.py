"""
Per-turn pipeline. Every step is defensive: a malformed or missing LLM field
degrades to a safe default rather than raising, because a 500 or a
schema-invalid response fails the hard eval outright regardless of how
good the reasoning upstream was. The only thing that must never fail
silently is `recommendations` containing an off-catalog item — that's
checked explicitly against `store.by_url`, not trusted from the LLM.
"""
from __future__ import annotations

import logging

from .catalog import CatalogStore
from .guardrails import looks_like_injection
from .history import format_shortlist_table, reconstruct_committed_shortlist
from .llm import LLMClient, LLMError
from .prompts import (
    GENERATE_SYSTEM,
    ROUTE_SYSTEM,
    build_generate_user_prompt,
    build_route_user_prompt,
)
from .retrieval import DEFAULT_COMPANIONS, RetrievalIndex
from .schemas import ChatMessage, ChatResponse, Recommendation

logger = logging.getLogger(__name__)

MAX_MESSAGES = 8  # literal spec: "8 turns including user & assistant"
FORCE_RECOMMEND_AT = MAX_MESSAGES  # this reply would be message #8 — last chance
SOFT_BIAS_AT = MAX_MESSAGES - 2  # nudge earlier, still allow one more clarify if truly needed

OFF_TOPIC_REPLY = (
    "I can only help with selecting SHL assessments from our catalog — I'm not able to advise on "
    "general hiring practices, legal or compliance questions, or anything outside assessment "
    "selection. Happy to help pick the right assessments for your role, though."
)
DEGRADED_REPLY = (
    "Sorry — I hit an internal error putting that together. Could you rephrase your request, or "
    "let me know the role and what you're assessing for?"
)


class Agent:
    def __init__(self, store: CatalogStore, index: RetrievalIndex, llm: LLMClient):
        self.store = store
        self.index = index
        self.llm = llm

    def handle(self, messages: list[ChatMessage]) -> ChatResponse:
        this_reply_index = len(messages) + 1
        force_recommend = this_reply_index >= FORCE_RECOMMEND_AT

        latest_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        if looks_like_injection(latest_user):
            return ChatResponse(reply=OFF_TOPIC_REPLY, recommendations=None, end_of_conversation=False)

        try:
            route = self._route(messages, force_recommend)
        except LLMError as e:
            logger.warning("route call failed: %s", e)
            return ChatResponse(reply=DEGRADED_REPLY, recommendations=None, end_of_conversation=False)

        intent = route.get("intent", "clarify")
        if force_recommend and intent != "refuse":
            intent = "recommend"  # hard override: code-enforced, not just prompt-requested

        if intent == "refuse":
            return self._handle_refuse(route)

        previous_shortlist = reconstruct_committed_shortlist(messages, self.store)

        if intent == "clarify":
            return self._handle_clarify(messages, route)

        if intent == "compare":
            return self._handle_compare(messages, route)

        # intent == "recommend" (covers fresh / refine / reconfirm)
        return self._handle_recommend(messages, route, previous_shortlist, force_recommend)

    # -- intent handlers -----------------------------------------------

    def _route(self, messages: list[ChatMessage], force_recommend: bool) -> dict:
        user_prompt = build_route_user_prompt(messages, force_recommend)
        return self.llm.complete_json(ROUTE_SYSTEM, user_prompt)

    def _handle_refuse(self, route: dict) -> ChatResponse:
        reply = OFF_TOPIC_REPLY
        try:
            gen = self.llm.complete_json(
                GENERATE_SYSTEM,
                build_generate_user_prompt(
                    messages=[], intent="refuse", clarifying_question=None,
                    refusal_reason=route.get("refusal_reason"), candidate_pool=[],
                    previous_shortlist=[], excluded_names=[],
                ),
            )
            reply = gen.get("reply") or reply
        except LLMError:
            pass  # fall back to canned OFF_TOPIC_REPLY — still schema-valid
        return ChatResponse(reply=reply, recommendations=None, end_of_conversation=False)

    def _handle_clarify(self, messages: list[ChatMessage], route: dict) -> ChatResponse:
        question = route.get("clarifying_question") or "Could you tell me more about the role and what matters most for it?"
        try:
            gen = self.llm.complete_json(
                GENERATE_SYSTEM,
                build_generate_user_prompt(
                    messages=messages, intent="clarify", clarifying_question=question,
                    refusal_reason=None, candidate_pool=[], previous_shortlist=[], excluded_names=[],
                ),
            )
            reply = gen.get("reply") or question
        except LLMError:
            reply = question
        return ChatResponse(reply=reply, recommendations=None, end_of_conversation=False)

    def _handle_compare(self, messages: list[ChatMessage], route: dict) -> ChatResponse:
        names = route.get("compare_items") or []
        resolved: list = []
        unresolved: list[str] = []
        for n in names:
            a = self.store.resolve_name(n)
            (resolved if a else unresolved).append(a or n)

        try:
            gen = self.llm.complete_json(
                GENERATE_SYSTEM,
                build_generate_user_prompt(
                    messages=messages, intent="compare", clarifying_question=None,
                    refusal_reason=None, candidate_pool=resolved, previous_shortlist=[],
                    excluded_names=[],
                ),
            )
            reply = gen.get("reply") or "I couldn't complete that comparison — could you name the assessments again?"
        except LLMError:
            reply = "I'm having trouble comparing those right now — could you try again?"

        if unresolved:
            reply += f"\n\n(Note: I couldn't find {', '.join(map(str, unresolved))} in the catalog.)"
        return ChatResponse(reply=reply, recommendations=None, end_of_conversation=False)

    def _handle_recommend(
        self, messages: list[ChatMessage], route: dict, previous_shortlist: list,
        force_recommend: bool,
    ) -> ChatResponse:
        query = route.get("retrieval_query") or " ".join(m.content for m in messages if m.role == "user")
        excluded_names = route.get("excluded_names") or []
        excluded_urls = {a.url for n in excluded_names if (a := self.store.resolve_name(n))}
        excluded_urls |= {a.url for a in previous_shortlist if a.name in excluded_names}

        test_type_focus = route.get("test_type_focus") or []
        job_level = route.get("job_level")
        language = route.get("language")
        pa_hint = route.get("personality_ability_hint")

        pool = {a.url: a for a in self.index.build_candidate_pool(
            query, job_level=job_level, language=language,
            exclude_urls=excluded_urls, extra_categories=test_type_focus,
            personality_ability_hint=pa_hint,
        )}
        for a in previous_shortlist:  # keep prior items visible as candidates for continuity
            if a.url not in excluded_urls:
                pool[a.url] = a
        candidate_pool = list(pool.values())

        # DEBUG: temporary — shows exactly what ROUTE extracted and what pool
        # GENERATE actually received, to diagnose live-vs-local discrepancies.
        logger.info("ROUTE output: %s", route)
        logger.info("retrieval_query used: %r", query)
        logger.info("candidate_pool (%d items): %s", len(candidate_pool), [a.name for a in candidate_pool])

        try:
            gen = self.llm.complete_json(
                GENERATE_SYSTEM,
                build_generate_user_prompt(
                    messages=messages, intent="recommend", clarifying_question=None,
                    refusal_reason=None, candidate_pool=candidate_pool,
                    previous_shortlist=previous_shortlist, excluded_names=excluded_names,
                ),
            )
        except LLMError:
            gen = {"reply": "Here's an updated shortlist based on your requirements.", "selected_urls": None}

        selected_urls = gen.get("selected_urls") or []
        pool_urls = {a.url for a in candidate_pool}
        final = [self.store.get_by_url(u) for u in selected_urls if u in pool_urls]
        final = [a for a in final if a is not None]

        if not final:  # safety net: never return recommend-intent with zero grounded items
            fallback_source = previous_shortlist or candidate_pool
            final = fallback_source[:5]
        final = final[:10]

        reply_text = gen.get("reply") or "Here's a shortlist based on your requirements."
        reply_text = reply_text.rstrip() + "\n\n" + format_shortlist_table(final)

        user_confirmed = bool(route.get("user_confirmed_shortlist"))
        end_of_conversation = user_confirmed or force_recommend

        recs = [
            Recommendation(name=a.name, url=a.url, test_type=",".join(a.test_types))
            for a in final
        ]
        return ChatResponse(reply=reply_text, recommendations=recs, end_of_conversation=end_of_conversation)
