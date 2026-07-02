"""
Retrieval over the ~377-item catalog. The catalog is small enough that a
full vector DB is overkill — an in-memory TF-IDF index covers it, loads in
milliseconds, and needs no model weights shipped in the Docker image (keeps
cold start well under the 2-minute grace period).

This module is intentionally LLM-free: it produces a candidate pool that the
agent layer then filters/reranks with conversation context. Keeping retrieval
deterministic and testable independent of the LLM is what makes the
eval_retrieval.py self-eval meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .catalog import Assessment, CatalogStore

# SHL's flagship general-purpose instruments. Derived from the gold traces:
# a generic personality measure appears in most committed shortlists even
# when the user never says "personality" (6/10 traces), and a general
# cognitive-ability test similarly appears by default for
# professional/graduate/senior roles (3/10 traces) — in both cases purely on
# role-seniority/selection-context grounds, not lexical overlap with the
# query. Kept as an explicit, visible list rather than inferred, since this
# is a domain convention, not something retrieval can discover from text.
DEFAULT_COMPANIONS = [
    "Occupational Personality Questionnaire OPQ32r",
    "SHL Verify Interactive G+",
]


@dataclass
class ScoredAssessment:
    assessment: Assessment
    score: float


class RetrievalIndex:
    def __init__(self, store: CatalogStore):
        self.store = store
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            min_df=1,
        )
        corpus = [a.search_text() for a in store.assessments]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
        job_level: str | None = None,
        language: str | None = None,
        category_codes: list[str] | None = None,
        exclude_urls: set[str] | None = None,
        include_default_companions: bool = False,
    ) -> list[ScoredAssessment]:
        """Rank catalog items against a free-text query, with optional hard
        filters. Filters are applied post-scoring so a near-miss on job_level
        spelling doesn't zero out an otherwise-relevant result silently —
        callers needing strict filtering should check `.assessment` fields
        themselves on the returned pool.

        `include_default_companions`: rides DEFAULT_COMPANIONS into the pool
        regardless of lexical score. These are near-universal companions in
        real assessment batteries by domain convention (a personality and a
        cognitive-ability instrument), not because the query text mentions
        "personality" — pure lexical/semantic similarity to a JD has no way
        to surface them otherwise (their score is ~0 against a query about,
        say, Rust and networking). Confirmed against the gold traces: a
        generic personality instrument shows up in most committed shortlists
        even when never mentioned by the user. The final include/exclude
        call still belongs to the reranking stage; this only guarantees
        they're *available* to choose, the same way a domain-expert recruiter
        would default-include them as candidates worth considering.
        """
        q_vec = self._vectorizer.transform([query])
        sims = linear_kernel(q_vec, self._matrix).flatten()

        exclude_urls = exclude_urls or set()

        def passes_filters(assessment) -> bool:
            # Only ever a hard exclusion for explicit exclude_urls. job_level
            # and language are NOT hard-filtered here — see score boost below
            # for why.
            return assessment.url not in exclude_urls

        def level_language_boost(assessment) -> float:
            # job_level/language come from an LLM's free-text guess, not the
            # catalog's own controlled vocabulary (e.g. catalog uses exactly
            # "Mid-Professional"; a reasonable model guess is "mid-level" —
            # these will never string-match). An earlier version of this
            # method HARD-EXCLUDED non-matches, which meant a routine
            # vocabulary mismatch could silently wipe out the entire
            # otherwise-correct candidate pool (observed in testing: a
            # Java/Spring/SQL query lost every relevant item to this filter
            # over a "mid-level" vs "Mid-Professional" mismatch alone).
            # Treating this as a soft boost instead means a wrong guess only
            # costs some ranking precision, never correctness.
            boost = 1.0
            if job_level and assessment.job_levels:
                boost *= 1.3 if job_level in assessment.job_levels else 1.0
            if language and assessment.languages:
                boost *= 1.3 if language in assessment.languages else 1.0
            return boost

        all_scored = []
        for sim, assessment in zip(sims, self.store.assessments):
            if not passes_filters(assessment):
                continue
            all_scored.append(ScoredAssessment(assessment, float(sim) * level_language_boost(assessment)))
        all_scored.sort(key=lambda s: s.score, reverse=True)

        primary = [
            s for s in all_scored
            if s.score > 0 and (not category_codes or set(s.assessment.test_types) & set(category_codes))
        ]
        pool = primary[:top_k]
        pool_urls = {s.assessment.url for s in pool}

        if include_default_companions:
            for name in DEFAULT_COMPANIONS:
                a = next((s.assessment for s in all_scored if s.assessment.name == name), None)
                if a and a.url not in pool_urls and passes_filters(a):
                    pool.append(ScoredAssessment(a, 0.0))
                    pool_urls.add(a.url)

        pool.sort(key=lambda s: s.score, reverse=True)
        return pool

    def build_candidate_pool(
        self,
        query: str,
        job_level: str | None = None,
        language: str | None = None,
        exclude_urls: set[str] | None = None,
        extra_categories: list[str] | None = None,
        personality_ability_hint: str | None = None,
    ) -> list[Assessment]:
        """Shared pool-construction logic used by both the live agent and
        the offline eval harness, so an eval improvement can never silently
        drift from what production actually does.

        Token-budget note: Personality (66 items) and Ability (32 items)
        often need domain judgment rather than lexical match (a
        safety-critical role needs DSI, not because "safety" appears in the
        query's own words, but because the role context implies it) — but
        those categories are still too large to dump into every GENERATE
        call wholesale on Groq's free tier (~6-12K tokens per call is the
        realistic ceiling; ~100 richly-detailed items alone would burn
        15-20K). Instead, `personality_ability_hint` is a short
        domain-reasoning phrase the ROUTE call produces when it judges the
        role needs something beyond the generic default (e.g. "safety
        dependability integrity reliability" for a HIPAA/trust-sensitive
        role) — a targeted keyword search *within* P/A using that phrase
        surfaces the right instrument at a fraction of the token cost of
        browsing the whole category. This is standard query-expansion: using
        the LLM's domain knowledge to bridge a semantic gap that lexical
        search on the raw user query structurally cannot close, rather than
        brute-forcing coverage by dumping everything."""
        exclude_urls = exclude_urls or set()
        pool = {a.assessment.url: a.assessment for a in self.search(
            query, top_k=12, job_level=job_level, language=language,
            exclude_urls=exclude_urls, include_default_companions=True,
        )}
        for code in extra_categories or []:  # explicit user ask, e.g. "add personality tests"
            for a in self.store.browse_category(code, job_level=job_level)[:10]:
                if a.url not in exclude_urls:
                    pool[a.url] = a
        if personality_ability_hint:
            for code in ("P", "A"):
                for s in self.search(
                    personality_ability_hint, top_k=5, category_codes=[code],
                    job_level=job_level, exclude_urls=exclude_urls,
                ):
                    pool[s.assessment.url] = s.assessment
        return list(pool.values())
