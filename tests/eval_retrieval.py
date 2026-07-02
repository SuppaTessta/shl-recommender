"""
Parses the 10 provided trace .md files, pulls out (a) all user turns as the
conversational query and (b) the FINAL markdown table as the ground-truth
expected shortlist, then measures pure-retrieval recall@10 — i.e. how good
the candidate pool is *before* any LLM re-ranking touches it.

This is deliberately retrieval-only: if recall@10 is bad here, no amount of
clever prompting later will fix it, since the LLM can only choose from what
retrieval hands it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.catalog import CatalogStore
from app.history import extract_shortlist_urls
from app.retrieval import RetrievalIndex

TRACE_DIR = Path(__file__).resolve().parent / "traces"
USER_TURN_RE = re.compile(r"\*\*User\*\*\s*\n\n>\s*(.+?)(?=\n\n\*\*Agent\*\*)", re.DOTALL)


def parse_trace(path: Path) -> tuple[str, set[str]]:
    text = path.read_text(encoding="utf-8")

    user_turns = [re.sub(r"\s+", " ", m.strip()) for m in USER_TURN_RE.findall(text)]
    query = " ".join(user_turns)

    # Ground truth = the LAST shortlist table in the whole trace (the final
    # committed/confirmed state), parsed with the exact same function that
    # runs in production — dogfooding means this eval catches regressions in
    # the real parser, not a second hand-rolled implementation that could
    # drift from it.
    expected_urls = set(extract_shortlist_urls(text))
    return query, expected_urls


def recall_at_k(retrieved_urls: list[str], expected: set[str], k: int = 10) -> float:
    if not expected:
        return float("nan")
    top_k = set(retrieved_urls[:k])
    return len(top_k & expected) / len(expected)


def main():
    store = CatalogStore.from_file(Path(__file__).resolve().parent.parent / "data" / "catalog_raw.json")
    index = RetrievalIndex(store)

    print("=== Pass 1: raw TF-IDF, pool=10, no companions ===")
    run_pass(store, index, lambda q: {a.assessment.url for a in index.search(q, top_k=10)})

    print("\n=== Pass 2: wider pool (20) + default companions ===")
    run_pass(store, index, lambda q: {a.assessment.url for a in index.search(
        q, top_k=20, include_default_companions=True)})

    print("\n=== Pass 3: production pool, no P/A hint (retrieval-only ceiling) ===")
    sizes = []
    def prod_pool_no_hint(q):
        items = index.build_candidate_pool(q)
        sizes.append(len(items))
        return {a.url for a in items}
    run_pass(store, index, prod_pool_no_hint)
    print(f"Avg pool size: {sum(sizes)/len(sizes):.0f} items (token-budget safe for Groq free tier)")

    print("\n=== Pass 4: production pool WITH hand-specified P/A hints ===")
    print("(hints simulate what a competent ROUTE call should produce per trace's role")
    print(" context — this measures the query-expansion MECHANISM, not LLM judgment quality,")
    print(" since the real hint is generated live by the LLM at request time, not by this")
    print(" offline harness)")
    # One hint per trace, written from the same role-context reasoning a
    # human would apply — e.g. C7's healthcare/HIPAA context implies
    # "safety dependability integrity", the same bridge a capable ROUTE
    # call is instructed to make (see prompts.py ROUTE_SYSTEM).
    HINTS = {
        "C1": "leadership executive strategic influence",
        "C6": "safety dependability integrity reliability",
        "C7": "safety dependability integrity reliability",
        "C10": "graduate scenarios situational judgment",
    }
    sizes2 = []
    def prod_pool_with_hint(q, trace_id):
        items = index.build_candidate_pool(q, personality_ability_hint=HINTS.get(trace_id))
        sizes2.append(len(items))
        return {a.url for a in items}
    run_pass(store, index, prod_pool_with_hint, hinted=True)
    print(f"Avg pool size: {sum(sizes2)/len(sizes2):.0f} items")


def run_pass(store, index, pool_fn, hinted=False):
    recalls = []
    print(f"{'trace':<8}{'expected':<10}{'found':<8}{'coverage':<10}missing")
    for path in sorted(TRACE_DIR.glob("C*.md"), key=lambda p: int(p.stem[1:])):
        query, expected = parse_trace(path)
        if not expected:
            continue
        retrieved_urls = pool_fn(query, path.stem) if hinted else pool_fn(query)
        r = len(retrieved_urls & expected) / len(expected)
        recalls.append(r)
        missing = expected - retrieved_urls
        missing_names = [store.get_by_url(u).name if store.get_by_url(u) else u for u in missing]
        print(f"{path.stem:<8}{len(expected):<10}{len(expected & retrieved_urls):<8}{r:<10.2f}{missing_names}")
    print(f"Mean pool-coverage: {sum(recalls)/len(recalls):.3f}")
    return recalls


if __name__ == "__main__":
    main()
