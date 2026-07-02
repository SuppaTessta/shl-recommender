# Approach Document — SHL Assessment Recommender

## Design choices

**Two-call architecture per turn: ROUTE then GENERATE.** ROUTE classifies
intent (refuse / clarify / compare / recommend) and extracts retrieval
signal from the conversation — it never sees catalog data, so it can't
hallucinate assessment facts, only misjudge intent. GENERATE receives a
server-built candidate pool and can only pick/write from it. The strongest
defense against hallucination is architectural — a validated pool — not a
prompt asking the model to behave.

**Stateless memory via self-describing replies.** The only memory across
turns is our own prior reply text. The agent always appends a
deterministic, code-generated markdown table whenever a shortlist is
delivered — never LLM-written — so the same code that formats it can parse
it back out of history next call (`app/history.py`). This guarantees
`recommendations` and the visible reply can never drift apart, and makes
refine reliable without asking the LLM to "remember" anything.

**`recommendations` is `null`, never `[]`, when empty — and it's `null`
if-and-only-if no shortlist table is shown this turn.** The PDF's own
example only shows the populated case, but checking all 10 gold traces
turn-by-turn found a 100%-consistent mechanical rule: the field mirrors
table-presence exactly, regardless of whether the turn is a refusal,
clarify, compare, or declined edit. I initially assumed this tracked
*intent category* and found a direct contradiction between two traces
before realizing the real rule is purely presentational — now enforced in
code (reply text and the structured field come from the same object), not
left to LLM judgment.

**Catalog `link` (URL) is the only safe join key.** Every URL referenced
across all 10 traces matches the catalog exactly, but several display
*names* differ in punctuation (e.g. trace's "SVAR Spoken English (US)" vs.
catalog's "SVAR - Spoken English (US)"). Every recommendation is built from
the catalog's own name/URL by URL lookup, never from LLM free text.

**Turn-cap enforced in code, not just prompted.** At message 7 of 8 (the
spec's "8 turns including user & assistant"), the service force-overrides
intent to `recommend` even if the LLM says `clarify` — applied *after* the
LLM call, so a model that ignores the instruction still can't violate the
cap.

## Retrieval setup

The catalog (377 items) is small enough that a full vector DB is overkill —
an in-memory TF-IDF index (scikit-learn, loads in milliseconds, no model
weights to ship) covers it. I measured retrieval directly against the 10
gold traces (`tests/eval_retrieval.py`, cumulative conversation as query,
final confirmed shortlist as ground truth) rather than assuming it worked:

| Pass | Mechanism | Mean pool-coverage |
|---|---|---|
| 1 | Raw TF-IDF, top-10 | 0.535 |
| 2 | + wider pool (20) + default-companion injection | 0.782 |
| 3 | + LLM-generated P/A query-expansion hint | 0.767* |

*(hand-specified hints simulating competent ROUTE output — measures the
mechanism, not live model judgment)*

Two structural gaps drove the improvement: (1) default-companion
instruments (OPQ32r, Verify G+) have ~zero lexical overlap with roles that
never say "personality," so they're injected explicitly by name — a
documented domain rule, not something retrieval "discovered"; (2)
role-to-instrument reasoning (HIPAA/patient-records implying "safety,
dependability, integrity" — the Dependability & Safety Instrument's own
vocabulary, not the query's) has no lexical bridge at all. Browsing the
entire Personality (66) + Ability (32) categories into every pool raised
coverage further (0.855), but at ~103 items/call that's 15-20K tokens —
past Groq's free-tier per-call budget (~6-12K tokens), rejected outright.
Replaced with LLM-driven query expansion instead: ROUTE produces a short
domain-vocabulary phrase (e.g. "safety dependability integrity
reliability") when role context implies a non-generic instrument, narrow-
searched within just P/A. Keeps the pool at ~15 items while closing most of
the gap — recall ceiling traded for operational reliability, deliberately.

**What didn't work:** raw lexical similarity for reasoning-based fits (no
TF-IDF tuning bridges "graduate financial analyst" to "Basic Statistics,"
or "senior Rust engineer" to "Smart Interview Live Coding" as a catalog-gap
substitute) — honest known limitations, not hidden behind a metric.

## Prompt design

System prompts are intent-conditioned rule lists, not few-shot examples —
explicit rules ("populate `user_confirmed_shortlist` only on explicit
acceptance language") proved more debuggable than pattern-matching, given
ambiguities already found in the gold data. Candidates sent to GENERATE use
two detail tiers: full detail for compare's always-tiny pool (one trace
hinges on a language-support distinction, so truncated fields would lose
the answer); a trimmed ~180-char-description tier for the larger recommend
pool, sized to fit token budget.

## Evaluation approach

Two independent, LLM-free test suites: `eval_retrieval.py` measures
retrieval quality in isolation against gold shortlists; `
test_http_pipeline.py` replays all 10 traces' user turns through the *real*
FastAPI app via HTTP with a scripted mock LLM, asserting schema compliance,
catalog-only URLs, and no internal errors across ~40 total turns. This
caught a real bug a narrower test would miss: an earlier edit had silently
deleted a method's `def` line, leaving its body as dead code — syntactically
valid, so it imported fine, and `/chat`'s top-level `try/except` caught the
resulting `AttributeError` and returned a schema-valid degraded reply, which
a schema-only test would call a pass. I added an explicit assertion that the
degraded-mode fallback string never appears in a successful replay, to make
this class of bug loud instead of silent.

**A second, more serious bug only surfaced under live testing against the
real model**, after both automated suites were already green — a reminder
that mock-LLM tests validate the *pipeline*, not model behavior. A live
query for "Java developer, mid-level, Spring and SQL" returned only one
generic, unrelated assessment, the model explicitly claiming no Java/Spring
/SQL match existed. Direct inspection showed retrieval *had* surfaced the
correct items — the live candidate pool was silently different from what my
offline testing used. Cause: `job_level` was passed from ROUTE's free-text
guess ("mid-level") directly into a hard, exact-string filter against the
catalog's own controlled vocabulary ("Mid-Professional") — the mismatch
hard-excluded nearly the entire relevant pool, leaving only untagged items.
A routine vocabulary mismatch was silently destroying correctness on most
recommend turns, not just this one. Fixed two ways: ROUTE now receives the
catalog's exact 10-value vocabulary instead of guessing, and — more
importantly — job-level/language matching became a soft ranking boost
instead of a hard filter, so a future mismatch can only cost ranking
precision, never correctness. Re-verified against the same failing input
before and after the fix.

## AI tool disclosure

Built in an agentic coding session with Claude (Anthropic), which wrote the
initial implementation across all modules under direction of a person
reviewing each design decision. All figures in this document were generated
by actually running the code against the provided catalog and traces, not
estimated — every claim above is reproducible via the two test scripts
above.
