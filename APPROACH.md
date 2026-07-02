# Approach Document — SHL Assessment Recommender

## Design choices

**Two-call architecture per turn: ROUTE then GENERATE.** ROUTE reads the
full conversation and classifies intent (refuse / clarify / compare /
recommend) plus extracts retrieval signal — it never sees catalog data, so
it cannot hallucinate assessment facts, only misjudge user intent. GENERATE
receives a server-built candidate pool and picks/writes from it — it cannot
name anything outside that pool. Splitting these means the strongest
defense against hallucination is architectural (a validated candidate pool),
not a prompt instruction asking the model to behave.

**Stateless memory via self-describing replies.** The API is stateless, so
the *only* memory across turns is our own prior reply text. I made the
agent always append a deterministic, code-generated markdown table
(name/type/duration/URL) whenever a shortlist is delivered — never
LLM-written — so the exact same code that formats it can parse it back out
of history on the next call (`app/history.py`). This guarantees the
structured `recommendations` field and the visible reply text can never
drift apart, and makes "refine" reliable without asking the LLM to
"remember" anything.

**`recommendations` is `null`, never `[]`, when empty — and it's `null`
if-and-only-if no shortlist table is shown this turn.** The assignment PDF's
own example only shows the populated case, but I checked all 10 gold traces
mechanically (line-by-line, every turn) and found a 100%-consistent rule:
the field mirrors table-presence exactly, with zero exceptions, regardless
of whether the turn is a refusal, a clarify, a compare, or a declined edit.
I initially assumed this correlated with *intent category* (e.g. "compare
never shows a table") and found a direct contradiction between two traces
before realizing the real rule is purely presentational. This is now
enforced in code, not left to LLM judgment: the reply text and the
structured field are built from the same object in the same code path.

**Catalog `link` (URL) is the only safe join key.** I cross-checked every
URL referenced across all 10 traces against the catalog — 100% match — but
several display *names* differ in punctuation from the catalog's own
strings (e.g. trace's "SVAR Spoken English (US)" vs. catalog's "SVAR -
Spoken English (US)"). Every recommendation is built from the catalog's own
name/URL by URL lookup, never from whatever string the LLM free-texts.

**Turn-cap is enforced in code, with a two-tier margin.** At message 7 of 8
(the literal "8 turns including user & assistant" from the spec), the
service force-overrides intent to `recommend` even if the LLM says
`clarify`, using best-available context — this is a hard override applied
*after* the LLM call, not just a prompt instruction, so a model that ignores
the instruction still can't violate the cap.

## Retrieval setup

The catalog (377 items) is small enough that a full vector DB is overkill —
an in-memory TF-IDF index (scikit-learn, loads in milliseconds) covers it
with no model weights to ship. I measured retrieval quality directly against
the 10 gold traces (`tests/eval_retrieval.py`, using each trace's cumulative
conversation as the query and its final confirmed shortlist as ground
truth) rather than assuming it worked:

| Pass | Mechanism | Mean pool-coverage |
|---|---|---|
| 1 | Raw TF-IDF, top-10 | 0.535 |
| 2 | + wider pool (20) + default-companion injection | 0.782 |
| 3 | + LLM-generated P/A query-expansion hint | 0.767* |

*(measured with hand-specified hints simulating competent ROUTE-call output,
since the real hint is generated live by the LLM, not by an offline
harness — this measures the mechanism, not model judgment quality)*

Two structural gaps drove most of the improvement: (1) default-companion
instruments (OPQ32r, Verify G+) have ~zero lexical overlap with role
descriptions that never say "personality," so they're injected explicitly by
name — a documented domain rule, not something retrieval "discovered"; (2)
role-context-to-instrument reasoning (a HIPAA/patient-records role implying
"safety, dependability, integrity" — the Dependability & Safety Instrument's
own vocabulary, not the query's) has no lexical bridge at all. I initially
fixed this by browsing the entire Personality (66) and Ability (32)
categories into every candidate pool, which raised coverage further
(0.855) — but at ~103 items per call, that's 15-20K tokens, which would
blow past Groq's free-tier per-call budget (~6-12K tokens depending on
model) and get rejected outright. I replaced it with LLM-driven query
expansion instead: ROUTE produces a short domain-vocabulary phrase (e.g.
"safety dependability integrity reliability") when role context implies a
non-generic instrument, and a narrow keyword search within just P/A uses
that phrase. This keeps the average pool at ~15 items (safely inside budget)
while still closing most of the gap — a real trade-off between recall
ceiling and operational reliability, made deliberately rather than by
accident.

**What didn't work:** relying on raw query lexical similarity alone for
reasoning-based fits (no amount of TF-IDF tuning bridges "graduate financial
analyst" to "Basic Statistics," or "senior Rust engineer" to "Smart
Interview Live Coding" as a catalog-gap substitute) — these remain honest
known limitations rather than something I forced a metric to hide.

## Prompt design

System prompts are intent-conditioned rule lists, not few-shot examples —
I found explicit rules ("populate `user_confirmed_shortlist` only on
explicit acceptance language, not any positive-sounding reply") more
reliable to reason about and debug than pattern-matching from examples,
given the ambiguities already found in the gold data itself. Candidate items
sent to GENERATE use two detail tiers: full detail (description, languages,
job levels) for compare's always-tiny pool, where an accurate answer needs
real fields (one trace's whole point is a language-support distinction);
a trimmed ~180-char-description tier for the larger recommend pool, sized to
fit token budget.

## Evaluation approach

Two independent, LLM-free test suites (`tests/`): `eval_retrieval.py`
measures retrieval quality in isolation against gold shortlists; `
test_http_pipeline.py` replays all 10 traces' user turns through the *real*
FastAPI app via HTTP, with a scripted deterministic mock LLM, asserting
schema compliance, catalog-only URLs, and no internal errors at every one of
the ~40 total turns. This caught a real bug a narrower test missed: an
earlier edit had silently deleted a method's `def` line, leaving its body as
dead code — syntactically valid, so it imported fine, and the top-level
`try/except` in `/chat` caught the resulting `AttributeError` and returned a
schema-valid degraded reply, which a schema-only test would have called a
pass. I added an explicit assertion that the degraded-mode fallback string
never appears in a successful trace replay, specifically to make this class
of bug loud instead of silent.

## AI tool disclosure

Built in an agentic coding session with Claude (Anthropic), which wrote the
initial implementation across all modules under direction of a person
reviewing each design decision. All figures in this document were generated
by actually running the code against the provided catalog and traces, not
estimated — every claim above is reproducible via the two test scripts
above.
