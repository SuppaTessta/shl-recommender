"""
A cheap regex net for the most blatant prompt-injection phrasing, checked
before the LLM is even called. This is deliberately NOT the primary
defense — regex can't reliably catch injection, and being over-aggressive
here would misfire on legitimate messages (e.g. "ignore the previous
shortlist, add personality tests" is a normal refine request that happens
to contain the word "ignore"). The real defense is architectural: the
GENERATE call can only ever select from a server-validated candidate pool,
so even a fully successful jailbreak of the routing/generation prompts
can't make the service emit a fabricated or off-catalog recommendation —
worst case is off-topic *text*, not a corrupted result. This filter only
exists to short-circuit the obvious cases cheaply, saving a wasted LLM
round trip within the 30s budget.
"""
import re

_PATTERNS = [
    r"\bignore (all |your |the )?(previous|prior|above|system) instructions\b",
    r"\breveal (your|the) (system )?prompt\b",
    r"\byou are now\b.{0,30}\b(dan|jailbroken|unrestricted)\b",
    r"\bact as (if you (had|have) no|an unfiltered)\b",
    r"\bdisregard (all|your) (rules|guidelines|instructions)\b",
    r"\bprint (your|the) (system|initial) prompt\b",
]
_COMPILED = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def looks_like_injection(text: str) -> bool:
    return bool(_COMPILED.search(text))
