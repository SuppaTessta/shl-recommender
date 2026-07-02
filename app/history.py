"""
The API is stateless — every /chat call gets the full history and nothing
else. Our own previous replies are the only memory that exists. This module
recovers the "currently committed shortlist" deterministically by parsing
markdown tables out of prior assistant messages, rather than asking the LLM
to remember/re-derive it from prose. This is the load-bearing trick that
keeps multi-turn refine ("actually, add personality tests") correct even
under a fresh, stateless LLM call each turn — and it's robust precisely
because it doesn't depend on the LLM being reliable.

Contract: whenever the agent delivers or reconfirms a shortlist, the reply
text MUST contain a markdown table with a URL per row, in this exact form
(see prompts.py / agent.py `format_shortlist_table`). This function is the
reverse of that formatter.
"""
from __future__ import annotations

import re

from .catalog import Assessment, CatalogStore
from .schemas import ChatMessage

URL_RE = re.compile(r"<(https://www\.shl\.com/[^\s>]+)>")


def extract_shortlist_urls(reply_text: str) -> list[str]:
    """Pull URLs out of the LAST contiguous block of markdown-table lines
    in this text, in row order. Returns [] if no table is present.

    Line-based rather than a single greedy regex: a regex requiring a
    trailing "\\n" after every row silently drops the final row whenever the
    table is the last thing in the message (no trailing newline) — exactly
    the common case, since replies end right after the table."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in reply_text.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        else:
            if current:
                blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    if not blocks:
        return []
    last_block = "\n".join(blocks[-1])
    return URL_RE.findall(last_block)


def reconstruct_committed_shortlist(
    messages: list[ChatMessage], store: CatalogStore
) -> list[Assessment]:
    """Scan assistant turns newest-to-oldest; the most recent one containing
    a shortlist table IS the current committed state (a later prose-only
    reply, e.g. a compare answer, does not erase it — it just doesn't touch
    it, matching the mechanical rule verified against all 10 gold traces:
    `recommendations` mirrors table-presence per turn, and an untouched
    shortlist persists silently in the background between table-bearing
    turns)."""
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        urls = extract_shortlist_urls(msg.content)
        if urls:
            resolved = [store.get_by_url(u) for u in urls]
            return [a for a in resolved if a is not None]
    return []


def format_shortlist_table(items: list[Assessment]) -> str:
    """Deterministic, code-generated markdown table — never LLM-generated —
    so structured `recommendations` and visible reply text can never drift
    apart, and so `reconstruct_committed_shortlist` can parse it back
    reliably on the next turn."""
    header = "| # | Name | Test Type | Duration | URL |\n|---|------|-----------|----------|-----|\n"
    rows = []
    for i, a in enumerate(items, start=1):
        duration = a.duration_raw or "—"
        rows.append(f"| {i} | {a.name} | {','.join(a.test_types)} | {duration} | <{a.url}> |")
    return header + "\n".join(rows)
