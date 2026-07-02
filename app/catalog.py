"""
Loads the SHL Individual Test Solutions catalog, repairs known scrape artifacts,
and exposes lookup helpers. The catalog `link` is the canonical join key —
trace data shows free-text names occasionally drift in punctuation/spacing
(e.g. "SVAR Spoken English (US)" vs catalog's "SVAR - Spoken English (US)"),
but URLs match exactly across every reference we've checked.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Single-letter test-type codes used in the API response schema, keyed by the
# human-readable category names the catalog stores.
CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

_DURATION_RE = re.compile(r"(\d+)")

# Known scrape artifacts: raw control characters split a field across lines,
# clobbering part of the text. Repair by entity_id rather than guessing from
# the mangled string, since the link slug gives us the true name unambiguously.
_NAME_REPAIRS = {
    "4207": "Microsoft Excel 365 (New)",
}


@dataclass
class Assessment:
    id: str
    name: str
    url: str
    description: str
    job_levels: list[str]
    languages: list[str]
    duration_raw: str
    duration_minutes: int | None
    remote_testing: bool
    adaptive_irt: bool
    categories: list[str]
    test_types: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.test_types:
            self.test_types = [CATEGORY_TO_CODE[c] for c in self.categories if c in CATEGORY_TO_CODE]

    def search_text(self) -> str:
        """Flattened text blob used for keyword/TF-IDF indexing."""
        parts = [
            self.name, self.name,  # weight name 2x
            self.description,
            " ".join(self.categories),
            " ".join(self.job_levels),
        ]
        return " ".join(p for p in parts if p)


def _parse_duration(raw: str) -> int | None:
    if not raw:
        return None
    m = _DURATION_RE.search(raw)
    return int(m.group(1)) if m else None


def _clean_name(entity_id: str, raw_name: str) -> str:
    if entity_id in _NAME_REPAIRS:
        return _NAME_REPAIRS[entity_id]
    # Defensive fallback for any other mangled entries we haven't seen yet:
    # collapse embedded control chars / repeated whitespace.
    return re.sub(r"\s+", " ", raw_name).strip()


def load_catalog(path: str | Path) -> list[Assessment]:
    raw_text = Path(path).read_text(encoding="utf-8")
    # strict=False: the source JSON has raw control characters embedded in a
    # couple of string fields (scrape artifact), which is invalid per the JSON
    # spec but tolerated by Python's lenient mode.
    rows = json.loads(raw_text, strict=False)

    catalog: list[Assessment] = []
    seen_urls: set[str] = set()
    for row in rows:
        url = row["link"].strip()
        if url in seen_urls:
            continue  # defensive: skip duplicate scrape rows, keep first
        seen_urls.add(url)
        catalog.append(
            Assessment(
                id=row["entity_id"],
                name=_clean_name(row["entity_id"], row["name"]),
                url=url,
                description=re.sub(r"\s+", " ", row.get("description", "")).strip(),
                job_levels=row.get("job_levels", []),
                languages=row.get("languages", []),
                duration_raw=row.get("duration", ""),
                duration_minutes=_parse_duration(row.get("duration", "")),
                remote_testing=row.get("remote") == "yes",
                adaptive_irt=row.get("adaptive") == "yes",
                categories=row.get("keys", []),
            )
        )
    return catalog


class CatalogStore:
    """In-memory catalog with lookup helpers. Loaded once at process startup."""

    def __init__(self, assessments: list[Assessment]):
        self.assessments = assessments
        self.by_url = {a.url: a for a in assessments}
        self.by_id = {a.id: a for a in assessments}

    @classmethod
    def from_file(cls, path: str | Path) -> "CatalogStore":
        return cls(load_catalog(path))

    def __len__(self) -> int:
        return len(self.assessments)

    def get_by_url(self, url: str) -> Assessment | None:
        return self.by_url.get(url.strip())

    def browse_category(self, code: str, job_level: str | None = None) -> list[Assessment]:
        """List every item in a test-type category (e.g. all 'P' personality
        instruments). Complements keyword retrieval: queries like "trust-
        sensitive healthcare role" or "leadership benchmark" have almost no
        lexical overlap with the catalog item that actually fits (DSI, OPQ
        Leadership Report), because the right answer is a domain-reasoning
        call over a short list of options, not a text-similarity match. A
        category has at most ~70 items, small enough for an LLM to reason
        over directly once narrowed this way."""
        items = [a for a in self.assessments if code in a.test_types]
        if job_level:
            items = [a for a in items if not a.job_levels or job_level in a.job_levels]
        return items

    def resolve_name(self, name: str, threshold: int = 80) -> Assessment | None:
        """Best-effort name -> Assessment resolution for free-text LLM output
        or user-typed names (e.g. in compare requests). Exact match first,
        then fuzzy. Returns None below threshold rather than guessing."""
        from rapidfuzz import fuzz, process

        name = name.strip()
        for a in self.assessments:
            if a.name.lower() == name.lower():
                return a
        match = process.extractOne(
            name, {a.id: a.name for a in self.assessments}.items(),
            scorer=fuzz.WRatio,
            processor=lambda x: x[1].lower() if isinstance(x, tuple) else x.lower(),
        )
        if match and match[1] >= threshold:
            matched_id = match[0][0]
            return self.by_id[matched_id]
        return None


if __name__ == "__main__":
    store = CatalogStore.from_file("data/catalog_raw.json")
    print(f"Loaded {len(store)} assessments")
    fixed = store.by_id["4207"]
    print("Repaired name check:", fixed.name, "|", fixed.url)
    print("Sample test_types:", fixed.test_types)
