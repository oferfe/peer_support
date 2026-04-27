"""Paper-grounded persona guidelines: load and expose.

The guideline content lives in `data/persona_guidelines.json` as a flat
`{"papers": [...]}` object. Each paper owns bibliographic fields
(`paper_citation`, `paper_summary`) and a list of `criteria`, where each
criterion in turn owns `criterion`, `explanation`, and
`citations_from_the_paper` (a list of verbatim quote strings).

The module is deliberately minimal in this first pass — English-only, no
language logic, no randomization, no prompt construction. The prompt builder
in `utils.intake.build_biography_prompt` reads the papers via `get_papers`
and renders them into the biography-drafting prompt.

Public API mirrors `utils.intake` / `utils.questionnaire` so the app can
treat all three assets alike:

- `load_guidelines()`   cached read of the JSON.
- `get_papers()`        returns the `papers[]` array for downstream use.

Hebrew is deferred to mini-step 14.7: once the researcher has authored the
English content, each authored string (`paper_summary`, `criterion`,
`explanation`) is wrapped into `{ "en": "...", "he": "..." }` and a
`language` parameter is reintroduced on `get_papers`. `paper_citation` and
`citations_from_the_paper` stay in their source language even after that
migration — citations and verbatim quotes are conventionally not translated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


_GUIDELINES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "persona_guidelines.json"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_guidelines() -> dict[str, Any]:
    """Read and cache the raw guidelines JSON from disk.

    Raises `FileNotFoundError` if the file is missing, and `ValueError` if
    the top-level `papers` key is absent (fail loudly rather than silently
    skipping the guidelines block).
    """
    if not _GUIDELINES_PATH.exists():
        raise FileNotFoundError(
            f"Guidelines file not found: {_GUIDELINES_PATH}"
        )
    with _GUIDELINES_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "papers" not in data:
        raise ValueError(
            "persona_guidelines.json must be an object with a 'papers' key"
        )
    return data


# ---------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------


def get_papers(guidelines: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the `papers` array from a loaded guidelines dict.

    Returns an empty list when `papers` is missing or not a list, so the
    prompt builder can gracefully emit no guideline block rather than
    crash — matching the "temporarily empty the array" smoke-test in the
    plan.
    """
    papers = guidelines.get("papers")
    if not isinstance(papers, list):
        return []
    return papers
