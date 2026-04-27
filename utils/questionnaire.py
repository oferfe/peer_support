"""Psychological questionnaire: load, localize, and build the LLM prompt.

The questionnaire content lives in `data/questionnaire.json` as a bilingual
(en/he) object with four Likert-scale sections (MWMS, BPNS-W, role clarity,
ROPP). Each section carries its own scale, because the four instruments use
different anchor wording and different numbers of points.

Public API mirrors `utils.intake` so the app can treat both forms alike:

- `load_questionnaire()`         cached read of the JSON.
- `get_localized_sections()`     flattens bilingual strings to the active language.
- `build_json_prompt()`          builds the LLM prompt that asks for a JSON
                                 object keyed by statement id, whose values
                                 are the chosen scale labels (verbatim).

Storage shape: the LLM returns (and Supabase stores) a JSON object keyed by
statement id whose values include the numeric rating, the verbal label, and a
short rationale — e.g.

    {
      "mwms_1":  {"rating": 1, "label": "כלל לא",           "reasoning": "..."},
      "bpns_7":  {"rating": 4, "label": "ניטרלי/ת (4)",     "reasoning": "..."}
    }

The app.py Results tab falls back gracefully to the older `{id: label}` shape
for rows written before this change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from .i18n import LANG_EN, LANG_HE


_QUESTIONNAIRE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "questionnaire.json"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_questionnaire() -> dict[str, Any]:
    """Read and cache the raw questionnaire JSON from disk."""
    if not _QUESTIONNAIRE_PATH.exists():
        raise FileNotFoundError(
            f"Questionnaire file not found: {_QUESTIONNAIRE_PATH}"
        )
    with _QUESTIONNAIRE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "sections" not in data:
        raise ValueError(
            "questionnaire.json must be an object with a 'sections' key"
        )
    return data


# ---------------------------------------------------------------------------
# Localization
# ---------------------------------------------------------------------------


def _pick(bilingual: dict[str, str] | str, language: str) -> str:
    """Pick a localized string from a `{en, he}` dict with safe fallbacks."""
    if isinstance(bilingual, str):
        return bilingual
    if not isinstance(bilingual, dict):
        return ""
    return (
        bilingual.get(language)
        or bilingual.get(LANG_EN)
        or bilingual.get(LANG_HE)
        or ""
    )


def get_localized_sections(
    questionnaire: dict[str, Any], language: str
) -> list[dict[str, Any]]:
    """Return sections with every user-facing string already picked.

    Output shape per section:
        {
            "id":    str,
            "title": str,
            "scale": list[str],
            "questions": [{"id": str, "question": str}, ...],
        }
    """
    lang = language if language in (LANG_EN, LANG_HE) else LANG_EN
    out: list[dict[str, Any]] = []
    for section in questionnaire.get("sections", []):
        scale_raw = section.get("scale") or []
        out.append(
            {
                "id": section.get("id", ""),
                "title": _pick(section.get("title", ""), lang),
                "scale": [_pick(label, lang) for label in scale_raw],
                "questions": [
                    {
                        "id": q["id"],
                        "question": _pick(q.get("question", ""), lang),
                    }
                    for q in section.get("questions", [])
                ],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    LANG_EN: "Respond entirely in English.",
    LANG_HE: "Respond entirely in Hebrew (עברית).",
}


def build_json_prompt(
    biography_text: str,
    questionnaire: dict[str, Any],
    language: str = LANG_EN,
) -> str:
    """Build the prompt that asks the LLM to answer the full questionnaire.

    For every statement in every section the model must:
      1. Pick a numeric rating on that section's scale (1..N, where N varies
         per section — e.g. MWMS and BPNS-W use 1..7, Role clarity and ROPP
         use 1..6).
      2. Copy the matching verbal label verbatim from the scale.
      3. Provide a short in-character rationale (1-2 sentences).

    Returns a single string suitable for OpenAI's `response_format=json_object`
    mode or Gemma's `response_mime_type="application/json"`.
    """
    sections = get_localized_sections(questionnaire, language)
    if not sections:
        raise ValueError("questionnaire has no sections")

    language_instruction = _LANGUAGE_INSTRUCTIONS.get(
        language, _LANGUAGE_INSTRUCTIONS[LANG_EN]
    )

    blocks: list[str] = []
    for section in sections:
        scale_lines = "\n".join(
            f'  {i}. "{label}"'
            for i, label in enumerate(section["scale"], start=1)
        )
        item_lines = "\n".join(
            f"- {q['id']}: {q['question']}" for q in section["questions"]
        )
        blocks.append(
            f"### {section['title']}\n"
            f"Scale for this section (pick one rating 1..{len(section['scale'])} "
            f"and copy its label verbatim):\n{scale_lines}\n"
            f"Statements:\n{item_lines}"
        )

    all_ids = [
        q["id"] for section in sections for q in section["questions"]
    ]
    id_hint = ", ".join(all_ids[:4]) + ", ..."

    return (
        "You are role-playing as the person described in the biography below.\n"
        "Answer every statement IN CHARACTER, in first person. For each\n"
        "statement, pick the rating on that section's scale that best\n"
        "reflects how the character would respond, and briefly explain why\n"
        "in the character's voice.\n"
        f"{language_instruction}\n"
        "\n"
        "## Biography\n"
        f"{biography_text.strip()}\n"
        "\n"
        "## Questionnaire\n"
        + "\n\n".join(blocks)
        + "\n\n"
        "## Output format\n"
        "Respond with ONLY a valid JSON object. No prose, no markdown, no\n"
        "code fences. The keys MUST be the statement ids (e.g. "
        f"{id_hint}). Each value MUST be an object with exactly three\n"
        "fields:\n"
        '  - "rating":    integer in the section\'s range (1..N).\n'
        '  - "label":     the scale label for that rating, copied verbatim\n'
        "                (including any number in parentheses).\n"
        '  - "reasoning": 1-2 sentences, in character, explaining the choice.\n'
        "The rating and label MUST be consistent — i.e. label must be the\n"
        "Nth entry in that section's scale when rating is N."
    )
