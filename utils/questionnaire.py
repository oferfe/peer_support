"""Psychological questionnaire: load, localize, and build the LLM prompt.

The questionnaire content lives in `data/questionnaire.json` as a bilingual
(en/he) object with four Likert-scale sections (MWMS, BPNS-W, role clarity,
ROPP). Each section carries its own scale, because the four instruments use
different anchor wording and different numbers of points.

Public API mirrors `utils.intake` so the app can treat both forms alike:

- `load_questionnaire()`         cached read of the JSON.
- `get_localized_sections()`     flattens bilingual strings to the active language.
- `build_character_system_prompt()`
                                 builds the system prompt that defines the
                                 character from the biography.
- `build_simulation_biography_prompt()`
                                 builds the conversion prompt that turns the
                                 saved biography into a direct persona prompt.
- `build_json_prompt()`          builds the first LLM prompt: answer as the
                                 character with ratings/labels only.
- `build_explanation_prompt()`   builds the second LLM prompt: explain the
                                 first LLM answers using academic context.

Storage shape: the app stores a JSON object keyed by statement id whose values
include the numeric rating, the verbal label, and a short explanation — e.g.

    {"mwms_1": {"rating": 1, "label": "כלל לא", "reasoning": "..."}}

The rating/label are produced by a persona LLM call. The explanation is
produced by a second expert LLM call using the biography, the first answers,
the model's peer-support knowledge, and `data/persona_guidelines.json`.

The app.py Results tab falls back gracefully to the older `{id: label}` shape
for rows written before this change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from . import guidelines
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


def _build_academic_context_block() -> str:
    """Render compact paper context for questionnaire-answer explanations."""
    try:
        data = guidelines.load_guidelines()
    except (FileNotFoundError, ValueError):
        return ""

    paper_blocks: list[str] = []
    for paper in guidelines.get_papers(data):
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("id") or "paper").strip()
        citation = str(paper.get("paper_citation") or "").strip()
        summary = str(paper.get("paper_summary") or "").strip()
        criteria_lines: list[str] = []
        for criterion in paper.get("criteria") or []:
            if not isinstance(criterion, dict):
                continue
            title = str(criterion.get("criterion") or "").strip()
            explanation = str(criterion.get("explanation") or "").strip()
            if title and explanation:
                criteria_lines.append(f"- {title}: {explanation}")
            elif title:
                criteria_lines.append(f"- {title}")
        parts = [f"### Paper {paper_id}"]
        if citation:
            parts.append(f"Citation: {citation}")
        if summary:
            parts.append(f"Summary: {summary}")
        if criteria_lines:
            parts.append("Relevant criteria:\n" + "\n".join(criteria_lines))
        if len(parts) > 1:
            paper_blocks.append("\n".join(parts))
    return "\n\n".join(paper_blocks)


def build_character_system_prompt(
    biography_text: str,
    language: str = LANG_EN,
) -> str:
    """Build the system prompt that defines the answering character."""
    return biography_text.strip()


def build_simulation_biography_prompt(
    biography_text: str,
    language: str = LANG_EN,
) -> str:
    """Build the prompt that prepares a biography for persona simulation."""
    lang = language if language in (LANG_EN, LANG_HE) else LANG_EN
    if lang == LANG_HE:
        instruction = (
            "המר/י את הביוגרפיה הבאה לפרומפט מערכת ישיר בגוף שני עבור "
            "סימולציה של פרסונה. הפרומפט צריך להישמע כמו: "
            "\"אתה אלון. נולדת...\" או \"את יעל. נולדת...\" בהתאם לטקסט. "
            "שמר/י את כל הפרטים העובדתיים, אל תוסיף/י עובדות חדשות, אל "
            "תסיר/י פרטים חשובים, ואל תסביר/י את ההמרה. החזר/י רק את "
            "פרומפט הפרסונה המומר בעברית."
        )
        header = "## ביוגרפיה לשימור ולהמרה"
    else:
        instruction = (
            "Convert the following biography into a direct second-person "
            "persona system prompt for simulation. The result should read "
            "like: \"You are Alon. You were born...\" Preserve every factual "
            "detail, do not add new facts, do not remove important details, "
            "and do not explain the conversion. Return only the converted "
            "persona prompt text in English."
        )
        header = "## Biography to preserve and convert"

    return f"{instruction}\n\n{header}\n{biography_text.strip()}"


def build_json_prompt(
    questionnaire: dict[str, Any],
    language: str = LANG_EN,
) -> str:
    """Build the questionnaire instruction prompt for the character LLM.

    This is the first LLM call. For every statement the character must:
      1. Pick a numeric rating on that section's scale (1..N, where N varies
         per section — e.g. MWMS and BPNS-W use 1..7, Role clarity and ROPP
         use 1..6).
      2. Copy the matching verbal label verbatim from the scale.

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
        "You are a survey respondent. For each statement, pick\n"
        "the rating on that section's scale that best reflects you.\n"
        f"{language_instruction}\n"
        "\n"
        "## Questionnaire\n"
        + "\n\n".join(blocks)
        + "\n\n"
        "## Output format\n"
        "Respond with ONLY a valid JSON object. No prose, no markdown, no\n"
        "code fences. The keys MUST be the statement ids (e.g. "
        f"{id_hint}). Each value MUST be an object with exactly two\n"
        "fields:\n"
        '  - "rating":    integer in the section\'s range (1..N).\n'
        '  - "label":     the scale label for that rating, copied verbatim\n'
        "                (including any number in parentheses).\n"
        "The rating and label MUST be consistent — i.e. label must be the\n"
        "Nth entry in that section's scale when rating is N."
    )


def build_explanation_prompt(
    biography_text: str,
    questionnaire: dict[str, Any],
    answers: dict[str, Any],
    language: str = LANG_EN,
) -> str:
    """Build the second-LLM prompt that explains fixed persona answers."""
    sections = get_localized_sections(questionnaire, language)
    if not sections:
        raise ValueError("questionnaire has no sections")

    language_instruction = _LANGUAGE_INSTRUCTIONS.get(
        language, _LANGUAGE_INSTRUCTIONS[LANG_EN]
    )
    academic_context = _build_academic_context_block()

    question_lines: list[str] = []
    for section in sections:
        question_lines.append(f"### {section['title']}")
        for q in section["questions"]:
            qid = q["id"]
            answer = answers.get(qid)
            if isinstance(answer, dict):
                rating = answer.get("rating", "")
                label = answer.get("label", "")
                answer_text = f"rating={rating}, label={label}"
            else:
                answer_text = str(answer)
            question_lines.append(
                f"- {qid}: {q['question']}\n  First LLM answer: {answer_text}"
            )

    all_ids = [
        q["id"] for section in sections for q in section["questions"]
    ]
    id_hint = ", ".join(all_ids[:4]) + ", ..."

    return (
        "You are an expert analyst of mental health peer support. Your task is to explain the persona answers.\n"
        f"{language_instruction}\n\n"
        "For each answer, write an explanation why this\n"
        "answer is plausible. Base the explanation on: (1) the biography,\n"
        "(2) your professional knowledge of peer support and mental health services,\n"
        "and (3) the academic context below from persona_guidelines.json. If "
        "you use information from a paper, cite the paper by its citation. "
        "Do not change the answer, only explain why it is plausible.\n"
        "## Biography\n"
        f"{biography_text.strip()}\n\n"
        "## First LLM answers to explain\n"
        + "\n".join(question_lines)
        + "\n\n"
        "## Academic peer-support context\n"
        + (academic_context or "No academic context was available.")
        + "\n\n"
        "## Output format\n"
        "Respond with ONLY a valid JSON object. No prose, no markdown, no code\n"
        f"fences. The keys MUST be the statement ids (e.g. {id_hint}). Each\n"
        "value MUST be a plain string explanation."
    )


def _answer_summary(answer: Any) -> str:
    """Return a compact answer summary for explanation prompts."""
    if isinstance(answer, dict):
        rating = answer.get("rating")
        label = answer.get("label")
        if rating is not None and label:
            return f"rating={rating}, label={label}"
        if label:
            return str(label)
        if rating is not None:
            return str(rating)
    return str(answer)


def build_change_explanation_prompt(
    *,
    current_biography_text: str,
    previous_biography_text: str,
    questionnaire: dict[str, Any],
    current_answers: dict[str, Any],
    previous_answers: dict[str, Any],
    language: str = LANG_EN,
) -> str:
    """Build an expert prompt explaining why changed answers changed."""
    sections = get_localized_sections(questionnaire, language)
    if not sections:
        raise ValueError("questionnaire has no sections")

    language_instruction = _LANGUAGE_INSTRUCTIONS.get(
        language, _LANGUAGE_INSTRUCTIONS[LANG_EN]
    )
    academic_context = _build_academic_context_block()

    changed_lines: list[str] = []
    for section in sections:
        section_lines: list[str] = []
        for q in section["questions"]:
            qid = q["id"]
            if qid not in current_answers or qid not in previous_answers:
                continue
            current = _answer_summary(current_answers[qid])
            previous = _answer_summary(previous_answers[qid])
            if current == previous:
                continue
            section_lines.append(
                f"- {qid}: {q['question']}\n"
                f"  Previous answer: {previous}\n"
                f"  Current answer: {current}"
            )
        if section_lines:
            changed_lines.append(
                f"### {section['title']}\n" + "\n".join(section_lines)
            )

    if not changed_lines:
        return ""

    changed_ids = [
        q["id"]
        for section in sections
        for q in section["questions"]
        if q["id"] in current_answers
        and q["id"] in previous_answers
        and _answer_summary(current_answers[q["id"]])
        != _answer_summary(previous_answers[q["id"]])
    ]
    id_hint = ", ".join(changed_ids[:4]) + (", ..." if len(changed_ids) > 4 else "")

    return (
        "You are the same expert analyst of mental health peer support who "
        "explains questionnaire answers. Your task now is to explain WHY the "
        "persona's answer changed between two biography revisions.\n"
        f"{language_instruction}\n\n"
        "For each changed answer, write 1-3 concise sentences explaining the "
        "most plausible reason for the change. Base the explanation on: "
        "(1) the differences between the previous and current biographies, "
        "(2) your professional knowledge of peer support and mental health "
        "services, and (3) the academic context below from "
        "persona_guidelines.json. Do not change any answer.\n\n"
        "## Previous biography\n"
        f"{previous_biography_text.strip()}\n\n"
        "## Current biography\n"
        f"{current_biography_text.strip()}\n\n"
        "## Changed questionnaire answers\n"
        + "\n\n".join(changed_lines)
        + "\n\n"
        "## Academic peer-support context\n"
        + (academic_context or "No academic context was available.")
        + "\n\n"
        "## Output format\n"
        "Respond with ONLY a valid JSON object. No prose, no markdown, no code "
        f"fences. The keys MUST be the changed statement ids (e.g. {id_hint}). "
        "Each value MUST be a plain string explaining why that answer changed."
    )
