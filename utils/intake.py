"""Structured intake form: load, localize, randomize, and prompt-build.

The raw intake content lives in `data/intake.json` as a bilingual (en/he)
dictionary. This module exposes four responsibilities used by the Streamlit
app:

- `load_intake()`           reads the JSON from disk (cached).
- `get_localized_sections()` flattens it into a UI-friendly shape with all
  strings pre-picked for the active language.
- `randomize_answers()`     produces a sensible random answer set for the
  researcher's "Randomize" button.
- `build_biography_prompt()` assembles the LLM prompt that turns a set of
  intake answers into a 1-2 paragraph first-person biography.

Answer shapes (keyed by `question_id` in the returned dict) store the
*verbal label* in the active language so the Supabase JSONB row is
human-readable without cross-referencing `intake.json`:

    open_ended                      -> str
    multiple_choice, boolean        -> str   (the selected option label)
        demo_q4 / children may also store {"choice": str, "number": str}
    boolean_with_text               -> {"choice": str, "elaboration": str}
    likert_with_open_elaboration    -> {"rating": str, "elaboration": str}

Note: `randomize_answers()` still returns *int-shaped* values because it
writes directly into Streamlit widget state (radios hold ints, sliders
hold ints). The conversion to verbal labels happens in the app layer
(`_collect_intake_answers`) before anything reaches `build_biography_prompt`
or the database.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import streamlit as st

from . import guidelines
from .i18n import LANG_EN, LANG_HE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INTAKE_PATH = Path(__file__).resolve().parent.parent / "data" / "intake.json"

QUESTION_TYPES = (
    "boolean",
    "multiple_choice",
    "open_ended",
    "boolean_with_text",
    "likert_with_open_elaboration",
)

LIKERT_MIN = 1
LIKERT_MAX = 5


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_intake() -> dict[str, Any]:
    """Read and cache the raw intake JSON from disk.

    Raises `FileNotFoundError` if the file is missing, and `ValueError` if the
    top-level `sections` key is absent (fail loudly rather than silently
    skipping the form).
    """
    if not _INTAKE_PATH.exists():
        raise FileNotFoundError(f"Intake file not found: {_INTAKE_PATH}")
    with _INTAKE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "sections" not in data:
        raise ValueError("intake.json must be an object with a 'sections' key")
    return data


# ---------------------------------------------------------------------------
# Localization
# ---------------------------------------------------------------------------


def _pick(bilingual: dict[str, str] | str, language: str) -> str:
    """Pick the right string from a `{en, he}` dict, with fallbacks.

    Accepts a bare string as well (so partially-translated sources don't
    crash) and falls back English -> Hebrew -> empty string.
    """
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
    intake: dict[str, Any], language: str
) -> list[dict[str, Any]]:
    """Return sections with every user-facing string already picked.

    Output shape per section:
        {
            "id": str,
            "title": str,
            "scale": list[str] | None,
            "questions": [
                {
                    "id": str,
                    "question": str,
                    "type": str,
                    "options": list[str],   # may be [] for open_ended / likert
                }
            ],
        }
    """
    lang = language if language in (LANG_EN, LANG_HE) else LANG_EN
    out: list[dict[str, Any]] = []
    for section in intake.get("sections", []):
        scale_raw = section.get("scale")
        scale = (
            [_pick(label, lang) for label in scale_raw]
            if isinstance(scale_raw, list)
            else None
        )
        questions: list[dict[str, Any]] = []
        for q in section.get("questions", []):
            options_raw = q.get("options") or []
            questions.append(
                {
                    "id": q["id"],
                    "question": _pick(q.get("question", ""), lang),
                    "type": q.get("type", ""),
                    "options": [_pick(opt, lang) for opt in options_raw],
                }
            )
        out.append(
            {
                "id": section.get("id", ""),
                "title": _pick(section.get("title", ""), lang),
                "scale": scale,
                "questions": questions,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def list_open_ended_question_ids(intake: dict[str, Any]) -> list[str]:
    """Return every `open_ended` question id in the intake, in order.

    Used by the app layer to validate that the researcher (or the LLM
    Randomize helper) has filled every free-text question before the
    biography can be drafted.
    """
    out: list[str] = []
    for section in intake.get("sections", []):
        for q in section.get("questions", []):
            if q.get("type") == "open_ended":
                qid = q.get("id")
                if isinstance(qid, str):
                    out.append(qid)
    return out


# ---------------------------------------------------------------------------
# Randomization
# ---------------------------------------------------------------------------


def randomize_answers(
    intake: dict[str, Any],
    *,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Return a randomized answer set keyed by `question_id`.

    - boolean / multiple_choice           -> uniform random index into options
    - likert_with_open_elaboration        -> uniform int in [LIKERT_MIN, LIKERT_MAX]
    - open_ended                          -> "" (LLM fills during drafting)
    - boolean_with_text                   -> random choice + "" elaboration

    Pass `rng` to make the result deterministic in tests.
    """
    r = rng or random
    answers: dict[str, Any] = {}
    for section in intake.get("sections", []):
        for q in section.get("questions", []):
            qid = q["id"]
            qtype = q.get("type")
            opts = q.get("options") or []
            if qtype in ("boolean", "multiple_choice"):
                if opts:
                    answers[qid] = r.randrange(len(opts))
            elif qtype == "boolean_with_text":
                if opts:
                    answers[qid] = {
                        "choice": r.randrange(len(opts)),
                        "elaboration": "",
                    }
            elif qtype == "likert_with_open_elaboration":
                answers[qid] = {
                    "rating": r.randint(LIKERT_MIN, LIKERT_MAX),
                    "elaboration": "",
                }
            elif qtype == "open_ended":
                answers[qid] = ""
    return answers


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _format_answer_line(
    localized_q: dict[str, Any],
    answer: Any,
    scale_labels: list[str] | None,
) -> str | None:
    """Render one (question, answer) pair as a single Markdown bullet line.

    `answer` is expected in the verbal-label shape produced by
    `_collect_intake_answers` in `app.py` (see the module docstring). For
    backward compatibility this function also accepts the older int-index
    shape — so rows persisted before the label migration still render cleanly
    if the researcher loads them back.

    Returns None when the answer is missing or empty so we can skip it in the
    prompt instead of cluttering it with blanks.
    """
    qtext = localized_q["question"].strip()
    qtype = localized_q["type"]
    opts: list[str] = localized_q["options"]

    def _resolve_choice(value: Any) -> str | None:
        """Turn a choice answer (label str or legacy int index) into a label."""
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if isinstance(value, int) and 0 <= value < len(opts):
            return opts[value]
        return None

    def _resolve_rating(value: Any) -> str | None:
        """Turn a rating answer (label str or legacy int index) into a label."""
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if isinstance(value, int) and LIKERT_MIN <= value <= LIKERT_MAX:
            if scale_labels and 1 <= value <= len(scale_labels):
                return scale_labels[value - 1]
            return f"{value}/{LIKERT_MAX}"
        return None

    if qtype in ("boolean", "multiple_choice"):
        if isinstance(answer, dict):
            choice = answer.get("choice")
            number = str(answer.get("number") or "").strip()
            if not isinstance(choice, str) or not choice.strip():
                return None
            line = f"- {qtext} {choice.strip()}"
            if number:
                line += f" — {number}"
            return line
        label = _resolve_choice(answer)
        if label is None:
            return None
        return f"- {qtext} {label}"

    if qtype == "boolean_with_text":
        if not isinstance(answer, dict):
            return None
        label = _resolve_choice(answer.get("choice"))
        if label is None:
            return None
        elab = (answer.get("elaboration") or "").strip()
        line = f"- {qtext} {label}"
        if elab:
            line += f" — {elab}"
        return line

    if qtype == "likert_with_open_elaboration":
        if not isinstance(answer, dict):
            return None
        label = _resolve_rating(answer.get("rating"))
        if label is None:
            return None
        elab = (answer.get("elaboration") or "").strip()
        line = f"- {qtext} {label}"
        if elab:
            line += f" — {elab}"
        return line

    if qtype == "open_ended":
        text = (answer or "").strip() if isinstance(answer, str) else ""
        if not text:
            return None
        return f"- {qtext} {text}"

    return None


# _BIO_INSTRUCTIONS: dict[str, str] = {
#     LANG_EN: (
#         "Write a coherent first-person biography based on the following intake answers. For enrichment, include the relevant text from the academic papers below. Note: Do not mention in the biography that the information comes from academic papers, the enrichment should be natural. Keep the biography grounded and concrete. The biography should be consistent with the intake answers and should provide concrete details about the person. The biography should be a single paragraph."
#         "Respond in English."
#     ),
#     LANG_HE: (
#         "כתוב/כתבי ביוגרפיה נרטיבית בגוף ראשון בהתבסס על תשובות השאלון המופיעות מטה. כדי להעשיר את הביוגרפיה, שלב בתוכה גם את הטקסט הרלוונטי מתוך המאמרים האקדמיים שמופיע מטה. שים לב שהטקסט צריך להיות רלוונטי לדמות הפרסונה - כלומר למאפיינים מתשובות השאלון. אין צורך לציין בביוגרפיה שהמידע מגיע ממאמרים, הוספת המידע צריכה להיות טבעית. שמור/שמרי על ביוגרפיה "
#         "מעוגנת וקונקרטית. הביוגרפיה צריכה להיות עקבית עם תשובות השאלון אך צריכה לספק בנוסף לתשובות גם קונקרטיזציה עקבית איתן. למשל, אם התשובה לשאלה היא איבדתי אדם קרוב - הביוגרפיה צריכה להרחיב על אותו האדם הספציפי, או אם התשובה היא שיש ילדים, הביוגרפיה צריכה להרחיב מספר פרטים אודותיהם וכדומה. המטרה היא ליצור ביוגרפיה אמינה, עשירה ושמתארת בן אדם קונקרטי ואמין שעקבי עם המענה על השאלות. השב/י בעברית."
#     ),

_BIO_INSTRUCTIONS: dict[str, str] = {
    LANG_EN: (
        "You are the persona that filled the questionnaire. You need to put on your shoes and tell me the story behind your answer to each question. Tell me the story of your life based on your answers to the questionnaire. Be specific, concrete, specific and not generic and technical. Expand on the story of your life."
    ),
    LANG_HE: (
        "אתה הדמות שמילאה את השאלון. עליך להיכנס לנעליה ולספר לי את הסיפור שעומר מאחוריי מילוי כל שאלה. ספר לי את סיפור חייך בתהאם למענה שלך על תשובות השאלון. עליך להיות ספציפי, קונקרטי,אותנטי ולא להישמע גנרי וטכני. פרט בהרחבה את סיפור חייך."
    ),
}


# ---------------------------------------------------------------------------
# Paper-grounded guidelines block
# ---------------------------------------------------------------------------
# English-only in this first pass. Mini-step 14.7 will wrap the authored
# strings into `{en, he}` and make the header / framing line language-aware.
# Until then the block is emitted in English regardless of the prompt's
# surrounding language — the LLM can read English criteria to inform a
# biography it ultimately writes in Hebrew.

_GUIDELINES_HEADER_EN = "## Paper-grounded persona criteria"
_GUIDELINES_FRAMING_EN = (
    "Use these paper-grounded criteria and their supporting quotes as "
    "inspiration to enrich the biography with concrete, plausible detail. "
    "Do NOT reproduce any paper citation or verbatim quote in the output "
    "biography, and do NOT answer any criterion literally."
)


def _build_guidelines_block() -> str:
    """Return the paper-grounded guidelines block, or empty string if none.

    Renders each paper as a sub-section with its citation and summary, then
    each criterion with its explanation and a bullet list of the verbatim
    supporting quotes. Missing fields are skipped silently so a half-filled
    entry still contributes whatever content it has.
    """
    try:
        data = guidelines.load_guidelines()
    except FileNotFoundError:
        return ""
    papers = guidelines.get_papers(data)
    if not papers:
        return ""

    paper_blocks: list[str] = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        citation = (paper.get("paper_citation") or "").strip()
        summary = (paper.get("paper_summary") or "").strip()
        criteria = paper.get("criteria") or []

        head_lines: list[str] = []
        if citation:
            head_lines.append(f"Citation: {citation}")
        if summary:
            head_lines.append(f"Summary: {summary}")

        criterion_blocks: list[str] = []
        for idx, c in enumerate(criteria, start=1):
            if not isinstance(c, dict):
                continue
            criterion = (c.get("criterion") or "").strip()
            explanation = (c.get("explanation") or "").strip()
            raw_quotes = c.get("citations_from_the_paper") or []
            quotes = [
                str(q).strip()
                for q in (raw_quotes if isinstance(raw_quotes, list) else [])
                if str(q).strip()
            ]
            if not (criterion or explanation or quotes):
                continue

            parts: list[str] = []
            if criterion:
                parts.append(f"Criterion {idx}: {criterion}")
            if explanation:
                parts.append(f"Explanation: {explanation}")
            if quotes:
                parts.append("Supporting quotes:")
                parts.extend(f"- {q}" for q in quotes)
            criterion_blocks.append("\n".join(parts))

        if not (head_lines or criterion_blocks):
            continue

        paper_id = str(paper.get("id") or "paper").strip() or "paper"
        body = "\n".join(head_lines)
        if criterion_blocks:
            body += ("\n\n" if body else "") + "\n\n".join(criterion_blocks)
        paper_blocks.append(f"### Paper: {paper_id}\n{body}")

    if not paper_blocks:
        return ""

    return (
        f"{_GUIDELINES_HEADER_EN}\n"
        f"{_GUIDELINES_FRAMING_EN}\n\n"
        + "\n\n".join(paper_blocks)
    )


def build_biography_prompt(
    answers: dict[str, Any],
    intake: dict[str, Any],
    language: str = LANG_EN,
) -> str:
    """Build the prompt that asks the LLM to draft a first-person biography.

    Only sections that have at least one non-empty answer are included, and
    within a section only answered questions are rendered — so a researcher
    who edits a few fields and hits "Draft biography" without randomizing the
    rest still gets a usable prompt.
    """
    lang = language if language in (LANG_EN, LANG_HE) else LANG_EN
    sections = get_localized_sections(intake, lang)
    instruction = _BIO_INSTRUCTIONS[lang]

    blocks: list[str] = []
    for section in sections:
        lines: list[str] = []
        for q in section["questions"]:
            qid = q["id"]
            if qid not in answers:
                continue
            line = _format_answer_line(q, answers[qid], section["scale"])
            if line:
                lines.append(line)
        if lines:
            title = section["title"].strip() or section["id"]
            blocks.append(f"### {title}\n" + "\n".join(lines))

    header_intake = "## Intake answers" if lang == LANG_EN else "## תשובות השאלון"
    intake_body = "\n\n".join(blocks) if blocks else (
        "(no answers provided)" if lang == LANG_EN else "(לא סופקו תשובות)"
    )
    guidelines_header = "## Academic papers" if lang == LANG_EN else "## מאמרים אקדמיים"
    guidelines_block = _build_guidelines_block()

    return (
        f"{instruction}\n"
        "\n"
        f"{header_intake}\n"
        f"{intake_body}\n"
        "\n"
        f"{guidelines_header}\n"
        + (f"\n{guidelines_block}\n" if guidelines_block else "")
        + "\n"
        + (
            "Output ONLY the biography prose. No preamble, no bullet points, "
            "no markdown headers, no quotation marks."
            if lang == LANG_EN
            else "החזר/החזירי אך ורק את טקסט הביוגרפיה. ללא הקדמה, ללא נקודות, "
            "ללא כותרות מרקדאון, ללא מירכאות."
        )
    )
