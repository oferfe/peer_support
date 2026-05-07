"""Unified LLM wrapper for OpenAI (ChatGPT) and Google GenAI (Gemma 3).

Three public functions are dispatched by a `model_label` string:

- `chat(...)`                 -> free-form chat turn, returns a string.
- `answer_questionnaire(...)` -> structured JSON answers, returns a dict.
- `generate_biography(...)`   -> free-form persona biography from intake
                                 answers, returns a string.

Provider differences hidden here:

- OpenAI supports a dedicated `system` role and a native JSON response format.
- Google's Gemma family on the GenAI API does NOT accept a `system` role, so
  the biography is prepended to the first user turn instead. The GenAI SDK
  also uses `role="model"` for assistant turns.
"""

from __future__ import annotations

import json
import re
from typing import Any

import streamlit as st
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

from .i18n import LANG_EN, LANG_HE
from .intake import build_biography_prompt, get_localized_sections
from .questionnaire import (
    build_character_system_prompt,
    build_explanation_prompt,
    build_json_prompt,
    build_simulation_biography_prompt,
)


# Accepted model labels from the UI radio. `OLLAMA` routes through a local
# Ollama server and is the current "development / open-source" option that
# replaced the earlier Gemma branch in the radio. `GEMMA` is kept as a
# module-level constant so the helper code below still compiles and can be
# re-exposed if Gemma is added back to the radio later.
CHATGPT = "ChatGPT"
OLLAMA = "Ollama"
GEMMA = "Gemma"


_LANGUAGE_DIRECTIVE: dict[str, str] = {
    LANG_EN: "Respond entirely in English.",
    LANG_HE: "Respond entirely in Hebrew (עברית).",
}


def _language_directive(language: str) -> str:
    return _LANGUAGE_DIRECTIVE.get(language, _LANGUAGE_DIRECTIVE[LANG_EN])


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _openai_client() -> OpenAI:
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


@st.cache_resource(show_spinner=False)
def _google_client() -> genai.Client:
    return genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])


@st.cache_resource(show_spinner=False)
def _ollama_client() -> OpenAI:
    """Return an `OpenAI` client pointed at the local Ollama server.

    Ollama exposes an OpenAI-compatible REST API at
    `http://localhost:11434/v1`, so we reuse the same SDK — and all the
    same response shapes — as the real ChatGPT branch. The `api_key`
    parameter is required by the OpenAI SDK but Ollama itself does not
    check it; any non-empty string works. Both the base URL and the
    placeholder key are overridable through `st.secrets` for users who
    run Ollama on a remote host or behind a reverse proxy.
    """
    return OpenAI(
        base_url=st.secrets.get(
            "OLLAMA_BASE_URL", "http://localhost:11434/v1"
        ),
        api_key=st.secrets.get("OLLAMA_API_KEY", "ollama"),
    )


def _openai_model() -> str:
    return st.secrets.get("OPENAI_MODEL", "gpt-4o")


def _gemma_model() -> str:
    return st.secrets.get("GEMMA_MODEL", "gemma-3-27b-it")


def _ollama_model() -> str:
    return st.secrets.get("OLLAMA_MODEL", "gemma3:12b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of an LLM response, tolerating stray prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _attach_explanations(
    answers: dict[str, Any],
    explanations: dict[str, Any],
) -> dict[str, Any]:
    """Attach second-LLM explanations to first-LLM questionnaire answers."""
    merged: dict[str, Any] = {}
    for qid, answer in answers.items():
        explanation = explanations.get(qid)
        if isinstance(explanation, str) and explanation.strip():
            if isinstance(answer, dict):
                enriched = dict(answer)
                enriched["reasoning"] = explanation.strip()
                merged[qid] = enriched
            else:
                merged[qid] = {
                    "label": answer,
                    "reasoning": explanation.strip(),
                }
        else:
            merged[qid] = answer
    return merged


def _history_to_genai(
    system_prompt: str,
    history: list[dict[str, str]],
    user_message: str,
) -> list[genai_types.Content]:
    """Convert OpenAI-style messages into GenAI `Content` objects.

    Gemma has no system role, so we fold `system_prompt` into the first user
    turn. OpenAI's `assistant` maps to GenAI's `model`.
    """
    contents: list[genai_types.Content] = []
    first_user_emitted = False

    def _user_text(text: str) -> str:
        nonlocal first_user_emitted
        if not first_user_emitted:
            first_user_emitted = True
            return f"{system_prompt.strip()}\n\n{text}"
        return text

    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            contents.append(
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text=_user_text(content))],
                )
            )
        elif role == "assistant":
            contents.append(
                genai_types.Content(
                    role="model",
                    parts=[genai_types.Part.from_text(text=content)],
                )
            )

    contents.append(
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=_user_text(user_message))],
        )
    )
    return contents


# ---------------------------------------------------------------------------
# OpenAI branch
# ---------------------------------------------------------------------------

def _openai_chat(
    system_prompt: str,
    history: list[dict[str, str]],
    user_message: str,
    language: str,
) -> str:
    full_system = f"{system_prompt}\n\n{_language_directive(language)}"
    messages: list[dict[str, str]] = [{"role": "system", "content": full_system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    resp = _openai_client().chat.completions.create(
        model=_openai_model(),
        messages=messages,
    )
    return resp.choices[0].message.content or ""


def _openai_biography(prompt: str) -> str:
    resp = _openai_client().chat.completions.create(
        model=_openai_model(),
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def _openai_questionnaire(
    biography_text: str,
    questionnaire: dict[str, Any],
    language: str,
    explanation_biography_text: str | None = None,
) -> dict[str, Any]:
    character_system_prompt = build_character_system_prompt(
        biography_text,
        language,
    )
    answer_prompt = build_json_prompt(questionnaire, language)
    resp = _openai_client().chat.completions.create(
        model=_openai_model(),
        messages=[
            {
                "role": "system",
                "content": character_system_prompt,
            },
            {"role": "user", "content": answer_prompt},
        ],
        response_format={"type": "json_object"},
    )
    answers = _parse_json(resp.choices[0].message.content or "{}")

    explanation_prompt = build_explanation_prompt(
        explanation_biography_text or biography_text,
        questionnaire,
        answers,
        language,
    )
    explanation_resp = _openai_client().chat.completions.create(
        model=_openai_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert academic explainer of mental health "
                    "peer support questionnaire responses. Return only valid "
                    f"JSON. {_language_directive(language)}"
                ),
            },
            {"role": "user", "content": explanation_prompt},
        ],
        response_format={"type": "json_object"},
    )
    explanations = _parse_json(
        explanation_resp.choices[0].message.content or "{}"
    )
    return _attach_explanations(answers, explanations)


# ---------------------------------------------------------------------------
# Google / Gemma branch
# ---------------------------------------------------------------------------

def _gemma_chat(
    system_prompt: str,
    history: list[dict[str, str]],
    user_message: str,
    language: str,
) -> str:
    full_system = f"{system_prompt}\n\n{_language_directive(language)}"
    contents = _history_to_genai(full_system, history, user_message)
    resp = _google_client().models.generate_content(
        model=_gemma_model(),
        contents=contents,
    )
    return resp.text or ""


def _gemma_biography(prompt: str) -> str:
    resp = _google_client().models.generate_content(
        model=_gemma_model(),
        contents=prompt,
    )
    return (resp.text or "").strip()


def _gemma_generate_json_text(prompt: str) -> str:
    """Call Gemma with a JSON-oriented prompt and return the raw text.

    Gemma 3 on the GenAI API does not universally support the
    `response_mime_type="application/json"` hint — on some account /
    model combinations it raises, on others it silently ignores it and
    returns plain prose that happens to contain JSON. We try the hinted
    call first, fall back to plain text generation on failure, and raise
    a clear `RuntimeError` with the upstream error text if both attempts
    fail or produce an empty response.
    """
    client = _google_client()
    last_exc: Exception | None = None

    try:
        resp = client.models.generate_content(
            model=_gemma_model(),
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        text = (resp.text or "").strip()
        if text:
            return text
        last_exc = RuntimeError("Gemma returned an empty JSON response")
    except Exception as exc:  # noqa: BLE001 — keep for the retry path
        last_exc = exc

    try:
        resp = client.models.generate_content(
            model=_gemma_model(),
            contents=prompt,
        )
    except Exception as exc:
        # Surface the original (mime-typed) error too when the retry also
        # fails — it's almost always the more informative of the two.
        raise RuntimeError(
            f"Gemma generation failed. Plain-text retry error: {exc!r}. "
            f"Original hinted-JSON error: {last_exc!r}"
        ) from exc

    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError(
            "Gemma returned an empty response on both the hinted-JSON "
            f"and plain-text attempts. Original hinted-JSON error: "
            f"{last_exc!r}"
        )
    return text


def _gemma_questionnaire(
    biography_text: str,
    questionnaire: dict[str, Any],
    language: str,
    explanation_biography_text: str | None = None,
) -> dict[str, Any]:
    character_system_prompt = build_character_system_prompt(
        biography_text,
        language,
    )
    answer_prompt = build_json_prompt(questionnaire, language)
    combined_answer_prompt = (
        f"## System prompt: character\n{character_system_prompt}\n\n"
        f"## System instructions: questionnaire\n{answer_prompt}"
    )
    text = _gemma_generate_json_text(combined_answer_prompt)
    try:
        answers = _parse_json(text)
    except Exception as exc:
        snippet = text[:400].replace("\n", " ")
        raise RuntimeError(
            f"Gemma returned content that is not valid JSON. Snippet: "
            f"{snippet!r}"
        ) from exc
    explanation_prompt = build_explanation_prompt(
        explanation_biography_text or biography_text,
        questionnaire,
        answers,
        language,
    )
    explanation_text = _gemma_generate_json_text(explanation_prompt)
    try:
        explanations = _parse_json(explanation_text)
    except Exception as exc:
        snippet = explanation_text[:400].replace("\n", " ")
        raise RuntimeError(
            "Gemma returned explanation content that is not valid JSON. "
            f"Snippet: {snippet!r}"
        ) from exc
    return _attach_explanations(answers, explanations)


# ---------------------------------------------------------------------------
# Ollama branch (local / open-source models via OpenAI-compatible API)
# ---------------------------------------------------------------------------
# Ollama's REST API accepts the exact same request shape as OpenAI, so every
# helper here is a near-copy of its `_openai_*` sibling but swaps in the
# Ollama-pointed client and model. Keeping them as separate functions (rather
# than parameterizing the OpenAI ones) preserves the tidy "one provider per
# section" layout of this file and lets `_openai_*` stay unaware of Ollama.

def _ollama_chat(
    system_prompt: str,
    history: list[dict[str, str]],
    user_message: str,
    language: str,
) -> str:
    full_system = f"{system_prompt}\n\n{_language_directive(language)}"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": full_system}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    resp = _ollama_client().chat.completions.create(
        model=_ollama_model(),
        messages=messages,
    )
    return resp.choices[0].message.content or ""


def _ollama_biography(prompt: str) -> str:
    resp = _ollama_client().chat.completions.create(
        model=_ollama_model(),
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def _ollama_json_chat(
    system_content: str,
    user_content: str,
) -> str:
    """Send a chat request to Ollama asking for a JSON response, falling
    back to a plain call if the local model does not accept
    `response_format={"type": "json_object"}`.

    Returns the raw text of the assistant turn; callers wrap `_parse_json`
    around it and attach their own error context on parse failure.
    """
    client = _ollama_client()
    model = _ollama_model()
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Some Ollama builds / models reject `response_format`. The prompt
        # itself is strict about JSON-only output, so a plain call is a
        # safe retry. If this also fails, the exception propagates.
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    return resp.choices[0].message.content or ""


def _ollama_questionnaire(
    biography_text: str,
    questionnaire: dict[str, Any],
    language: str,
    explanation_biography_text: str | None = None,
) -> dict[str, Any]:
    answer_system_content = build_character_system_prompt(
        biography_text,
        language,
    )
    answer_prompt = build_json_prompt(questionnaire, language)
    text = _ollama_json_chat(answer_system_content, answer_prompt)
    try:
        answers = _parse_json(text)
    except Exception as exc:
        snippet = text[:400].replace("\n", " ") if text else "<empty>"
        raise RuntimeError(
            f"Ollama returned content that is not valid JSON. Snippet: "
            f"{snippet!r}"
        ) from exc
    explanation_prompt = build_explanation_prompt(
        explanation_biography_text or biography_text,
        questionnaire,
        answers,
        language,
    )
    explanation_system_content = (
        "You are an expert academic explainer of mental health peer support "
        "questionnaire responses. Return only valid JSON. "
        f"{_language_directive(language)}"
    )
    explanation_text = _ollama_json_chat(
        explanation_system_content,
        explanation_prompt,
    )
    try:
        explanations = _parse_json(explanation_text)
    except Exception as exc:
        snippet = (
            explanation_text[:400].replace("\n", " ")
            if explanation_text
            else "<empty>"
        )
        raise RuntimeError(
            "Ollama returned explanation content that is not valid JSON. "
            f"Snippet: {snippet!r}"
        ) from exc
    return _attach_explanations(answers, explanations)


def _ollama_json(prompt: str, language: str) -> dict[str, Any]:
    """Generic JSON-only prompt runner used by `generate_open_ended_answers`."""
    system_content = (
        "You return only valid JSON matching the user's schema. "
        f"{_language_directive(language)}"
    )
    text = _ollama_json_chat(system_content, prompt)
    try:
        return _parse_json(text)
    except Exception as exc:
        snippet = text[:400].replace("\n", " ") if text else "<empty>"
        raise RuntimeError(
            f"Ollama returned content that is not valid JSON. Snippet: "
            f"{snippet!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def chat(
    model_label: str,
    system_prompt: str,
    history: list[dict[str, str]],
    user_message: str,
    language: str = LANG_EN,
) -> str:
    """Send one chat turn; return the assistant's reply as plain text."""
    if model_label == CHATGPT:
        return _openai_chat(system_prompt, history, user_message, language)
    if model_label == OLLAMA:
        return _ollama_chat(system_prompt, history, user_message, language)
    if model_label == GEMMA:
        return _gemma_chat(system_prompt, history, user_message, language)
    raise ValueError(f"Unknown model_label: {model_label!r}")


def answer_questionnaire(
    model_label: str,
    biography_text: str,
    questionnaire: dict[str, Any],
    language: str = LANG_EN,
    *,
    explanation_biography_text: str | None = None,
) -> dict[str, Any]:
    """Ask the model to answer the full questionnaire in character.

    Returns a dict keyed by statement id (e.g. ``"mwms_1"``) whose values are
    verbatim labels drawn from each section's scale.
    """
    if model_label == CHATGPT:
        return _openai_questionnaire(
            biography_text,
            questionnaire,
            language,
            explanation_biography_text,
        )
    if model_label == OLLAMA:
        return _ollama_questionnaire(
            biography_text,
            questionnaire,
            language,
            explanation_biography_text,
        )
    if model_label == GEMMA:
        return _gemma_questionnaire(
            biography_text,
            questionnaire,
            language,
            explanation_biography_text,
        )
    raise ValueError(f"Unknown model_label: {model_label!r}")


def convert_biography_for_simulation(
    model_label: str,
    biography_text: str,
    language: str = LANG_EN,
) -> str:
    """Convert a saved biography into a direct persona system prompt.

    This is an internal simulation-preparation step. The returned prompt is
    used as the character system prompt for questionnaire answering, while the
    original saved biography remains unchanged in the database and is still
    used by the explanation LLM.
    """
    prompt = build_simulation_biography_prompt(biography_text, language)
    if model_label == CHATGPT:
        return _openai_biography(prompt)
    if model_label == OLLAMA:
        return _ollama_biography(prompt)
    if model_label == GEMMA:
        return _gemma_biography(prompt)
    raise ValueError(f"Unknown model_label: {model_label!r}")


def _collect_open_ended_questions(
    intake: dict[str, Any],
    language: str,
    partial_answers: dict[str, Any],
) -> list[tuple[str, str]]:
    """Return [(qid, localized question text), ...] for every `open_ended`
    question whose current answer in `partial_answers` is missing/empty."""
    pending: list[tuple[str, str]] = []
    for section in get_localized_sections(intake, language):
        for q in section["questions"]:
            if q["type"] != "open_ended":
                continue
            qid = q["id"]
            current = partial_answers.get(qid)
            if isinstance(current, str) and current.strip():
                continue
            pending.append((qid, q["question"]))
    return pending


def _build_open_ended_answers_prompt(
    pending: list[tuple[str, str]],
    partial_answers: dict[str, Any],
    intake: dict[str, Any],
    language: str,
) -> str:
    """Build the JSON-only prompt for `generate_open_ended_answers`."""
    lang = language if language in (LANG_EN, LANG_HE) else LANG_EN
    context = build_biography_prompt(partial_answers, intake, lang)

    questions_block = "\n".join(
        f'- "{qid}": {qtext}' for qid, qtext in pending
    )
    expected_keys = ", ".join(f'"{qid}"' for qid, _ in pending)

    if lang == LANG_HE:
        instruction = (
            "יש לנו את תשובות השאלון החלקיות של פרסונה, ואת הקשר הרקע המלא "
            "המופיע מטה. עבור כל שאלה פתוחה שרשומה למטה, הפק/י תשובה "
            "אמינה וקצרה (משפט עד שניים) בגוף ראשון, עקבית עם שאר הפרטים. "
            f"החזר/י אובייקט JSON בלבד עם המפתחות: {expected_keys}. כל ערך "
            "הוא מחרוזת. אין להוסיף טקסט חופשי מחוץ ל-JSON."
        )
        questions_header = "## שאלות פתוחות להשלמה"
        context_header = "## הקשר"
    else:
        instruction = (
            "You have a partially-filled intake for a persona (see context "
            "below). For each open-ended question listed, generate a short "
            "(1-2 sentence) first-person answer that is consistent with the "
            f"rest of the intake. Return ONLY a JSON object with keys: "
            f"{expected_keys}. Each value must be a plain string. No prose "
            "outside the JSON."
        )
        questions_header = "## Open-ended questions to answer"
        context_header = "## Context"

    return (
        f"{instruction}\n\n"
        f"{questions_header}\n{questions_block}\n\n"
        f"{context_header}\n{context}\n"
    )


def _openai_json(prompt: str, language: str) -> dict[str, Any]:
    """Run a JSON-only prompt through OpenAI's response_format=json_object."""
    resp = _openai_client().chat.completions.create(
        model=_openai_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "You return only valid JSON matching the user's schema. "
                    f"{_language_directive(language)}"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return _parse_json(resp.choices[0].message.content or "{}")


def _gemma_json(prompt: str) -> dict[str, Any]:
    """Run a JSON-only prompt through Gemma; regex-fallback for stray prose.

    Delegates the provider-specific retry + empty-response handling to
    `_gemma_generate_json_text` and wraps JSON-parse failures with a
    truncated snippet of the raw text so the caller can see what Gemma
    actually produced.
    """
    text = _gemma_generate_json_text(prompt)
    try:
        return _parse_json(text)
    except Exception as exc:
        snippet = text[:400].replace("\n", " ")
        raise RuntimeError(
            f"Gemma returned content that is not valid JSON. Snippet: "
            f"{snippet!r}"
        ) from exc


def generate_open_ended_answers(
    model_label: str,
    intake: dict[str, Any],
    partial_answers: dict[str, Any],
    language: str = LANG_EN,
) -> dict[str, str]:
    """Generate short plausible answers for every empty `open_ended` question.

    Invoked by the Randomize flow to satisfy the rule that `open_ended`
    intake answers must be filled (manually or by the LLM) before a
    biography can be drafted. For each `open_ended` question whose answer
    in `partial_answers` is missing or empty, the selected LLM is asked to
    produce a 1-2 sentence first-person answer consistent with the rest of
    the intake. Answers for questions the researcher already filled are
    left untouched: only *new* answers are returned, keyed by question id.

    The function is tolerant of a malformed LLM response — any question id
    that does not come back as a non-empty string is silently omitted so
    the caller can detect it via `list_open_ended_question_ids` still not
    being satisfied.
    """
    open_ended_q = _collect_open_ended_questions(intake, language, partial_answers)
    if not open_ended_q:
        return {}

    prompt = _build_open_ended_answers_prompt(
        open_ended_q, partial_answers, intake, language
    )
    if model_label == CHATGPT:
        raw = _openai_json(prompt, language)
    elif model_label == OLLAMA:
        raw = _ollama_json(prompt, language)
    elif model_label == GEMMA:
        raw = _gemma_json(prompt)
    else:
        raise ValueError(f"Unknown model_label: {model_label!r}")

    out: dict[str, str] = {}
    for qid, _qtext in open_ended_q:
        value = raw.get(qid)
        if isinstance(value, str) and value.strip():
            out[qid] = value.strip()
    return out


def generate_biography(
    model_label: str,
    intake: dict[str, Any],
    answers: dict[str, Any],
    language: str = LANG_EN,
) -> str:
    """Draft a third-person persona biography from the intake answers.

    The prompt is assembled by `utils.intake.build_biography_prompt` and sent
    through the selected provider in plain-text mode (no JSON). The prompt
    itself instructs the model to return only the biography prose; any
    leading/trailing whitespace is stripped before returning.
    """
    prompt = build_biography_prompt(answers, intake, language)
    if model_label == CHATGPT:
        return _openai_biography(prompt)
    if model_label == OLLAMA:
        return _ollama_biography(prompt)
    if model_label == GEMMA:
        return _gemma_biography(prompt)
    raise ValueError(f"Unknown model_label: {model_label!r}")
