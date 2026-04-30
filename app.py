"""Persona Chatbot & Evaluation Dashboard.

Streamlit entry point. This file owns page layout and session state; all
external calls (Supabase, OpenAI, Google GenAI) live in the `utils` package.
All user-facing strings are routed through `utils.i18n.t(...)` so the UI
switches between English and Hebrew (with RTL layout) based on the researcher's
selection on the entry screen.
"""

from __future__ import annotations

from collections.abc import Mapping
import difflib
import html
import uuid
from typing import Any

import streamlit as st

from utils import db, llm
from utils.i18n import (
    LANG_EN,
    LANG_HE,
    LANGUAGE_LABELS,
    inject_rtl_css,
    t,
)
from utils.intake import (
    get_localized_sections,
    list_open_ended_question_ids,
    load_intake,
    randomize_answers,
)
from utils.llm import CHATGPT, OLLAMA
from utils.questionnaire import (
    get_localized_sections as get_questionnaire_sections,
    load_questionnaire,
)


# ---------------------------------------------------------------------------
# Intake-form helpers (Step 13)
# ---------------------------------------------------------------------------
# Widget state lives under predictable keys so both Randomize and Draft-bio
# buttons can read/write a single source of truth.

def _choice_key(qid: str) -> str:
    return f"intake_choice_{qid}"


def _text_key(qid: str) -> str:
    return f"intake_text_{qid}"


def _rating_key(qid: str) -> str:
    return f"intake_rating_{qid}"


def _answer_comment_key(questionnaire_id: str, qid: str) -> str:
    return f"answer_comment_{questionnaire_id}_{qid}"


def _render_intake_question(
    question: dict[str, object],
    scale_labels: list[str] | None,
    *,
    disabled: bool = False,
) -> None:
    """Render one intake widget bound to stable session_state keys."""
    qid = str(question["id"])
    qtype = str(question["type"])
    label = str(question["question"])
    options = [str(opt) for opt in (question.get("options") or [])]  # type: ignore[arg-type]

    if qtype in ("boolean", "multiple_choice"):
        st.radio(
            label,
            options=list(range(len(options))),
            format_func=lambda i: options[i],
            horizontal=len(options) <= 3,
            index=None,
            key=_choice_key(qid),
            disabled=disabled,
        )
    elif qtype == "boolean_with_text":
        st.radio(
            label,
            options=list(range(len(options))),
            format_func=lambda i: options[i],
            horizontal=True,
            index=None,
            key=_choice_key(qid),
            disabled=disabled,
        )
        # Show the elaboration text area only when the "yes" option (index 0,
        # per the JSON convention of listing the affirmative first) is picked.
        if st.session_state.get(_choice_key(qid)) == 0:
            st.text_area(
                t("intake_details_label"),
                key=_text_key(qid),
                height=80,
                disabled=disabled,
            )
    elif qtype == "likert_with_open_elaboration":
        if scale_labels:
            st.select_slider(
                label,
                options=list(range(1, len(scale_labels) + 1)),
                format_func=lambda r: scale_labels[r - 1],
                key=_rating_key(qid),
                disabled=disabled,
            )
        st.text_area(
            t("intake_elaborate_label"),
            key=_text_key(qid),
            height=80,
            disabled=disabled,
        )
    elif qtype == "open_ended":
        st.text_area(
            label,
            key=_text_key(qid),
            height=140,
            disabled=disabled,
        )


def _apply_random_intake_answers(
    intake: dict[str, object],
    model_label: str,
    language: str,
) -> None:
    """Overwrite intake widget state with a fresh random draw.

    Must be invoked via `st.button(on_click=...)` so it runs *before* the
    widgets re-render — Streamlit disallows mutating a widget-bound key
    after its widget has been instantiated in the same run.

    After the structured random picks are applied, every `open_ended`
    question that is still empty is filled via an LLM call so the
    "open-ended questions must be filled" rule is always satisfied after
    Randomize. The LLM call is best-effort: if it fails for any reason
    (network, quota, ...) the researcher still sees the structured
    randomization and can fill the open-ended fields manually.
    """
    for qid, value in randomize_answers(intake).items():
        if isinstance(value, str):
            st.session_state[_text_key(qid)] = value
        elif isinstance(value, int):
            st.session_state[_choice_key(qid)] = value
        elif isinstance(value, dict) and "rating" in value:
            st.session_state[_rating_key(qid)] = value["rating"]
            st.session_state[_text_key(qid)] = value.get("elaboration", "")
        elif isinstance(value, dict) and "choice" in value:
            st.session_state[_choice_key(qid)] = value["choice"]
            st.session_state[_text_key(qid)] = value.get("elaboration", "")

    # Collect the intake answers *as they stand after* the structured
    # randomization so the LLM prompt sees the full context when it
    # generates the open-ended answers.
    partial = _collect_intake_answers(intake, language)
    try:
        with st.spinner(
            t("intake_randomize_llm_spinner", model=model_label)
        ):
            generated = llm.generate_open_ended_answers(
                model_label, intake, partial, language
            )
    except Exception as exc:  # noqa: BLE001 — surface LLM failure to the UI
        st.toast(f"{t('intake_randomize_done')} ({exc})", icon="🎲")
        return
    for qid, text in generated.items():
        st.session_state[_text_key(qid)] = text
    st.toast(t("intake_randomize_done"), icon="🎲")


def _missing_open_ended_ids(
    intake: dict[str, object],
) -> list[str]:
    """Return the question ids of `open_ended` questions the user hasn't
    filled yet (neither by typing nor by Randomize). Used to disable the
    **Draft biography** button and show a targeted "please fill" banner."""
    missing: list[str] = []
    for qid in list_open_ended_question_ids(intake):
        value = st.session_state.get(_text_key(qid), "") or ""
        if not value.strip():
            missing.append(qid)
    return missing


def _seed_intake_widget_state_from_answers(
    intake: dict[str, object],
    answers: dict[str, object],
    language: str,
) -> None:
    """Populate intake widget state from a saved intake snapshot.

    The snapshot is the first revision's `intake_answers` JSONB payload. Once a
    persona is saved, these values are the fixed source of truth for the
    persona. We seed widget state before rendering the disabled widgets so the
    researcher can review the original answers without being able to alter
    them for later biography revisions.
    """
    for section in get_localized_sections(intake, language):
        for q in section["questions"]:
            qid = q["id"]
            st.session_state.pop(_choice_key(qid), None)
            st.session_state.pop(_text_key(qid), None)
            st.session_state.pop(_rating_key(qid), None)

    for section in get_localized_sections(intake, language):
        scale = section["scale"]
        for q in section["questions"]:
            qid = q["id"]
            qtype = q["type"]
            if qid not in answers:
                continue

            value = answers[qid]
            options = q["options"]
            if qtype in ("boolean", "multiple_choice"):
                if isinstance(value, str) and value in options:
                    st.session_state[_choice_key(qid)] = options.index(value)
            elif qtype == "boolean_with_text" and isinstance(value, dict):
                choice = value.get("choice")
                if isinstance(choice, str) and choice in options:
                    st.session_state[_choice_key(qid)] = options.index(choice)
                st.session_state[_text_key(qid)] = (
                    value.get("elaboration") or ""
                )
            elif (
                qtype == "likert_with_open_elaboration"
                and isinstance(value, dict)
            ):
                rating = value.get("rating")
                if isinstance(rating, str) and scale and rating in scale:
                    st.session_state[_rating_key(qid)] = (
                        scale.index(rating) + 1
                    )
                st.session_state[_text_key(qid)] = (
                    value.get("elaboration") or ""
                )
            elif qtype == "open_ended" and isinstance(value, str):
                st.session_state[_text_key(qid)] = value


def _collect_intake_answers(
    intake: dict[str, object], language: str
) -> dict[str, object]:
    """Read intake widget state into the canonical answer dict.

    Answers are stored as the *verbal label* in the active language (never the
    raw widget index) so the Supabase JSONB row is human-readable without
    cross-referencing `intake.json`. Shapes:

        open_ended                      -> str
        boolean / multiple_choice       -> str (the selected option label)
        boolean_with_text               -> {"choice": str, "elaboration": str}
        likert_with_open_elaboration    -> {"rating": str, "elaboration": str}

    Questions the researcher never touched are omitted.
    """
    out: dict[str, object] = {}
    for section in get_localized_sections(intake, language):
        scale = section["scale"]
        for q in section["questions"]:
            qid = q["id"]
            qtype = q["type"]
            options = q["options"]
            if qtype in ("boolean", "multiple_choice"):
                idx = st.session_state.get(_choice_key(qid))
                if isinstance(idx, int) and 0 <= idx < len(options):
                    out[qid] = options[idx]
            elif qtype == "boolean_with_text":
                idx = st.session_state.get(_choice_key(qid))
                if isinstance(idx, int) and 0 <= idx < len(options):
                    out[qid] = {
                        "choice": options[idx],
                        "elaboration": st.session_state.get(_text_key(qid), "") or "",
                    }
            elif qtype == "likert_with_open_elaboration":
                rating = st.session_state.get(_rating_key(qid))
                if (
                    isinstance(rating, int)
                    and scale
                    and 1 <= rating <= len(scale)
                ):
                    out[qid] = {
                        "rating": scale[rating - 1],
                        "elaboration": st.session_state.get(_text_key(qid), "") or "",
                    }
            elif qtype == "open_ended":
                text = st.session_state.get(_text_key(qid), "") or ""
                if text.strip():
                    out[qid] = text
    return out


def _answer_display(value: object) -> str:
    """Compact human-readable answer for current/previous comparisons."""
    if isinstance(value, dict):
        rating = value.get("rating")
        label = value.get("label")
        if rating is not None and label:
            return f"{rating} — {label}"
        if label:
            return str(label)
        if rating is not None:
            return str(rating)
    return str(value)


def _answer_reasoning(value: object) -> str:
    """Return the LLM rationale when present in an answer object."""
    if isinstance(value, dict):
        reasoning = value.get("reasoning")
        if isinstance(reasoning, str):
            return reasoning
    return ""


def _merge_answer_reasonings(
    answers: dict[str, Any],
    reasonings: dict[str, str] | None,
) -> dict[str, Any]:
    """Attach stored reasonings back to the DB answer shape for display."""
    if not reasonings:
        return answers
    merged: dict[str, Any] = {}
    for qid, value in answers.items():
        if isinstance(value, dict):
            enriched = dict(value)
            if qid in reasonings:
                enriched["reasoning"] = reasonings[qid]
            merged[qid] = enriched
        else:
            merged[qid] = value
    return merged


def _word_diff_html(previous: str, current: str) -> str:
    """Render a small inline word diff with red deletions / green additions."""
    tokens = difflib.ndiff(previous.split(), current.split())
    parts: list[str] = []
    for token in tokens:
        prefix = token[:2]
        text = html.escape(token[2:])
        if prefix == "- ":
            parts.append(
                "<del style='background:#ffd8d8;color:#8a1f1f;"
                "padding:0 2px;border-radius:3px'>"
                f"{text}</del>"
            )
        elif prefix == "+ ":
            parts.append(
                "<ins style='background:#d8f5d0;color:#176b2c;"
                "padding:0 2px;border-radius:3px;text-decoration:none'>"
                f"{text}</ins>"
            )
        elif prefix == "  ":
            parts.append(text)
    return " ".join(parts)


def _render_biography_change(
    current_bio: str,
    previous_simulation: dict[str, Any] | None,
) -> None:
    """Show the previous biography beneath the current one with a diff.

    The current biography is already rendered by the caller as the primary
    biography text area. This helper intentionally renders only the previous
    version and the highlighted difference below it so the left column reads:
    current biography -> previous biography -> what changed.
    """
    if not previous_simulation:
        return
    previous_bio = previous_simulation.get("biography") or {}
    previous_text = previous_bio.get("biography_text") or ""
    if not previous_text or previous_text == current_bio:
        return
    previous_rev = previous_bio.get("revision_number", "?")
    st.divider()
    st.markdown(f"#### {t('simulation_bio_changes_header', n=previous_rev)}")
    st.caption(t("simulation_previous_bio_header", n=previous_rev))
    st.text_area(
        t("simulation_previous_bio_label"),
        value=previous_text,
        height=220,
        disabled=True,
        label_visibility="collapsed",
    )
    with st.expander(t("simulation_bio_diff_header"), expanded=True):
        st.markdown(_word_diff_html(previous_text, current_bio), unsafe_allow_html=True)


def _render_questionnaire_results(
    answers: dict[str, Any] | None,
    questionnaire: dict[str, Any],
    language: str,
    *,
    model_label: str,
    biography_id: str | None,
    questionnaire_id: str | None,
    persona_id: str | None,
    researcher_name: str,
    previous_simulation: dict[str, Any] | None = None,
) -> None:
    """Render questionnaire answers, highlighting changes from last run."""
    if not answers:
        st.info(t("q_no_answers"))
        return

    st.caption(
        t(
            "q_model_caption",
            model=model_label,
            id=biography_id,
        )
    )

    previous_answers: dict[str, Any] = {}
    previous_rev: object = "?"
    if previous_simulation:
        previous_answers = _merge_answer_reasonings(
            previous_simulation.get("answers") or {},
            previous_simulation.get("reasonings"),
        )
        previous_bio = previous_simulation.get("biography") or {}
        previous_rev = previous_bio.get("revision_number", "?")
        st.info(t("simulation_comparison_info", n=previous_rev))

    rendered_question_ids: list[str] = []
    for section in get_questionnaire_sections(questionnaire, language):
        st.markdown(f"### {section['title']}")
        for q in section["questions"]:
            qid = q["id"]
            answer = answers.get(qid)
            if answer is None:
                continue
            rendered_question_ids.append(qid)

            previous = previous_answers.get(qid)
            current_text = _answer_display(answer)
            previous_text = (
                _answer_display(previous) if previous is not None else ""
            )
            changed = bool(previous_text and previous_text != current_text)

            st.markdown(f"**{q['question']}**")
            if changed:
                st.markdown(
                    f":orange[{t('simulation_answer_changed')}] "
                    f"**{current_text}**"
                )
                st.caption(
                    t(
                        "simulation_previous_answer",
                        answer=previous_text,
                    )
                )
            else:
                st.markdown(f"**{current_text}**")

            reasoning = _answer_reasoning(answer)
            if reasoning:
                st.caption(reasoning)
            if questionnaire_id:
                st.text_area(
                    t("answer_comment_label"),
                    key=_answer_comment_key(questionnaire_id, qid),
                    height=80,
                    placeholder=t("answer_comment_placeholder"),
                )
        st.divider()

    if not questionnaire_id:
        st.caption(t("answer_comments_unavailable"))
        return

    if st.button(
        t("save_answer_comments_button"),
        use_container_width=True,
        key=f"save_answer_comments_{questionnaire_id}",
    ):
        comments = {
            qid: st.session_state.get(
                _answer_comment_key(questionnaire_id, qid), ""
            )
            for qid in rendered_question_ids
        }
        try:
            saved_count = db.save_answer_comments(
                questionnaire_id=questionnaire_id,
                biography_id=biography_id or "",
                persona_id=persona_id,
                researcher_name=researcher_name,
                comments=comments,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(t("answer_comments_save_failed", error=exc))
        else:
            if saved_count:
                st.toast(
                    t("answer_comments_saved", count=saved_count), icon="💬"
                )
            else:
                st.info(t("answer_comments_none_to_save"))


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

def _secret_bool(key: str, default: bool = False) -> bool:
    """Read a boolean-like Streamlit secret.

    Streamlit secrets may come from TOML booleans (`true`) or string values
    (`"true"`) depending on deployment configuration. This keeps deployment
    flags forgiving without adding another config dependency.
    """
    value = st.secrets.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


# Language must be resolved before we call `t("page_title")`, so initialize it
# in session state first. In deployment mode the researcher-facing UI is locked
# to Hebrew only; local development keeps the language switch.
_DEPLOYMENT_MODE = _secret_bool("DEPLOYMENT_MODE")
st.session_state.setdefault("language", LANG_EN)
if _DEPLOYMENT_MODE:
    st.session_state.language = LANG_HE
    st.session_state["model_label"] = CHATGPT
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("auth_username", None)
st.session_state.setdefault("auth_researcher_name", "")

st.set_page_config(
    page_title=t("page_title"),
    page_icon="🧠",
    layout="wide",
)
inject_rtl_css()


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------

def _configured_researchers() -> dict[str, dict[str, str]]:
    """Return researcher login records from `[researchers.<username>]`.

    Expected secrets shape:

        [researchers.ofer]
        name = "Ofer"
        password = "..."

    Passwords are compared as plain strings. This is intentionally lightweight
    for the current internal Streamlit workflow; use Supabase Auth before
    exposing this app as a public, multi-user service.
    """
    raw = st.secrets.get("researchers", {})
    if not isinstance(raw, Mapping):
        return {}

    researchers: dict[str, dict[str, str]] = {}
    for username, config in raw.items():
        if not isinstance(config, Mapping):
            continue
        password = config.get("password")
        if not isinstance(password, str) or not password:
            continue
        display_name = config.get("name")
        researchers[str(username)] = {
            "name": str(display_name or username),
            "password": password,
        }
    return researchers


def _clear_persona_session_state() -> None:
    """Clear persona/browser state when the authenticated researcher changes."""
    st.session_state.update(
        persona_id=None,
        biography_id=None,
        biography_revision_number=0,
        biography_text="",
        latest_saved_biography_text="",
        initial_intake_answers={},
        biography_edit_mode=False,
        persona_is_final=False,
        questionnaire_answers=None,
        previous_simulation=None,
        session_id=None,
        messages=[],
        active_persona_loaded_for=None,
    )
    for key in ("bio_unsaved_area", "bio_edit_area", "bio_readonly_area"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Sidebar: language, researcher login, model choice, system status
# ---------------------------------------------------------------------------
with st.sidebar:
    is_authenticated = bool(st.session_state.authenticated)
    if not _DEPLOYMENT_MODE:
        st.header(t("sb_language_header"))
        # Language is editable only until the researcher logs in; after that
        # it's locked to keep UI + LLM output consistent within a session.
        # We always render the radio (disabled after login) so Streamlit's
        # stale-widget cleanup does not drop `st.session_state.language`.
        language_codes = [LANG_EN, LANG_HE]
        st.radio(
            t("sb_language_header"),
            options=language_codes,
            format_func=lambda code: LANGUAGE_LABELS[code],
            horizontal=True,
            key="language",
            label_visibility="collapsed",
            disabled=is_authenticated,
        )
        if is_authenticated:
            st.caption(
                t(
                    "sb_language_locked",
                    lang=LANGUAGE_LABELS[st.session_state.language],
                )
            )

    st.header(t("sb_researcher_header"))
    researchers = _configured_researchers()
    if st.session_state.authenticated:
        researcher_name = st.session_state.auth_researcher_name
        st.success(t("login_success", name=researcher_name))
        if st.button(t("logout_button"), use_container_width=True):
            _clear_persona_session_state()
            st.session_state.update(
                authenticated=False,
                auth_username=None,
                auth_researcher_name="",
            )
            st.rerun()
    else:
        researcher_name = ""
        if not researchers:
            st.error(t("login_no_researchers_configured"))
        login_username = st.text_input(
            t("login_username_label"),
            key="login_username",
            placeholder=t("login_username_placeholder"),
        ).strip()
        login_password = st.text_input(
            t("login_password_label"),
            type="password",
            key="login_password",
        )
        if st.button(t("login_button"), type="primary", use_container_width=True):
            record = researchers.get(login_username)
            if record and login_password == record["password"]:
                _clear_persona_session_state()
                st.session_state.update(
                    authenticated=True,
                    auth_username=login_username,
                    auth_researcher_name=record["name"],
                )
                st.rerun()
            else:
                st.error(t("login_failed"))

    if _DEPLOYMENT_MODE:
        model_label = CHATGPT
        st.session_state["model_label"] = CHATGPT
    else:
        st.header(t("sb_llm_header"))
        # Default model is environment-controlled: set `DEFAULT_MODEL` in
        # `st.secrets` to flip the initial selection. Local development
        # defaults to Ollama so no external API is required.
        _MODEL_OPTIONS = [CHATGPT, OLLAMA]
        _default_model = st.secrets.get("DEFAULT_MODEL", OLLAMA)
        if _default_model not in _MODEL_OPTIONS:
            _default_model = OLLAMA
        if st.session_state.get("model_label") not in _MODEL_OPTIONS:
            st.session_state["model_label"] = _default_model
        model_label = st.radio(
            t("sb_llm_label"),
            _MODEL_OPTIONS,
            horizontal=True,
            key="model_label",
        )

        st.header(t("sb_status_header"))
        if db.health_check():
            st.markdown(t("sb_supabase_connected"))
        else:
            st.markdown(t("sb_supabase_unreachable"))
            st.caption(t("sb_secrets_hint"))

# Gate: nothing below renders until the researcher logs in.
if not st.session_state.authenticated:
    st.title(t("page_title"))
    st.info(t("gate_login_info"))
    st.stop()


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
_DEFAULT_STATE: dict[str, object] = {
    # `persona_id` groups every biography revision produced while the
    # researcher iterates on the same character. `biography_id` is the
    # *current* revision's id (what chat / questionnaire pin to); it is
    # refreshed every time the researcher clicks "Save biography" or
    # "Save changes". See docs/persona_lifecycle_redesign.md.
    "persona_id": None,
    "biography_id": None,
    "biography_revision_number": 0,
    "biography_text": "",
    "latest_saved_biography_text": "",
    "initial_intake_answers": {},
    "biography_edit_mode": False,
    "persona_is_final": False,
    "questionnaire_answers": None,
    "current_questionnaire_id": None,
    "previous_simulation": None,
    "session_id": None,
    "messages": [],
    "active_persona_loaded_for": None,
}
for _key, _value in _DEFAULT_STATE.items():
    st.session_state.setdefault(_key, _value)

if st.session_state.active_persona_loaded_for != researcher_name:
    # Researcher identity is the persistent owner key for the current persona.
    # When a researcher enters their name (or switches to another name in the
    # same browser session), restore that researcher's latest unfinished
    # persona from Supabase. The biography should only disappear after Finish,
    # never just because Streamlit reran or the browser session was refreshed.
    st.session_state.update(
        persona_id=None,
        biography_id=None,
        biography_revision_number=0,
        biography_text="",
        latest_saved_biography_text="",
        initial_intake_answers={},
        biography_edit_mode=False,
        persona_is_final=False,
        questionnaire_answers=None,
        current_questionnaire_id=None,
        previous_simulation=None,
        session_id=None,
        messages=[],
        active_persona_loaded_for=researcher_name,
    )
    st.session_state.pop("bio_unsaved_area", None)
    st.session_state.pop("bio_edit_area", None)
    st.session_state.pop("bio_readonly_area", None)

    try:
        active_persona = db.fetch_active_persona_for_researcher(
            researcher_name
        )
    except Exception as exc:  # noqa: BLE001 — keep the app usable if DB fails
        st.warning(f"Could not load active persona for this researcher: {exc}")
        active_persona = None

    if active_persona:
        bio_text = active_persona.get("biography_text") or ""
        active_persona_id = (
            active_persona.get("persona_id") or active_persona.get("id")
        )
        active_biography_id = active_persona.get("id")
        existing_simulation = None
        previous_simulation = None
        if isinstance(active_biography_id, str):
            existing_simulation = db.fetch_latest_questionnaire_for_biography(
                active_biography_id
            )
        if existing_simulation and isinstance(active_persona_id, str):
            previous_simulation = db.fetch_previous_questionnaire_for_persona(
                active_persona_id,
                current_biography_id=active_biography_id,
            )
        st.session_state.update(
            persona_id=active_persona_id,
            biography_id=active_biography_id,
            biography_revision_number=active_persona.get(
                "revision_number", 1
            ),
            biography_text=bio_text,
            latest_saved_biography_text=bio_text,
            initial_intake_answers=active_persona.get(
                "initial_intake_answers"
            )
            or active_persona.get("intake_answers")
            or {},
            biography_edit_mode=False,
            persona_is_final=bool(active_persona.get("is_final")),
            questionnaire_answers=(
                _merge_answer_reasonings(
                    existing_simulation.get("answers") or {},
                    existing_simulation.get("reasonings"),
                )
                if existing_simulation
                else None
            ),
            current_questionnaire_id=(
                existing_simulation.get("id") if existing_simulation else None
            ),
            previous_simulation=previous_simulation,
            session_id=str(uuid.uuid4()),
            messages=[],
        )

# Browser sessions created before `latest_saved_biography_text` existed may
# already have an active persona plus `biography_text` but an empty saved copy.
# Backfill it once so the read-only view keeps showing the current persona's
# latest biography until the researcher explicitly clicks "Finish persona".
if (
    st.session_state.persona_id is not None
    and not st.session_state.latest_saved_biography_text
    and st.session_state.biography_text
):
    st.session_state.latest_saved_biography_text = (
        st.session_state.biography_text
    )

language = st.session_state.language
questionnaire = load_questionnaire()


# ---------------------------------------------------------------------------
# Main layout: wide intake area (70%) + narrow tabs sidebar (30%)
# ---------------------------------------------------------------------------
# The Persona Setup area is wide because the researcher spends most of their
# time in the intake form; the Chat / Questionnaire / Log tabs live in a
# narrow side panel for reviewing results. The `direction: rtl` flip applied
# by `inject_rtl_css()` automatically reverses the visual order of these
# columns for Hebrew — so in English the intake is on the visual LEFT and
# the tabs on the RIGHT, while in Hebrew the intake appears on the visual
# RIGHT (reader's start side) and the tabs on the LEFT, without any
# language-specific Python branching here.
st.title(t("app_title"))

intake_col, tabs_col = st.columns([7, 3], gap="large")

with intake_col:
    st.subheader(t("persona_setup"))
    st.caption(t("persona_caption"))
    persona_exists = st.session_state.persona_id is not None
    if persona_exists:
        st.info(
            t(
                "active_persona_continue_info",
                name=researcher_name,
            )
        )

    # --- Intake form -------------------------------------------------------
    # After a simulation has been generated, hide the intake form so the
    # right-side simulation panel is visually paired only with the biography.
    # Saving a new biography revision clears `questionnaire_answers`, which
    # brings the locked intake snapshot back until the next simulation run.
    show_intake_form = st.session_state.questionnaire_answers is None
    try:
        intake_data = load_intake()
    except (FileNotFoundError, ValueError) as exc:
        intake_data = None
        st.warning(t("intake_load_failed", error=exc))

    if intake_data and show_intake_form:
        st.subheader(t("intake_form_header"))
        if persona_exists:
            st.caption(t("intake_locked_caption"))
            saved_intake_answers = st.session_state.initial_intake_answers
            if isinstance(saved_intake_answers, dict):
                _seed_intake_widget_state_from_answers(
                    intake_data, saved_intake_answers, language
                )
        else:
            st.caption(t("intake_form_caption"))

            # Randomize lives at the top so the researcher can fill the whole
            # form in one click before scrolling, then tweak individual answers
            # as needed. `on_click` fires before widgets re-instantiate, which
            # is required to mutate their session_state keys. After the
            # structured random picks, the callback also calls the LLM to
            # generate short answers for every `open_ended` question — that
            # way the "open-ended must be filled" rule is always satisfied
            # once Randomize returns.
            st.button(
                t("intake_randomize_button"),
                use_container_width=True,
                type="primary",
                key="intake_randomize_btn",
                on_click=_apply_random_intake_answers,
                args=(intake_data, model_label, language),
            )

        for section in get_localized_sections(intake_data, language):
            st.markdown(f"##### {section['title']}")
            for question in section["questions"]:
                _render_intake_question(
                    question,
                    section["scale"],
                    disabled=persona_exists,
                )
            st.divider()

        # Block drafting until every open-ended question has text — either
        # typed by the researcher or generated by Randomize. The banner
        # lists which question ids are still missing so the researcher
        # doesn't have to hunt for them.
        if not persona_exists:
            missing_open_ended = _missing_open_ended_ids(intake_data)
            if missing_open_ended:
                st.info(
                    t(
                        "intake_open_ended_missing",
                        ids=", ".join(missing_open_ended),
                    )
                )

            # Draft biography stays at the bottom because it consumes the
            # current state of all answers above it.
            draft_clicked = st.button(
                t("intake_draft_button"),
                use_container_width=True,
                type="secondary",
                key="intake_draft_btn",
                disabled=bool(missing_open_ended),
            )

            if draft_clicked:
                answers_payload = _collect_intake_answers(
                    intake_data, language
                )
                try:
                    with st.spinner(
                        t("intake_drafting_spinner", model=model_label)
                    ):
                        drafted = llm.generate_biography(
                            model_label,
                            intake_data,
                            answers_payload,
                            language,
                        )
                except Exception as exc:  # noqa: BLE001 — surface LLM errors to the UI
                    st.error(t("intake_draft_failed", error=exc))
                else:
                    st.session_state.biography_text = drafted
                    st.session_state.pop("bio_unsaved_area", None)
                    st.toast(t("intake_draft_success"), icon="✅")
                    st.rerun()

    # --- Biography panel: three mutually-exclusive states -----------------
    # The persona lifecycle is:
    #   (1) no persona yet          -> editable bio + Save biography
    #   (2) saved, read-only view   -> Edit biography / Finish persona /
    #                                  Generate questionnaire responses
    #   (3) editing a saved bio     -> editable bio + Save changes /
    #                                  Cancel edit / Finish persona
    # Each "save" in states 1 and 3 inserts a new row in `biographies`
    # sharing the same `persona_id`, with `revision_number` monotonically
    # increasing. See docs/persona_lifecycle_redesign.md.
    edit_mode = bool(st.session_state.biography_edit_mode)

    if not persona_exists:
        # --- State 1: initial draft, not yet saved ------------------------
        if "bio_unsaved_area" not in st.session_state:
            st.session_state.bio_unsaved_area = st.session_state.biography_text
        bio = st.text_area(
            t("bio_label"),
            height=400,
            placeholder=t("bio_placeholder"),
            key="bio_unsaved_area",
        )
        st.session_state.biography_text = bio

        if st.button(
            t("save_bio_button"),
            type="primary",
            disabled=not bio.strip(),
            use_container_width=True,
            key="save_bio_btn",
        ):
            intake_payload = (
                _collect_intake_answers(intake_data, language)
                if intake_data
                else {}
            )
            try:
                with st.spinner(t("save_bio_spinner")):
                    revision_id, new_persona_id = db.insert_biography(
                        researcher_name,
                        bio,
                        persona_id=None,
                        revision_number=1,
                        intake_answers=intake_payload or None,
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(t("save_bio_failed", error=exc))
            else:
                st.session_state.update(
                    persona_id=new_persona_id,
                    biography_id=revision_id,
                    biography_revision_number=1,
                    biography_text=bio,
                    latest_saved_biography_text=bio,
                    initial_intake_answers=intake_payload or {},
                    biography_edit_mode=False,
                    persona_is_final=False,
                    questionnaire_answers=None,
                    current_questionnaire_id=None,
                    previous_simulation=None,
                    session_id=str(uuid.uuid4()),
                    messages=[],
                )
                st.session_state.pop("bio_unsaved_area", None)
                st.session_state.pop("bio_edit_area", None)
                st.session_state.pop("bio_readonly_area", None)
                st.toast(t("save_bio_success_toast"), icon="✅")
                st.rerun()

    elif edit_mode:
        # --- State 3: editing a saved biography ---------------------------
        if "bio_edit_area" not in st.session_state:
            st.session_state.bio_edit_area = st.session_state.biography_text
        bio = st.text_area(
            t("bio_label"),
            height=400,
            placeholder=t("bio_placeholder"),
            key="bio_edit_area",
        )
        # Keep the in-progress edit visible across reruns. Simulation actions
        # are hidden in edit mode, so this draft text cannot be used before
        # the researcher explicitly clicks "Save changes".
        st.session_state.biography_text = bio
        st.caption(
            t("bio_edit_hint", n=st.session_state.biography_revision_number)
        )
        col_save, col_cancel = st.columns(2)
        if col_save.button(
            t("save_changes_button"),
            type="primary",
            disabled=not bio.strip(),
            use_container_width=True,
            key="save_changes_btn",
        ):
            if bio.strip() == st.session_state.latest_saved_biography_text.strip():
                st.session_state.update(
                    biography_edit_mode=False,
                    biography_text=st.session_state.latest_saved_biography_text,
                )
                st.session_state.pop("bio_edit_area", None)
                st.session_state.pop("bio_readonly_area", None)
                st.toast(t("save_changes_noop_toast"), icon="ℹ️")
                st.rerun()
            next_rev = st.session_state.biography_revision_number + 1
            intake_payload = st.session_state.initial_intake_answers
            try:
                with st.spinner(t("save_changes_spinner")):
                    revision_id, _ = db.insert_biography(
                        researcher_name,
                        bio,
                        persona_id=st.session_state.persona_id,
                        revision_number=next_rev,
                        intake_answers=intake_payload or None,
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(t("save_changes_failed", error=exc))
            else:
                st.session_state.update(
                    biography_id=revision_id,
                    biography_revision_number=next_rev,
                    biography_text=bio,
                    latest_saved_biography_text=bio,
                    biography_edit_mode=False,
                    questionnaire_answers=None,
                    current_questionnaire_id=None,
                    previous_simulation=None,
                )
                # Drop the widget-state slot so next edit re-seeds from
                # `biography_text` instead of the now-saved-but-stale value.
                st.session_state.pop("bio_edit_area", None)
                st.session_state.pop("bio_readonly_area", None)
                st.toast(
                    t("save_changes_success_toast", n=next_rev), icon="✅"
                )
                st.rerun()
        if col_cancel.button(
            t("cancel_edit_button"),
            use_container_width=True,
            key="cancel_edit_btn",
        ):
            st.session_state.update(
                biography_edit_mode=False,
                biography_text=st.session_state.latest_saved_biography_text,
            )
            # Drop the in-flight edit so re-entering edit mode later starts
            # from the last saved biography, not the discarded changes.
            st.session_state.pop("bio_edit_area", None)
            st.session_state.pop("bio_readonly_area", None)
            st.rerun()

    else:
        # --- State 2: saved view, read-only -------------------------------
        st.text_area(
            t("bio_label"),
            value=st.session_state.latest_saved_biography_text,
            height=400,
            placeholder=t("bio_placeholder"),
            disabled=True,
        )
        st.caption(
            t(
                "bio_readonly_hint",
                n=st.session_state.biography_revision_number,
            )
        )
        _render_biography_change(
            st.session_state.latest_saved_biography_text,
            st.session_state.previous_simulation,
        )
        col_edit, col_finish = st.columns(2)
        if col_edit.button(
            t("edit_bio_button"),
            type="primary",
            use_container_width=True,
            key="edit_bio_btn",
        ):
            st.session_state.biography_edit_mode = True
            st.session_state.biography_text = (
                st.session_state.latest_saved_biography_text
            )
            # Reset any stale widget state so the edit area seeds from the
            # current `biography_text` rather than an older cached value.
            st.session_state.pop("bio_edit_area", None)
            st.session_state.pop("bio_readonly_area", None)
            st.rerun()
        if col_finish.button(
            t("finish_persona_button"),
            use_container_width=True,
            key="finish_persona_btn",
        ):
            try:
                with st.spinner(t("finish_persona_spinner")):
                    db.finalize_persona(st.session_state.persona_id)
            except Exception as exc:  # noqa: BLE001
                st.error(t("finish_persona_failed", error=exc))
            else:
                # Reset every persona-scoped session key so the next render
                # lands back in State 1 with a clean slate.
                st.session_state.update(
                    persona_id=None,
                    biography_id=None,
                    biography_revision_number=0,
                    biography_text="",
                    latest_saved_biography_text="",
                    initial_intake_answers={},
                    biography_edit_mode=False,
                    persona_is_final=False,
                    questionnaire_answers=None,
                    current_questionnaire_id=None,
                    previous_simulation=None,
                    session_id=None,
                    messages=[],
                )
                st.session_state.pop("bio_unsaved_area", None)
                st.session_state.pop("bio_edit_area", None)
                st.session_state.pop("bio_readonly_area", None)
                st.toast(t("finish_persona_success_toast"), icon="🏁")
                st.rerun()

        # Generate questionnaire responses — unchanged behavior, but only
        # reachable in the saved read-only view (edit mode blocks it so
        # unsaved edits can't leak into the LLM call).
        if st.session_state.questionnaire_answers is None:
            if st.button(
                t("generate_q_button"),
                type="primary",
                use_container_width=True,
                key="generate_q_btn",
            ):
                try:
                    with st.spinner(
                        t("generate_q_spinner", model=model_label)
                    ):
                        previous_simulation = (
                            db.fetch_previous_questionnaire_for_persona(
                                st.session_state.persona_id,
                                current_biography_id=st.session_state.biography_id,
                            )
                        )
                        answers = llm.answer_questionnaire(
                            model_label,
                            st.session_state.latest_saved_biography_text,
                            questionnaire,
                            language,
                        )
                        # {qid: {"rating", "label", "reasoning"}}. Split
                        # rationales into their own JSONB column so the
                        # `answers` payload stays compact and the
                        # rationales are easy to read / hide in Supabase.
                        structured_answers: dict[str, Any] = {}
                        reasonings: dict[str, str] = {}
                        for qid, value in answers.items():
                            if isinstance(value, dict):
                                rating = value.get("rating")
                                label = value.get("label")
                                reasoning = value.get("reasoning") or ""
                                structured_answers[qid] = {
                                    "rating": rating,
                                    "label": label,
                                }
                                if reasoning:
                                    reasonings[qid] = reasoning
                            else:
                                # Legacy bare-string answers.
                                structured_answers[qid] = value
                        questionnaire_id = db.insert_questionnaire(
                            st.session_state.biography_id,
                            model_label,
                            structured_answers,
                            reasonings or None,
                            persona_id=st.session_state.persona_id,
                        )
                except Exception as exc:  # noqa: BLE001
                    st.error(t("generate_q_failed", error=exc))
                else:
                    st.session_state.questionnaire_answers = answers
                    st.session_state.current_questionnaire_id = questionnaire_id
                    st.session_state.previous_simulation = previous_simulation
                    st.toast(t("generate_q_success_toast"), icon="✅")
                    st.rerun()

    if persona_exists:
        st.caption(
            t(
                "active_persona_revision",
                id=st.session_state.persona_id,
                n=st.session_state.biography_revision_number,
                final_suffix=(
                    t("active_persona_final_suffix")
                    if st.session_state.persona_is_final
                    else ""
                ),
            )
        )

with tabs_col:
    # Wrap the side panel in a bordered container so it reads as a clearly
    # separated workspace — distinct from the intake form on the main side.
    side_panel = st.container(border=True)
with side_panel:
    # The simulation result now lives directly in the side panel next to the
    # biography instead of behind a tab, so after clicking "Generate
    # questionnaire responses" the researcher immediately sees the answers.
    # Biography version comparison is rendered beneath the biography itself.
    st.subheader(t("simulation_results_header"))
    _render_questionnaire_results(
        st.session_state.questionnaire_answers,
        questionnaire,
        language,
        model_label=model_label,
        biography_id=st.session_state.biography_id,
        questionnaire_id=st.session_state.current_questionnaire_id,
        persona_id=st.session_state.persona_id,
        researcher_name=researcher_name,
        previous_simulation=st.session_state.previous_simulation,
    )

    # -----------------------------------------------------------------------
    # Log History
    # -----------------------------------------------------------------------
    st.divider()
    with st.expander(t("tab_history"), expanded=False):
        st.caption(t("log_caption", name=researcher_name))
        try:
            rows = db.fetch_recent_biographies(researcher_name, limit=10)
        except Exception as exc:  # noqa: BLE001
            st.error(t("log_error", error=exc))
        else:
            if not rows:
                st.info(t("log_empty"))
            else:
                st.dataframe(
                    rows,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "id": st.column_config.TextColumn(
                            t("log_col_id"), width="small"
                        ),
                        "persona_id": st.column_config.TextColumn(
                            t("log_col_persona"), width="small"
                        ),
                        "revision_number": st.column_config.NumberColumn(
                            t("log_col_revision"), width="small"
                        ),
                        "is_final": st.column_config.CheckboxColumn(
                            t("log_col_final"), width="small"
                        ),
                        "researcher_name": st.column_config.TextColumn(
                            t("log_col_researcher"), width="small"
                        ),
                        "biography_text": st.column_config.TextColumn(
                            t("log_col_biography"), width="large"
                        ),
                        "created_at": st.column_config.DatetimeColumn(
                            t("log_col_created"), format="YYYY-MM-DD HH:mm"
                        ),
                    },
                )
