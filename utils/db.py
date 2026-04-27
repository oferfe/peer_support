"""Supabase data-access layer.

All calls go through a single cached client; each helper is a thin wrapper
around one `.insert()` or `.select()` that returns plain Python types so the
Streamlit UI never touches the Supabase SDK directly.

UUIDs are generated client-side with `uuid.uuid4()` so that inserts can return
the new row's id without an extra round-trip.
"""

from __future__ import annotations

from datetime import UTC, datetime
import uuid
from typing import Any

import streamlit as st
from supabase import Client, create_client


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_client() -> Client:
    """Return a cached Supabase client built from Streamlit secrets."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def health_check() -> bool:
    """Return True iff a trivial query against `biographies` succeeds."""
    try:
        get_client().table("biographies").select("id").limit(1).execute()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Inserts
# ---------------------------------------------------------------------------

def insert_biography(
    researcher_name: str,
    biography_text: str,
    *,
    persona_id: str | None = None,
    revision_number: int = 1,
    intake_answers: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Insert a biography revision. Returns `(revision_id, persona_id)`.

    A *persona* is the group of all revisions produced while the researcher
    iterates on the same character. The first save of a persona calls this
    with `persona_id=None` / `revision_number=1`; a fresh UUID is generated
    and returned. Each subsequent "Save changes" passes the same
    `persona_id` back in together with the next `revision_number`, producing
    an append-only revision history under that persona key.

    `intake_answers` is the structured intake payload (`question_id -> value`)
    that produced the biography; see `utils.intake` for the canonical shape.
    Pass `None` when the biography was authored free-form without going
    through the intake form — the column is left NULL.
    """
    revision_id = str(uuid.uuid4())
    resolved_persona_id = persona_id or str(uuid.uuid4())
    row: dict[str, Any] = {
        "id": revision_id,
        "researcher_name": researcher_name,
        "biography_text": biography_text,
        "persona_id": resolved_persona_id,
        "revision_number": revision_number,
    }
    if intake_answers is not None:
        row["intake_answers"] = intake_answers
    get_client().table("biographies").insert(row).execute()
    return revision_id, resolved_persona_id


def finalize_persona(persona_id: str) -> None:
    """Mark the latest revision of a persona as final.

    "Latest" is resolved by the highest `revision_number` within the
    persona — the one we just updated `is_final` / `finalized_at` on. Older
    revisions are left alone so the full history is preserved and still
    queryable.
    """
    client = get_client()
    latest = (
        client.table("biographies")
        .select("id")
        .eq("persona_id", persona_id)
        .order("revision_number", desc=True)
        .limit(1)
        .execute()
    )
    rows = latest.data or []
    if not rows:
        return
    latest_id = rows[0]["id"]
    client.table("biographies").update(
        {
            "is_final": True,
            "finalized_at": datetime.now(UTC).isoformat(),
        }
    ).eq("id", latest_id).execute()


def insert_questionnaire(
    biography_id: str,
    model_used: str,
    answers: dict[str, Any],
    reasonings: dict[str, str] | None = None,
    *,
    persona_id: str | None = None,
) -> str:
    """Persist a structured questionnaire response and return its id.

    `answers` is the per-statement structured choice — currently
    `{statement_id: {"rating": int, "label": str}}`.
    `reasonings` is the sibling map `{statement_id: rationale_text}` produced
    by the LLM in the persona's voice; stored in its own column so it stays
    easy to read and query in Supabase. Pass `None` to leave the column NULL
    (e.g. for older code paths or models that did not return rationales).
    `persona_id` is an optional grouping column that makes it cheap to fetch
    every questionnaire run for a persona regardless of which revision was
    active at the time.
    """
    questionnaire_id = str(uuid.uuid4())
    row: dict[str, Any] = {
        "id": questionnaire_id,
        "biography_id": biography_id,
        "model_used": model_used,
        "answers": answers,
    }
    if reasonings is not None:
        row["reasonings"] = reasonings
    if persona_id is not None:
        row["persona_id"] = persona_id
    get_client().table("questionnaires").insert(row).execute()
    return questionnaire_id


def save_answer_comments(
    *,
    questionnaire_id: str,
    biography_id: str,
    persona_id: str | None,
    researcher_name: str,
    comments: dict[str, str],
) -> int:
    """Upsert researcher comments for answers in one questionnaire run.

    `comments` is `{question_id: comment_text}`. Empty comments are ignored.
    Returns the number of comments written. The table has a unique constraint
    on `(questionnaire_id, question_id, researcher_name)`, so saving again
    updates the existing comment for that answer instead of creating duplicate
    rows.
    """
    now = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for question_id, comment in comments.items():
        text = comment.strip()
        if not text:
            continue
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "questionnaire_id": questionnaire_id,
                "biography_id": biography_id,
                "persona_id": persona_id,
                "researcher_name": researcher_name,
                "question_id": question_id,
                "comment_text": text,
                "updated_at": now,
            }
        )
    if not rows:
        return 0
    get_client().table("answer_comments").upsert(
        rows,
        on_conflict="questionnaire_id,question_id,researcher_name",
    ).execute()
    return len(rows)


def insert_chat_message(
    biography_id: str,
    session_id: str,
    role: str,
    content: str,
    model_used: str,
    *,
    persona_id: str | None = None,
) -> None:
    """Persist a single chat turn (user or assistant).

    `persona_id` is an optional grouping column that mirrors the one on
    `questionnaires` — it lets callers query "every message ever logged for
    persona X" without joining through `biographies`.
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
    row: dict[str, Any] = {
        "biography_id": biography_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "model_used": model_used,
    }
    if persona_id is not None:
        row["persona_id"] = persona_id
    get_client().table("chat_logs").insert(row).execute()


# ---------------------------------------------------------------------------
# Fetches
# ---------------------------------------------------------------------------

def fetch_active_persona_for_researcher(
    researcher_name: str,
    *,
    limit: int = 200,
) -> dict[str, Any] | None:
    """Return the latest unfinished persona revision for this researcher.

    The `biographies` table stores revisions, not personas. A persona is
    considered active when its latest revision has not been finalized. Since
    only the latest revision gets `is_final = true` on Finish, we fetch recent
    rows newest-first, keep the first row we see for each `persona_id` (that is
    that persona's latest revision), and return the newest persona whose latest
    revision is still unfinished.
    """
    resp = (
        get_client()
        .table("biographies")
        .select(
            "id, persona_id, revision_number, is_final, intake_answers, "
            "researcher_name, biography_text, created_at"
        )
        .eq("researcher_name", researcher_name)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    seen_personas: set[str] = set()
    for row in resp.data or []:
        persona_id = row.get("persona_id") or row.get("id")
        if not isinstance(persona_id, str) or persona_id in seen_personas:
            continue
        seen_personas.add(persona_id)
        if not row.get("is_final"):
            initial = (
                get_client()
                .table("biographies")
                .select("intake_answers")
                .eq("persona_id", persona_id)
                .order("revision_number", desc=False)
                .limit(1)
                .execute()
            )
            initial_rows = initial.data or []
            if initial_rows:
                row["initial_intake_answers"] = initial_rows[0].get(
                    "intake_answers"
                )
            return row
    return None


def fetch_recent_biographies(
    researcher_name: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return the most recent biographies created by this researcher."""
    resp = (
        get_client()
        .table("biographies")
        .select(
            "id, persona_id, revision_number, is_final, "
            "researcher_name, biography_text, created_at"
        )
        .eq("researcher_name", researcher_name)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data or []


def fetch_latest_questionnaire_for_biography(
    biography_id: str,
) -> dict[str, Any] | None:
    """Return the latest questionnaire run for one biography revision."""
    resp = (
        get_client()
        .table("questionnaires")
        .select(
            "id, biography_id, persona_id, model_used, "
            "answers, reasonings, created_at"
        )
        .eq("biography_id", biography_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    return rows[0] if rows else None


def fetch_previous_questionnaire_for_persona(
    persona_id: str,
    *,
    current_biography_id: str,
    limit: int = 20,
) -> dict[str, Any] | None:
    """Return the latest questionnaire run for an earlier revision.

    Used after simulating the current biography revision: we look backward
    within the same persona and skip questionnaire rows tied to the current
    `biography_id`. The returned dict includes both the questionnaire payload
    and the previous biography row so the UI can show biography and answer
    diffs together.
    """
    client = get_client()
    q_resp = (
        client.table("questionnaires")
        .select("id, biography_id, model_used, answers, reasonings, created_at")
        .eq("persona_id", persona_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    for row in q_resp.data or []:
        previous_bio_id = row.get("biography_id")
        if not previous_bio_id or previous_bio_id == current_biography_id:
            continue

        bio_resp = (
            client.table("biographies")
            .select("id, persona_id, revision_number, biography_text, created_at")
            .eq("id", previous_bio_id)
            .limit(1)
            .execute()
        )
        bio_rows = bio_resp.data or []
        if not bio_rows:
            continue
        row["biography"] = bio_rows[0]
        return row
    return None
