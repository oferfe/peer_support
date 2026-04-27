-- Schema for the Persona Chatbot research dashboard.
-- Run this once in the Supabase SQL editor (or via `psql`) on a fresh project.

-- gen_random_uuid() lives in pgcrypto.
create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------------
-- biographies: one row per persona definition created by a researcher.
-- ---------------------------------------------------------------------------
create table if not exists biographies (
    id              uuid        primary key default gen_random_uuid(),
    researcher_name text        not null,
    biography_text  text        not null,
    created_at      timestamptz not null default now()
);

create index if not exists biographies_researcher_idx
    on biographies (researcher_name, created_at desc);

-- ---------------------------------------------------------------------------
-- questionnaires: structured LLM answers to the fixed instrument, one row
-- per (biography, model) run.
-- ---------------------------------------------------------------------------
create table if not exists questionnaires (
    id           uuid        primary key default gen_random_uuid(),
    biography_id uuid        not null references biographies(id) on delete cascade,
    model_used   text        not null,
    answers      jsonb       not null,
    created_at   timestamptz not null default now()
);

create index if not exists questionnaires_biography_idx
    on questionnaires (biography_id, created_at desc);

-- ---------------------------------------------------------------------------
-- chat_logs: every chat turn (user + assistant) for every session.
-- session_id groups the messages of a single conversation.
-- ---------------------------------------------------------------------------
create table if not exists chat_logs (
    id           uuid        primary key default gen_random_uuid(),
    biography_id uuid        not null references biographies(id) on delete cascade,
    session_id   uuid        not null,
    role         text        not null check (role in ('user', 'assistant')),
    content      text        not null,
    model_used   text        not null,
    created_at   timestamptz not null default now()
);

create index if not exists chat_logs_biography_idx
    on chat_logs (biography_id, created_at);

create index if not exists chat_logs_session_idx
    on chat_logs (session_id, created_at);

-- ---------------------------------------------------------------------------
-- Migrations
-- ---------------------------------------------------------------------------
-- These `alter table ... add column if not exists` statements are idempotent,
-- so re-running this whole file on an existing project is safe.

-- Step 13 — Structured intake form.
-- `intake_answers` stores the full set of intake answers (question_id -> value)
-- that produced the biography. See utils/intake.py for the canonical shape.
alter table biographies
    add column if not exists intake_answers jsonb;

-- Step 13 — Per-statement reasoning for questionnaire answers.
-- `answers`     keeps only the structured choice ({rating, label}) per statement.
-- `reasonings`  is a sibling jsonb mapping statement_id -> short rationale text
-- the LLM produced in the persona's voice. Split out into its own column so it
-- is easy to read / hide in the Supabase table view and to query independently.
alter table questionnaires
    add column if not exists reasonings jsonb;

-- Step 15 — Persona lifecycle (revisions + finalization).
-- One persona can have many biography revisions. Every save of the biography
-- (initial save + each "Save changes" while editing) inserts a new row in
-- `biographies` that shares the same `persona_id` as the previous revision of
-- the same persona, with `revision_number` monotonically increasing. When the
-- researcher clicks "Finish persona", the latest revision flips to
-- `is_final = true` and `finalized_at = now()`. See
-- docs/persona_lifecycle_redesign.md for the full design.
alter table biographies
    add column if not exists persona_id      uuid,
    add column if not exists revision_number integer not null default 1,
    add column if not exists is_final        boolean not null default false,
    add column if not exists finalized_at    timestamptz;

-- Backfill existing rows so each pre-migration biography becomes its own
-- single-revision persona. Idempotent: only touches rows missing persona_id.
update biographies
    set persona_id = id
    where persona_id is null;

alter table biographies
    alter column persona_id set not null;

create index if not exists biographies_persona_idx
    on biographies (persona_id, revision_number);

-- Sibling grouping columns so chat / questionnaire activity can be queried
-- by persona without joining through biographies. Nullable and unconstrained
-- (no FK) — the integrity anchor is still the existing biography_id FK.
alter table chat_logs
    add column if not exists persona_id uuid;

alter table questionnaires
    add column if not exists persona_id uuid;

-- Step 16 — Researcher feedback on questionnaire answers.
-- Stores optional free-text comments written by the researcher for specific
-- simulation answers. Each row is tied to the exact questionnaire run and
-- answer id, while also carrying biography_id/persona_id for easier filtering.
create table if not exists answer_comments (
    id               uuid        primary key default gen_random_uuid(),
    questionnaire_id uuid        not null references questionnaires(id) on delete cascade,
    biography_id     uuid        not null references biographies(id) on delete cascade,
    persona_id       uuid,
    researcher_name  text        not null,
    question_id      text        not null,
    comment_text     text        not null,
    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now(),
    unique (questionnaire_id, question_id, researcher_name)
);

create index if not exists answer_comments_questionnaire_idx
    on answer_comments (questionnaire_id, question_id);

create index if not exists answer_comments_persona_idx
    on answer_comments (persona_id, created_at desc);
