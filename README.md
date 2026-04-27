# Persona Chatbot & Evaluation Dashboard

A Streamlit dashboard for psychology researchers. Define a character's
**biography**, pick an LLM (**ChatGPT** for deployment or **Ollama** for a
local open-source model during development), and have it answer a fixed
psychological questionnaire *in character*. Every biography revision and
questionnaire result is logged to Supabase for later analysis.

---

## Prerequisites

- **Python 3.11+**
- A **Supabase** project (free tier is fine) — [supabase.com](https://supabase.com)
- An **OpenAI API key** (for GPT-4o) — [platform.openai.com](https://platform.openai.com)
- **Ollama** installed locally for the open-source path — [ollama.com](https://ollama.com)
  - After install, pull the default model: `ollama pull gemma3:12b`
  - Any Ollama-compatible model works; set `OLLAMA_MODEL` in
    `.streamlit/secrets.toml` to pick a different one.

## Setup

```bash
git clone <this-repo>
cd peer_support

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Supabase configuration (one-time)

1. Create a new project at [supabase.com/dashboard](https://supabase.com/dashboard).
2. Open **SQL Editor → New query**, paste the entire contents of
   [`supabase/schema.sql`](supabase/schema.sql), and click **Run**.
3. Confirm the **Table Editor** lists three tables: `biographies`,
   `questionnaires`, `chat_logs`.
4. Open **Project Settings → API** and copy the **Project URL** and either
   the `anon` or `service_role` **API key** — you'll paste these into
   `secrets.toml` below.

## Secrets configuration

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Open the new `.streamlit/secrets.toml` and fill in all values:

```toml
SUPABASE_URL   = "https://<project-ref>.supabase.co"
SUPABASE_KEY   = "<anon or service_role key>"

OPENAI_API_KEY = "sk-..."

OPENAI_MODEL   = "gpt-4o"

# Deployment mode locks the researcher-facing UI to Hebrew + ChatGPT and hides
# language/model/system-status controls. Use this in Streamlit Cloud.
# DEPLOYMENT_MODE = true

# Ollama: reads OLLAMA_BASE_URL (default http://localhost:11434/v1),
# OLLAMA_MODEL (default gemma3:12b), and OLLAMA_API_KEY (default "ollama").
OLLAMA_MODEL   = "gemma3:12b"

# Default model for the sidebar radio. Leave unset locally to default to
# Ollama; set to "ChatGPT" in the Streamlit Cloud secrets of the deployed
# app so it starts on OpenAI.
# DEFAULT_MODEL = "Ollama"

# Researcher logins. Add one section per researcher. The section name is the
# login username; `name` is the display name saved in Supabase.
[researchers.ofer]
name = "Ofer"
password = "replace-with-a-private-password"
```

The real `secrets.toml` is gitignored and must never be committed.

### Running Ollama locally

Start the Ollama daemon (it typically runs automatically after install; if
not, `ollama serve`) and confirm the default model is available:

```bash
ollama pull gemma3:12b
ollama list
```

The app talks to Ollama over its OpenAI-compatible endpoint at
`http://localhost:11434/v1`, so no extra Python dependency is needed.

## Run the app

**From the activated venv:**

```bash
streamlit run app.py
```

If `streamlit` prints an ImportError about `supabase`, make sure the venv is
active (`which streamlit` should print a path ending in `.venv/bin/streamlit`)
or launch directly with `./.venv/bin/streamlit run app.py`.

The app opens at [http://localhost:8501](http://localhost:8501).

## Usage

1. Type your name in the sidebar to unlock the main UI.
2. Select **ChatGPT** or **Ollama** in the sidebar (the default follows
   `DEFAULT_MODEL` in `secrets.toml`, falling back to Ollama for local dev).
3. Fill the intake form (or click **Randomize** to have the selected model
   populate every field including the open-ended questions), then click
   **Draft biography** followed by **Save biography** to persist the first
   revision of the persona.
4. Make changes via **Edit biography** → **Save changes** — each save writes
   a new revision under the same `persona_id`. See
   [docs/persona_lifecycle_redesign.md](docs/persona_lifecycle_redesign.md)
   for the full lifecycle.
5. Switch between tabs on the right:
   - **Questionnaire Results** — structured JSON answers for the current
     persona revision.
   - **Log History** — the researcher's 10 most recent biography revisions,
     with `persona_id`, `revision_number`, and `is_final` columns.

## File layout

```
peer_support/
├── app.py                      # Streamlit entry point (UI + session state)
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── secrets.toml.example    # Template to copy
│   └── secrets.toml            # Real credentials — gitignored
├── supabase/
│   └── schema.sql              # Run once in Supabase SQL editor
└── utils/
    ├── __init__.py
    ├── db.py                   # Supabase client + insert/fetch helpers
    ├── llm.py                  # Unified wrapper for OpenAI + Google GenAI
    └── questionnaire.py        # QUESTIONS list + JSON prompt builder
```

## Schema reference

| Table            | Columns                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| `biographies`    | `id (uuid pk)`, `researcher_name`, `biography_text`, `created_at`                                |
| `questionnaires` | `id (uuid pk)`, `biography_id → biographies.id`, `model_used`, `answers (jsonb)`, `created_at`   |
| `chat_logs`      | `id (uuid pk)`, `biography_id → biographies.id`, `session_id`, `role`, `content`, `model_used`, `created_at` |

See [`supabase/schema.sql`](supabase/schema.sql) for the full DDL, including
the `pgcrypto` extension used for `gen_random_uuid()` and the indexes on
`researcher_name`, `biography_id`, and `session_id`.

## Replacing the questionnaire

The placeholder items live in
[`utils/questionnaire.py`](utils/questionnaire.py) under the
`QUESTIONS: list[str]` constant. Replace them with the validated instrument
(e.g., BFI-10, PHQ-9, custom) — keep each item as a full sentence because the
exact string is used as the JSON key in the stored `questionnaires.answers`.

No other code changes are required.

## Troubleshooting

- **`PGRST205 Could not find the table 'public.biographies'`** — the schema
  wasn't run against the Supabase project whose URL/key is in
  `secrets.toml`. Re-run [`supabase/schema.sql`](supabase/schema.sql) in the
  correct project.
- **`ImportError: cannot import name 'Client' from 'supabase'`** — your
  shell is using a Streamlit installed outside the venv. Activate the venv or
  launch with `./.venv/bin/streamlit run app.py`.
- **Supabase status shows `unreachable`** — double-check `SUPABASE_URL` and
  `SUPABASE_KEY` in `.streamlit/secrets.toml`.
