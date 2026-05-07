"""Internationalization (i18n) helpers.

Every user-facing string in `app.py` goes through `t(key)`, which looks up
the selected language from `st.session_state["language"]` and falls back to
English if a key is missing from the Hebrew table.

Hebrew selection also flips the page into right-to-left via `inject_rtl_css()`.
"""

from __future__ import annotations

import streamlit as st


LANG_EN = "en"
LANG_HE = "he"

LANGUAGE_LABELS: dict[str, str] = {
    LANG_EN: "English",
    LANG_HE: "עברית",
}


TRANSLATIONS: dict[str, dict[str, str]] = {
    LANG_EN: {
        # Page / title
        "page_title": "Persona Chatbot",
        "app_title": "Persona Chatbot & Evaluation Dashboard",

        # Sidebar
        "sb_language_header": "Language",
        "sb_language_locked": "Language: {lang}",
        "sb_researcher_header": "Researcher",
        "sb_researcher_label": "Researcher Name",
        "sb_researcher_placeholder": "e.g., Dr. Cohen",
        "sb_llm_header": "LLM",
        "sb_llm_label": "Select LLM",
        "sb_status_header": "System status",
        "sb_supabase_connected": "Supabase: :green[connected]",
        "sb_supabase_unreachable": "Supabase: :red[unreachable]",
        "sb_secrets_hint": "Check SUPABASE_URL / SUPABASE_KEY in `.streamlit/secrets.toml`.",
        "login_username_label": "Username",
        "login_username_placeholder": "Researcher username",
        "login_password_label": "Password",
        "login_button": "Log in",
        "logout_button": "Log out",
        "login_success": "Logged in as **{name}**",
        "login_failed": "Invalid username or password.",
        "login_no_researchers_configured": (
            "No researchers are configured in `.streamlit/secrets.toml`."
        ),

        # Entry gate
        "gate_info": "Enter your name in the sidebar to begin.",
        "gate_login_info": "Log in from the sidebar to begin.",

        # Left panel — Persona setup
        "persona_setup": "Persona Setup",
        "persona_caption": (
            "Paste or edit the biography, pick the model in the sidebar, "
            "then click the button below."
        ),
        "bio_label": "Biography / System Prompt",
        "bio_placeholder": (
            "Describe the persona in first or third person: background, "
            "personality, relevant life events, speech style, etc."
        ),
        "save_bio_button": "Save biography",
        "save_bio_spinner": "Saving biography...",
        "save_bio_failed": "Saving biography failed: {error}",
        "save_bio_success_toast": "Biography saved. Chat reset.",
        "generate_q_button": "Generate questionnaire responses",
        "generate_q_spinner": "Running questionnaire via {model}...",
        "generate_q_failed": "Questionnaire generation failed: {error}",
        "generate_q_success_toast": "Questionnaire answers generated.",
        "active_persona": "Active persona id: `{id}`",
        "active_persona_revision": (
            "Active persona: `{id}` (revision {n}{final_suffix})"
        ),
        "active_persona_final_suffix": " — finalized",
        "edit_bio_button": "Edit biography",
        "cancel_edit_button": "Cancel edit",
        "save_changes_button": "Save changes",
        "save_changes_spinner": "Saving changes...",
        "save_changes_failed": "Saving changes failed: {error}",
        "save_changes_success_toast": "Revision {n} saved.",
        "save_changes_noop_toast": "No biography changes detected.",
        "finish_persona_button": "Finish persona",
        "finish_persona_spinner": "Finalizing persona...",
        "finish_persona_failed": "Finalizing persona failed: {error}",
        "finish_persona_success_toast": (
            "Persona finalized. You can start a new one."
        ),
        "bio_readonly_hint": (
            "Biography saved (revision {n}). Click 'Edit biography' to "
            "make changes — each save creates a new revision under the "
            "same persona."
        ),
        "bio_edit_hint": (
            "Editing revision {n}. Clicking 'Save changes' records a new "
            "revision under the same persona id."
        ),
        "active_persona_continue_info": (
            "Welcome back, {name}. Please continue editing the biography, "
            "save your changes when needed, or press Finish persona when "
            "you are done."
        ),
        "intake_open_ended_missing": (
            "Please answer every open-ended question before drafting — "
            "type an answer or click Randomize to have the LLM fill them. "
            "Missing: {ids}."
        ),
        "intake_locked_caption": (
            "These intake answers are locked from the first saved version of "
            "this persona. Later biography revisions keep the same intake "
            "snapshot."
        ),
        "intake_randomize_llm_spinner": (
            "Generating open-ended answers via {model}..."
        ),
        "chat_edit_mode_blocked": (
            "Save or cancel your edits before chatting — the chat runs "
            "against the most recent saved revision."
        ),
        "questionnaire_edit_mode_blocked": (
            "Save or cancel your edits before generating questionnaire "
            "answers — they run against the most recent saved revision."
        ),

        # Intake form (Step 13)
        "intake_form_header": "Intake form",
        "intake_form_caption": (
            "Answer any subset, or use Randomize to fill the remaining "
            "fields. Click Draft biography to turn the answers into prose "
            "that populates the biography field below — you can still edit "
            "it before committing."
        ),
        "intake_load_failed": "Could not load intake form: {error}",
        "intake_randomize_button": "Randomize",
        "intake_randomize_done": "Randomized.",
        "intake_draft_button": "Draft biography",
        "intake_drafting_spinner": "Drafting biography via {model}...",
        "intake_draft_failed": "Biography draft failed: {error}",
        "intake_draft_success": "Biography drafted — review and edit as needed.",
        "intake_details_label": "Please elaborate",
        "intake_elaborate_label": "Elaborate (optional)",

        # Right tabs
        "tab_chat": "💬 Chat",
        "tab_questionnaire": "📋 Questionnaire Results",
        "tab_history": "📜 Log History",

        # Chat tab
        "chat_no_persona": (
            "Set a persona in the control panel first — then you can chat "
            "with it here."
        ),
        "chat_input_placeholder": "Talk to the persona",
        "chat_thinking": "Thinking...",
        "chat_error": "Model call failed: {error}",

        # Questionnaire tab
        "q_no_answers": (
            "Run the persona update in the control panel to generate "
            "questionnaire answers."
        ),
        "q_model_caption": "Model: **{model}** — persona `{id}`",
        "simulation_results_header": "Simulation results",
        "simulation_bio_changes_header": (
            "Biography changes since previous simulated revision {n}"
        ),
        "simulation_previous_bio_header": "Previous biography (revision {n})",
        "simulation_current_bio_header": "Current biography (revision {n})",
        "simulation_previous_bio_label": "Previous biography",
        "simulation_current_bio_label": "Current biography",
        "simulation_bio_diff_header": "Highlighted biography difference",
        "simulation_comparison_info": (
            "Comparing with the previous simulated biography revision {n}. "
            "Changed answers are highlighted and include the previous answer."
        ),
        "simulation_answer_changed": "Changed from previous simulation",
        "simulation_previous_answer": "Previous answer: {answer}",
        "simulation_change_explanation_prefix": "Why this changed:",
        "answer_comment_label": "Researcher comment / feedback (optional)",
        "answer_explanation_prefix": "Explanation:",
        "answer_comment_placeholder": (
            "Add an optional note about this answer..."
        ),
        "save_answer_comments_button": "Save answer comments",
        "answer_comments_saved": "Saved {count} answer comment(s).",
        "answer_comments_none_to_save": "No non-empty comments to save.",
        "answer_comments_save_failed": "Saving comments failed: {error}",
        "answer_comments_unavailable": (
            "Comments can be saved after a new simulation is generated."
        ),
        "go_saved_personas_button": "Go to saved personas",
        "view_current_results_button": "View current simulation results",
        "back_to_intake_button": "Back to intake",
        "back_to_edit_persona_button": "Back to edit current persona",
        "current_persona_results_title": "Current persona simulation results",
        "current_persona_results_caption": (
            "Results for the current persona. Go back to edit the biography, "
            "save changes, and run another simulation."
        ),
        "saved_personas_title": "Saved personas",
        "saved_personas_caption": (
            "Saved biographies and simulation answers for **{name}**."
        ),
        "saved_personas_empty": "No saved personas yet.",
        "saved_personas_load_failed": "Could not load saved personas: {error}",
        "saved_persona_heading": (
            "Persona {persona} - version {version} - {status}"
        ),
        "saved_persona_revision": "Revision {n}",
        "saved_persona_latest": "Latest biography",
        "saved_persona_final": "completed",
        "saved_persona_active": "active",
        "saved_persona_no_simulations": (
            "No simulation answers saved for this persona yet."
        ),
        "saved_persona_tab_overview": "Overview",
        "saved_persona_tab_versions": "Versions",
        "saved_persona_tab_simulations": "Simulations",
        "saved_persona_load_version_button": "Load this version",
        "saved_persona_version_loaded": "Loaded revision {n}.",
        "saved_persona_simulation": "Simulation from revision {n}",
        "saved_persona_simulation_tab": "Results for revision {n}",
        "saved_persona_created": "Created: {date}",
        "saved_persona_previous_versions": "Previous versions",
        "saved_persona_previous_tab": "Revision {n}",

        # Log tab
        "log_caption": "Most recent biographies created by **{name}** (up to 10).",
        "log_error": "Could not load history: {error}",
        "log_empty": "No biographies yet. Create one in the control panel.",
        "log_col_id": "ID",
        "log_col_persona": "Persona",
        "log_col_revision": "Revision",
        "log_col_final": "Final",
        "log_col_researcher": "Researcher",
        "log_col_biography": "Biography",
        "log_col_created": "Created",
    },
    LANG_HE: {
        # Page / title
        "page_title": "מחקר פרסונות סינתטיות - עמיתים נותני שירות",
        "app_title": "יצירת פרסונה סינתטית וסימולציית תשובות לשאלון",

        # Sidebar
        "sb_language_header": "שפה",
        "sb_language_locked": "שפה: {lang}",
        "sb_researcher_header": "חוקר/ת",
        "sb_researcher_label": "שם החוקר/ת",
        "sb_researcher_placeholder": "למשל, ד\"ר כהן",
        "sb_llm_header": "מודל שפה",
        "sb_llm_label": "בחר/י מודל",
        "sb_status_header": "סטטוס מערכת",
        "sb_supabase_connected": "Supabase: :green[מחובר]",
        "sb_supabase_unreachable": "Supabase: :red[לא זמין]",
        "sb_secrets_hint": "יש לבדוק את SUPABASE_URL / SUPABASE_KEY בקובץ `.streamlit/secrets.toml`.",
        "login_username_label": "שם משתמש",
        "login_username_placeholder": "שם משתמש של החוקר/ת",
        "login_password_label": "סיסמה",
        "login_button": "כניסה",
        "logout_button": "יציאה",
        "login_success": "מחובר/ת כ־**{name}**",
        "login_failed": "שם המשתמש או הסיסמה אינם נכונים.",
        "login_no_researchers_configured": (
            "לא הוגדרו חוקרים/ות בקובץ `.streamlit/secrets.toml`."
        ),

        # Entry gate
        "gate_info": "הזן/י את שמך בסרגל הצד כדי להתחיל.",
        "gate_login_info": "יש להתחבר דרך סרגל הצד כדי להתחיל.",

        # Left panel — Persona setup
        "persona_setup": "הגדרת פרסונה",
        "persona_caption": (
            "הדביק/י או ערוך/י את הביוגרפיה, בחר/י מודל בסרגל הצד, ולחץ/י "
            "על הכפתור שמופיע מטה."
          
        ),
        "bio_label": "ביוגרפיה / פרומפט מערכת",
        "bio_placeholder": (
            "תאר/י את הפרסונה בגוף שלישי: רקע, אישיות, אירועי "
            "חיים רלוונטיים, סגנון דיבור וכדומה."
        ),
        "save_bio_button": "שמירת ביוגרפיה",
        "save_bio_spinner": "שומר ביוגרפיה...",
        "save_bio_failed": "שמירת הביוגרפיה נכשלה: {error}",
        "save_bio_success_toast": "הביוגרפיה נשמרה. השיחה אופסה.",
        "generate_q_button": "יצירת תשובות לשאלון",
        "generate_q_spinner": "מריץ שאלון באמצעות {model}...",
        "generate_q_failed": "יצירת תשובות לשאלון נכשלה: {error}",
        "generate_q_success_toast": "תשובות השאלון נוצרו.",
        "active_persona": "מזהה פרסונה פעילה: `{id}`",
        "active_persona_revision": (
            "פרסונה פעילה: `{id}` (גרסה {n}{final_suffix})"
        ),
        "active_persona_final_suffix": " — הושלמה",
        "edit_bio_button": "עריכת ביוגרפיה",
        "cancel_edit_button": "ביטול עריכה",
        "save_changes_button": "שמירת שינויים",
        "save_changes_spinner": "שומר שינויים...",
        "save_changes_failed": "שמירת השינויים נכשלה: {error}",
        "save_changes_success_toast": "גרסה {n} נשמרה.",
        "save_changes_noop_toast": "לא זוהו שינויים בביוגרפיה.",
        "finish_persona_button": "סיום פרסונה",
        "finish_persona_spinner": "מסיים פרסונה...",
        "finish_persona_failed": "סיום הפרסונה נכשל: {error}",
        "finish_persona_success_toast": (
            "הפרסונה הושלמה. ניתן להתחיל פרסונה חדשה."
        ),
        "bio_readonly_hint": (
            "הביוגרפיה נשמרה (גרסה {n}). ללחוץ על 'עריכת ביוגרפיה' כדי "
            "לבצע שינויים — כל שמירה יוצרת גרסה חדשה תחת אותה פרסונה."
        ),
        "bio_edit_hint": (
            "עורך/ת את גרסה {n}. לחיצה על 'שמירת שינויים' תשמור גרסה "
            "חדשה תחת אותו מזהה פרסונה."
        ),
        "active_persona_continue_info": (
            "ברוך/ה השב/ה, {name}. ניתן להמשיך לערוך את הביוגרפיה, "
            "לשמור שינויים לפי הצורך, או ללחוץ על 'סיום פרסונה' כאשר "
            "העבודה הסתיימה."
        ),
        "intake_open_ended_missing": (
            "יש למלא את כל השאלות הפתוחות לפני יצירת הביוגרפיה — ניתן "
            "לכתוב תשובה ידנית או ללחוץ על 'מילוי אקראי' כדי שמודל השפה "
            "ימלא אותן. חסרות: {ids}."
        ),
        "intake_locked_caption": (
            "תשובות שאלון הקליטה נעולות לפי הגרסה הראשונה שנשמרה עבור "
            "הפרסונה הזו. גרסאות ביוגרפיה מאוחרות יותר שומרות על אותו "
            "צילום מצב של התשובות."
        ),
        "intake_randomize_llm_spinner": (
            "מייצר תשובות לשאלות הפתוחות באמצעות {model}..."
        ),
        "chat_edit_mode_blocked": (
            "יש לשמור או לבטל את העריכה לפני השיחה — השיחה פועלת מול "
            "הגרסה האחרונה שנשמרה."
        ),
        "questionnaire_edit_mode_blocked": (
            "יש לשמור או לבטל את העריכה לפני יצירת תשובות לשאלון — הן "
            "פועלות מול הגרסה האחרונה שנשמרה."
        ),

        # Intake form (Step 13)
        "intake_form_header": "שאלון קליטה",
        "intake_form_caption": (
            "ניתן לענות על תת-קבוצה של שאלות, או ללחוץ על 'מילוי אקראי' "
            "כדי להשלים את השאר. לחיצה על 'יצירת ביוגרפיה' תהפוך את התשובות "
            "לפסקה שתמלא את שדה הביוגרפיה למטה — ניתן עדיין לערוך אותה לפני "
            "השמירה."
        ),
        "intake_load_failed": "טעינת שאלון הקליטה נכשלה: {error}",
        "intake_randomize_button": "מילוי אקראי",
        "intake_randomize_done": "מולא אקראית.",
        "intake_draft_button": "יצירת ביוגרפיה",
        "intake_drafting_spinner": "מנסח/ת ביוגרפיה באמצעות {model}...",
        "intake_draft_failed": "יצירת הביוגרפיה נכשלה: {error}",
        "intake_draft_success": "הטיוטה מוכנה — ניתן לערוך לפי הצורך.",
        "intake_details_label": "פרט/י",
        "intake_elaborate_label": "פירוט (רשות)",

        # Right tabs
        "tab_chat": "💬 שיחה",
        "tab_questionnaire": "📋 תוצאות שאלון",
        "tab_history": "📜 היסטוריית רישום",

        # Chat tab
        "chat_no_persona": (
            "יש להגדיר פרסונה בפאנל הבקרה תחילה — לאחר מכן ניתן לשוחח איתה "
            "כאן."
        ),
        "chat_input_placeholder": "דבר/י עם הפרסונה",
        "chat_thinking": "חושב...",
        "chat_error": "קריאה למודל נכשלה: {error}",

        # Questionnaire tab
        "q_no_answers": (
            "הרץ/י עדכון פרסונה בפאנל הבקרה כדי לקבל תשובות לשאלון."
        ),
        "q_model_caption": "מודל: **{model}** — פרסונה `{id}`",
        "simulation_results_header": "תוצאות סימולציה",
        "simulation_bio_changes_header": (
            "שינויים בביוגרפיה מאז הגרסה המדומה הקודמת {n}"
        ),
        "simulation_previous_bio_header": "ביוגרפיה קודמת (גרסה {n})",
        "simulation_current_bio_header": "ביוגרפיה נוכחית (גרסה {n})",
        "simulation_previous_bio_label": "ביוגרפיה קודמת",
        "simulation_current_bio_label": "ביוגרפיה נוכחית",
        "simulation_bio_diff_header": "הבדלים מודגשים בביוגרפיה",
        "simulation_comparison_info": (
            "השוואה לגרסת הביוגרפיה המדומה הקודמת {n}. תשובות שהשתנו "
            "מודגשות ומציגות גם את התשובה הקודמת."
        ),
        "simulation_answer_changed": "השתנה מהסימולציה הקודמת",
        "simulation_previous_answer": "תשובה קודמת: {answer}",
        "simulation_change_explanation_prefix": "למה זה השתנה:",
        "answer_comment_label": "הערה / משוב של החוקר/ת (רשות)",
        "answer_explanation_prefix": "הסבר:",
        "answer_comment_placeholder": "אפשר להוסיף הערה על התשובה הזו...",
        "save_answer_comments_button": "שמירת הערות לתשובות",
        "answer_comments_saved": "נשמרו {count} הערות לתשובות.",
        "answer_comments_none_to_save": "אין הערות שאינן ריקות לשמירה.",
        "answer_comments_save_failed": "שמירת ההערות נכשלה: {error}",
        "answer_comments_unavailable": (
            "ניתן לשמור הערות לאחר יצירת סימולציה חדשה."
        ),
        "go_saved_personas_button": "מעבר לפרסונות שמורות",
        "view_current_results_button": "צפייה בתוצאות הסימולציה הנוכחית",
        "back_to_intake_button": "חזרה לטופס",
        "back_to_edit_persona_button": "חזרה לעריכת הפרסונה הנוכחית",
        "current_persona_results_title": "תוצאות סימולציה לפרסונה הנוכחית",
        "current_persona_results_caption": (
            "תוצאות עבור הפרסונה הנוכחית. ניתן לחזור לעריכת הביוגרפיה, "
            "לשמור שינויים ולהריץ סימולציה נוספת."
        ),
        "saved_personas_title": "פרסונות שמורות",
        "saved_personas_caption": (
            "ביוגרפיות ותשובות סימולציה שנשמרו עבור **{name}**."
        ),
        "saved_personas_empty": "עדיין אין פרסונות שמורות.",
        "saved_personas_load_failed": "טעינת הפרסונות השמורות נכשלה: {error}",
        "saved_persona_heading": (
            "פרסונה {persona} - גרסה {version} - {status}"
        ),
        "saved_persona_revision": "גרסה {n}",
        "saved_persona_latest": "ביוגרפיה אחרונה",
        "saved_persona_final": "הושלמה",
        "saved_persona_active": "פעילה",
        "saved_persona_no_simulations": (
            "עדיין אין תשובות סימולציה שמורות עבור הפרסונה הזו."
        ),
        "saved_persona_tab_overview": "סקירה",
        "saved_persona_tab_versions": "גרסאות",
        "saved_persona_tab_simulations": "סימולציות",
        "saved_persona_load_version_button": "טעינת גרסה זו",
        "saved_persona_version_loaded": "גרסה {n} נטענה.",
        "saved_persona_simulation": "סימולציה מגרסה {n}",
        "saved_persona_simulation_tab": "תוצאות עבור גרסה {n}",
        "saved_persona_created": "נוצר: {date}",
        "saved_persona_previous_versions": "גרסאות קודמות",
        "saved_persona_previous_tab": "גרסה {n}",

        # Log tab
        "log_caption": "הביוגרפיות האחרונות שיצר/ה **{name}** (עד 10).",
        "log_error": "טעינת ההיסטוריה נכשלה: {error}",
        "log_empty": "עדיין אין ביוגרפיות. צור/י אחת בפאנל הבקרה.",
        "log_col_id": "מזהה",
        "log_col_persona": "פרסונה",
        "log_col_revision": "גרסה",
        "log_col_final": "הושלמה",
        "log_col_researcher": "חוקר/ת",
        "log_col_biography": "ביוגרפיה",
        "log_col_created": "נוצר",
    },
}


def current_language() -> str:
    """Return the active language code, defaulting to English."""
    return st.session_state.get("language", LANG_EN)


def is_rtl() -> bool:
    """True iff the active language renders right-to-left."""
    return current_language() == LANG_HE


def t(key: str, **kwargs: object) -> str:
    """Look up a translation, falling back to English then to the key itself.

    Any `**kwargs` are formatted into the string via `str.format`.
    """
    lang = current_language()
    table = TRANSLATIONS.get(lang, {})
    template = table.get(key) or TRANSLATIONS[LANG_EN].get(key, key)
    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError):
            return template
    return template


_RTL_CSS = """
<style>
    /* Flip the whole app to right-to-left for Hebrew. */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    [data-testid="stMain"],
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        direction: rtl !important;
        text-align: right !important;
    }

    /* Most user-visible Streamlit text: markdown, headings, labels, captions,
       alerts, tabs, expanders and button text. Streamlit uses several nested
       generated containers, so we target stable data-testid/BaseWeb selectors
       plus the common element tags. */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stCaptionContainer"],
    [data-testid="stAlert"],
    [data-testid="stAlert"] *,
    [data-testid="stExpander"],
    [data-testid="stExpander"] *,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stRadio"],
    [data-testid="stRadio"] *,
    [data-testid="stSelectSlider"],
    [data-testid="stSelectSlider"] *,
    [data-baseweb="tab"],
    [data-baseweb="tab"] *,
    [data-baseweb="button"],
    [data-baseweb="button"] *,
    h1, h2, h3, h4, h5, h6,
    p, li, label, span {
        direction: rtl !important;
        text-align: right !important;
    }

    /* Streamlit wraps inputs in BaseWeb components that apply their own
       direction: ltr with high specificity. Override them with !important
       on every likely selector so free-text fields mirror the rest of the
       Hebrew UI. `unicode-bidi: plaintext` lets a single field contain
       mixed-direction text without auto-detecting the wrong direction from
       an English placeholder. */
    .stTextInput input,
    .stTextInput [data-baseweb="input"] input,
    .stTextArea textarea,
    .stTextArea [data-baseweb="textarea"] textarea,
    .stChatInput textarea,
    .stChatInput [data-baseweb="textarea"] textarea,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: plaintext;
    }

    /* Multi-line free-text boxes should behave like real paragraph boxes:
       wrap long text, hide horizontal overflow, and scroll vertically when
       the answer is longer than the visible area. */
    .stTextArea textarea,
    .stTextArea [data-baseweb="textarea"] textarea,
    .stChatInput textarea,
    .stChatInput [data-baseweb="textarea"] textarea,
    [data-baseweb="textarea"] textarea {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        min-height: 180px !important;
        white-space: pre-wrap !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        line-height: 1.5 !important;
        resize: vertical !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder,
    .stChatInput textarea::placeholder,
    [data-baseweb="input"] input::placeholder,
    [data-baseweb="textarea"] textarea::placeholder {
        direction: rtl !important;
        text-align: right !important;
    }

    /* Tables/dataframes often render in virtualized grids; align all headers
       and cells to the right in Hebrew mode. */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] *,
    [role="grid"],
    [role="grid"] *,
    [role="columnheader"],
    [role="gridcell"],
    table,
    thead,
    tbody,
    tr,
    th,
    td {
        direction: rtl !important;
        text-align: right !important;
    }

    /* Keep inline icons / status glyphs visually stable while their text stays
       RTL. */
    svg {
        direction: ltr;
    }

    /* Keep code / JSON blocks readable (left-to-right). */
    code,
    pre,
    .stCode,
    .stCode *,
    [data-testid="stCodeBlock"],
    [data-testid="stCodeBlock"] * {
        direction: ltr !important;
        text-align: left !important;
    }
</style>
"""


def inject_rtl_css() -> None:
    """Apply RTL styling when the active language is Hebrew."""
    if is_rtl():
        st.markdown(_RTL_CSS, unsafe_allow_html=True)
