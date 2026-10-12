"""SQLite storage. One file, no server, works on Windows out of the box."""
import os
import sqlite3

DB_PATH = os.getenv("ATUNSE_DB", "atunse.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_hash TEXT,
    language TEXT,                -- detected language of the user's last message
    messages_json TEXT,           -- full chat (system + turns) sent to N-ATLaS, OpenAI-style role/content list
    response TEXT,                -- what N-ATLaS answered
    response_language TEXT,       -- detected language of the answer
    model_variant TEXT,
    tokens_in INTEGER,            -- prompt tokens, counted by the N-ATLaS tokenizer (via Ollama)
    tokens_out INTEGER,           -- answer tokens, counted by the N-ATLaS tokenizer
    words_out INTEGER,            -- answer word count, so we can compute tokens per word
    drift_retry INTEGER DEFAULT 0, -- 1 if the language lock had to re-ask the model
    regenerate_of INTEGER,        -- set when this answer is a regeneration of an earlier one
    consent INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER REFERENCES interactions(id),
    type TEXT,                    -- thumbs_down | thumbs_up | regenerate | rephrase | language_drift
    reason TEXT,                  -- wrong_language | wrong_facts | unnatural | biased | misunderstood
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER UNIQUE REFERENCES interactions(id),
    fixed_text TEXT,
    reviewer TEXT,
    status TEXT DEFAULT 'fixed',  -- fixed | rejected (reviewer says the original was fine)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts: row["language"]
    return conn


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
