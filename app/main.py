"""Atunse: feedback flywheel for N-ATLaS.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs   Demo UI: http://localhost:8000/
"""
import hashlib
import json
import os
import re
from difflib import SequenceMatcher

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db
from .lang import NAMES, detect

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("NATLAS_MODEL", "hf.co/tosinamuda/N-ATLaS-GGUF:Q4_K_M")
MOCK = os.getenv("ATUNSE_MOCK") == "1"  # only for running tests without the model

# System prompt from the official N-ATLaS model card.
DEFAULT_SYSTEM = (
    "you are a large language model trained by Awarri AI technologies. "
    "You are a friendly assistant and you are here to help."
)

SIGNAL_TYPES = {"thumbs_down", "thumbs_up", "regenerate", "rephrase", "language_drift"}
REASONS = {"wrong_language", "wrong_facts", "unnatural", "biased", "misunderstood"}
# How much each signal raises an item's place in the review queue.
WEIGHTS = {"thumbs_down": 3, "language_drift": 2, "regenerate": 1, "rephrase": 1}

app = FastAPI(title="Atunse", description="Feedback flywheel for N-ATLaS. Powered by Awarri.")
db.init()


# ---------- model call ----------

def call_natlas(messages: list[dict], temperature: float = 0.1) -> dict:
    """Send a chat to N-ATLaS through Ollama. Returns text plus token counts from the N-ATLaS tokenizer."""
    if MOCK:
        last = messages[-1]["content"]
        return {"text": f"[mock answer to: {last}]", "tokens_in": len(last.split()), "tokens_out": 5}
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            # Same decoding settings as the model card example.
            "options": {"temperature": temperature, "repeat_penalty": 1.12, "num_predict": 700},
        },
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    return {
        "text": data["message"]["content"],
        "tokens_in": data.get("prompt_eval_count", 0),
        "tokens_out": data.get("eval_count", 0),
    }


def norm(text: str) -> str:
    """Normalise a prompt for duplicate grouping: lowercase, strip punctuation and extra spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


# ---------- request bodies ----------

class ChatIn(BaseModel):
    messages: list[dict]               # [{"role": "user", "content": "..."}], system prompt optional
    session_id: str = "anonymous"
    regenerate_of: int | None = None   # id of the answer being regenerated
    consent: bool = True


class SignalIn(BaseModel):
    interaction_id: int
    type: str
    reason: str | None = None


class ReviewIn(BaseModel):
    interaction_id: int
    fixed_text: str = ""
    reviewer: str
    status: str = "fixed"              # or "rejected" if the original answer was actually fine


# ---------- endpoints ----------

@app.post("/v1/chat")
def chat(body: ChatIn):
    messages = list(body.messages)
    if not messages or messages[-1].get("role") != "user":
        raise HTTPException(400, "last message must be from the user")
    if messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})

    user_text = messages[-1]["content"]
    user_lang = detect(user_text)
    temperature = 0.7 if body.regenerate_of else 0.1  # a regeneration should differ from the first try

    out = call_natlas(messages, temperature)
    resp_lang = detect(out["text"])

    # Language lock: if the user wrote in a known language and the answer came back in another one,
    # ask once more with an explicit instruction, and remember that drift happened.
    drift = user_lang != "unknown" and resp_lang not in ("unknown", user_lang)
    if drift:
        locked = messages[:-1] + [
            {"role": "system", "content": f"Reply only in {NAMES[user_lang]}."},
            messages[-1],
        ]
        out = call_natlas(locked, temperature)
        resp_lang = detect(out["text"])

    session_hash = hashlib.sha256(body.session_id.encode()).hexdigest()[:16]  # no raw identifiers stored
    with db.connect() as conn:
        # Rephrase detection: same session, previous question very similar but not identical.
        prev = conn.execute(
            "SELECT id, messages_json FROM interactions WHERE session_hash=? ORDER BY id DESC LIMIT 1",
            (session_hash,),
        ).fetchone()
        cur = conn.execute(
            """INSERT INTO interactions (session_hash, language, messages_json, response, response_language,
               model_variant, tokens_in, tokens_out, words_out, drift_retry, regenerate_of, consent)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (session_hash, user_lang, json.dumps(messages, ensure_ascii=False), out["text"], resp_lang,
             MODEL, out["tokens_in"], out["tokens_out"], len(out["text"].split()), int(drift),
             body.regenerate_of, int(body.consent)),
        )
        new_id = cur.lastrowid
        if drift:
            conn.execute("INSERT INTO signals (interaction_id, type) VALUES (?, 'language_drift')", (new_id,))
        if body.regenerate_of:
            conn.execute("INSERT INTO signals (interaction_id, type) VALUES (?, 'regenerate')", (body.regenerate_of,))
        elif prev:
            prev_q = json.loads(prev["messages_json"])[-1]["content"]
            ratio = SequenceMatcher(None, norm(prev_q), norm(user_text)).ratio()
            if 0.6 < ratio < 1.0:
                conn.execute("INSERT INTO signals (interaction_id, type) VALUES (?, 'rephrase')", (prev["id"],))

    return {"interaction_id": new_id, "response": out["text"], "language": user_lang,
            "response_language": resp_lang, "language_lock_triggered": drift,
            "tokens_in": out["tokens_in"], "tokens_out": out["tokens_out"]}


@app.post("/v1/signal")
def signal(body: SignalIn):
    if body.type not in SIGNAL_TYPES:
        raise HTTPException(400, f"type must be one of {sorted(SIGNAL_TYPES)}")
    if body.reason and body.reason not in REASONS:
        raise HTTPException(400, f"reason must be one of {sorted(REASONS)}")
    with db.connect() as conn:
        conn.execute("INSERT INTO signals (interaction_id, type, reason) VALUES (?,?,?)",
                     (body.interaction_id, body.type, body.reason))
    return {"ok": True}


@app.get("/v1/queue")
def queue(language: str | None = Query(None), limit: int = 50):
    """Ranked review queue. Duplicate questions are grouped so a reviewer fixes one item per cluster."""
    sql = """
        SELECT i.id, i.language, i.messages_json, i.response, i.created_at,
               SUM(CASE s.type WHEN 'thumbs_down' THEN ? WHEN 'language_drift' THEN ?
                               WHEN 'regenerate' THEN ? WHEN 'rephrase' THEN ? ELSE 0 END) AS priority,
               GROUP_CONCAT(DISTINCT s.type) AS signal_types,
               GROUP_CONCAT(DISTINCT s.reason) AS reasons
        FROM interactions i
        JOIN signals s ON s.interaction_id = i.id
        LEFT JOIN reviews r ON r.interaction_id = i.id
        WHERE r.id IS NULL AND i.consent = 1 AND (? IS NULL OR i.language = ?)
        GROUP BY i.id
        HAVING priority > 0
        ORDER BY priority DESC, i.id DESC
    """
    w = WEIGHTS
    with db.connect() as conn:
        rows = conn.execute(sql, (w["thumbs_down"], w["language_drift"], w["regenerate"], w["rephrase"],
                                  language, language)).fetchall()
    clusters: dict[str, dict] = {}
    for row in rows:
        msgs = json.loads(row["messages_json"])
        key = row["language"] + "|" + norm(msgs[-1]["content"])
        if key not in clusters:  # rows are already sorted, so the first one is the most urgent
            clusters[key] = {"interaction_id": row["id"], "language": row["language"],
                             "question": msgs[-1]["content"], "response": row["response"],
                             "signals": row["signal_types"], "reasons": row["reasons"],
                             "priority": row["priority"], "cluster_size": 1}
        else:
            clusters[key]["cluster_size"] += 1
            clusters[key]["priority"] += row["priority"]
    items = sorted(clusters.values(), key=lambda c: c["priority"], reverse=True)
    return {"count": len(items), "items": items[:limit]}


@app.post("/v1/review")
def review(body: ReviewIn):
    if body.status not in ("fixed", "rejected"):
        raise HTTPException(400, "status must be fixed or rejected")
    if body.status == "fixed" and not body.fixed_text.strip():
        raise HTTPException(400, "fixed_text is required when status is fixed")
    with db.connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO reviews (interaction_id, fixed_text, reviewer, status) VALUES (?,?,?,?)",
            (body.interaction_id, body.fixed_text, body.reviewer, body.status),
        )
    return {"ok": True}


# ---------- exports ----------

def render_llama3(messages: list[dict]) -> str:
    """Render a chat in the Llama-3 template N-ATLaS was trained on (see the model card's sample output)."""
    out = "<|begin_of_text|>"
    for m in messages:
        out += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
    return out


def _fixed_rows(conn):
    return conn.execute(
        """SELECT i.id, i.language, i.messages_json, i.response, r.fixed_text, r.reviewer
           FROM reviews r JOIN interactions i ON i.id = r.interaction_id
           WHERE r.status = 'fixed' AND i.consent = 1"""
    ).fetchall()


@app.get("/v1/export/sft", response_class=PlainTextResponse)
def export_sft(fmt: str = Query("messages", pattern="^(messages|natlas_text)$")):
    """Reviewer-fixed examples. fmt=messages for TRL SFTTrainer, fmt=natlas_text for raw Llama-3 template text."""
    lines = []
    with db.connect() as conn:
        for row in _fixed_rows(conn):
            msgs = json.loads(row["messages_json"]) + [{"role": "assistant", "content": row["fixed_text"]}]
            rec = {"messages": msgs} if fmt == "messages" else {"text": render_llama3(msgs)}
            rec["language"] = row["language"]
            lines.append(json.dumps(rec, ensure_ascii=False))
    return "\n".join(lines)


@app.get("/v1/export/dpo", response_class=PlainTextResponse)
def export_dpo():
    """Preference pairs for DPO / RLHF, in TRL's conversational format.

    Two sources: reviewer fixes (fix = chosen, original = rejected), and regenerations
    the user liked (new answer = chosen, flagged original = rejected).
    """
    lines = []
    with db.connect() as conn:
        for row in _fixed_rows(conn):
            msgs = json.loads(row["messages_json"])
            lines.append(json.dumps({
                "prompt": msgs,
                "chosen": [{"role": "assistant", "content": row["fixed_text"]}],
                "rejected": [{"role": "assistant", "content": row["response"]}],
                "language": row["language"], "source": "reviewer_fix",
            }, ensure_ascii=False))
        pairs = conn.execute(
            """SELECT o.messages_json, o.response AS rejected, n.response AS chosen, o.language
               FROM interactions n JOIN interactions o ON o.id = n.regenerate_of
               WHERE n.consent = 1
                 AND EXISTS (SELECT 1 FROM signals s WHERE s.interaction_id = n.id AND s.type = 'thumbs_up')"""
        ).fetchall()
        for p in pairs:
            lines.append(json.dumps({
                "prompt": json.loads(p["messages_json"]),
                "chosen": [{"role": "assistant", "content": p["chosen"]}],
                "rejected": [{"role": "assistant", "content": p["rejected"]}],
                "language": p["language"], "source": "regenerate_preferred",
            }, ensure_ascii=False))
    return "\n".join(lines)


# ---------- stats + mini eval ----------

@app.get("/v1/stats")
def stats():
    """Per-language health report: usage, flag rate, language drift, token cost, and the correction eval."""
    report = {}
    with db.connect() as conn:
        for lang in ["en", "ha", "ig", "yo", "unknown"]:
            row = conn.execute(
                """SELECT COUNT(*) n, COALESCE(SUM(drift_retry),0) drift,
                          COALESCE(SUM(tokens_out),0) t, COALESCE(SUM(words_out),0) w
                   FROM interactions WHERE language = ?""", (lang,)).fetchone()
            if row["n"] == 0:
                continue
            flagged = conn.execute(
                """SELECT COUNT(DISTINCT i.id) FROM interactions i JOIN signals s ON s.interaction_id = i.id
                   WHERE i.language = ? AND s.type IN ('thumbs_down','regenerate','rephrase','language_drift')""",
                (lang,)).fetchone()[0]
            fixes = conn.execute(
                """SELECT i.response, r.fixed_text FROM reviews r JOIN interactions i ON i.id = r.interaction_id
                   WHERE i.language = ? AND r.status = 'fixed'""", (lang,)).fetchall()
            # Mini eval: how much did reviewers have to change? 0 = untouched, 1 = fully rewritten.
            dists = [1 - SequenceMatcher(None, f["response"], f["fixed_text"]).ratio() for f in fixes]
            report[NAMES.get(lang, lang)] = {
                "interactions": row["n"],
                "flag_rate": round(flagged / row["n"], 3),
                "language_drift_rate": round(row["drift"] / row["n"], 3),
                "tokens_per_word": round(row["t"] / row["w"], 2) if row["w"] else None,
                "reviewer_fixes": len(fixes),
                "mean_correction_distance": round(sum(dists) / len(dists), 3) if dists else None,
            }
    return report


# Demo UI and widget, served at /
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "static"), html=True), name="static")
