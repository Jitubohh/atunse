# Atunse: feedback flywheel for N-ATLaS

*Àtúnṣe* is Yoruba for "correction". Atunse is developer infrastructure that turns everyday use of N-ATLaS into training data that improves N-ATLaS.

Any app built on N-ATLaS can add Atunse. Users give one tap signals (thumbs down, try again). Atunse also catches failures on its own: answers in the wrong language, and users rephrasing because they weren't understood. Flags are ranked and grouped so a small team of bilingual reviewers only fixes what matters. The fixes export as ready to train files in N-ATLaS's own chat format, for supervised fine tuning and for preference training (DPO / RLHF).

The N-ATLaS model card lists RLHF, more datasets and bias mitigation as future work, and reports Yoruba as its weakest language (2.69/5). Atunse is the pipeline that collects that data from real usage, split by its four languages.

## How it works

```
 App user ──► Atunse SDK / widget ──► N-ATLaS (via Ollama, GGUF Q4_K_M)
                │
                ├─ language lock: answer in wrong language? re-ask once, log drift
                ├─ signals: thumbs down + reason, thumbs up, try again, rephrase (auto)
                ▼
          SQLite store ──► ranked + deduplicated review queue ──► bilingual reviewers fix answers
                │
                ├─► /v1/export/sft   chat messages or raw Llama-3 template text
                ├─► /v1/export/dpo   chosen / rejected preference pairs
                └─► /v1/stats        per language flag rate, drift rate, tokens per word, correction eval
```

## N-ATLaS integration

| What | How |
|---|---|
| Model | `tosinamuda/N-ATLaS-GGUF` Q4_K_M, a GGUF conversion of `NCAIR1/N-ATLaS`, served locally by Ollama |
| System prompt and decoding | Same as the official model card: Awarri system prompt, temperature 0.1, repetition penalty 1.12 |
| Token counts | `prompt_eval_count` / `eval_count` from Ollama, counted by the N-ATLaS tokenizer itself |
| Languages | Everything is split by the four N-ATLaS languages: English, Hausa, Igbo, Yoruba |
| Export format | `fmt=natlas_text` renders the Llama-3 chat template N-ATLaS was trained on (`<|start_header_id|>` / `<|eot_id|>`); `fmt=messages` and the DPO file use TRL's conversational format |

## Features

- **Chat endpoint with language lock.** Detects the user's language (en / ha / ig / yo). If N-ATLaS answers in another language, Atunse re-asks once with an explicit instruction and records a `language_drift` signal.
- **Low effort signals.** Thumbs up / down with an optional reason (wrong language, wrong facts, unnatural, biased, misunderstood). "Try again" records a `regenerate` signal on the first answer. A similar follow up question in the same session records a `rephrase` signal automatically.
- **Review queue.** Priority score per answer (thumbs down 3, drift 2, regenerate 1, rephrase 1). Identical questions are grouped into one item with a cluster size, so reviewers fix one example per cluster. Reviewers edit the answer in place or mark the original as fine.
- **Exports.** SFT from reviewer fixes. DPO pairs from reviewer fixes (fix chosen, original rejected) and from regenerations the user liked (new answer chosen, flagged original rejected).
- **Mini eval.** `/v1/stats` reports per language: interactions, flag rate, language drift rate, tokens per word, number of fixes, and mean correction distance (0 means reviewers changed nothing, 1 means a full rewrite).
- **Python SDK.** `sdk/atunse_client.py`.
- **Privacy.** No names, emails or phone numbers stored. Session IDs are hashed. Consent notice in the widget; non consented items are excluded from the queue and exports.

## Setup (Windows, macOS, Linux)

Requirements: Python 3.10+, [Ollama](https://ollama.com), about 5 GB free disk, a GPU with 6 GB+ VRAM recommended (CPU works, slower).

```powershell
ollama pull hf.co/tosinamuda/N-ATLaS-GGUF:Q4_K_M
git clone <this repo> ; cd atunse
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000 for the demo (Chat, Review queue, Insights & export) and http://localhost:8000/docs for the interactive API docs.

Config via environment variables (see `.env.example`): `OLLAMA_URL`, `NATLAS_MODEL`, `ATUNSE_DB`.

On some RTX 20 series cards Ollama's CUDA backend crashes on long outputs. Setting the environment variable `OLLAMA_LLM_LIBRARY=vulkan` and restarting Ollama fixes it.

## Trying Atunse without installing the model

A developer can test the SDK against a running Atunse instance without downloading N-ATLaS. The host exposes their server with a free tunnel:

```powershell
cloudflared tunnel --url http://localhost:8000
```

The tester only needs `requests` and `sdk/atunse_client.py`, then points `Atunse("https://<tunnel-url>")` at it.

## SDK usage

```python
from atunse_client import Atunse

a = Atunse("http://localhost:8000")
r = a.chat("menene ake nufi da gwagwarmaya")
print(r["response"])
a.thumbs_down(r["interaction_id"], reason="unnatural")
r2 = a.regenerate(r["interaction_id"])
a.thumbs_up(r2["interaction_id"])
print(a.stats())
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat` | `{messages, session_id, regenerate_of?}` → answer, interaction_id, languages, token counts |
| POST | `/v1/signal` | `{interaction_id, type, reason?}` |
| GET | `/v1/queue?language=yo` | ranked, grouped review items |
| POST | `/v1/review` | `{interaction_id, fixed_text, reviewer, status}` |
| GET | `/v1/export/sft?fmt=messages\|natlas_text` | JSONL |
| GET | `/v1/export/dpo` | JSONL |
| GET | `/v1/stats` | per language report |

## Status and roadmap

Working: chat with language lock, signals, rephrase detection, review queue with grouping, both exports, stats, Python SDK, demo UI.

Next: embedding based clustering for near duplicate questions, two reviewer agreement for bias and safety items, a drop-in JS widget package, hosted deployment, and a reference fine tuning notebook that trains on the exports and reports the change against base N-ATLaS.

Known limits: language detection is a character and stopword heuristic and can be wrong on very short or code switched text; Pidgin is not a separate tag.

## License and attribution

N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.

Atunse source code is released under the MIT License (see `LICENSE`). The N-ATLaS model is not bundled and is used under the N-ATLaS Terms of Use. Contributions welcome, see `CONTRIBUTING.md`. Built by G2 Tech Solutions.
