"""Atunse Python SDK: add the N-ATLaS feedback flywheel to any app in a few lines.

    from atunse_client import Atunse
    a = Atunse("http://localhost:8000")
    r = a.chat("Bawo ni?")
    a.thumbs_down(r["interaction_id"], reason="unnatural")
"""
import uuid

import requests


class Atunse:
    def __init__(self, base_url: str = "http://localhost:8000", session_id: str | None = None):
        self.base = base_url.rstrip("/")
        self.session_id = session_id or str(uuid.uuid4())
        self.history: list[dict] = []

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(self.base + path, json=body, timeout=300)
        r.raise_for_status()
        return r.json()

    def chat(self, text: str) -> dict:
        """Ask N-ATLaS. Keeps conversation history and returns the answer plus its interaction_id."""
        msgs = self.history + [{"role": "user", "content": text}]
        r = self._post("/v1/chat", {"messages": msgs, "session_id": self.session_id})
        self.history = msgs + [{"role": "assistant", "content": r["response"]}]
        return r

    def regenerate(self, interaction_id: int) -> dict:
        """Ask again for the last question. Atunse records that the first answer was not good enough."""
        msgs = self.history[:-1]
        r = self._post("/v1/chat", {"messages": msgs, "session_id": self.session_id, "regenerate_of": interaction_id})
        self.history = msgs + [{"role": "assistant", "content": r["response"]}]
        return r

    def thumbs_down(self, interaction_id: int, reason: str | None = None) -> dict:
        return self._post("/v1/signal", {"interaction_id": interaction_id, "type": "thumbs_down", "reason": reason})

    def thumbs_up(self, interaction_id: int) -> dict:
        return self._post("/v1/signal", {"interaction_id": interaction_id, "type": "thumbs_up"})

    def stats(self) -> dict:
        return requests.get(self.base + "/v1/stats", timeout=30).json()
