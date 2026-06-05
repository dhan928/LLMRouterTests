from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_env(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        site_url: str | None = None,
        app_name: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL")
        self.app_name = app_name or os.getenv("OPENROUTER_APP_NAME", "LLMRouterTests")

    def chat(
        self,
        *,
        model: str,
        query: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> Mapping[str, object]:
        if not self.api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY in .env before calling OpenRouter.")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        request = urllib.request.Request(
            OPENROUTER_CHAT_URL,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed: {exc.code} {body}") from exc
