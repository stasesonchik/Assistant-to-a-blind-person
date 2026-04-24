from __future__ import annotations

import base64
import json
import mimetypes
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable


TokenCallback = Callable[[str], None]


class VLMClientError(RuntimeError):
    pass


class OpenAICompatibleVLM:
    """Small OpenAI-compatible Chat Completions client.

    It intentionally avoids the official OpenAI SDK so the same code works in a
    fresh Colab, on Windows, and with llama.cpp's server.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 180.0, api_key: str = "EMPTY"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    def describe_image(
        self,
        image_path: str | Path,
        prompt: str,
        memory_context: str = "",
        stream: bool = True,
        cancel_event: threading.Event | None = None,
        on_token: TokenCallback | None = None,
        max_tokens: int = 512,
    ) -> str:
        system_prompt = (
            "Ты русскоязычный ассистент для человека, который показывает предметы на камеру. "
            "Отвечай кратко, полезно и уверенно. Описывай видимые предметы, состояние, текст, "
            "цвета и возможное назначение. Если информации не хватает, явно скажи, что не видно. "
            "Используй память только когда она действительно помогает.\n\n"
            f"{memory_context}".strip()
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            },
        ]
        return self.chat(messages, stream=stream, cancel_event=cancel_event, on_token=on_token, max_tokens=max_tokens)

    def chat(
        self,
        messages: list[dict],
        stream: bool = True,
        cancel_event: threading.Event | None = None,
        on_token: TokenCallback | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if stream:
                    return self._read_stream(response, cancel_event, on_token)
                data = json.loads(response.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                if on_token:
                    on_token(content)
                return content
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise VLMClientError(f"VLM HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise VLMClientError(f"Cannot reach VLM server at {self.base_url}: {exc}") from exc

    def _read_stream(
        self,
        response,
        cancel_event: threading.Event | None,
        on_token: TokenCallback | None,
    ) -> str:
        chunks: list[str] = []
        for raw_line in response:
            if cancel_event and cancel_event.is_set():
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            token = extract_delta(event)
            if token:
                chunks.append(token)
                if on_token:
                    on_token(token)
        return "".join(chunks).strip()


def extract_delta(event: dict) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


def image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def print_token(token: str) -> None:
    print(token, end="", flush=True)
    if token.endswith((".", "!", "?", "\n")):
        sys.stdout.flush()

