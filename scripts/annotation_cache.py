"""Cache teacher labels by source content and the exact annotation request."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dataset_generation import ChatCompletionClient

# Increment when annotation semantics change beyond the request itself.
CACHE_VERSION = 1


def teacher_fingerprint(
    conversation: dict[str, Any],
    client: ChatCompletionClient,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> str:
    """Hash source, provider, model and prompts without including credentials."""
    value = {
        "version": CACHE_VERSION,
        "source": conversation,
        "provider": client.provider_name,
        "endpoint": client.api_url,
        "model": client.model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "temperature": temperature,
    }
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TeacherAnnotationCache:
    """Atomically persist each label together with its request fingerprint."""

    def __init__(self, path: Path) -> None:
        """Load versioned entries; legacy annotation-only files are not used."""
        self.path = path
        self.entries: dict[str, Any] = {}
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"标注缓存顶层必须是对象：{path}")
            if value.get("version") == CACHE_VERSION:
                entries = value.get("entries")
                if not isinstance(entries, dict):
                    raise ValueError(f"标注缓存 entries 必须是对象：{path}")
                self.entries = entries

    def get(self, identity: str, fingerprint: str) -> dict[str, Any] | None:
        """Return a label only when its complete request fingerprint matches."""
        entry = self.entries.get(identity)
        if isinstance(entry, dict) and entry.get("fingerprint") == fingerprint:
            label = entry.get("label")
            if isinstance(label, dict):
                return label
        return None

    def put(
        self, identity: str, fingerprint: str, label: dict[str, Any]
    ) -> None:
        """Save a validated label and its fingerprint in a single replacement."""
        self.entries[identity] = {"fingerprint": fingerprint, "label": label}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"version": CACHE_VERSION, "entries": self.entries},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
