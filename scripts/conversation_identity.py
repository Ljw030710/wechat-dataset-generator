"""Shared, portable conversation IDs and screenshot filenames."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


def image_filename(conversation_id: Any) -> str:
    """Validate an ID and return its unchanged stem plus the PNG suffix."""
    if not isinstance(conversation_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,120}", conversation_id
    ):
        raise ValueError(
            f"无效 conversation_id {conversation_id!r}："
            "必须由 1–120 个英文字母、数字、下划线或连字符组成；"
            "请修改源 JSON 中的 ID 后重新渲染，不会自动改名"
        )
    if re.fullmatch(
        r"CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9]", conversation_id, re.I
    ):
        raise ValueError(
            f"conversation_id {conversation_id!r} 是系统保留文件名"
        )
    return f"{conversation_id}.png"


def validate_conversation_ids(conversations: Iterable[dict[str, Any]]) -> None:
    """Reject invalid or colliding IDs before a batch writes any output.

    IDs differing only by case also collide on common macOS/Windows filesystems.
    Errors identify both one-based source positions.
    """
    seen: dict[str, tuple[int, str]] = {}
    for index, row in enumerate(conversations, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"第 {index} 条会话必须是 JSON 对象")
        identity = row.get("conversation_id")
        try:
            filename = image_filename(identity)
        except ValueError as error:
            raise ValueError(f"第 {index} 条会话：{error}") from error
        key = filename.casefold()
        if key in seen:
            first_index, first_id = seen[key]
            raise ValueError(
                f"conversation_id 重复或文件名冲突：第 {first_index} 条 "
                f"{first_id!r} 与第 {index} 条 {identity!r}；ID 不得仅大小写不同"
            )
        seen[key] = (index, identity)
