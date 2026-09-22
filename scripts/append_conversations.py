#!/usr/bin/env python3
"""Validate and append a batch of conversations to an existing dataset."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
from typing import Any
from typing import Literal

import dataset_generation

ChatType = Literal["private", "group"]


def load_array(path: pathlib.Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSON array and reject non-object entries."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(f"{path} 必须是 JSON 对象数组")
    return value


def ending_key(row: dict[str, Any]) -> str:
    """Normalize the final message for duplicate-ending checks."""
    return re.sub(r"[\s，。！？!?～~]+", "", row["messages"][-1]["text"])


def validate_row(row: dict[str, Any], chat_type: ChatType) -> None:
    """Check schema, calendar references, dialogue, and group ending rules."""
    dataset_generation.validate_conversation(row, chat_type)
    dataset_generation.validate_calendar_mentions(row)
    dataset_generation.validate_natural_dialogue(row)
    if chat_type == "group":
        dataset_generation.validate_closed_group_ending(row)


def atomic_write(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    """Replace the dataset with UTF-8 JSON using a sibling temporary file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_batch(
    dataset: pathlib.Path, batch: pathlib.Path, chat_type: ChatType
) -> int:
    """Validate and append new conversations, saving each accepted row.

    Args:
        dataset: Existing JSON array to extend in place on disk.
        batch: JSON array containing candidate conversations.
        chat_type: Schema and ending rules to apply.

    Returns:
        The total number of conversations after appending the batch.

    Raises:
        ValueError: A row is invalid or repeats an ID, topic, ending, or group
            name. Rows accepted before the failure remain saved.
    """
    existing = load_array(dataset)
    additions = load_array(batch)
    ids = {row["conversation_id"] for row in existing}
    topics = {row["topic"] for row in existing}
    endings = {ending_key(row) for row in existing}
    group_names = (
        {row["group_name"] for row in existing}
        if chat_type == "group"
        else set()
    )

    for index, row in enumerate(additions, start=1):
        validate_row(row, chat_type)
        conversation_id = row["conversation_id"]
        topic = row["topic"]
        final_key = ending_key(row)
        if conversation_id in ids:
            raise ValueError(f"conversation_id 重复：{conversation_id}")
        if topic in topics:
            raise ValueError(f"topic 重复：{topic}")
        if final_key in endings:
            raise ValueError(f"结尾重复：{row['messages'][-1]['text']}")
        if chat_type == "group":
            group_name = row["group_name"]
            if group_name in group_names:
                raise ValueError(f"group_name 重复：{group_name}")
            group_names.add(group_name)
        ids.add(conversation_id)
        topics.add(topic)
        endings.add(final_key)
        existing.append(row)
        atomic_write(dataset, existing)
        print(f"[{index}/{len(additions)}] {conversation_id} 校验并保存")
    return len(existing)


def main() -> int:
    """Run the command-line interface and return its exit status."""
    parser = argparse.ArgumentParser(
        description="校验会话 JSON 批次并追加到已有数据集"
    )
    parser.add_argument("chat_type", choices=("private", "group"))
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("batch", type=pathlib.Path)
    args = parser.parse_args()
    total = append_batch(args.dataset, args.batch, args.chat_type)
    print(f"完成：{args.chat_type} 当前共 {total} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
