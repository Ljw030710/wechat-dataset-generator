#!/usr/bin/env python3
"""Export group-chat screenshots as LLaMA-Factory multimodal SFT data.

This is a read-only consumer of the existing group JSON and PNG screenshots.
It does not import, call, or modify the screenshot rendering workflow.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import random
import re
import shutil
import sys
import time
from typing import Any

import annotation_cache
import dataset_generation

LABEL_TEMPERATURE = 0.1

DEFAULT_DATASET_NAME = "wechat_group_green_schedule"
DEFAULT_USER_PROMPT = """<image>请读取这张微信群聊截图，只提取右侧绿色气泡用户明确参与、接受或负责的最终日程。
其他成员的白色气泡只能用于理解上下文，不能单独给绿色用户新增事项。
忽略仍在讨论、绿色用户未确认、已拒绝、已取消或已被替代的安排。
仅输出合法 JSON，格式固定为 {"items":[{"title":"事项标题","date":"YYYY-MM-DD","time":"HH:MM","note":"补充提醒"}]}。
没有符合条件的日程时输出 {"items":[]}，不要输出解释或 Markdown。"""

LABEL_SYSTEM_PROMPT = """你是微信群聊截图微调数据标注员。请根据结构化的截图源消息生成准确监督标签，只输出合法 JSON。

核心定义：participants[0] 是截图右侧绿色气泡用户。目标不是提取群内所有安排，而是提取应该进入这个用户个人日历的安排。

应当标注：
1. 绿色用户主动提出并最终确定的安排。
2. 绿色用户明确表示会参加、会到、同意该时间，或以“到时见”等方式确认参与的共同活动。
3. 群内分配给绿色用户，且绿色用户明确接受或确认的个人任务。
4. 绿色用户负责组织、预订、提醒、提交或携带物品，并有确定日期和时间的任务。

不得标注：
1. 只属于其他成员的任务，或只有其他成员确认的活动。
2. 绿色用户没有回应、只发表意见、只开玩笑、明确拒绝或表示无法参加的安排。
3. 仍在提议、投票、等待确认，或者缺少可确定日期或时间的事项。
4. 已取消、已改期前的旧安排、已被后文替代的时间地点。
5. 仅凭白色气泡推断绿色用户会参加的事项。

上下文规则：
- “收到”“可以”“好”只有在紧邻上下文能明确证明绿色用户接受共同活动或个人任务时才算确认。
- 追踪整段群聊的状态，以聊天结束时最后有效版本为准；相同事件合并。
- 相对日期以对应消息 time 的日期为基准换算为 YYYY-MM-DD；time 使用 HH:MM。
- title 简短具体，不包含日期时间；note 只写截图中出现的地点、携带物、条件或提醒，没有则为空字符串。
- 顶层字段只能是 items；每项字段及顺序只能是 title, date, time, note；最多 3 项。
- 没有符合条件的事项时输出 {"items":[]}。"""


def participant_name(participant: str) -> str:
    """Return the name before any parenthesized participant description."""
    return re.split(r"[（(]", participant, maxsplit=1)[0].strip()


def validate_target(value: Any) -> dict[str, Any]:
    """Validate labels and strip title/note whitespace in place.

    Raises:
        ValueError: The label schema, dates, or item uniqueness are invalid.
    """
    if not isinstance(value, dict) or tuple(value.keys()) != ("items",):
        raise ValueError("标签顶层字段必须只有 items")
    items = value["items"]
    if not isinstance(items, list) or len(items) > 3:
        raise ValueError("items 必须是数组且最多 3 项")
    seen: set[tuple[str, str, str]] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or tuple(item.keys()) != (
            "title",
            "date",
            "time",
            "note",
        ):
            raise ValueError(
                f"items[{index}] 字段及顺序必须为 title, date, time, note"
            )
        if not isinstance(item["title"], str) or not item["title"].strip():
            raise ValueError(f"items[{index}].title 必须是非空字符串")
        if not isinstance(item["note"], str):
            raise ValueError(f"items[{index}].note 必须是字符串")
        try:
            datetime.datetime.strptime(item["date"], "%Y-%m-%d")
            datetime.datetime.strptime(item["time"], "%H:%M")
        except (TypeError, ValueError) as error:
            raise ValueError(f"items[{index}] 日期或时间格式错误") from error
        item["title"] = item["title"].strip()
        item["note"] = item["note"].strip()
        key = (item["title"], item["date"], item["time"])
        if key in seen:
            raise ValueError(f"items[{index}] 与前项重复")
        seen.add(key)
    return value


def annotation_payload(conversation: dict[str, Any]) -> dict[str, Any]:
    """Build teacher input with each message marked as green or white."""
    green_name = participant_name(conversation["participants"][0])
    return {
        "conversation_id": conversation["conversation_id"],
        "group_name": conversation["group_name"],
        "green_sender": green_name,
        "participants": conversation["participants"],
        "messages": [
            {
                "side": "green"
                if message["speaker"] == green_name
                else "white",
                "speaker": message["speaker"],
                "text": message["text"],
                "time": message["time"],
            }
            for message in conversation["messages"]
        ],
    }


def label_from_schedule(conversation: dict[str, Any]) -> dict[str, Any]:
    """Conservative local baseline: only tasks explicitly owned by green."""
    green_name = participant_name(conversation["participants"][0])
    return validate_target(
        {
            "items": [
                {
                    "title": item["task"],
                    "date": item["date"],
                    "time": item["time"],
                    "note": "",
                }
                for item in conversation["schedule"]
                if item["owner"] == green_name
            ]
        }
    )


def annotation_prompt(conversation: dict[str, Any]) -> str:
    """Build the exact user prompt used by annotation and cache checks."""
    payload = annotation_payload(conversation)
    return f"请标注以下群聊源消息：{dataset_generation.compact_json(payload)}"


def label_with_teacher(
    client: dataset_generation.ChatCompletionClient,
    conversation: dict[str, Any],
    *,
    attempts: int = 5,
) -> dict[str, Any]:
    """Ask the teacher for a validated group label with bounded retries."""
    user_prompt = annotation_prompt(conversation)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            label = client.complete(
                LABEL_SYSTEM_PROMPT,
                user_prompt,
                temperature=LABEL_TEMPERATURE,
            )
            return validate_target(label)
        except (RuntimeError, ValueError) as error:
            last_error = error
            print(
                f"  标签第 {attempt}/{attempts} 次失败：{error}",
                file=sys.stderr,
            )
            if attempt < attempts:
                time.sleep(0.5)
    raise RuntimeError(f"标签连续 {attempts} 次失败：{last_error}")


def atomic_json_write(path: pathlib.Path, value: Any) -> None:
    """Write UTF-8 JSON via a sibling temporary file, replacing the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_group_conversations(path: pathlib.Path) -> list[dict[str, Any]]:
    """Read a JSON array and validate each entry as a group conversation."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("群聊源数据顶层必须是数组")
    return [
        dataset_generation.validate_conversation(item, "group")
        for item in value
    ]


def build_sft_record(image_path: str, label: dict[str, Any]) -> dict[str, Any]:
    """Pair a screenshot reference with its prompt and JSON answer."""
    return {
        "messages": [
            {"role": "user", "content": DEFAULT_USER_PROMPT},
            {
                "role": "assistant",
                "content": json.dumps(label, ensure_ascii=False, indent=2),
            },
        ],
        "images": [image_path],
    }


def dataset_info_entry(file_name: str) -> dict[str, Any]:
    """Return the ShareGPT column mapping for a LLaMA-Factory data file."""
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }


def export_dataset(
    source: pathlib.Path,
    images: pathlib.Path,
    output: pathlib.Path,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    label_mode: str = "schedule",
    eval_ratio: float = 0.1,
    seed: int = 42,
    delay: float = 0.2,
    client: dataset_generation.ChatCompletionClient | None = None,
) -> tuple[int, int]:
    """Export screenshots and labels as training and evaluation splits.

    Args:
        source: JSON array of source conversations.
        images: Directory of screenshots named by conversation ID.
        output: Destination for labels, copied images, splits, and registry.
        dataset_name: Prefix for split filenames and registry entries.
        eval_ratio: Fraction assigned to evaluation, in [0, 1).
        seed: Seed for reproducible shuffling.
        delay: Pause in seconds after each new teacher annotation.
        label_mode: Use local schedule fields or request teacher annotations.
        client: Optional teacher client, used only in teacher mode.

    Returns:
        A tuple containing training and evaluation row counts.

    Raises:
        ValueError: The ratio, mode, conversations, or labels are invalid.
        FileNotFoundError: A conversation's screenshot is missing.
        RuntimeError: Teacher annotation retries are exhausted.

    Schedule labels are recomputed. Teacher labels are reused only when
    source content, provider, model and annotation prompts match the cache.
    """
    if label_mode not in ("schedule", "teacher"):
        raise ValueError("label_mode 必须是 schedule 或 teacher")
    if not 0 <= eval_ratio < 1:
        raise ValueError("eval_ratio 必须在 [0, 1) 范围内")

    conversations = load_group_conversations(source)
    api = (
        (client or dataset_generation.ChatCompletionClient())
        if label_mode == "teacher"
        else None
    )
    annotations_path = output / f"annotations_{label_mode}.json"
    annotations: dict[str, dict[str, Any]] = {}
    teacher_cache = (
        annotation_cache.TeacherAnnotationCache(output / "teacher_cache.json")
        if api is not None
        else None
    )

    for index, conversation in enumerate(conversations, 1):
        conversation_id = conversation["conversation_id"]
        image_source = images / f"{conversation_id}.png"
        if not image_source.is_file():
            raise FileNotFoundError(f"缺少对应截图：{image_source}")
        if api is not None and teacher_cache is not None:
            fingerprint = annotation_cache.teacher_fingerprint(
                conversation,
                api,
                system_prompt=LABEL_SYSTEM_PROMPT,
                user_prompt=annotation_prompt(conversation),
                temperature=LABEL_TEMPERATURE,
            )
            label = teacher_cache.get(conversation_id, fingerprint)
            if label is None:
                print(
                    f"[{index}/{len(conversations)}] 标注绿色气泡日程（teacher）"
                )
                label = label_with_teacher(api, conversation)
                teacher_cache.put(conversation_id, fingerprint, label)
                if delay > 0:
                    time.sleep(delay)
            annotations[conversation_id] = validate_target(label)
        else:
            annotations[conversation_id] = label_from_schedule(conversation)
        atomic_json_write(annotations_path, annotations)

    # Also clear stale labels when exporting an empty source dataset.
    atomic_json_write(annotations_path, annotations)

    image_output = output / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, dict[str, Any]]] = []
    for conversation in conversations:
        conversation_id = conversation["conversation_id"]
        file_name = f"{conversation_id}.png"
        shutil.copy2(images / file_name, image_output / file_name)
        records.append(
            (
                conversation_id,
                build_sft_record(
                    f"images/{file_name}", annotations[conversation_id]
                ),
            )
        )

    random.Random(seed).shuffle(records)
    eval_count = round(len(records) * eval_ratio)
    if eval_ratio > 0 and records:
        eval_count = max(1, eval_count)
    eval_records = [record for _, record in records[:eval_count]]
    train_records = [record for _, record in records[eval_count:]]
    train_file = f"{dataset_name}_train.json"
    eval_file = f"{dataset_name}_eval.json"
    atomic_json_write(output / train_file, train_records)
    atomic_json_write(output / eval_file, eval_records)
    atomic_json_write(
        output / "dataset_info.json",
        {
            f"{dataset_name}_train": dataset_info_entry(train_file),
            f"{dataset_name}_eval": dataset_info_entry(eval_file),
        },
    )
    return len(train_records), len(eval_records)


def main() -> int:
    """Run the command-line interface and return its exit status."""
    parser = argparse.ArgumentParser(
        description="导出群聊绿色气泡日程的 LLaMA-Factory 多模态数据"
    )
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=pathlib.Path("data/group_dataset.json"),
    )
    parser.add_argument(
        "--images", type=pathlib.Path, default=pathlib.Path("output/group")
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("llamafactory_group_data"),
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--label-mode",
        choices=("schedule", "teacher"),
        default="schedule",
        help="schedule 完全本地且保守；teacher 会把合成源消息发送给所选模型服务",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.2)
    dataset_generation.add_model_arguments(parser)
    args = parser.parse_args()
    try:
        client = (
            dataset_generation.build_generation_client(
                args.provider, model=args.model, base_url=args.base_url
            )
            if args.label_mode == "teacher"
            else None
        )
        train_count, eval_count = export_dataset(
            args.source,
            args.images,
            args.output,
            dataset_name=args.dataset_name,
            label_mode=args.label_mode,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
            delay=args.delay,
            client=client,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    print(
        f"完成：训练集 {train_count} 条，验证集 {eval_count} 条 -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
