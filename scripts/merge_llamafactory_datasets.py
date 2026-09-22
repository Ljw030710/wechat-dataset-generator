#!/usr/bin/env python3
"""Merge private and group multimodal exports for LLaMA-Factory."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import shutil
from typing import Any

DEFAULT_DATASET_NAME = "wechat_all_green_schedule"


def atomic_json_write(path: pathlib.Path, value: Any) -> None:
    """Write UTF-8 JSON via a sibling temporary file, replacing the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    """Validate SFT records with one relative image reference per record."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} 顶层必须是 JSON 数组")
    records: list[dict[str, Any]] = []
    for index, record in enumerate(value):
        if not isinstance(record, dict) or tuple(record.keys()) != (
            "messages",
            "images",
        ):
            raise ValueError(f"{path}[{index}] 字段必须为 messages, images")
        messages = record["messages"]
        images = record["images"]
        if (
            not isinstance(messages, list)
            or not isinstance(images, list)
            or len(images) != 1
        ):
            raise ValueError(f"{path}[{index}] 必须包含消息和恰好一张图片")
        image_tokens = sum(
            message.get("content", "").count("<image>")
            for message in messages
            if isinstance(message, dict)
            and isinstance(message.get("content"), str)
        )
        if image_tokens != len(images):
            raise ValueError(f"{path}[{index}] 的 <image> 数量与 images 不一致")
        image_ref = images[0]
        if (
            not isinstance(image_ref, str)
            or pathlib.Path(image_ref).is_absolute()
        ):
            raise ValueError(f"{path}[{index}] 图片必须使用相对路径")
        records.append(record)
    return records


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


def merge_exports(
    private_dir: pathlib.Path,
    group_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    seed: int = 42,
) -> tuple[int, int, int]:
    """Merge exports without changing training and evaluation membership.

    Args:
        private_dir: Private-chat export with split files and annotations.
        group_dir: Group-chat export with split files and annotations.
        output_dir: Destination for merged JSON files and copied images.
        dataset_name: Prefix for split files and dataset registry entries.
        seed: Seed used to shuffle rows within each existing split.

    Returns:
        A tuple of training row count, evaluation row count, and image count.

    Raises:
        ValueError: Records are malformed, annotation IDs overlap, or image
            names refer to different contents.
        FileNotFoundError: A referenced source image is missing.
    """
    sources = (
        (
            private_dir,
            private_dir / "wechat_private_green_schedule_train.json",
            private_dir / "wechat_private_green_schedule_eval.json",
        ),
        (
            group_dir,
            group_dir / "wechat_group_green_schedule_train.json",
            group_dir / "wechat_group_green_schedule_eval.json",
        ),
    )
    train_with_sources: list[tuple[pathlib.Path, dict[str, Any]]] = []
    eval_with_sources: list[tuple[pathlib.Path, dict[str, Any]]] = []
    annotations: dict[str, Any] = {}

    for source_dir, train_path, eval_path in sources:
        train_with_sources.extend(
            (source_dir, row) for row in load_records(train_path)
        )
        eval_with_sources.extend(
            (source_dir, row) for row in load_records(eval_path)
        )
        annotation_path = source_dir / "annotations_schedule.json"
        source_annotations = json.loads(
            annotation_path.read_text(encoding="utf-8")
        )
        if not isinstance(source_annotations, dict):
            raise ValueError(f"{annotation_path} 顶层必须是对象")
        duplicate_ids = set(annotations) & set(source_annotations)
        if duplicate_ids:
            raise ValueError(f"标注 ID 重复：{sorted(duplicate_ids)[:3]}")
        annotations.update(source_annotations)

    random.Random(seed).shuffle(train_with_sources)
    random.Random(seed + 1).shuffle(eval_with_sources)
    all_rows = train_with_sources + eval_with_sources
    image_output = output_dir / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    copied_names: set[str] = set()
    for source_dir, record in all_rows:
        image_ref = record["images"][0]
        source_image = source_dir / image_ref
        if not source_image.is_file():
            raise FileNotFoundError(f"缺少图片：{source_image}")
        image_name = pathlib.Path(image_ref).name
        destination = image_output / image_name
        if image_name in copied_names:
            if (
                not destination.is_file()
                or source_image.read_bytes() != destination.read_bytes()
            ):
                raise ValueError(f"图片文件名冲突：{image_name}")
            continue
        shutil.copy2(source_image, destination)
        copied_names.add(image_name)

    train_file = f"{dataset_name}_train.json"
    eval_file = f"{dataset_name}_eval.json"
    atomic_json_write(
        output_dir / train_file, [row for _, row in train_with_sources]
    )
    atomic_json_write(
        output_dir / eval_file, [row for _, row in eval_with_sources]
    )
    atomic_json_write(output_dir / "annotations_schedule.json", annotations)
    atomic_json_write(
        output_dir / "dataset_info.json",
        {
            f"{dataset_name}_train": dataset_info_entry(train_file),
            f"{dataset_name}_eval": dataset_info_entry(eval_file),
        },
    )
    return len(train_with_sources), len(eval_with_sources), len(copied_names)


def main() -> int:
    """Run the command-line interface and return its exit status."""
    parser = argparse.ArgumentParser(
        description="合并私聊和群聊的 LLaMA-Factory 多模态数据"
    )
    parser.add_argument(
        "--private",
        type=pathlib.Path,
        default=pathlib.Path("llamafactory_data"),
    )
    parser.add_argument(
        "--group",
        type=pathlib.Path,
        default=pathlib.Path("llamafactory_group_data"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("llamafactory_combined_data"),
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        train_count, eval_count, image_count = merge_exports(
            args.private,
            args.group,
            args.output,
            dataset_name=args.dataset_name,
            seed=args.seed,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"完成：训练集 {train_count} 条，验证集 {eval_count} 条，"
        f"图片 {image_count} 张 -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
