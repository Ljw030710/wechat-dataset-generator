#!/usr/bin/env python3
"""Consolidate raw conversations and all local LLaMA-Factory exports.

The organizer is intentionally offline.  It validates and deduplicates raw
private/group conversations, reuses existing screenshots and annotations,
renders only screenshots that are missing, and writes one deterministic
LLaMA-Factory multimodal dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import shutil
from collections.abc import Iterable
from typing import Any

import dataset_generation
import export_group_llamafactory_dataset as group_export
import export_llamafactory_dataset as private_export
import merge_llamafactory_datasets as merge_datasets
import wechat_screenshot

DATASET_NAME = "wechat_schedule_combined"


def load_json(path: pathlib.Path) -> Any:
    """Read a UTF-8 JSON value from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    """Serialize JSON with sorted object keys for content comparisons."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def deduplicate_conversations(
    paths: Iterable[pathlib.Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return validated private and group rows, rejecting conflicting IDs."""
    rows_by_id: dict[str, dict[str, Any]] = {}
    kinds_by_id: dict[str, str] = {}
    for path in paths:
        value = load_json(path)
        if not isinstance(value, list):
            raise ValueError(f"{path} 顶层必须是 JSON 数组")
        for index, row in enumerate(value):
            if not isinstance(row, dict):
                raise ValueError(f"{path}[{index}] 必须是对象")
            chat_type = "group" if "group_name" in row else "private"
            validated = dataset_generation.validate_conversation(row, chat_type)
            conversation_id = validated["conversation_id"]
            if conversation_id in rows_by_id:
                if canonical(rows_by_id[conversation_id]) != canonical(
                    validated
                ):
                    raise ValueError(f"会话 ID 存在内容冲突：{conversation_id}")
                continue
            rows_by_id[conversation_id] = validated
            kinds_by_id[conversation_id] = chat_type

    def sort_key(row: dict[str, Any]) -> tuple[str, str]:
        return row["messages"][0]["time"], row["conversation_id"]

    private_rows = sorted(
        (
            row
            for identity, row in rows_by_id.items()
            if kinds_by_id[identity] == "private"
        ),
        key=sort_key,
    )
    group_rows = sorted(
        (
            row
            for identity, row in rows_by_id.items()
            if kinds_by_id[identity] == "group"
        ),
        key=sort_key,
    )
    return private_rows, group_rows


def record_id(record: dict[str, Any], source: pathlib.Path) -> str:
    """Derive a record ID from its single image filename."""
    images = record.get("images")
    if (
        not isinstance(images, list)
        or len(images) != 1
        or not isinstance(images[0], str)
    ):
        raise ValueError(f"{source} 中存在无效的 images 字段")
    return pathlib.Path(images[0]).stem


def load_existing_llamafactory(
    directories: Iterable[pathlib.Path],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, pathlib.Path],
]:
    """Load compatible annotations, SFT records, and image paths by ID.

    Missing directories are skipped. Conflicting content for the same ID,
    missing images, or unmatched record and annotation IDs are rejected.

    Returns:
        A tuple of annotation, record, and image-path dictionaries.

    Raises:
        ValueError: An export is malformed or conflicts with another export.
        FileNotFoundError: A referenced image is missing.
    """
    annotations: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    images: dict[str, pathlib.Path] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        annotation_paths = sorted(directory.glob("annotations*.json"))
        if len(annotation_paths) != 1:
            raise ValueError(f"{directory} 应当恰好包含一个 annotations*.json")
        source_annotations = load_json(annotation_paths[0])
        if not isinstance(source_annotations, dict):
            raise ValueError(f"{annotation_paths[0]} 顶层必须是对象")
        for identity, label in source_annotations.items():
            private_export.validate_target(label)
            previous = annotations.get(identity)
            if previous is not None and canonical(previous) != canonical(label):
                raise ValueError(f"已有标注内容冲突：{identity}")
            annotations[identity] = label

        registry_path = directory / "dataset_info.json"
        registry = load_json(registry_path)
        if not isinstance(registry, dict):
            raise ValueError(f"{registry_path} 顶层必须是对象")
        for entry in registry.values():
            if not isinstance(entry, dict) or not isinstance(
                entry.get("file_name"), str
            ):
                raise ValueError(f"{registry_path} 中存在无效条目")
            record_path = directory / entry["file_name"]
            source_records = load_json(record_path)
            if not isinstance(source_records, list):
                raise ValueError(f"{record_path} 顶层必须是数组")
            for record in source_records:
                if not isinstance(record, dict):
                    raise ValueError(f"{record_path} 中存在非对象记录")
                identity = record_id(record, record_path)
                previous = records.get(identity)
                if previous is not None and canonical(previous) != canonical(
                    record
                ):
                    raise ValueError(f"已有 SFT 记录内容冲突：{identity}")
                records[identity] = record
                image_path = directory / record["images"][0]
                if not image_path.is_file():
                    raise FileNotFoundError(f"缺少已有截图：{image_path}")
                images.setdefault(identity, image_path)
    if set(records) != set(annotations):
        missing_records = sorted(set(annotations) - set(records))[:5]
        missing_labels = sorted(set(records) - set(annotations))[:5]
        raise ValueError(
            f"已有记录与标注不一致：缺记录={missing_records}，缺标注={missing_labels}"
        )
    return annotations, records, images


def find_output_images(output_root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Index PNGs by filename stem, keeping the first path in sorted order."""
    images: dict[str, pathlib.Path] = {}
    if not output_root.is_dir():
        return images
    for path in sorted(output_root.rglob("*.png")):
        images.setdefault(path.stem, path)
    return images


def link_or_copy(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Hard-link an image, falling back to a metadata-preserving copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_images(
    private_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    existing_images: dict[str, pathlib.Path],
    output_images: dict[str, pathlib.Path],
    destination: pathlib.Path,
) -> tuple[int, int]:
    """Populate a new image directory, rendering only missing screenshots.

    Existing export images take precedence over loose output screenshots.

    Returns:
        A tuple containing the reused image count and the rendered image count.
    """
    destination.mkdir(parents=True, exist_ok=False)
    all_rows = private_rows + group_rows
    reused = 0
    for row in all_rows:
        identity = row["conversation_id"]
        source = existing_images.get(identity) or output_images.get(identity)
        if source is None:
            continue
        link_or_copy(source, destination / f"{identity}.png")
        reused += 1

    missing_private = [
        row
        for row in private_rows
        if not (destination / f"{row['conversation_id']}.png").is_file()
    ]
    missing_group = [
        row
        for row in group_rows
        if not (destination / f"{row['conversation_id']}.png").is_file()
    ]
    render_kwargs = {
        "width": 900,
        "font_path": wechat_screenshot.resolve_font(None),
        "cli_self": None,
        "background": wechat_screenshot.parse_color("#ededed"),
        "fixed_height": round(900 * wechat_screenshot.IOS_SCREEN_RATIO),
        "avatar_sheet": wechat_screenshot.DEFAULT_AVATAR_SHEET,
    }
    if missing_private:
        wechat_screenshot.generate_batch(
            missing_private, destination, **render_kwargs
        )
    if missing_group:
        wechat_screenshot.generate_batch(
            missing_group, destination, **render_kwargs
        )
    return reused, len(missing_private) + len(missing_group)


def build_dataset(
    private_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    existing_annotations: dict[str, dict[str, Any]],
    existing_records: dict[str, dict[str, Any]],
    output_dir: pathlib.Path,
    *,
    seed: int,
    eval_ratio: float,
) -> tuple[int, int, int]:
    """Write shuffled splits, preferring existing labels to schedule labels.

    Records found only in existing exports are retained. The sorted IDs and
    local random seed make the split reproducible for identical inputs.

    Returns:
        A tuple of training row count, evaluation row count, and label count.

    Raises:
        ValueError: The evaluation ratio or a selected label is invalid.
    """
    if not 0 <= eval_ratio < 1:
        raise ValueError("eval_ratio 必须在 [0, 1) 范围内")
    annotations: dict[str, dict[str, Any]] = {}
    records_by_id: dict[str, dict[str, Any]] = {}

    for row in private_rows:
        identity = row["conversation_id"]
        label = existing_annotations.get(
            identity
        ) or private_export.label_from_schedule(row)
        private_export.validate_target(label)
        annotations[identity] = label
        records_by_id[identity] = private_export.build_sft_record(
            f"images/{identity}.png", label
        )
    for row in group_rows:
        identity = row["conversation_id"]
        label = existing_annotations.get(
            identity
        ) or group_export.label_from_schedule(row)
        group_export.validate_target(label)
        annotations[identity] = label
        records_by_id[identity] = group_export.build_sft_record(
            f"images/{identity}.png", label
        )

    raw_ids = set(records_by_id)
    for identity in sorted(set(existing_records) - raw_ids):
        record = existing_records[identity]
        label = existing_annotations[identity]
        records_by_id[identity] = record
        annotations[identity] = label

    identities = sorted(records_by_id)
    random.Random(seed).shuffle(identities)
    eval_count = round(len(identities) * eval_ratio)
    if eval_ratio > 0 and identities:
        eval_count = max(1, eval_count)
    eval_ids = identities[:eval_count]
    train_ids = identities[eval_count:]

    train_file = f"{DATASET_NAME}_train.json"
    eval_file = f"{DATASET_NAME}_eval.json"
    merge_datasets.atomic_json_write(
        output_dir / train_file, [records_by_id[item] for item in train_ids]
    )
    merge_datasets.atomic_json_write(
        output_dir / eval_file, [records_by_id[item] for item in eval_ids]
    )
    merge_datasets.atomic_json_write(
        output_dir / "annotations.json", annotations
    )
    merge_datasets.atomic_json_write(
        output_dir / "dataset_info.json",
        {
            f"{DATASET_NAME}_train": merge_datasets.dataset_info_entry(
                train_file
            ),
            f"{DATASET_NAME}_eval": merge_datasets.dataset_info_entry(
                eval_file
            ),
        },
    )
    return len(train_ids), len(eval_ids), len(annotations)


def validate_output(output_dir: pathlib.Path, expected: int) -> None:
    """Verify record schemas, labels, image references, and expected counts."""
    registry = load_json(output_dir / "dataset_info.json")
    records: list[dict[str, Any]] = []
    for entry in registry.values():
        records.extend(load_json(output_dir / entry["file_name"]))
    annotations = load_json(output_dir / "annotations.json")
    identities: list[str] = []
    for record in records:
        identity = record_id(record, output_dir)
        identities.append(identity)
        if tuple(record.keys()) != ("messages", "images"):
            raise ValueError(f"SFT 记录字段错误：{identity}")
        messages = record["messages"]
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
            or messages[0].get("content")
            not in (
                private_export.DEFAULT_USER_PROMPT,
                group_export.DEFAULT_USER_PROMPT,
            )
        ):
            raise ValueError(f"SFT 消息格式错误：{identity}")
        label = json.loads(messages[1]["content"])
        private_export.validate_target(label)
        if canonical(label) != canonical(annotations[identity]):
            raise ValueError(f"SFT 答案与标注不一致：{identity}")
        if not (output_dir / record["images"][0]).is_file():
            raise FileNotFoundError(f"SFT 图片引用无效：{identity}")
    if len(records) != expected or len(set(identities)) != expected:
        raise ValueError(
            f"SFT 样本数量或唯一性错误：rows={len(records)}, unique={len(set(identities))}"
        )
    if set(identities) != set(annotations):
        raise ValueError("SFT 样本、图片与标注 ID 不一致")
    image_count = len(list((output_dir / "images").glob("*.png")))
    if image_count != expected:
        raise ValueError(f"图片数量错误：{image_count} != {expected}")


def main() -> int:
    """Run the command-line interface and return its exit status."""
    parser = argparse.ArgumentParser(
        description="整理全部本地会话并合并为一个 LLaMA-Factory 数据集"
    )
    parser.add_argument(
        "--data", type=pathlib.Path, default=pathlib.Path("data")
    )
    parser.add_argument(
        "--work", type=pathlib.Path, default=pathlib.Path("work")
    )
    parser.add_argument(
        "--output-images", type=pathlib.Path, default=pathlib.Path("output")
    )
    parser.add_argument(
        "--llamafactory-source",
        action="append",
        type=pathlib.Path,
        default=None,
        help="已有 LLaMA-Factory 数据目录，可重复指定",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("llamafactory_organized"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"输出目录已存在，为避免覆盖请先处理：{args.output}")
    sources = sorted(args.data.glob("*.json")) + sorted(
        args.work.rglob("*.json")
    )
    try:
        private_rows, group_rows = deduplicate_conversations(sources)
        llama_sources = args.llamafactory_source or [
            pathlib.Path("llamafactory_merged"),
            pathlib.Path("llamafactory_private_100"),
        ]
        existing_annotations, existing_records, existing_images = (
            load_existing_llamafactory(llama_sources)
        )
        output_dir = args.output
        output_dir.mkdir(parents=True)
        reused, rendered = prepare_images(
            private_rows,
            group_rows,
            existing_images,
            find_output_images(args.output_images),
            output_dir / "images",
        )
        llama_only_images = 0
        raw_ids = {row["conversation_id"] for row in private_rows + group_rows}
        for identity in sorted(set(existing_images) - raw_ids):
            link_or_copy(
                existing_images[identity],
                output_dir / "images" / f"{identity}.png",
            )
            llama_only_images += 1
        train_count, eval_count, total = build_dataset(
            private_rows,
            group_rows,
            existing_annotations,
            existing_records,
            output_dir,
            seed=args.seed,
            eval_ratio=args.eval_ratio,
        )
        validate_output(output_dir, total)
        merge_datasets.atomic_json_write(
            args.data / "private_dataset.json", private_rows
        )
        merge_datasets.atomic_json_write(
            args.data / "group_dataset.json", group_rows
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    print(
        "整理完成："
        f"原始私聊 {len(private_rows)} 条，原始群聊 {len(group_rows)} 条；"
        f"复用截图 {reused} 张，补渲染 {rendered} 张，保留 LLaMA-only 截图 {llama_only_images} 张；"
        f"训练集 {train_count} 条，验证集 {eval_count} 条，总计 {total} 条 -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
