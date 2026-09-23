#!/usr/bin/env python3
"""Generate conversations, render screenshots, and export a paired dataset."""

from __future__ import annotations

import argparse
import pathlib
import sys

import chat_render_cli
import dataset_generation
import export_group_llamafactory_dataset as group_export
import export_llamafactory_dataset as private_export
import generate_group_dataset
import generate_private_dataset
import wechat_screenshot


def build_parser() -> argparse.ArgumentParser:
    """Describe the shared generation, rendering and export options."""
    parser = argparse.ArgumentParser(
        description="一次完成会话生成、截图渲染和 LLaMA-Factory 数据导出"
    )
    parser.add_argument("chat_type", choices=("private", "group"))
    dataset_generation.add_model_arguments(parser)
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="本次结果目录，默认 output/pipeline/<private 或 group>",
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "-n",
        "--count",
        type=int,
        help="生成目标总条数，包含已保存会话；默认 5",
    )
    inputs.add_argument(
        "--source",
        type=pathlib.Path,
        help="使用已有会话 JSON，跳过模型生成；与 -n 互斥",
    )
    parser.add_argument("--direction", help="模型生成的创作方向")
    parser.add_argument("--font", help="本地中文 .ttf 或 .ttc 字体路径")
    parser.add_argument(
        "--width", type=int, default=900, help="截图宽度，默认 900"
    )
    parser.add_argument(
        "--label-mode",
        choices=("schedule", "teacher"),
        default="schedule",
        help="schedule 本地提取标签（默认）；teacher 使用所选模型标注",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run all stages in order, preserving completed work if a stage fails."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source is not None and args.direction is not None:
        parser.error("--source 使用已有数据，不能同时指定 --direction")
    output = args.output or pathlib.Path("output/pipeline") / args.chat_type
    source = output / "conversations.json"
    images = output / "images"
    exported = output / "llamafactory"
    exporter = private_export if args.chat_type == "private" else group_export
    generator = (
        generate_private_dataset.generate_private_one
        if args.chat_type == "private"
        else generate_group_dataset.generate_group_one
    )
    stage = "配置检查"
    try:
        count = 5 if args.count is None else args.count
        if count < 1:
            raise ValueError("-n 必须至少为 1")
        if not 0 <= args.eval_ratio < 1:
            raise ValueError("--eval-ratio 必须在 [0, 1) 范围内")
        if args.width < 600:
            raise ValueError("--width 不能小于 600")
        if args.delay < 0:
            raise ValueError("--delay 不能为负数")
        font = wechat_screenshot.resolve_font(args.font)
        # Check render resources before incurring generation costs.
        wechat_screenshot.load_fonts(font, args.width / 900)
        for asset in (
            wechat_screenshot.DEFAULT_AVATAR_SHEET,
            wechat_screenshot.DEFAULT_FOOTER_IMAGE,
        ):
            if not asset.is_file():
                raise FileNotFoundError(f"缺少渲染素材：{asset}")
        client = (
            dataset_generation.build_generation_client(
                args.provider, model=args.model, base_url=args.base_url
            )
            if args.source is None or args.label_mode == "teacher"
            else None
        )

        stage = "会话生成" if args.source is None else "会话读取"
        print(f"[1/3] {stage}", flush=True)
        if args.source is None:
            rows = dataset_generation.generate_dataset(
                args.chat_type,
                count,
                source,
                generator=generator,
                direction=args.direction,
                delay=args.delay,
                client=client,
            )
        else:
            loader = (
                private_export.load_private_conversations
                if args.chat_type == "private"
                else group_export.load_group_conversations
            )
            rows = loader(args.source)
            private_export.atomic_json_write(source, rows)

        stage = "截图渲染"
        print("[2/3] 截图渲染", flush=True)
        for index, row in enumerate(rows):
            chat_render_cli.validate_for_single_screen(
                row, args.chat_type, index
            )
        wechat_screenshot.generate_batch(
            rows,
            images,
            width=args.width,
            font_path=font,
            cli_self=None,
            background=(237, 237, 237),
            fixed_height=round(args.width * wechat_screenshot.IOS_SCREEN_RATIO),
            avatar_sheet=wechat_screenshot.DEFAULT_AVATAR_SHEET,
        )

        stage = "训练数据导出"
        print("[3/3] 训练数据导出", flush=True)
        train_count, eval_count = exporter.export_dataset(
            source,
            images,
            exported,
            label_mode=args.label_mode,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
            delay=args.delay,
            client=client,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"{stage}失败：{error}", file=sys.stderr)
        print(
            f"已完成的数据保留在 {output}，修正问题后可用同一命令重跑。",
            file=sys.stderr,
        )
        return 1

    print(
        f"完成：{len(rows)} 条会话，训练集 {train_count} 条，验证集 {eval_count} 条。"
    )
    print(f"会话：{source}\n截图：{images}\n训练数据：{exported}")
    if train_count == 0 and rows:
        print(
            "本次训练集为空，适合检查流程；正式训练请增加样本数量或调整 --eval-ratio。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
