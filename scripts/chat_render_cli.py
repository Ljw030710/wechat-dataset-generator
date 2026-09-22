#!/usr/bin/env python3
"""Typed CLI adapters for private and group screenshot rendering."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Literal

import wechat_screenshot

ChatType = Literal["private", "group"]


def _participant_name(value: object) -> str:
    return re.split(r"[（(]", str(value), maxsplit=1)[0].strip()


def validate_for_single_screen(
    unit: dict, chat_type: ChatType, index: int
) -> None:
    """Reject conversations that do not fit the single-screen contract."""
    identity = str(unit.get("conversation_id") or f"第 {index + 1} 条会话")
    participants = unit.get("participants")
    if not isinstance(participants, list):
        raise ValueError(f"{identity}: participants 必须是数组")
    if chat_type == "private" and len(participants) != 2:
        raise ValueError(f"{identity}: 私聊必须恰好有 2 位参与者")
    if chat_type == "group" and not 3 <= len(participants) <= 5:
        raise ValueError(f"{identity}: 群聊必须有 3–5 位参与者")

    messages = unit.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 12:
        raise ValueError(f"{identity}: 当前渲染器最多接受 12 条消息")
    names = {_participant_name(item) for item in participants}
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"{identity}: messages[{message_index}] 必须是对象"
            )
        if str(message.get("speaker", "")).strip() not in names:
            raise ValueError(
                f"{identity}: messages[{message_index}].speaker 不在 participants 中"
            )
        if len(str(message.get("text", ""))) > 28:
            raise ValueError(
                f"{identity}: messages[{message_index}] 超过 28 字，不适合手机聊天气泡"
            )


def render_cli(chat_type: ChatType, argv: list[str] | None = None) -> int:
    """Render the selected chat type; return 1 for input or rendering errors."""
    label = "私聊" if chat_type == "private" else "群聊"
    parser = argparse.ArgumentParser(
        description=f"把固定格式 JSON 渲染为微信风格{label}单屏截图"
    )
    parser.add_argument(
        "input", type=pathlib.Path, help="JSON/JSONL 文件或数据目录"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(f"output/{chat_type}"),
    )
    parser.add_argument(
        "--self",
        dest="self_name",
        help="右侧绿色气泡对应的参与者；默认 participants[0]",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=900,
        help="图片宽度（默认：900；截图采用 iPhone X 的 1125:2436 比例）",
    )
    parser.add_argument("--font", help="可选中文字体路径")
    parser.add_argument("--background", default="#ededed", help="聊天背景色")
    parser.add_argument(
        "--avatar-sheet", type=pathlib.Path, help="九宫格虚构头像素材板"
    )
    args = parser.parse_args(argv)
    try:
        units = wechat_screenshot.load_conversations(args.input)
        for index, unit in enumerate(units):
            validate_for_single_screen(unit, chat_type, index)
        avatar_sheet = (
            args.avatar_sheet or wechat_screenshot.DEFAULT_AVATAR_SHEET
        )
        generated = wechat_screenshot.generate_batch(
            units,
            args.output,
            width=args.width,
            font_path=wechat_screenshot.resolve_font(args.font),
            cli_self=args.self_name,
            background=wechat_screenshot.parse_color(args.background),
            fixed_height=round(args.width * wechat_screenshot.IOS_SCREEN_RATIO),
            avatar_sheet=avatar_sheet,
        )
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    print(f"完成：共生成 {len(generated)} 张{label}截图")
    return 0
