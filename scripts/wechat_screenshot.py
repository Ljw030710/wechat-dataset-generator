#!/usr/bin/env python3
"""Batch-generate WeChat-style PNG screenshots from conversation JSON."""

from __future__ import annotations

import argparse
import colorsys
import dataclasses
import datetime
import hashlib
import json
import pathlib
import re
import sys
from collections.abc import Iterable
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

IOS_SCREEN_RATIO = 2436 / 1125

FONT_CANDIDATES = (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)

DEFAULT_AVATAR_SHEET = (
    pathlib.Path(__file__).resolve().parent.parent
    / "assets"
    / "avatars"
    / "wechat-fictional-avatar-sheet.png"
)
DEFAULT_FOOTER_IMAGE = (
    pathlib.Path(__file__).resolve().parent.parent
    / "assets"
    / "ui"
    / "wechat-footer.png"
)


@dataclasses.dataclass
class Fonts:
    """Font faces for the five text roles in a screenshot.

    Attributes:
        status: Status-bar clock font.
        title: Conversation title font.
        message: Chat-bubble body font.
        time: Message timestamp font.
        speaker: Incoming group-message sender font.
    """

    status: ImageFont.FreeTypeFont
    title: ImageFont.FreeTypeFont
    message: ImageFont.FreeTypeFont
    time: ImageFont.FreeTypeFont
    speaker: ImageFont.FreeTypeFont


@dataclasses.dataclass
class MessageLayout:
    """Measured geometry for one chat message, in output pixels.

    Attributes:
        message: Source speaker, text, and timestamp fields.
        outgoing: Whether to draw the message on the right in green.
        lines: Wrapped text lines inside the bubble.
        line_height: Distance between consecutive text baselines.
        bubble_width: Bubble width including horizontal padding.
        bubble_height: Bubble height including vertical padding.
        row_height: Height reserved for the avatar, sender label, and bubble.
        speaker_label: Whether an incoming group sender label is needed.
        y: Top of the row, measured from the top of the image.
    """

    message: dict[str, str]
    outgoing: bool
    lines: list[str]
    line_height: int
    bubble_width: int
    bubble_height: int
    row_height: int
    speaker_label: bool
    y: int


def resolve_font(font_path: str | None = None) -> pathlib.Path:
    """Find a Chinese font or raise FileNotFoundError if none is available."""
    if font_path:
        path = pathlib.Path(font_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"字体文件不存在：{path}")
        return path
    for candidate in FONT_CANDIDATES:
        path = pathlib.Path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError(
        "找不到中文字体，请通过 --font 指定 .ttf 或 .ttc 字体文件"
    )


def load_fonts(path: pathlib.Path, scale: float, dense: bool = False) -> Fonts:
    """Load scaled fonts, using smaller body text for dense conversations."""

    def size(value: float) -> int:
        return max(10, round(value * scale))

    return Fonts(
        status=ImageFont.truetype(str(path), size(36)),
        title=ImageFont.truetype(str(path), size(36)),
        message=ImageFont.truetype(str(path), size(29 if dense else 34)),
        time=ImageFont.truetype(str(path), size(19 if dense else 21)),
        speaker=ImageFont.truetype(str(path), size(17 if dense else 19)),
    )


def extract_units(payload: Any, source: str) -> list[dict[str, Any]]:
    """Unwrap supported conversation containers into a list of objects."""
    if isinstance(payload, list):
        units = payload
    elif isinstance(payload, dict) and isinstance(
        payload.get("conversations"), list
    ):
        units = payload["conversations"]
    elif isinstance(payload, dict) and isinstance(
        payload.get("messages"), list
    ):
        units = [payload]
    else:
        raise ValueError(
            f"{source} 不是会话对象、会话数组或 conversations 包装对象"
        )
    if not all(isinstance(unit, dict) for unit in units):
        raise ValueError(f"{source} 中的每个会话单元都必须是 JSON 对象")
    return units


def load_json_file(path: pathlib.Path) -> list[dict[str, Any]]:
    """Read JSON or JSONL, reporting malformed JSON with its source location."""
    if path.suffix.lower() == ".jsonl":
        units: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} JSON 格式错误：{error.msg}"
                ) from error
            units.extend(extract_units(payload, f"{path}:{line_number}"))
        return units
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path} JSON 格式错误：{error.msg}（第 {error.lineno} 行）"
        ) from error
    return extract_units(payload, str(path))


def load_conversations(input_path: pathlib.Path) -> list[dict[str, Any]]:
    """Load a file or recursively load JSON and JSONL files in sorted order."""
    if input_path.is_dir():
        files = sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        )
        if not files:
            raise ValueError(f"目录中没有 .json 或 .jsonl 文件：{input_path}")
        units: list[dict[str, Any]] = []
        for path in files:
            units.extend(load_json_file(path))
        return units
    if not input_path.is_file():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")
    return load_json_file(input_path)


def strip_description(value: str) -> str:
    """Remove a parenthesized role or description from a participant name."""
    return re.split(r"[（(]", value, maxsplit=1)[0].strip()


def normalize_conversation(unit: dict[str, Any], index: int) -> dict[str, Any]:
    """Return a normalized copy with message strings and missing defaults."""
    conversation_id = str(
        unit.get("conversation_id") or f"conversation_{index + 1:04d}"
    )
    raw_messages = unit.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(f"{conversation_id}: messages 必须是非空数组")

    messages: list[dict[str, str]] = []
    speakers: list[str] = []
    for message_index, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise ValueError(
                f"{conversation_id}: messages[{message_index}] 必须是对象"
            )
        missing = [
            key
            for key in ("speaker", "text", "time")
            if not str(raw.get(key, "")).strip()
        ]
        if missing:
            raise ValueError(
                f"{conversation_id}: messages[{message_index}] 缺少 {', '.join(missing)}"
            )
        message = {
            key: str(raw[key]).strip() for key in ("speaker", "text", "time")
        }
        messages.append(message)
        if message["speaker"] not in speakers:
            speakers.append(message["speaker"])

    raw_participants = unit.get("participants")
    participants = (
        [str(item).strip() for item in raw_participants]
        if isinstance(raw_participants, list)
        else speakers
    )
    if not participants:
        participants = speakers

    return {
        **unit,
        "conversation_id": conversation_id,
        "messages": messages,
        "participants": participants,
        "topic": str(unit.get("topic") or conversation_id),
        "group_name": str(unit.get("group_name") or "").strip(),
    }


def parse_message_time(value: str) -> datetime.datetime | None:
    """Parse a supported minute-resolution timestamp, or return None."""
    for pattern in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y/%m/%d %H:%M"):
        try:
            return datetime.datetime.strptime(value[:16], pattern)
        except ValueError:
            continue
    return None


def show_time_chip(
    current: dict[str, str], previous: dict[str, str] | None
) -> bool:
    """Show timestamps at the start, on new days, or after 30-minute gaps."""
    if previous is None:
        return True
    current_time = parse_message_time(current["time"])
    previous_time = parse_message_time(previous["time"])
    if not current_time or not previous_time:
        return current["time"] != previous["time"]
    return (
        current_time.date() != previous_time.date()
        or (current_time - previous_time).total_seconds() >= 1800
    )


def format_time(value: str, *, date: bool = True) -> str:
    """Format a timestamp for display, preserving unrecognized input."""
    parsed = parse_message_time(value)
    if not parsed:
        return value
    return (
        f"{parsed.month}月{parsed.day}日 {parsed:%H:%M}"
        if date
        else parsed.strftime("%H:%M")
    )


def text_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont
) -> int:
    """Measure glyph width in pixels; empty text has zero width."""
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    """Wrap text to a pixel width while preserving blank lines."""
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            candidate = current + char
            if current and text_width(draw, candidate, font) > max_width:
                lines.append(current.rstrip())
                current = char.lstrip() if char.isspace() else char
            else:
                current = candidate
        lines.append(current)
    return lines or [""]


def avatar_color(speaker: str) -> tuple[int, int, int]:
    """Return a stable, muted avatar background color for a participant."""
    digest = hashlib.sha256(strip_description(speaker).encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535
    saturation = 0.28 + digest[2] / 255 * 0.16
    value = 0.68 + digest[3] / 255 * 0.12
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(round(channel * 255) for channel in rgb)


def matches_self(speaker: str, self_name: str) -> bool:
    """Compare sender names while ignoring parenthesized descriptions."""
    return (
        speaker == self_name
        or strip_description(self_name) == speaker
        or strip_description(speaker) == strip_description(self_name)
    )


def choose_self(conversation: dict[str, Any], cli_self: str | None) -> str:
    """Resolve the right-side sender, defaulting to the first participant."""
    requested = cli_self or str(conversation.get("self") or "").strip()
    participants = conversation["participants"]
    if requested:
        for participant in participants:
            if matches_self(strip_description(participant), requested):
                return participant
        raise ValueError(
            f"{conversation['conversation_id']}: 找不到指定身份 {requested!r}"
        )
    return participants[0]


def other_title(conversation: dict[str, Any], self_name: str) -> str:
    """Return the peer name or a group title with its member count."""
    participants = conversation["participants"]
    others = [item for item in participants if item != self_name]
    if len(participants) == 2 and others:
        return strip_description(others[0])
    group_name = str(conversation.get("group_name", "")).strip() or "好友小分队"
    return f"{group_name}（{len(participants)}）"


def safe_filename(value: str) -> str:
    """Sanitize and shorten a conversation ID for use as a filename."""
    cleaned = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")
    return cleaned[:120] or "conversation"


def parse_color(value: str) -> tuple[int, int, int]:
    """Parse a six-digit RGB hex color, allowing an optional leading hash."""
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", value.strip())
    if not match:
        raise ValueError(
            f"颜色必须是 6 位十六进制，例如 #ededed；收到 {value!r}"
        )
    raw = match.group(1)
    return tuple(int(raw[index : index + 2], 16) for index in (0, 2, 4))


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    """Center text at xy using its font bounding box."""
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2 - box[1]),
        text,
        font=font,
        fill=fill,
    )


def draw_avatar(
    image: Image.Image,
    xy: tuple[int, int],
    size: int,
    speaker: str,
    radius: int,
    avatar_sheet: Image.Image | None = None,
    avatar_slot: int | None = None,
) -> None:
    """Paste an avatar from the sheet, or draw a deterministic illustration."""
    if avatar_sheet is not None and avatar_slot is not None:
        columns = rows = 3
        column, row = avatar_slot % columns, (avatar_slot // columns) % rows
        sheet_width, sheet_height = avatar_sheet.size
        inset_x = max(2, round(sheet_width * 0.008))
        inset_y = max(2, round(sheet_height * 0.008))
        left = round(column * sheet_width / columns) + inset_x
        top = round(row * sheet_height / rows) + inset_y
        right = round((column + 1) * sheet_width / columns) - inset_x
        bottom = round((row + 1) * sheet_height / rows) - inset_y
        crop = avatar_sheet.crop((left, top, right, bottom))
        avatar = ImageOps.fit(
            crop,
            (size, size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.46),
        )
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, size - 1, size - 1), radius=radius, fill=255
        )
        image.paste(avatar, xy, mask)
        ImageDraw.Draw(image).rounded_rectangle(
            (xy[0], xy[1], xy[0] + size - 1, xy[1] + size - 1),
            radius=radius,
            outline=(198, 198, 198),
            width=max(1, round(size / 72)),
        )
        return

    name = strip_description(speaker)
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    supersample = 3
    side = size * supersample
    s = supersample
    top_color = avatar_color(name)
    bottom_color = tuple(max(35, channel - 38) for channel in top_color)
    avatar = Image.new("RGB", (side, side), top_color)
    avatar_draw = ImageDraw.Draw(avatar)

    for row in range(side):
        blend = row / max(1, side - 1)
        color = tuple(
            round(a * (1 - blend) + b * blend)
            for a, b in zip(top_color, bottom_color)
        )
        avatar_draw.line((0, row, side, row), fill=color)

    style = digest[1] % 4
    if style in (0, 1):
        skin_palette = (
            (246, 211, 181),
            (231, 188, 151),
            (207, 158, 120),
            (249, 221, 195),
        )
        hair_palette = ((44, 35, 31), (75, 49, 36), (32, 39, 48), (94, 62, 42))
        shirt_palette = (
            (241, 239, 229),
            (71, 107, 153),
            (190, 91, 93),
            (68, 139, 117),
        )
        skin = skin_palette[digest[2] % len(skin_palette)]
        hair = hair_palette[digest[3] % len(hair_palette)]
        shirt = shirt_palette[digest[4] % len(shirt_palette)]
        head_x = (50 + (digest[5] % 7 - 3)) * s
        head_y = 47 * s

        avatar_draw.ellipse(
            (8 * s, 8 * s, 40 * s, 40 * s), fill=(255, 232, 166)
        )
        avatar_draw.ellipse((-7 * s, 71 * s, 107 * s, 127 * s), fill=shirt)
        avatar_draw.rounded_rectangle(
            ((head_x - 10 * s), 65 * s, (head_x + 10 * s), 84 * s),
            radius=5 * s,
            fill=skin,
        )
        avatar_draw.ellipse(
            (
                head_x - 27 * s,
                head_y - 31 * s,
                head_x + 27 * s,
                head_y + 31 * s,
            ),
            fill=hair,
        )
        avatar_draw.ellipse(
            (
                head_x - 23 * s,
                head_y - 25 * s,
                head_x + 23 * s,
                head_y + 29 * s,
            ),
            fill=skin,
        )
        if style == 0:
            avatar_draw.pieslice(
                (
                    head_x - 28 * s,
                    head_y - 34 * s,
                    head_x + 28 * s,
                    head_y + 15 * s,
                ),
                180,
                358,
                fill=hair,
            )
            avatar_draw.polygon(
                (
                    (head_x - 25 * s, head_y - 12 * s),
                    (head_x - 6 * s, head_y - 30 * s),
                    (head_x + 2 * s, head_y - 12 * s),
                ),
                fill=hair,
            )
        else:
            avatar_draw.arc(
                (
                    head_x - 25 * s,
                    head_y - 31 * s,
                    head_x + 25 * s,
                    head_y + 19 * s,
                ),
                190,
                350,
                fill=hair,
                width=8 * s,
            )
        eye_y = head_y + 1 * s
        for eye_x in (head_x - 9 * s, head_x + 9 * s):
            avatar_draw.ellipse(
                (eye_x - 2 * s, eye_y - 2 * s, eye_x + 2 * s, eye_y + 2 * s),
                fill=(55, 45, 40),
            )
        avatar_draw.arc(
            (head_x - 7 * s, head_y + 7 * s, head_x + 7 * s, head_y + 18 * s),
            12,
            168,
            fill=(142, 75, 70),
            width=s,
        )
    elif style == 2:
        cream = (247, 230, 197)
        ink = (64, 57, 49)
        avatar_draw.polygon(
            ((22 * s, 36 * s), (34 * s, 10 * s), (45 * s, 36 * s)), fill=cream
        )
        avatar_draw.polygon(
            ((55 * s, 36 * s), (67 * s, 10 * s), (79 * s, 36 * s)), fill=cream
        )
        avatar_draw.ellipse((18 * s, 22 * s, 82 * s, 87 * s), fill=cream)
        avatar_draw.ellipse((31 * s, 47 * s, 38 * s, 54 * s), fill=ink)
        avatar_draw.ellipse((62 * s, 47 * s, 69 * s, 54 * s), fill=ink)
        avatar_draw.polygon(
            ((47 * s, 59 * s), (53 * s, 59 * s), (50 * s, 64 * s)),
            fill=(184, 112, 103),
        )
        avatar_draw.arc(
            (39 * s, 59 * s, 50 * s, 70 * s), 4, 104, fill=ink, width=2 * s
        )
        avatar_draw.arc(
            (50 * s, 59 * s, 61 * s, 70 * s), 76, 176, fill=ink, width=2 * s
        )
        for offset in (-6, 0, 6):
            avatar_draw.line(
                (28 * s, (62 + offset) * s, 43 * s, (64 + offset) * s),
                fill=ink,
                width=s,
            )
            avatar_draw.line(
                (57 * s, (64 + offset) * s, 72 * s, (62 + offset) * s),
                fill=ink,
                width=s,
            )
    else:
        sun_x = (18 + digest[4] % 65) * s
        peak_one = (22 + digest[5] % 22) * s
        peak_two = (58 + digest[6] % 24) * s
        avatar_draw.ellipse(
            (sun_x - 12 * s, 11 * s, sun_x + 13 * s, 36 * s),
            fill=(248, 214, 117),
        )
        avatar_draw.polygon(
            ((0, 67 * s), (peak_one, 27 * s), (62 * s, 67 * s)),
            fill=(79, 115, 108),
        )
        avatar_draw.polygon(
            ((29 * s, 69 * s), (peak_two, 24 * s), (100 * s, 69 * s)),
            fill=(55, 87, 91),
        )
        water = (
            168 + digest[7] % 28,
            194 + digest[8] % 25,
            178 + digest[9] % 24,
        )
        avatar_draw.polygon(
            (
                (0, 72 * s),
                (35 * s, 57 * s),
                (58 * s, 77 * s),
                (100 * s, 51 * s),
                (100 * s, 100 * s),
                (0, 100 * s),
            ),
            fill=water,
        )
        avatar_draw.line(
            (0, 79 * s, 100 * s, 79 * s), fill=(224, 239, 231), width=2 * s
        )

    mask = Image.new("L", (side, side), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, side - 1, side - 1), radius=radius * s, fill=255
    )
    avatar = avatar.resize((size, size), Image.Resampling.LANCZOS)
    mask = mask.resize((size, size), Image.Resampling.LANCZOS)
    image.paste(avatar, xy, mask)
    ImageDraw.Draw(image).rounded_rectangle(
        (xy[0], xy[1], xy[0] + size - 1, xy[1] + size - 1),
        radius=radius,
        outline=(198, 198, 198),
        width=max(1, round(size / 72)),
    )


def build_avatar_slots(
    conversation: dict[str, Any], capacity: int = 9
) -> dict[str, int]:
    """Assign distinct avatar cells that remain stable across rerenders."""
    names: list[str] = []
    for raw in conversation["participants"]:
        name = strip_description(str(raw))
        if name and name not in names:
            names.append(name)
    for message in conversation["messages"]:
        name = strip_description(message["speaker"])
        if name and name not in names:
            names.append(name)
    if len(names) > capacity:
        raise ValueError(
            f"{conversation['conversation_id']}: 头像素材格只有 {capacity} 个，参与者数量过多"
        )

    result: dict[str, int] = {}
    used: set[int] = set()
    conversation_id = conversation["conversation_id"]
    for name in names:
        digest = hashlib.sha256(
            f"{conversation_id}|{name}".encode("utf-8")
        ).digest()
        slot = int.from_bytes(digest[:2], "big") % capacity
        # Probe the next cell on collisions so rerenders keep distinct avatars.
        while slot in used:
            slot = (slot + 1) % capacity
        result[name] = slot
        used.add(slot)
    return result


def draw_status_bar(
    draw: ImageDraw.ImageDraw,
    width: int,
    fonts: Fonts,
    latest_time: str,
) -> None:
    """Draw the clock, signal, Wi-Fi, and battery at the reference scale."""
    ink = (22, 22, 22)

    def to_pixels(value: float) -> int:
        return round(value * width / 1125)

    draw.text(
        (to_pixels(95), to_pixels(36)),
        format_time(latest_time, date=False),
        font=fonts.status,
        fill=ink,
    )

    signal_x, signal_top = to_pixels(869), to_pixels(48)
    for x, y, bar_width, bar_height in (
        (0, 27, 9, 9),
        (13, 20, 9, 16),
        (26, 12, 9, 24),
        (39, 3, 9, 33),
    ):
        draw.rounded_rectangle(
            (
                signal_x + to_pixels(x),
                signal_top + to_pixels(y),
                signal_x + to_pixels(x + bar_width),
                signal_top + to_pixels(y + bar_height),
            ),
            radius=max(1, to_pixels(1.5)),
            fill=(0, 0, 0),
        )

    draw_wifi(
        draw, (to_pixels(941), to_pixels(48)), to_pixels(48), to_pixels(36)
    )

    battery_left, battery_top = to_pixels(1007), to_pixels(54)
    draw.rounded_rectangle(
        (
            battery_left,
            battery_top,
            battery_left + to_pixels(60),
            battery_top + to_pixels(24),
        ),
        radius=max(2, to_pixels(5)),
        outline=(0, 0, 0),
        width=max(2, to_pixels(2.5)),
    )
    fill_left = battery_left + to_pixels(4)
    fill_top = battery_top + to_pixels(4)
    draw.rounded_rectangle(
        (
            fill_left,
            fill_top,
            fill_left + to_pixels(31),
            battery_top + to_pixels(20),
        ),
        radius=max(1, to_pixels(2)),
        fill=(0, 0, 0),
    )
    draw.rounded_rectangle(
        (
            battery_left + to_pixels(63),
            battery_top + to_pixels(7),
            battery_left + to_pixels(67),
            battery_top + to_pixels(17),
        ),
        radius=max(1, to_pixels(2)),
        fill=(0, 0, 0),
    )


def draw_header(
    draw: ImageDraw.ImageDraw,
    width: int,
    scale: float,
    fonts: Fonts,
    title: str,
    latest_time: str,
) -> None:
    """Draw the navigation header and status bar onto the supplied canvas."""

    def to_pixels(value: float) -> int:
        return round(value * scale)

    draw.rectangle((0, 0, width, to_pixels(210)), fill=(237, 237, 237))
    draw_status_bar(draw, width, fonts, latest_time)
    y = to_pixels(151)
    draw.line(
        (
            to_pixels(54),
            y - to_pixels(18),
            to_pixels(36),
            y,
            to_pixels(54),
            y + to_pixels(18),
        ),
        fill=(24, 24, 24),
        width=max(3, to_pixels(4)),
        joint="curve",
    )
    draw_centered(draw, (width / 2, y), title, fonts.title, (15, 15, 15))
    for offset in (-to_pixels(18), 0, to_pixels(18)):
        draw.ellipse(
            (
                width - to_pixels(55) + offset,
                y - to_pixels(4),
                width - to_pixels(47) + offset,
                y + to_pixels(4),
            ),
            fill=(20, 20, 20),
        )
    draw.line(
        (0, to_pixels(209), width, to_pixels(209)),
        fill=(215, 215, 215),
        width=1,
    )


def cubic_bezier_points(
    start: tuple[float, float],
    control_one: tuple[float, float],
    control_two: tuple[float, float],
    end: tuple[float, float],
    steps: int = 20,
) -> list[tuple[float, float]]:
    """Sample a cubic Bezier curve, including both endpoints."""
    points: list[tuple[float, float]] = []
    for index in range(steps + 1):
        t = index / steps
        one_minus = 1 - t
        x = (
            one_minus**3 * start[0]
            + 3 * one_minus**2 * t * control_one[0]
            + 3 * one_minus * t**2 * control_two[0]
            + t**3 * end[0]
        )
        y = (
            one_minus**3 * start[1]
            + 3 * one_minus**2 * t * control_one[1]
            + 3 * one_minus * t**2 * control_two[1]
            + t**3 * end[1]
        )
        points.append((x, y))
    return points


def draw_wifi(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    width: int,
    height: int,
) -> None:
    """Reproduce the reference Wi-Fi glyph from its 24×18 vector geometry."""
    sx, sy = width / 24, height / 18

    def map_points(
        points: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        return [(xy[0] + x * sx, xy[1] + y * sy) for x, y in points]

    outer = (
        cubic_bezier_points((12, 2), (7.8, 2), (4, 3.7), (1.2, 6.5))
        + cubic_bezier_points((2.7, 8), (5, 5.8), (8.3, 4.5), (12, 4.5))
        + cubic_bezier_points((12, 4.5), (15.7, 4.5), (19, 5.8), (21.3, 8))
        + cubic_bezier_points((22.8, 6.5), (19.9, 3.7), (16.2, 2), (12, 2))
    )
    middle = (
        cubic_bezier_points((12, 7), (9.1, 7), (6.5, 8.1), (4.6, 10))
        + cubic_bezier_points((6.1, 11.5), (7.8, 9.8), (9.8, 9), (12, 9))
        + cubic_bezier_points((12, 9), (14.2, 9), (16.2, 9.8), (17.9, 11.5))
        + cubic_bezier_points((19.4, 10), (17.5, 8.1), (14.9, 7), (12, 7))
    )
    inner = (
        cubic_bezier_points((12, 12), (10.3, 12), (8.8, 12.7), (7.7, 13.8))
        + cubic_bezier_points((9.2, 15.3), (9.9, 14.5), (10.9, 14), (12, 14))
        + cubic_bezier_points((12, 14), (13.1, 14), (14.1, 14.5), (14.8, 15.3))
        + cubic_bezier_points((16.3, 13.8), (15.2, 12.7), (13.7, 12), (12, 12))
    )
    for polygon in (outer, middle, inner):
        draw.polygon(map_points(polygon), fill=(0, 0, 0))
    draw.ellipse(
        (
            xy[0] + (12 - 1.5) * sx,
            xy[1] + (17 - 1.5) * sy,
            xy[0] + (12 + 1.5) * sx,
            xy[1] + (17 + 1.5) * sy,
        ),
        fill=(0, 0, 0),
    )


def draw_footer(image: Image.Image, width: int, top: int) -> None:
    """Paste the complete footer bitmap without redrawing its UI elements."""
    if not DEFAULT_FOOTER_IMAGE.is_file():
        raise FileNotFoundError(f"底栏图片不存在：{DEFAULT_FOOTER_IMAGE}")
    with Image.open(DEFAULT_FOOTER_IMAGE) as source:
        footer = source.convert("RGB")
    target_height = round(footer.height * width / footer.width)
    if footer.size != (width, target_height):
        footer = footer.resize((width, target_height), Image.Resampling.LANCZOS)
    image.paste(footer, (0, top))


def build_layouts(
    conversation: dict[str, Any],
    self_name: str,
    draw: ImageDraw.ImageDraw,
    fonts: Fonts,
    width: int,
    scale: float,
    dense: bool = False,
) -> tuple[list[tuple[str, Any]], int]:
    """Measure timestamps and message rows before drawing.

    Args:
        conversation: Normalized conversation with messages and participants.
        self_name: Participant shown in right-side green bubbles.
        draw: Drawing context used to measure text.
        fonts: Loaded fonts at the output scale.
        width: Output image width in pixels.
        scale: Ratio of output width to the 900-pixel reference layout.
        dense: Whether to use compact spacing for long conversations.

    Returns:
        A tuple of tagged timestamp/message layouts and their bottom y-coordinate.
        Each message payload is a MessageLayout; each time payload is (text, y).
    """

    def to_pixels(value: float) -> int:
        return round(value * scale)

    header_height = to_pixels(210)
    y = header_height + to_pixels(12 if dense else 31)
    max_text_width = min(
        to_pixels(600 if dense else 540),
        width - to_pixels(270 if dense else 330),
    )
    # WeChat lays messages out on a font baseline.  Using a mixed-glyph ink
    # bounding box as the line box makes the last line leave extra space below
    # it, which in turn makes the text look too high inside the bubble.
    ascent, descent = fonts.message.getmetrics()
    font_line_height = ascent + descent
    line_height = font_line_height + to_pixels(8 if dense else 15)
    vertical_padding = to_pixels(18 if dense else 30)
    participant_count = len(conversation["participants"])
    layouts: list[tuple[str, Any]] = []

    for index, message in enumerate(conversation["messages"]):
        previous = conversation["messages"][index - 1] if index else None
        if show_time_chip(message, previous):
            layouts.append(("time", (message["time"], y)))
            # Dense 10–12 message captures can contain several date separators.
            # Trim timestamp spacing so dense conversations are less likely
            # to overflow the fixed screen.
            y += to_pixels(34 if dense else 62)

        lines = wrap_text(draw, message["text"], fonts.message, max_text_width)
        widest = max(text_width(draw, line, fonts.message) for line in lines)
        horizontal_padding = to_pixels(44 if dense else 60)
        bubble_width = min(
            max_text_width + horizontal_padding,
            max(to_pixels(90 if dense else 104), widest + horizontal_padding),
        )
        text_block_height = font_line_height + (len(lines) - 1) * line_height
        bubble_height = text_block_height + vertical_padding * 2
        outgoing = matches_self(message["speaker"], self_name)
        speaker_label = participant_count > 2 and not outgoing
        label_height = to_pixels(22 if dense else 29) if speaker_label else 0
        row_height = max(
            to_pixels(82 if dense else 96), label_height + bubble_height
        )
        layout = MessageLayout(
            message=message,
            outgoing=outgoing,
            lines=lines,
            line_height=line_height,
            bubble_width=bubble_width,
            bubble_height=bubble_height,
            row_height=row_height,
            speaker_label=speaker_label,
            y=y,
        )
        layouts.append(("message", layout))
        y += row_height + to_pixels(12 if dense else 30)
    return layouts, y


def render_conversation(
    conversation: dict[str, Any],
    output_path: pathlib.Path,
    *,
    width: int = 900,
    font_path: pathlib.Path,
    self_name: str,
    background: tuple[int, int, int] = (237, 237, 237),
    fixed_height: int | None = None,
    avatar_sheet: pathlib.Path | None = None,
) -> None:
    """Render a normalized conversation to a PNG file.

    Args:
        conversation: Conversation with nonempty messages and participants.
        output_path: PNG destination; missing parent directories are created.
        width: Image width in pixels, at least 600.
        font_path: Chinese font file used for all text roles.
        self_name: Participant shown in right-side green bubbles.
        background: RGB background color.
        fixed_height: Required height in pixels, or None to size to the content.
        avatar_sheet: Optional nine-cell avatar image; otherwise draw avatars.

    Raises:
        ValueError: Width is too small or content exceeds fixed_height.
        FileNotFoundError: A required avatar sheet or footer bitmap is missing.
    """
    if width < 600:
        raise ValueError("图片宽度不能小于 600 像素")
    scale = width / 900

    def to_pixels(value: float) -> int:
        return round(value * scale)

    dense = len(conversation["messages"]) >= 10
    fonts = load_fonts(font_path, scale, dense)
    probe = Image.new("RGB", (width, 10), background)
    probe_draw = ImageDraw.Draw(probe)
    layouts, content_bottom = build_layouts(
        conversation, self_name, probe_draw, fonts, width, scale, dense
    )
    footer_height = to_pixels(221)
    required_height = content_bottom + footer_height + to_pixels(12)
    if fixed_height is not None:
        if fixed_height < required_height:
            raise ValueError(
                f"对话内容需要至少 {required_height}px 高度，超过固定手机截图 {fixed_height}px；请减少消息或缩短文字"
            )
        total_height = fixed_height
    else:
        total_height = max(to_pixels(1150), required_height)
    footer_top = total_height - footer_height

    image = Image.new("RGB", (width, total_height), background)
    draw = ImageDraw.Draw(image)
    avatar_sheet_image: Image.Image | None = None
    avatar_slots: dict[str, int] = {}
    if avatar_sheet is not None:
        if not avatar_sheet.is_file():
            raise FileNotFoundError(f"头像素材板不存在：{avatar_sheet}")
        with Image.open(avatar_sheet) as source:
            avatar_sheet_image = source.convert("RGB")
        avatar_slots = build_avatar_slots(conversation)
    draw.rectangle((0, to_pixels(210), width, footer_top), fill=background)
    draw_header(
        draw,
        width,
        scale,
        fonts,
        other_title(conversation, self_name),
        conversation["messages"][-1]["time"],
    )

    margin = to_pixels(29)
    avatar_size = to_pixels(82 if dense else 96)
    gap = to_pixels(14 if dense else 18)
    bubble_green = (152, 233, 112)
    message_ascent, _ = fonts.message.getmetrics()
    message_top_padding = to_pixels(18 if dense else 30)
    for kind, payload in layouts:
        if kind == "time":
            raw_time, y = payload
            label = format_time(raw_time)
            draw_centered(
                draw,
                (width / 2, y + to_pixels(19)),
                label,
                fonts.time,
                (166, 166, 166),
            )
            continue

        layout: MessageLayout = payload
        message = layout.message
        label_height = (
            to_pixels(22 if dense else 29) if layout.speaker_label else 0
        )
        bubble_y = layout.y + label_height
        if layout.outgoing:
            avatar_x = width - margin - avatar_size
            bubble_x = avatar_x - gap - layout.bubble_width
            fill = bubble_green
            draw.polygon(
                (
                    (
                        bubble_x + layout.bubble_width - 1,
                        bubble_y + to_pixels(24 if dense else 34),
                    ),
                    (
                        bubble_x
                        + layout.bubble_width
                        + to_pixels(11 if dense else 13),
                        bubble_y + to_pixels(36 if dense else 48),
                    ),
                    (
                        bubble_x + layout.bubble_width - 1,
                        bubble_y + to_pixels(48 if dense else 62),
                    ),
                ),
                fill=fill,
            )
        else:
            avatar_x = margin
            bubble_x = avatar_x + avatar_size + gap
            fill = (255, 255, 255)
            draw.polygon(
                (
                    (bubble_x + 1, bubble_y + to_pixels(24 if dense else 34)),
                    (
                        bubble_x - to_pixels(11 if dense else 13),
                        bubble_y + to_pixels(36 if dense else 48),
                    ),
                    (bubble_x + 1, bubble_y + to_pixels(48 if dense else 62)),
                ),
                fill=fill,
            )
            if layout.speaker_label:
                draw.text(
                    (bubble_x + to_pixels(5), layout.y),
                    message["speaker"],
                    font=fonts.speaker,
                    fill=(128, 128, 128),
                )

        draw.rounded_rectangle(
            (
                bubble_x,
                bubble_y,
                bubble_x + layout.bubble_width,
                bubble_y + layout.bubble_height,
            ),
            radius=to_pixels(12),
            fill=fill,
            outline=(135, 213, 94) if layout.outgoing else (224, 224, 224),
            width=max(1, to_pixels(1)),
        )
        text_baseline = bubble_y + message_top_padding + message_ascent
        for line in layout.lines:
            draw.text(
                (bubble_x + to_pixels(22 if dense else 30), text_baseline),
                line,
                font=fonts.message,
                fill=(20, 20, 20),
                anchor="ls",
            )
            text_baseline += layout.line_height
        speaker_key = strip_description(message["speaker"])
        draw_avatar(
            image,
            (avatar_x, layout.y),
            avatar_size,
            message["speaker"],
            to_pixels(9 if dense else 10),
            avatar_sheet_image,
            avatar_slots.get(speaker_key),
        )

    draw_footer(image, width, footer_top)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def unique_output_path(
    output_dir: pathlib.Path, conversation_id: str, used: set[pathlib.Path]
) -> pathlib.Path:
    """Reserve a path in used, suffixing duplicate names within the batch."""
    base = safe_filename(conversation_id)
    candidate = output_dir / f"{base}.png"
    suffix = 2
    while candidate in used:
        candidate = output_dir / f"{base}_{suffix}.png"
        suffix += 1
    used.add(candidate)
    return candidate


def generate_batch(
    conversations: Iterable[dict[str, Any]],
    output_dir: pathlib.Path,
    *,
    width: int,
    font_path: pathlib.Path,
    cli_self: str | None,
    background: tuple[int, int, int],
    fixed_height: int | None = None,
    avatar_sheet: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    """Normalize and render conversations, returning their PNG paths.

    Duplicate filenames within the batch receive numeric suffixes. Existing
    files at the selected paths are overwritten; images already saved remain
    on disk if a later conversation fails.
    """
    generated: list[pathlib.Path] = []
    used: set[pathlib.Path] = set()
    for index, unit in enumerate(conversations):
        conversation = normalize_conversation(unit, index)
        self_name = choose_self(conversation, cli_self)
        output_path = unique_output_path(
            output_dir, conversation["conversation_id"], used
        )
        render_conversation(
            conversation,
            output_path,
            width=width,
            font_path=font_path,
            self_name=self_name,
            background=background,
            fixed_height=fixed_height,
            avatar_sheet=avatar_sheet,
        )
        generated.append(output_path)
        print(f"✓ {conversation['conversation_id']} -> {output_path}")
    return generated


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for generic conversation rendering."""
    parser = argparse.ArgumentParser(
        description="从 JSON 批量生成微信风格聊天截图 PNG"
    )
    parser.add_argument(
        "input",
        type=pathlib.Path,
        help="单个 JSON/JSONL 文件，或包含这些文件的目录",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("output"),
        help="PNG 输出目录（默认：output）",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=900,
        help="输出图片宽度，至少 600（默认：900）",
    )
    parser.add_argument(
        "--self",
        dest="self_name",
        help="指定显示在右侧绿色气泡的参与者；默认使用 participants[0]",
    )
    parser.add_argument(
        "--font", help="中文 .ttf/.ttc 字体路径；默认自动查找系统字体"
    )
    parser.add_argument(
        "--background", default="#ededed", help="聊天背景色，例如 #f8d7df"
    )
    parser.add_argument(
        "--avatar-sheet", type=pathlib.Path, help="九宫格虚构头像素材板"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return its exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        font_path = resolve_font(args.font)
        background = parse_color(args.background)
        conversations = load_conversations(args.input)
        avatar_sheet = args.avatar_sheet or DEFAULT_AVATAR_SHEET
        generated = generate_batch(
            conversations,
            args.output,
            width=args.width,
            font_path=font_path,
            cli_self=args.self_name,
            background=background,
            fixed_height=round(args.width * IOS_SCREEN_RATIO),
            avatar_sheet=avatar_sheet,
        )
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    print(f"完成：共生成 {len(generated)} 张 PNG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
