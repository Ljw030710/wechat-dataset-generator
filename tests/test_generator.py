import json
import pathlib

import pytest
from PIL import Image
from PIL import ImageDraw

import chat_render_cli
import dataset_generation
import export_llamafactory_dataset as private_export
import generate_group_dataset
import generate_private_dataset
import wechat_screenshot


def sample_unit(conversation_id: str = "conv_test") -> dict:
    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "speaker": "林志远",
                "text": "我们先确认今天的计划。",
                "time": "2026-03-23 14:05",
            },
            {
                "speaker": "周海",
                "text": "好的，我来安排现场。",
                "time": "2026-03-23 14:12",
            },
        ],
        "participants": ["林志远（负责流程）", "周海（负责现场）"],
        "schedule": [],
        "topic": "测试会话",
    }


def test_load_single_and_array_json(tmp_path: pathlib.Path) -> None:
    single = tmp_path / "single.json"
    single.write_text(json.dumps(sample_unit()), encoding="utf-8")
    assert len(wechat_screenshot.load_conversations(single)) == 1

    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps([sample_unit("one"), sample_unit("two")]), encoding="utf-8"
    )
    assert [
        item["conversation_id"]
        for item in wechat_screenshot.load_conversations(batch)
    ] == ["one", "two"]


def test_load_jsonl_and_directory(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    jsonl = source / "batch.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps(sample_unit(value), ensure_ascii=False)
            for value in ("a", "b")
        ),
        encoding="utf-8",
    )
    assert len(wechat_screenshot.load_conversations(source)) == 2


def test_normalization_and_self_selection() -> None:
    conversation = wechat_screenshot.normalize_conversation(sample_unit(), 0)
    assert wechat_screenshot.choose_self(conversation, None).startswith(
        "林志远"
    )
    assert wechat_screenshot.choose_self(conversation, "周海").startswith(
        "周海"
    )
    with pytest.raises(ValueError, match="找不到指定身份"):
        wechat_screenshot.choose_self(conversation, "不存在的人")


def test_invalid_message_is_rejected() -> None:
    unit = sample_unit()
    del unit["messages"][0]["text"]
    with pytest.raises(ValueError, match="缺少 text"):
        wechat_screenshot.normalize_conversation(unit, 0)


def test_batch_generates_png(tmp_path: pathlib.Path) -> None:
    output = tmp_path / "output"
    generated = wechat_screenshot.generate_batch(
        [sample_unit("one"), sample_unit("two")],
        output,
        width=600,
        font_path=wechat_screenshot.resolve_font(),
        cli_self=None,
        background=wechat_screenshot.parse_color("#ededed"),
        fixed_height=1000,
    )
    assert len(generated) == 2
    with Image.open(generated[0]) as image:
        assert image.format == "PNG"
        assert image.width == 600
        assert image.height == 1000


def test_message_text_uses_balanced_vertical_bubble_metrics() -> None:
    conversation = wechat_screenshot.normalize_conversation(sample_unit(), 0)
    fonts = wechat_screenshot.load_fonts(wechat_screenshot.resolve_font(), 1)
    probe = Image.new("RGB", (900, 10))
    layouts, _ = wechat_screenshot.build_layouts(
        conversation,
        conversation["participants"][0],
        ImageDraw.Draw(probe),
        fonts,
        900,
        1,
    )
    messages = [payload for kind, payload in layouts if kind == "message"]
    ascent, descent = fonts.message.getmetrics()

    assert messages[0].bubble_height == ascent + descent + 60
    assert abs(messages[0].bubble_height - 96) <= 4


def short_unit(chat_type: str) -> dict:
    participants = ["林轩（发起人）", "周哲（朋友）"]
    names = ["林轩", "周哲"]
    if chat_type == "group":
        participants.append("安然（朋友）")
        names.append("安然")
    unit = {
        "conversation_id": f"conv_{chat_type}_test",
        "messages": [
            {
                "speaker": names[index % len(names)],
                "text": text,
                "time": f"2026-07-12 10:{15 + index:02d}",
            }
            for index, text in enumerate(
                (
                    "明天的安排先取消吧。",
                    "好，那改到周六？",
                    "周六下午可以。",
                    "我先看看还有没有位置。",
                    "五点那场还剩后排。",
                    "后排也行，别太靠边。",
                    "我来重新订位。",
                    "订好后把取票码发群里。",
                    "已经订好，两张连座。",
                    "收到，周六见。",
                )
            )
        ],
        "participants": participants,
        "schedule": [
            {
                "date": "2026-07-18",
                "owner": names[1],
                "task": "重新订位",
                "time": "15:00",
            }
        ],
        "topic": "日常行程改期",
    }
    if chat_type == "group":
        unit["group_name"] = "周末搭子群"
    return unit


def extraction_ready_private_unit() -> dict:
    unit = short_unit("private")
    unit["messages"][8]["text"] = "7月18日下午3点我来重新订位"
    unit["schedule"][0] = {
        "date": "2026-07-18",
        "owner": "林轩",
        "task": "重新订位",
        "time": "15:00",
    }
    return unit


def extraction_ready_group_unit() -> dict:
    unit = short_unit("group")
    unit["messages"][6]["text"] = "7月18日下午三点我来重新订位"
    unit["schedule"][0] = {
        "date": "2026-07-18",
        "owner": "林轩",
        "task": "重新订位",
        "time": "15:00",
    }
    return unit


@pytest.mark.parametrize("chat_type", ["private", "group"])
def test_fixed_dataset_schema(chat_type: str) -> None:
    assert (
        dataset_generation.validate_conversation(
            short_unit(chat_type), chat_type
        )["topic"]
        == "日常行程改期"
    )


def test_private_training_schedule_is_visible_in_green_bubbles() -> None:
    unit = extraction_ready_private_unit()
    dataset_generation.validate_private_training_schedule(unit)

    unit["messages"][8]["text"] = "好，那就按刚才说的"
    with pytest.raises(ValueError, match="M月D日"):
        dataset_generation.validate_private_training_schedule(unit)

    unit = extraction_ready_private_unit()
    unit["messages"][8]["text"] = "7月18日上午3点我来重新订位"
    with pytest.raises(ValueError, match="具体钟点"):
        dataset_generation.validate_private_training_schedule(unit)

    unit = extraction_ready_private_unit()
    unit["schedule"][0]["time"] = "15:40"
    unit["messages"][8]["text"] = "7月18日下午三点四十我来重新订位"
    dataset_generation.validate_private_training_schedule(unit)


def test_private_training_schedule_requires_green_owner_and_future_item() -> (
    None
):
    unit = extraction_ready_private_unit()
    unit["schedule"][0]["owner"] = "周哲"
    with pytest.raises(ValueError, match=r"participants\[0\]"):
        dataset_generation.validate_private_training_schedule(unit)

    unit = extraction_ready_private_unit()
    unit["schedule"] = []
    with pytest.raises(ValueError, match="有且仅有 1 项"):
        dataset_generation.validate_private_training_schedule(unit)


def test_group_training_schedule_is_visible_in_green_bubbles() -> None:
    unit = extraction_ready_group_unit()
    dataset_generation.validate_group_training_schedule(unit)

    unit["messages"][6]["text"] = "好，我也参加"
    with pytest.raises(ValueError, match="M月D日"):
        dataset_generation.validate_group_training_schedule(unit)


def test_schema_rejects_extra_keys_and_long_chat() -> None:
    unit = short_unit("private")
    unit["chat_type"] = "private"
    with pytest.raises(ValueError, match="顶层字段不固定"):
        dataset_generation.validate_conversation(unit, "private")

    unit = short_unit("private")
    unit["messages"] *= 2
    with pytest.raises(ValueError, match="10–12"):
        dataset_generation.validate_conversation(unit, "private")


def test_schema_requires_formal_two_or_three_character_chinese_names() -> None:
    unit = short_unit("private")
    unit["participants"][0] = "林轩同学（发起人）"
    unit["messages"][0]["speaker"] = "林轩同学"
    with pytest.raises(ValueError, match="2–3 个汉字的正式姓名"):
        dataset_generation.validate_conversation(unit, "private")


def test_group_roster_is_rebuilt_from_formal_speaker_names() -> None:
    unit = short_unit("group")
    unit["participants"].append("陈小雨（未发言成员）")
    rebuilt = dataset_generation.rebuild_group_participants(unit)

    assert rebuilt["participants"] == [
        "林轩（发起人）",
        "周哲（朋友）",
        "安然（朋友）",
    ]
    assert dataset_generation.validate_conversation(rebuilt, "group") == rebuilt


def group_beat_plan() -> dict:
    names = ["林轩", "周哲", "安然"]
    return {
        "participants": [
            {
                "name": name,
                "group_role": role,
                "speaking_style": "简洁",
                "activity_level": "活跃",
            }
            for name, role in zip(
                names, ("组织者", "执行者", "提意见的人"), strict=True
            )
        ],
        "beats": [
            {
                "speaker": names[index % len(names)],
                "stage": "推进",
                "reply_to": "",
                "key_detail": "测试",
                "message_form": "文本",
            }
            for index in range(10)
        ],
        "speaking_order_note": "不机械轮流",
        "final_summary": "确认安排",
    }


def private_beat_plan() -> dict:
    names = ["林轩", "周哲"]
    return {
        "participants": [
            {"name": "林轩", "identity": "同学", "speaking_style": "直接"},
            {"name": "周哲", "identity": "同学", "speaking_style": "耐心"},
        ],
        "beats": [
            {
                "speaker": names[index % len(names)],
                "purpose": "推进",
                "key_detail": "测试",
                "emotion": "自然",
                "message_form": "文本",
            }
            for index in range(10)
        ],
        "minor_turn": "确认细节",
        "ending_state": "达成一致",
    }


def test_group_plan_constraints_override_invented_speakers() -> None:
    unit = short_unit("group")
    for message in unit["messages"]:
        message["speaker"] = "小林"
    unit["participants"] = ["小林（组织者）", "周哲（执行者）"]
    unit["schedule"][0]["owner"] = "周哲同学"

    constrained = dataset_generation.apply_group_plan_constraints(
        unit, group_beat_plan()
    )

    assert constrained["participants"] == [
        "林轩（组织者）",
        "周哲（执行者）",
        "安然（提意见的人）",
    ]
    assert [message["speaker"] for message in constrained["messages"]] == [
        beat["speaker"] for beat in group_beat_plan()["beats"]
    ]
    assert constrained["schedule"][0]["owner"] == "周哲"
    assert (
        dataset_generation.validate_conversation(constrained, "group")
        == constrained
    )


def test_private_plan_constraints_override_redundant_identity_fields() -> None:
    unit = short_unit("private")
    for message in unit["messages"]:
        message["speaker"] = "小林"
    unit["participants"] = ["小林（同学）", "老周（同学）"]

    constrained = dataset_generation.apply_private_plan_constraints(
        unit, private_beat_plan()
    )

    assert constrained["participants"] == ["林轩（同学）", "周哲（同学）"]
    assert [message["speaker"] for message in constrained["messages"]] == [
        beat["speaker"] for beat in private_beat_plan()["beats"]
    ]
    assert (
        dataset_generation.validate_conversation(constrained, "private")
        == constrained
    )


def test_brief_contracts_reject_nickname_rosters() -> None:
    private_brief = {"participants": private_beat_plan()["participants"]}
    group_brief = {"participants": group_beat_plan()["participants"]}
    assert dataset_generation.private_brief_contract(private_brief)[0] == [
        "林轩",
        "周哲",
    ]
    assert dataset_generation.group_brief_contract(group_brief)[0] == [
        "林轩",
        "周哲",
        "安然",
    ]

    private_brief["participants"][0]["name"] = "小林"
    with pytest.raises(ValueError, match="正式中文姓名"):
        dataset_generation.private_brief_contract(private_brief)


def test_private_schedule_is_locked_in_the_first_prompt_link() -> None:
    brief = {
        "confirmed_schedule": {
            "owner": "林轩",
            "task": "重新订位",
            "date": "2026-07-18",
            "time": "15:00",
            "note": "带学生证",
        }
    }
    item = generate_private_dataset.private_schedule_brief_contract(
        brief, ["林轩", "周哲"]
    )
    assert item["owner"] == "林轩"

    item["owner"] = "周哲"
    with pytest.raises(ValueError, match=r"participants\[0\]"):
        generate_private_dataset.private_schedule_brief_contract(
            brief, ["林轩", "周哲"]
        )


def test_group_schedule_is_locked_in_the_first_prompt_link() -> None:
    brief = {
        "confirmed_schedule": {
            "owner": "林轩",
            "task": "重新订位",
            "date": "2026-07-18",
            "time": "15:00",
            "note": "带群旗",
        }
    }
    item = generate_group_dataset.group_schedule_brief_contract(
        brief, ["林轩", "周哲", "安然"]
    )
    assert item["owner"] == "林轩"

    item["owner"] = "周哲"
    with pytest.raises(ValueError, match=r"participants\[0\]"):
        generate_group_dataset.group_schedule_brief_contract(
            brief, ["林轩", "周哲", "安然"]
        )


def test_private_plan_requires_both_people_to_speak() -> None:
    plan = private_beat_plan()
    for beat in plan["beats"]:
        beat["speaker"] = "林轩"

    with pytest.raises(ValueError, match="两位参与者"):
        dataset_generation.private_plan_contract(plan)


def test_schedule_datetime_normalization_is_deterministic() -> None:
    unit = short_unit("private")
    unit["schedule"] = [
        {
            "date": "2026年2月30日",
            "owner": "林轩",
            "task": "确认安排",
            "time": "9：05",
        }
    ]

    normalized = dataset_generation.normalize_schedule_datetimes(unit)

    assert normalized["schedule"][0]["date"] == "2026-02-28"
    assert normalized["schedule"][0]["time"] == "09:05"
    assert (
        dataset_generation.validate_conversation(normalized, "private")
        == normalized
    )

    unit["schedule"][0]["time"] = "下午3点半"
    assert (
        dataset_generation.normalize_schedule_datetimes(unit)["schedule"][0][
            "time"
        ]
        == "15:30"
    )

    unit["schedule"][0]["time"] = "下午三点一刻"
    assert (
        dataset_generation.normalize_schedule_datetimes(unit)["schedule"][0][
            "time"
        ]
        == "15:15"
    )


def test_plain_text_planning_markers_are_removed() -> None:
    unit = short_unit("private")
    unit["messages"][0]["text"] = "[文字] 明天下午三点见"
    unit["messages"][1]["text"] = "【文本】好，我准时到"

    normalized = dataset_generation.normalize_message_markers(unit)

    assert normalized["messages"][0]["text"] == "明天下午三点见"
    assert normalized["messages"][1]["text"] == "好，我准时到"


def test_long_message_is_shortened_at_complete_clause_boundary() -> None:
    unit = short_unit("private")
    unit["messages"][0]["text"] = (
        "陈老师，孩子这次数学只考了七十二分，比上次少了十三分，我有点担心。"
    )

    from dataset_generation import shorten_message_texts

    shortened = shorten_message_texts(unit)["messages"][0]["text"]

    assert shortened == "陈老师，孩子这次数学只考了七十二分，比上次少了十三分。"
    assert not shortened.endswith("…")


def test_group_generated_ending_cannot_reopen_the_decision() -> None:
    unit = short_unit("group")
    unit["messages"][-1]["text"] = "建议改到七点半，大家觉得呢？"

    with pytest.raises(ValueError, match="最终结果"):
        dataset_generation.validate_closed_group_ending(unit)


def test_generated_dialogue_does_not_force_a_short_message_quota() -> None:
    unit = short_unit("private")
    dataset_generation.validate_natural_dialogue(unit)

    for message in unit["messages"]:
        message["text"] = "这是一条没有短回应的完整消息"
    dataset_generation.validate_natural_dialogue(unit)


def test_generated_dialogue_rejects_telegram_style() -> None:
    unit = short_unit("private")
    for message in unit["messages"][:7]:
        message["text"] = "知道了"

    with pytest.raises(ValueError, match="最多保留 6 条"):
        dataset_generation.validate_natural_dialogue(unit)


def test_generated_dialogue_rejects_schedule_as_task_list() -> None:
    unit = short_unit("group")
    unit["schedule"] = [
        {
            "date": "2025-06-19",
            "owner": "林轩",
            "task": "到场",
            "time": "12:00",
        },
        {
            "date": "2025-06-19",
            "owner": "周哲",
            "task": "到场",
            "time": "12:00",
        },
    ]

    with pytest.raises(ValueError, match="schedule 最多保留 1 项"):
        dataset_generation.validate_natural_dialogue(unit)


def test_human_dialogue_style_rejects_glued_calendar_time() -> None:
    unit = extraction_ready_private_unit()
    unit["messages"][8]["text"] = "7月18日15:00我来重新订位"

    with pytest.raises(ValueError, match="日历记录"):
        dataset_generation.validate_human_dialogue_style(unit)


def test_human_dialogue_style_accepts_natural_chinese_time() -> None:
    dataset_generation.validate_human_dialogue_style(
        extraction_ready_private_unit()
    )


def test_generated_dialogue_allows_contextual_service_closing() -> None:
    unit = short_unit("private")
    unit["messages"][-1]["text"] = "随时联系我，祝你们活动顺利"

    dataset_generation.validate_natural_dialogue(unit)


def test_generated_dialogue_rejects_meeting_minutes_closing() -> None:
    unit = short_unit("group")
    unit["messages"][-1]["text"] = "方案通过，大家一致同意。"

    with pytest.raises(ValueError, match="模型式表达"):
        dataset_generation.validate_natural_dialogue(unit)


def test_generated_dialogue_allows_natural_favor_closing() -> None:
    unit = short_unit("private")
    unit["messages"][-1]["text"] = "回头请你吃火锅。"

    dataset_generation.validate_natural_dialogue(unit)


def test_generated_dialogue_allows_contextual_contact_offer() -> None:
    unit = short_unit("private")
    unit["messages"][-1]["text"] = "术前记得禁食8小时，有问题随时联系我"

    dataset_generation.validate_natural_dialogue(unit)


def test_simulation_label_is_removed_from_ordinary_topic() -> None:
    unit = short_unit("private")
    unit["topic"] = "推荐小学生科普书（模拟案例）"

    assert (
        dataset_generation.normalize_simulation_topic(unit)["topic"]
        == "推荐小学生科普书"
    )

    unit["topic"] = "客服售后服务评价（模拟案例）"
    assert (
        dataset_generation.normalize_simulation_topic(unit)["topic"]
        == "客服售后服务评价（模拟案例）"
    )


def test_generated_calendar_mentions_must_match_schedule() -> None:
    unit = short_unit("group")
    unit["schedule"] = [
        {
            "date": "2025-06-19",
            "owner": "林轩",
            "task": "参加聚会",
            "time": "12:00",
        }
    ]
    unit["messages"][0]["text"] = "6月19日周四中午聚会。"
    unit["messages"][-1]["text"] = "最终周四中午见。"
    dataset_generation.validate_calendar_mentions(unit)

    unit["messages"][0]["text"] = "6月19日周六中午聚会。"
    with pytest.raises(ValueError, match="应为周四"):
        dataset_generation.validate_calendar_mentions(unit)


def test_wrong_weekday_is_removed_without_changing_numeric_date() -> None:
    unit = short_unit("group")
    unit["schedule"] = [
        {
            "date": "2025-06-19",
            "owner": "林轩",
            "task": "参加聚会",
            "time": "12:00",
        }
    ]
    unit["messages"][0]["text"] = "6月19日周六中午聚会。"
    unit["messages"][-1]["text"] = "最终周六中午见。"

    normalized = dataset_generation.normalize_calendar_mentions(unit)

    assert normalized["messages"][0]["text"] == "6月19日中午聚会。"
    assert normalized["messages"][-1]["text"] == "最终中午见。"
    dataset_generation.validate_calendar_mentions(normalized)


def test_group_plan_contract_rejects_unknown_speaker() -> None:
    plan = group_beat_plan()
    plan["beats"][4]["speaker"] = "成员A"

    with pytest.raises(ValueError, match="无法映射"):
        dataset_generation.group_plan_contract(plan)


def test_group_plan_contract_recovers_roster_from_formal_beat_speakers() -> (
    None
):
    plan = group_beat_plan()
    plan["participants"] = []

    names, speakers, roles = dataset_generation.group_plan_contract(plan)

    assert names == ["林轩", "周哲", "安然"]
    assert speakers == [beat["speaker"] for beat in plan["beats"]]
    assert roles == {"林轩": "群成员", "周哲": "群成员", "安然": "群成员"}


def test_schema_accepts_a_story_without_cancellation_or_rescheduling() -> None:
    unit = short_unit("private")
    unit["messages"] = [
        {
            "speaker": "林轩",
            "text": "第12题我还是卡在辅助线。",
            "time": "2026-07-12 20:15",
        },
        {
            "speaker": "周哲",
            "text": "先连圆心，别急着套公式。",
            "time": "2026-07-12 20:16",
        },
        {
            "speaker": "林轩",
            "text": "懂了，原来半径相等。",
            "time": "2026-07-12 20:18",
        },
        {
            "speaker": "周哲",
            "text": "对，再自己重做一遍就稳了。",
            "time": "2026-07-12 20:19",
        },
        {
            "speaker": "林轩",
            "text": "第二步角度关系怎么写？",
            "time": "2026-07-12 20:20",
        },
        {
            "speaker": "周哲",
            "text": "先标同弧，再写圆周角相等。",
            "time": "2026-07-12 20:21",
        },
        {
            "speaker": "林轩",
            "text": "这样就能接上全等条件了。",
            "time": "2026-07-12 20:22",
        },
        {
            "speaker": "周哲",
            "text": "对，最后别漏对应边。",
            "time": "2026-07-12 20:23",
        },
        {
            "speaker": "林轩",
            "text": "我现在重新完整写一遍。",
            "time": "2026-07-12 20:24",
        },
        {
            "speaker": "周哲",
            "text": "写完拍给我，我帮你看。",
            "time": "2026-07-12 20:25",
        },
    ]
    unit["schedule"] = []
    unit["topic"] = "同学讨论几何错题"

    assert (
        dataset_generation.validate_conversation(unit, "private")["topic"]
        == "同学讨论几何错题"
    )


def test_typed_renderer_rejects_wrong_chat_type() -> None:
    with pytest.raises(ValueError, match="私聊必须"):
        chat_render_cli.validate_for_single_screen(
            short_unit("group"), "private", 0
        )


def test_group_header_uses_group_name_instead_of_topic() -> None:
    conversation = wechat_screenshot.normalize_conversation(
        short_unit("group"), 0
    )
    assert (
        wechat_screenshot.other_title(
            conversation, conversation["participants"][0]
        )
        == "周末搭子群（3）"
    )

    del conversation["group_name"]
    assert (
        wechat_screenshot.other_title(
            conversation, conversation["participants"][0]
        )
        == "好友小分队（3）"
    )


def test_extract_fenced_json() -> None:
    assert dataset_generation.extract_json_object(
        '```json\n{"ok": true}\n```'
    ) == {"ok": True}


def test_llamafactory_target_and_image_record() -> None:
    label = private_export.validate_target(
        {
            "items": [
                {
                    "title": "在静安书店见面",
                    "date": "2026-07-18",
                    "time": "15:00",
                    "note": "记得带上方案",
                }
            ]
        }
    )
    record = private_export.build_sft_record("images/example.png", label)

    assert record["messages"][0] == {
        "role": "user",
        "content": private_export.DEFAULT_USER_PROMPT,
    }
    assert json.loads(record["messages"][1]["content"]) == label
    assert record["images"] == ["images/example.png"]
    assert record["messages"][0]["content"].count("<image>") == len(
        record["images"]
    )


def test_annotation_marks_first_participant_as_green() -> None:
    source = private_export.annotation_payload(short_unit("private"))
    assert source["green_sender"] == "林轩"
    assert source["messages"][0]["side"] == "green"
    assert source["messages"][1]["side"] == "white"


def test_local_schedule_label_keeps_only_green_owner() -> None:
    unit = short_unit("private")
    unit["schedule"] = [
        {
            "date": "2026-07-18",
            "owner": "林轩",
            "task": "在静安书店见面",
            "time": "15:00",
        },
        {
            "date": "2026-07-19",
            "owner": "周哲",
            "task": "准备方案",
            "time": "20:00",
        },
    ]

    assert private_export.label_from_schedule(unit) == {
        "items": [
            {
                "title": "在静安书店见面",
                "date": "2026-07-18",
                "time": "15:00",
                "note": "",
            }
        ]
    }


def test_batch_generation_uses_open_request_and_avoids_recent_topics(
    tmp_path: pathlib.Path,
) -> None:
    directions: list[str] = []

    def generator(_client: object, direction: str, sequence: int) -> dict:
        directions.append(direction)
        return {"topic": f"开放主题{sequence}"}

    rows = dataset_generation.generate_dataset(
        "private",
        2,
        tmp_path / "open.json",
        generator=generator,  # type: ignore[arg-type]
        delay=0,
        client=object(),  # type: ignore[arg-type]
    )

    assert len(rows) == 2
    assert "不要从预设主题列表" in directions[0]
    assert "开放主题1" in directions[1]


def test_group_batch_passes_recent_names_and_retries_exact_duplicate(
    tmp_path: pathlib.Path,
) -> None:
    directions: list[str] = []
    generated_names = iter(("三号楼夜猫子", "三号楼夜猫子", "厨房门口见"))

    def generator(_client: object, direction: str, sequence: int) -> dict:
        directions.append(direction)
        return {
            "topic": f"群聊主题{sequence}",
            "group_name": next(generated_names),
        }

    rows = dataset_generation.generate_dataset(
        "group",
        2,
        tmp_path / "group_open.json",
        generator=generator,  # type: ignore[arg-type]
        delay=0,
        client=object(),  # type: ignore[arg-type]
    )

    assert [row["group_name"] for row in rows] == ["三号楼夜猫子", "厨房门口见"]
    assert "最近已经使用的群名" in directions[1]
    assert len(directions) == 3


def test_batch_retries_near_duplicate_topic(tmp_path: pathlib.Path) -> None:
    topics = iter(
        ("学生作文引用错误", "学生作文引用错误引发讨论", "同事分享阳台种花经验")
    )
    calls = 0

    def generator(_client: object, _direction: str, _sequence: int) -> dict:
        nonlocal calls
        calls += 1
        return {"topic": next(topics)}

    rows = dataset_generation.generate_dataset(
        "private",
        2,
        tmp_path / "private_open.json",
        generator=generator,  # type: ignore[arg-type]
        delay=0,
        client=object(),  # type: ignore[arg-type]
    )

    assert [row["topic"] for row in rows] == [
        "学生作文引用错误",
        "同事分享阳台种花经验",
    ]
    assert calls == 3


def test_batch_retries_same_life_domain_and_passes_reason_to_next_attempt(
    tmp_path: pathlib.Path,
) -> None:
    topics = iter(("借一本旧书", "寻找绝版诗集", "同事帮忙调试路由器"))
    directions: list[str] = []

    def generator(_client: object, direction: str, _sequence: int) -> dict:
        directions.append(direction)
        return {"topic": next(topics)}

    rows = dataset_generation.generate_dataset(
        "private",
        2,
        tmp_path / "private_domains.json",
        generator=generator,  # type: ignore[arg-type]
        delay=0,
        client=object(),  # type: ignore[arg-type]
    )

    assert [row["topic"] for row in rows] == [
        "借一本旧书",
        "同事帮忙调试路由器",
    ]
    assert "生活领域与最近数据重复" in directions[2]


def test_batch_detects_repeated_domain_hidden_in_message_text(
    tmp_path: pathlib.Path,
) -> None:
    outputs = iter(
        (
            {
                "topic": "随手聊聊旧物",
                "messages": [{"text": "这本旧书批注挺有意思"}],
            },
            {
                "topic": "最近的一件小事",
                "messages": [{"text": "刚找到一本绝版诗集"}],
            },
            {
                "topic": "桌面收纳",
                "messages": [{"text": "键盘线先绕到显示器后面"}],
            },
        )
    )

    def generator(_client: object, _direction: str, _sequence: int) -> dict:
        return next(outputs)

    rows = dataset_generation.generate_dataset(
        "private",
        2,
        tmp_path / "hidden_domain.json",
        generator=generator,  # type: ignore[arg-type]
        delay=0,
        client=object(),  # type: ignore[arg-type]
    )

    assert [row["topic"] for row in rows] == ["随手聊聊旧物", "桌面收纳"]


def test_brief_domain_is_rejected_before_drafting() -> None:
    direction = '本地分类器判定最近 2 条已覆盖这些领域：["图书出版"]。本条不得继续使用。'

    with pytest.raises(ValueError, match="无需继续写草稿"):
        dataset_generation.reject_forbidden_brief_domains(
            "朋友聊刚买到的旧书和诗集", direction
        )

    dataset_generation.reject_forbidden_brief_domains(
        "邻居聊阳台番茄苗", direction
    )


def test_scheduled_brief_is_rejected_when_recent_schedule_is_full() -> None:
    direction = "最近 4 条已有 2 条包含未来日程，最终 schedule 必须为 []。"

    with pytest.raises(ValueError, match="日程密度已满"):
        dataset_generation.reject_forbidden_brief_schedule(
            "朋友邀约周末去看展", direction
        )

    dataset_generation.reject_forbidden_brief_schedule(
        "朋友吐槽刚买的杯子漏水", direction
    )


def test_group_batch_does_not_force_empty_schedules(
    tmp_path: pathlib.Path,
) -> None:
    outputs = iter(
        {"topic": f"话题{index}", "schedule": [{}], "group_name": f"群{index}"}
        for index in range(1, 4)
    )
    directions: list[str] = []

    def generator(_client: object, direction: str, _sequence: int) -> dict:
        directions.append(direction)
        return next(outputs)

    rows = dataset_generation.generate_dataset(
        "group",
        3,
        tmp_path / "schedule_density.json",
        generator=generator,  # type: ignore[arg-type]
        delay=0,
        client=object(),  # type: ignore[arg-type]
    )

    assert [row["topic"] for row in rows] == ["话题1", "话题2", "话题3"]
    assert all(
        "最终 schedule 必须为 []" not in direction for direction in directions
    )


def test_private_batch_does_not_force_empty_schedules(
    tmp_path: pathlib.Path,
) -> None:
    outputs = iter(
        {"topic": f"话题{index}", "schedule": [{}]} for index in range(1, 4)
    )
    directions: list[str] = []

    def generator(_client: object, direction: str, _sequence: int) -> dict:
        directions.append(direction)
        return next(outputs)

    rows = dataset_generation.generate_dataset(
        "private",
        3,
        tmp_path / "private_schedule_density.json",
        generator=generator,  # type: ignore[arg-type]
        delay=0,
        client=object(),  # type: ignore[arg-type]
    )

    assert [row["topic"] for row in rows] == ["话题1", "话题2", "话题3"]
    assert all(
        "最终 schedule 必须为 []" not in direction for direction in directions
    )


def test_batch_retries_high_risk_topic(tmp_path: pathlib.Path) -> None:
    outputs = iter(
        (
            {"topic": "儿童误食硬币后的处理", "group_name": "夜班搭子"},
            {"topic": "同事分享桌面收纳办法", "group_name": "工位附近"},
        )
    )

    def generator(_client: object, _direction: str, _sequence: int) -> dict:
        return next(outputs)

    rows = dataset_generation.generate_dataset(
        "group",
        1,
        tmp_path / "low_risk.json",
        generator=generator,  # type: ignore[arg-type]
        delay=0,
        client=object(),  # type: ignore[arg-type]
    )

    assert rows[0]["topic"] == "同事分享桌面收纳办法"


def test_prompt_chain_passes_prior_outputs_to_later_links() -> None:
    final_unit = short_unit("private")
    final_unit["messages"][8]["text"] = "7月18日下午3点我来重新订位"
    final_unit["schedule"][0]["owner"] = "林轩"
    replies = [
        {
            "participants": private_beat_plan()["participants"],
            "relationship": "蓝图关系标记",
            "purpose": "确认复习方法",
            "event": "一次错题复盘",
            "emotional_arc": "担心 → 理解 → 放松",
            "detail_clues": ["第12题", "周三晚八点"],
            "everyday_detail": "刚下地铁",
            "confirmed_schedule": {
                "owner": "林轩",
                "task": "重新订位",
                "date": "2026-07-18",
                "time": "15:00",
                "note": "",
            },
            "safety_note": "虚构",
        },
        private_beat_plan(),
        final_unit,
        final_unit,
    ]

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, float]] = []

        def complete(
            self, system: str, user: str, *, temperature: float
        ) -> dict:
            self.calls.append((system, user, temperature))
            return replies[len(self.calls) - 1]

    client = RecordingClient()
    result = generate_private_dataset.generate_private_one(
        client, "学习教育类", 1
    )  # type: ignore[arg-type]

    assert len(client.calls) == 4
    assert "不要从预设主题列表中选择" in client.calls[0][1]
    assert "正式中文姓名" in client.calls[0][1]
    assert "不能只是聊家常" in client.calls[0][1]
    assert "蓝图关系标记" in client.calls[1][1]
    assert "最终明确约定" in client.calls[1][1]
    assert "蓝图关系标记" in client.calls[2][1]
    assert "minor_turn" in client.calls[2][1]
    assert "蓝图关系标记" in client.calls[3][1]
    assert "minor_turn" in client.calls[3][1]
    assert "成员姓名白名单" in client.calls[2][1]
    assert "自然但无歧义的钟点" in client.calls[2][1]
    assert result["conversation_id"].startswith("conv_private_")


def test_group_prompt_chain_uses_group_roles_and_prior_outputs() -> None:
    final_unit = extraction_ready_group_unit()
    replies = [
        {
            "group_name": "宿舍拼单群",
            "group_relationship": "宿舍群蓝图标记",
            "common_task": "采购日用品",
            "participants": group_beat_plan()["participants"],
            "temporary_conflict": "价格超过预算",
            "detail_clues": ["三人", "46元"],
            "final_result": "确认拼单",
            "confirmed_schedule": {
                "owner": "林轩",
                "task": "重新订位",
                "date": "2026-07-18",
                "time": "15:00",
                "note": "",
            },
            "safety_note": "虚构",
        },
        group_beat_plan(),
        final_unit,
        final_unit,
    ]

    class RecordingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, float]] = []

        def complete(
            self, system: str, user: str, *, temperature: float
        ) -> dict:
            self.calls.append((system, user, temperature))
            return replies[len(self.calls) - 1]

    client = RecordingClient()
    result = generate_group_dataset.generate_group_one(client, "宿舍群", 1)  # type: ignore[arg-type]

    assert len(client.calls) == 4
    assert (
        "群聊关系 + 眼前事项 + 成员状态 + 明确约定 + 自然落点"
        in client.calls[0][1]
    )
    assert "不要从预设群聊类型中选择" in client.calls[0][1]
    assert "正式中文姓名" in client.calls[0][1]
    assert "宿舍群蓝图标记" in client.calls[1][1]
    assert "最终明确约定" in client.calls[1][1]
    assert "speaking_order_note" in client.calls[2][1]
    assert "不要严格轮流" in client.calls[2][1]
    assert "整段只安排 3–5 条 2–7 个字" in client.calls[2][1]
    assert "最后一条不必复述" in client.calls[2][1]
    assert "宿舍群蓝图标记" in client.calls[3][1]
    assert "成员姓名白名单" in client.calls[2][1]
    assert "自然但无歧义的钟点" in client.calls[2][1]
    assert result["conversation_id"].startswith("conv_group_")
