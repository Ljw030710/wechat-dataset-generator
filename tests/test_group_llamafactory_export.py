import json

import pytest

import export_group_llamafactory_dataset as group_export


def group_unit() -> dict:
    names = ["林轩", "周哲", "安然"]
    texts = (
        "周六下午三点在静安书店见。",
        "我可以到，顺便带投影仪。",
        "二楼靠窗的位置已经订好了。",
        "那就按三点，大家别迟到。",
        "我会提前十分钟过去。",
        "方案由周哲负责打印。",
        "收到，我今晚打印完成。",
        "结束后要不要一起吃饭？",
        "到时再看，先把复盘完成。",
        "好的，周六见。",
    )
    return {
        "conversation_id": "conv_group_export_test",
        "messages": [
            {
                "speaker": names[index % len(names)],
                "text": text,
                "time": f"2026-07-16 10:{index:02d}",
            }
            for index, text in enumerate(texts)
        ],
        "participants": ["林轩（组织者）", "周哲（执行者）", "安然（参与者）"],
        "schedule": [
            {
                "date": "2026-07-18",
                "owner": "林轩",
                "task": "在静安书店参加复盘",
                "time": "15:00",
            },
            {
                "date": "2026-07-17",
                "owner": "周哲",
                "task": "打印复盘方案",
                "time": "20:00",
            },
        ],
        "topic": "确认周末复盘安排",
        "group_name": "项目复盘组",
    }


def test_group_annotation_identifies_green_sender() -> None:
    payload = group_export.annotation_payload(group_unit())

    assert payload["green_sender"] == "林轩"
    assert payload["group_name"] == "项目复盘组"
    assert payload["messages"][0]["side"] == "green"
    assert payload["messages"][1]["side"] == "white"


def test_local_group_label_only_keeps_green_owned_task() -> None:
    assert group_export.label_from_schedule(group_unit()) == {
        "items": [
            {
                "title": "在静安书店参加复盘",
                "date": "2026-07-18",
                "time": "15:00",
                "note": "",
            }
        ]
    }


def test_group_sft_record_matches_llamafactory_image_count() -> None:
    label = group_export.validate_target(
        {
            "items": [
                {
                    "title": "在静安书店参加复盘",
                    "date": "2026-07-18",
                    "time": "15:00",
                    "note": "二楼靠窗，带投影仪",
                }
            ]
        }
    )
    record = group_export.build_sft_record("images/group.png", label)

    assert record["messages"][0] == {
        "role": "user",
        "content": group_export.DEFAULT_USER_PROMPT,
    }
    assert record["messages"][0]["content"].count("<image>") == len(
        record["images"]
    )
    assert json.loads(record["messages"][1]["content"]) == label


def test_group_target_rejects_extra_fields() -> None:
    with pytest.raises(ValueError, match="顶层字段"):
        group_export.validate_target({"items": [], "owner": "林轩"})
