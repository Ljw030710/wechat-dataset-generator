import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_generator import extraction_ready_group_unit
from test_generator import group_beat_plan
from test_generator import private_beat_plan
from test_generator import short_unit

import dataset_generation
import schedule_evidence


def supported(index, text):
    return {
        "supported": True,
        "issues": [],
        "evidence": {
            field: [{"message_index": index, "quote": text}]
            for field in ("task", "date", "time")
        },
    }


@pytest.fixture
def example():
    return json.loads(
        (
            Path(__file__).resolve().parents[1] / "examples/private_demo.json"
        ).read_text()
    )[0]


def test_audit_hides_background_and_keeps_only_dialogue_and_label(example):
    example["participants"][0] += "（隐藏身份设定）"
    example["topic"] = "隐藏的任务用途"
    client = Mock()
    client.complete.return_value = supported(6, example["messages"][5]["text"])
    schedule_evidence.validate_schedule_evidence(client, example)
    payload = client.complete.call_args.args[1]
    assert "隐藏" not in payload
    assert "time" not in json.loads(payload)["messages"][0]
    assert set(json.loads(payload)) == {"messages", "schedule"}


@pytest.mark.parametrize(
    "fault",
    [
        "false",
        "contradiction",
        "string_bool",
        "missing",
        "forged",
        "wrong_index",
        "white_only",
    ],
)
def test_audit_rejects_unsupported_or_invalid_evidence(example, fault):
    result = supported(6, example["messages"][5]["text"])
    if fault == "false":
        result.update(supported=False, issues=["正文未提到出租平台"])
    elif fault == "contradiction":
        result["issues"] = ["用途未表达"]
    elif fault == "string_bool":
        result["supported"] = "true"
    elif fault == "missing":
        del result["evidence"]["task"]
    elif fault == "forged":
        result["evidence"]["task"][0]["quote"] = "用于出租平台"
    elif fault == "wrong_index":
        result["evidence"]["task"][0]["message_index"] = 999
    elif fault == "white_only":
        result["evidence"]["task"] = [
            {"message_index": 1, "quote": example["messages"][0]["text"]}
        ]
    client = Mock()
    client.complete.return_value = result
    with pytest.raises(ValueError):
        schedule_evidence.validate_schedule_evidence(client, example)


@pytest.mark.parametrize("kind", ["private", "group"])
def test_failed_audit_repairs_then_audits_again(kind):
    if kind == "private":
        good = short_unit("private")
        good["messages"][8]["text"] = "7月18日下午3点我来重新订位"
        good["schedule"][0]["owner"] = "林轩"
        plan = {"private_plan": private_beat_plan()}
        index = 9
    else:
        good = extraction_ready_group_unit()
        plan = {"group_plan": group_beat_plan()}
        index = 7
    bad = copy.deepcopy(good)
    bad["schedule"][0]["task"] += "用于出租平台"
    client = Mock()
    client.complete.side_effect = [
        {"supported": False, "issues": ["正文未提到出租平台"]},
        good,
        supported(index, good["messages"][index - 1]["text"]),
    ]
    result = dataset_generation.validate_with_repair(client, bad, kind, **plan)
    assert result["schedule"][0]["task"] == "重新订位"
    assert client.complete.call_count == 3
    assert "正文未提到出租平台" in client.complete.call_args_list[1].args[1]
    assert (
        "用于出租平台"
        not in json.loads(client.complete.call_args_list[2].args[1])[
            "schedule"
        ][0]["task"]
    )


def test_audit_failure_at_repair_limit_rejects_sample():
    unit = extraction_ready_group_unit()
    client = Mock()
    client.complete.return_value = {
        "supported": False,
        "issues": ["正文未提到出租平台"],
    }
    with pytest.raises(ValueError, match="仍不合格.*出租平台"):
        dataset_generation.validate_with_repair(
            client, unit, "group", max_repairs=0, group_plan=group_beat_plan()
        )
    assert client.complete.call_count == 1
