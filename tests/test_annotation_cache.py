import copy
import json
from pathlib import Path

import pytest
from PIL import Image

import annotation_cache
import dataset_generation
import export_group_llamafactory_dataset as group_export
import export_llamafactory_dataset as private_export


class FakeTeacher(dataset_generation.ChatCompletionClient):
    def __init__(self):
        super().__init__(
            api_key="secret-test-key",
            provider_name="TestProvider",
            api_url="https://example.test/v1/chat/completions",
            model="test-model",
        )
        self.calls = 0

    def complete(self, system, user, *, temperature):
        self.calls += 1
        return {
            "items": [
                {
                    "title": f"标注结果{self.calls}",
                    "date": "2026-09-23",
                    "time": "15:00",
                    "note": "",
                }
            ]
        }


@pytest.fixture(params=[private_export, group_export], ids=["private", "group"])
def export_case(request, tmp_path):
    module = request.param
    kind = "group" if module is group_export else "private"
    rows = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "examples"
            / f"{kind}_demo.json"
        ).read_text(encoding="utf-8")
    )
    source = tmp_path / "source.json"
    images = tmp_path / "images"
    images.mkdir()
    output = tmp_path / "export"

    def run(values, *, mode="schedule", client=None):
        source.write_text(
            json.dumps(values, ensure_ascii=False), encoding="utf-8"
        )
        for row in values:
            Image.new("RGB", (16, 16)).save(
                images / f"{row['conversation_id']}.png"
            )
        module.export_dataset(
            source,
            images,
            output,
            label_mode=mode,
            client=client,
            eval_ratio=0,
            delay=0,
        )
        records = json.loads(
            (output / f"{module.DEFAULT_DATASET_NAME}_train.json").read_text()
        )
        return [
            json.loads(record["messages"][1]["content"]) for record in records
        ]

    return module, rows, output, run


def test_schedule_recomputes_changed_source_and_removes_deleted_labels(
    export_case,
):
    module, rows, output, run = export_case
    removed = copy.deepcopy(rows[0])
    removed["conversation_id"] += "_removed"
    run([*rows, removed])
    rows[0]["schedule"][0]["task"] = "领取新的海报"
    rows[0]["schedule"][0]["time"] = "16:30"
    labels = run(rows)
    expected = module.label_from_schedule(rows[0])
    assert labels == [expected]
    annotations = json.loads((output / "annotations_schedule.json").read_text())
    assert annotations == {rows[0]["conversation_id"]: expected}
    rows[0]["schedule"] = []
    assert run(rows) == [{"items": []}]
    assert not (output / "teacher_cache.json").exists()
    assert run([]) == []
    assert json.loads((output / "annotations_schedule.json").read_text()) == {}


def test_schedule_does_not_read_old_annotation_file(export_case):
    _, rows, output, run = export_case
    output.mkdir()
    (output / "annotations_schedule.json").write_text("obsolete invalid JSON")
    assert run(rows)[0]["items"]


def test_teacher_reuses_unchanged_request_without_storing_key(export_case):
    _, rows, output, run = export_case
    teacher = FakeTeacher()
    first = run(rows, mode="teacher", client=teacher)
    teacher.api_key = "rotated-secret-key"
    assert run(rows, mode="teacher", client=teacher) == first
    assert teacher.calls == 1
    cache_text = (output / "teacher_cache.json").read_text()
    assert "secret-test-key" not in cache_text
    assert "rotated-secret-key" not in cache_text


@pytest.mark.parametrize(
    "change",
    [
        "message",
        "schedule",
        "model",
        "endpoint",
        "provider",
        "system_prompt",
        "user_prompt",
        "temperature",
        "version",
    ],
)
def test_teacher_invalidates_changed_request(export_case, monkeypatch, change):
    module, rows, _, run = export_case
    teacher = FakeTeacher()
    first = run(rows, mode="teacher", client=teacher)
    if change == "message":
        rows[0]["messages"][0]["text"] = "我们再确认一下具体时间。"
    elif change == "schedule":
        rows[0]["schedule"][0]["time"] = "16:30"
    elif change == "model":
        teacher.model = "another-model"
    elif change == "endpoint":
        teacher.api_url = "https://another.test/v1/chat/completions"
    elif change == "provider":
        teacher.provider_name = "AnotherProvider"
    elif change == "system_prompt":
        monkeypatch.setattr(module, "LABEL_SYSTEM_PROMPT", "新的标注规则")
    elif change == "user_prompt":
        original = module.annotation_prompt
        monkeypatch.setattr(
            module,
            "annotation_prompt",
            lambda row: original(row) + "\n核对日期",
        )
    elif change == "temperature":
        monkeypatch.setattr(module, "LABEL_TEMPERATURE", 0.2)
    else:
        monkeypatch.setattr(annotation_cache, "CACHE_VERSION", 2)
    updated = run(rows, mode="teacher", client=teacher)
    assert teacher.calls == 2
    assert updated != first
    assert run(rows, mode="teacher", client=teacher) == updated
    assert teacher.calls == 2


def test_teacher_regenerates_legacy_annotations_without_fingerprint(
    export_case,
):
    _, rows, output, run = export_case
    output.mkdir()
    (output / "annotations_teacher.json").write_text(
        json.dumps({rows[0]["conversation_id"]: {"items": []}})
    )
    teacher = FakeTeacher()
    assert run(rows, mode="teacher", client=teacher)[0]["items"]
    assert teacher.calls == 1


def test_teacher_resumes_after_interruption_and_updates_only_changed_row(
    export_case,
    monkeypatch,
):
    module, rows, output, run = export_case
    second = copy.deepcopy(rows[0])
    second["conversation_id"] += "_second"
    rows.append(second)
    teacher = FakeTeacher()
    name = (
        "label_with_teacher" if module is group_export else "label_conversation"
    )
    original = getattr(module, name)

    def interrupted(client, conversation):
        if conversation["conversation_id"] == second["conversation_id"]:
            raise RuntimeError("simulated interruption")
        return original(client, conversation)

    monkeypatch.setattr(module, name, interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(rows, mode="teacher", client=teacher)
    assert teacher.calls == 1
    monkeypatch.setattr(module, name, original)
    run(rows, mode="teacher", client=teacher)
    assert teacher.calls == 2
    rows[0]["messages"][0]["text"] = "记得带上画册。"
    run(rows, mode="teacher", client=teacher)
    assert teacher.calls == 3
    run([rows[0]], mode="teacher", client=teacher)
    assert teacher.calls == 3
    annotations = json.loads((output / "annotations_teacher.json").read_text())
    assert set(annotations) == {rows[0]["conversation_id"]}
