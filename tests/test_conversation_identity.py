import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

import dataset_generation
import export_group_llamafactory_dataset as group_export
import export_llamafactory_dataset as private_export
import wechat_screenshot
from conversation_identity import image_filename
from conversation_identity import validate_conversation_ids


@pytest.fixture(params=["private", "group"])
def case(request):
    kind = request.param
    row = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "examples"
            / f"{kind}_demo.json"
        ).read_text()
    )[0]
    exporter = private_export if kind == "private" else group_export
    return row, exporter


@pytest.mark.parametrize(
    "identity",
    [
        "demo:private",
        "../outside",
        "/tmp/outside",
        "a/b",
        "a\\b",
        "a b",
        "",
        None,
        123,
        "a" * 121,
        "CON",
        "lpt1",
    ],
)
def test_invalid_ids_fail_explicitly(identity):
    with pytest.raises(ValueError, match="conversation_id"):
        image_filename(identity)


@pytest.mark.parametrize("second_id", ["sample_1", "SAMPLE_1"])
def test_export_rejects_duplicate_ids_before_writes_or_api(
    case, tmp_path, second_id
):
    row, exporter = case
    row["conversation_id"] = "sample_1"
    second = copy.deepcopy(row)
    second["conversation_id"] = second_id
    source = tmp_path / "source.json"
    source.write_text(json.dumps([row, second]))
    output = tmp_path / "export"
    client = Mock()
    with pytest.raises(ValueError, match="第 1 条.*第 2 条"):
        exporter.export_dataset(
            source,
            tmp_path / "images",
            output,
            label_mode="teacher",
            client=client,
        )
    assert not output.exists()
    client.complete.assert_not_called()


def test_special_id_rejected_before_export(case, tmp_path):
    row, exporter = case
    row["conversation_id"] = "demo:private"
    source = tmp_path / "source.json"
    source.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="第 1 条.*demo:private"):
        exporter.export_dataset(
            source, tmp_path / "images", tmp_path / "export"
        )
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("second_id", ["sample", "SAMPLE", "sample:2"])
def test_renderer_preflights_whole_batch_before_writing(
    case, tmp_path, monkeypatch, second_id
):
    row, _ = case
    row["conversation_id"] = "sample"
    second = copy.deepcopy(row)
    second["conversation_id"] = second_id
    render = Mock()
    monkeypatch.setattr(wechat_screenshot, "render_conversation", render)
    with pytest.raises(ValueError, match="conversation_id"):
        wechat_screenshot.generate_batch(
            [row, second],
            tmp_path,
            width=900,
            font_path=Path("unused.ttf"),
            cli_self=None,
            background=(237, 237, 237),
        )
    render.assert_not_called()
    assert not list(tmp_path.glob("*.png"))


def test_render_and_export_keep_distinct_images_and_labels(
    case, tmp_path, monkeypatch
):
    row, exporter = case
    row["conversation_id"] = "sample_1"
    other = copy.deepcopy(row)
    other["conversation_id"] = "sample-2"
    other["schedule"][0]["task"] = "整理书桌"
    other["messages"][0]["text"] = "书桌上的东西摆好了。"
    rows = [row, other]
    source = tmp_path / "source.json"
    source.write_text(json.dumps(rows))
    images = tmp_path / "images"

    def render(conversation, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        color = (
            "red" if conversation["conversation_id"] == "sample_1" else "blue"
        )
        Image.new("RGB", (16, 16), color).save(output_path)

    monkeypatch.setattr(wechat_screenshot, "render_conversation", render)
    paths = wechat_screenshot.generate_batch(
        rows,
        images,
        width=900,
        font_path=Path("unused.ttf"),
        cli_self=None,
        background=(237, 237, 237),
    )
    assert {p.name for p in paths} == {"sample_1.png", "sample-2.png"}
    output = tmp_path / "export"
    assert exporter.export_dataset(source, images, output, eval_ratio=0.5) == (
        1,
        1,
    )
    records = [
        record
        for p in output.glob("*_*.json")
        if p.name.endswith(("_train.json", "_eval.json"))
        for record in json.loads(p.read_text())
    ]
    by_image = {
        record["images"][0]: json.loads(record["messages"][1]["content"])
        for record in records
    }
    assert len(by_image) == 2
    assert (
        by_image["images/sample_1.png"]["items"][0]["title"]
        == row["schedule"][0]["task"]
    )
    assert by_image["images/sample-2.png"]["items"][0]["title"] == "整理书桌"
    assert (output / "images/sample_1.png").read_bytes() != (
        output / "images/sample-2.png"
    ).read_bytes()


def test_resume_rejects_duplicate_ids_without_changing_saved_data(
    case, tmp_path
):
    row, _ = case
    source = tmp_path / "source.json"
    original = json.dumps([row, row])
    source.write_text(original)
    generator = Mock()
    kind = "group" if "group_name" in row else "private"
    with pytest.raises(ValueError, match="第 1 条.*第 2 条"):
        dataset_generation.generate_dataset(
            kind, 3, source, generator=generator, client=Mock()
        )
    generator.assert_not_called()
    assert source.read_text() == original


def test_generation_does_not_save_duplicate_candidate(
    case, tmp_path, monkeypatch
):
    row, _ = case
    generator = Mock(side_effect=[copy.deepcopy(row), copy.deepcopy(row)])
    monkeypatch.setattr(dataset_generation, "MAX_GENERATION_ATTEMPTS", 1)
    source = tmp_path / "generated.json"
    kind = "group" if "group_name" in row else "private"
    with pytest.raises(RuntimeError, match="conversation_id"):
        dataset_generation.generate_dataset(
            kind, 2, source, generator=generator, client=Mock(), delay=0
        )
    saved = json.loads(source.read_text())
    validate_conversation_ids(saved)
    assert len(saved) == 1
    assert generator.call_count == 2
