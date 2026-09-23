import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

import run_pipeline


@pytest.fixture
def render_stub(monkeypatch):
    monkeypatch.setattr(
        run_pipeline.wechat_screenshot,
        "resolve_font",
        lambda _: Path("font.ttf"),
    )
    monkeypatch.setattr(
        run_pipeline.wechat_screenshot, "load_fonts", lambda *args: None
    )

    def render(rows, output, **kwargs):
        assert kwargs["cli_self"] is None
        output.mkdir(parents=True, exist_ok=True)
        for row in rows:
            Image.new("RGB", (16, 16)).save(
                output / f"{row['conversation_id']}.png"
            )

    stub = Mock(side_effect=render)
    monkeypatch.setattr(run_pipeline.wechat_screenshot, "generate_batch", stub)
    return stub


@pytest.mark.parametrize("kind", ["private", "group"])
def test_existing_source_exports_without_model(
    kind, tmp_path, monkeypatch, render_stub
):
    client = Mock(
        side_effect=AssertionError("offline mode must not create client")
    )
    monkeypatch.setattr(
        run_pipeline.dataset_generation, "build_generation_client", client
    )
    source = (
        Path(__file__).resolve().parents[1] / "examples" / f"{kind}_demo.json"
    )
    output = tmp_path / kind
    assert (
        run_pipeline.main([kind, "--source", str(source), "-o", str(output)])
        == 0
    )
    assert json.loads(
        (output / "conversations.json").read_text()
    ) == json.loads(source.read_text())
    exported = output / "llamafactory"
    records = json.loads(next(exported.glob("*_eval.json")).read_text())
    assert len(records) == 1
    assert (exported / records[0]["images"][0]).is_file()
    assert records[0]["messages"][0]["content"].count("<image>") == 1
    assert json.loads(records[0]["messages"][1]["content"])["items"]
    assert (exported / "dataset_info.json").is_file()
    client.assert_not_called()


def test_render_failure_keeps_generated_source_and_skips_export(
    tmp_path, monkeypatch, render_stub
):
    rows = json.loads(
        (
            Path(__file__).resolve().parents[1] / "examples/private_demo.json"
        ).read_text()
    )
    client = object()
    factory = Mock(return_value=client)
    monkeypatch.setattr(
        run_pipeline.dataset_generation, "build_generation_client", factory
    )

    def generate(kind, count, output, **kwargs):
        assert kind == "private" and count == 1
        assert kwargs["client"] is client
        assert (
            kwargs["generator"]
            is run_pipeline.generate_private_dataset.generate_private_one
        )
        run_pipeline.private_export.atomic_json_write(output, rows)
        return rows

    monkeypatch.setattr(
        run_pipeline.dataset_generation, "generate_dataset", generate
    )
    render_stub.side_effect = ValueError("image too tall")
    export = Mock()
    monkeypatch.setattr(run_pipeline.private_export, "export_dataset", export)
    assert (
        run_pipeline.main(
            [
                "private",
                "--provider",
                "deepseek",
                "-n",
                "1",
                "-o",
                str(tmp_path),
            ]
        )
        == 1
    )
    factory.assert_called_once_with("deepseek", model=None, base_url=None)
    assert json.loads((tmp_path / "conversations.json").read_text()) == rows
    export.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [["-n", "0"], ["--width", "599"], ["--eval-ratio", "1"], ["--delay", "-1"]],
)
def test_invalid_options_fail_before_api(args, monkeypatch):
    factory = Mock()
    monkeypatch.setattr(
        run_pipeline.dataset_generation, "build_generation_client", factory
    )
    assert run_pipeline.main(["private", *args]) == 1
    factory.assert_not_called()


@pytest.mark.parametrize("extra", [["-n", "1"], ["--direction", "朋友聚会"]])
def test_source_rejects_generation_options(extra):
    with pytest.raises(SystemExit) as error:
        run_pipeline.main(["private", "--source", "existing.json", *extra])
    assert error.value.code == 2
