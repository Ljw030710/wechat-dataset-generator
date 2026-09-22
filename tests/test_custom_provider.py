import io
import json
import sys

import pytest

import dataset_generation
import export_group_llamafactory_dataset as group_export
import export_llamafactory_dataset as private_export


@pytest.fixture
def custom_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "custom-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-unrelated")


def test_custom_request_uses_selected_endpoint_key_and_model(
    monkeypatch: pytest.MonkeyPatch, custom_environment: None
) -> None:
    def urlopen(request, *, timeout):
        assert request.full_url == "https://example.test/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer custom-test-key"
        payload = json.loads(request.data)
        assert payload["model"] == "custom-model"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["messages"][1]["content"] == "test input"
        return io.BytesIO(
            json.dumps(
                {"choices": [{"message": {"content": '{"ok": true}'}}]}
            ).encode()
        )

    monkeypatch.setattr(dataset_generation.urllib.request, "urlopen", urlopen)
    client = dataset_generation.build_generation_client("custom")
    assert client.complete("system", "test input", temperature=0.4) == {
        "ok": True
    }


@pytest.mark.parametrize(
    "base_url",
    [
        "https://override.test/api/v2",
        "https://override.test/api/v2/chat/completions/",
    ],
)
def test_explicit_configuration_overrides_environment(
    custom_environment: None, base_url: str
) -> None:
    client = dataset_generation.build_generation_client(
        "custom", model="override-model", base_url=base_url
    )
    assert client.api_url == "https://override.test/api/v2/chat/completions"
    assert client.model == "override-model"


@pytest.mark.parametrize(
    "variable", ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"]
)
def test_missing_custom_configuration_does_not_use_deepseek(
    monkeypatch: pytest.MonkeyPatch, custom_environment: None, variable: str
) -> None:
    monkeypatch.setenv(variable, "   ")
    with pytest.raises(ValueError, match=variable):
        dataset_generation.build_generation_client("custom")


@pytest.mark.parametrize(
    "base_url",
    [
        "file:///tmp/model",
        "example.test/v1",
        "https://user:secret@example.test/v1",
        "https://example.test/v1?key=secret",
        "https://example.test/v1#fragment",
        "https://example.test:bad/v1",
        "https://example.test/invalid path",
    ],
)
def test_invalid_endpoint_is_rejected_without_echoing_it(
    custom_environment: None, base_url: str
) -> None:
    with pytest.raises(ValueError, match="HTTP") as error:
        dataset_generation.build_generation_client("custom", base_url=base_url)
    assert base_url not in str(error.value)


def test_custom_environment_does_not_change_default_provider(
    custom_environment: None,
) -> None:
    client = dataset_generation.build_generation_client("deepseek")
    assert client.api_key == "sk-unrelated"
    assert client.api_url == dataset_generation.DEEPSEEK_API_URL
    with pytest.raises(ValueError, match="仅适用于"):
        dataset_generation.build_generation_client(
            "deepseek", base_url="https://example.test/v1"
        )


@pytest.mark.parametrize("chat_type", ["private", "group"])
def test_generation_cli_passes_custom_client(
    monkeypatch: pytest.MonkeyPatch, custom_environment: None, chat_type: str
) -> None:
    def generate(*args, **kwargs):
        client = kwargs["client"]
        assert client.model == "cli-model"
        assert client.api_url == "https://cli.test/v1/chat/completions"
        assert client.api_key == "custom-test-key"
        return [{}]

    monkeypatch.setattr(dataset_generation, "generate_dataset", generate)
    assert (
        dataset_generation.dataset_cli(
            chat_type,
            lambda *args: {},
            [
                "--provider",
                "custom",
                "--model",
                "cli-model",
                "--base-url",
                "https://cli.test/v1",
                "-n",
                "1",
            ],
        )
        == 0
    )


@pytest.mark.parametrize("module", [private_export, group_export])
@pytest.mark.parametrize("label_mode", ["teacher", "schedule"])
def test_export_cli_uses_provider_only_for_teacher(
    monkeypatch: pytest.MonkeyPatch,
    custom_environment: None,
    module,
    label_mode: str,
) -> None:
    if label_mode == "schedule":
        monkeypatch.delenv("LLM_API_KEY")
        monkeypatch.delenv("DEEPSEEK_API_KEY")

    def export(*args, **kwargs):
        client = kwargs["client"]
        if label_mode == "teacher":
            assert client.api_key == "custom-test-key"
            assert client.model == "cli-model"
        else:
            assert client is None
        return 1, 0

    monkeypatch.setattr(module, "export_dataset", export)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export",
            "--label-mode",
            label_mode,
            "--provider",
            "custom",
            "--model",
            "cli-model",
        ],
    )
    assert module.main() == 0


def test_custom_model_listing_does_not_contact_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def list_models():
        pytest.fail("Custom model listing must not call the local proxy")

    monkeypatch.setattr(dataset_generation, "list_cliproxy_models", list_models)
    assert (
        dataset_generation.dataset_cli(
            "private",
            lambda *args: {},
            ["--provider", "custom", "--list-models"],
        )
        == 1
    )
