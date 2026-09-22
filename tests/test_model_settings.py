import argparse
import os

import pytest

import dataset_generation


def test_env_file_selects_provider_and_builds_client_from_another_directory(
    monkeypatch,
    tmp_path,
):
    dataset_generation.ENV_FILE.write_text(
        'LLM_PROVIDER=custom\nLLM_API_KEY="test-${LITERAL}-key"\n'
        'LLM_BASE_URL=https://example.test/v1\nLLM_MODEL="test-model"\n',
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text("LLM_MODEL=wrong-directory\n")
    monkeypatch.chdir(elsewhere)
    parser = argparse.ArgumentParser()
    dataset_generation.add_model_arguments(parser)
    args = parser.parse_args([])
    client = dataset_generation.build_generation_client(args.provider)
    assert args.provider == "custom"
    assert client.api_key == "test-${LITERAL}-key"
    assert client.api_url == "https://example.test/v1/chat/completions"
    assert client.model == "test-model"
    assert "LLM_API_KEY" not in os.environ


def test_explicit_options_then_environment_then_file(monkeypatch):
    dataset_generation.ENV_FILE.write_text(
        "LLM_PROVIDER=custom\nLLM_MODEL=file-model\n"
        "LLM_API_KEY=file-key\nLLM_BASE_URL=https://file.test/v1\n"
        "DEEPSEEK_API_KEY=sk-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    monkeypatch.setenv("LLM_API_KEY", "environment-key")
    client = dataset_generation.build_generation_client("custom")
    assert client.model == "environment-model"
    assert client.api_key == "environment-key"
    explicit = dataset_generation.build_generation_client(
        "custom", model="cli-model", base_url="https://cli.test/v1"
    )
    assert explicit.model == "cli-model"
    assert explicit.api_url == "https://cli.test/v1/chat/completions"
    parser = argparse.ArgumentParser()
    dataset_generation.add_model_arguments(parser)
    assert parser.parse_args(["--provider", "deepseek"]).provider == "deepseek"


def test_default_is_deepseek_without_file():
    parser = argparse.ArgumentParser()
    dataset_generation.add_model_arguments(parser)
    assert parser.parse_args([]).provider == "deepseek"


@pytest.mark.parametrize("provider", ["deepseek", "cliproxy"])
def test_existing_providers_can_read_credentials_and_model_from_file(provider):
    prefix = "DEEPSEEK" if provider == "deepseek" else "CLIPROXY"
    dataset_generation.ENV_FILE.write_text(
        f"{prefix}_API_KEY=sk-file-key\n{prefix}_MODEL=file-model\n"
    )
    client = dataset_generation.build_generation_client(provider)
    assert client.api_key == "sk-file-key"
    assert client.model == "file-model"


def test_file_edits_are_read_again_and_empty_environment_does_not_fall_back(
    monkeypatch,
):
    for key in ("sk-first", "sk-second"):
        dataset_generation.ENV_FILE.write_text(f"DEEPSEEK_API_KEY={key}\n")
        assert dataset_generation.ChatCompletionClient().api_key == key
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        dataset_generation.ChatCompletionClient()
