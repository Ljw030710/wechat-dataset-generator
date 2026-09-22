import pytest

import dataset_generation


def test_environment_key_is_read_when_client_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "  sk-first  ")
    first = dataset_generation.build_generation_client("deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-second")
    second = dataset_generation.ChatCompletionClient()

    assert first.api_key == "sk-first"
    assert second.api_key == "sk-second"


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_environment_key_fails_before_request(
    monkeypatch: pytest.MonkeyPatch, key: str | None
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    if key is not None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", key)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY 环境变量"):
        dataset_generation.ChatCompletionClient()


def test_explicit_key_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-environment")

    client = dataset_generation.ChatCompletionClient(api_key=" sk-explicit ")

    assert client.api_key == "sk-explicit"


def test_empty_explicit_key_does_not_fall_back_to_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-environment")

    with pytest.raises(ValueError):
        dataset_generation.ChatCompletionClient(api_key="")


def test_invalid_key_is_not_in_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "invalid-test-key"
    monkeypatch.setenv("DEEPSEEK_API_KEY", key)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY 格式无效") as error:
        dataset_generation.ChatCompletionClient()

    assert key not in str(error.value)


def test_proxy_uses_its_own_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("CLIPROXY_API_KEY", "proxy-test-key")

    client = dataset_generation.build_generation_client("cliproxy")

    assert client.api_key == "proxy-test-key"
    assert client.api_url == dataset_generation.CLIPROXY_API_URL


def test_other_provider_does_not_inherit_deepseek_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    with pytest.raises(ValueError, match="CLIProxyAPI API Key"):
        dataset_generation.ChatCompletionClient(provider_name="CLIProxyAPI")
