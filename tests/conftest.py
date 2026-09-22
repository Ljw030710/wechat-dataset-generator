import pytest

import dataset_generation


@pytest.fixture(autouse=True)
def isolate_model_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_generation, "ENV_FILE", tmp_path / ".env")
    for name in (
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_MODEL",
        "CLIPROXY_API_KEY",
        "CLIPROXY_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
