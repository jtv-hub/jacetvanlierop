import pytest

from crypto_trading_bot import config as bot_config
from crypto_trading_bot.config import CONFIG, ConfigurationError
from crypto_trading_bot.safety.confirmation import require_live_confirmation


@pytest.fixture
def live_mode_config():
    live_cfg = CONFIG.setdefault("live_mode", {})
    original = dict(live_cfg)
    try:
        yield live_cfg
    finally:
        live_cfg.clear()
        live_cfg.update(original)


def _reset_confirmation_state(live_cfg):
    live_cfg.pop("confirmation_acknowledged", None)
    live_cfg.pop("_force_logged", None)


def test_confirmation_not_required_in_paper_mode(monkeypatch, live_mode_config, tmp_path):
    confirmation_path = tmp_path / "confirm_live"
    live_mode_config["confirmation_file"] = str(confirmation_path)
    live_mode_config["force_override"] = False
    _reset_confirmation_state(live_mode_config)

    monkeypatch.setattr(bot_config, "is_live", False, raising=False)

    assert require_live_confirmation()


def test_live_mode_allows_when_file_present(monkeypatch, live_mode_config, tmp_path):
    confirmation_path = tmp_path / "confirm_live"
    confirmation_path.write_text("ok", encoding="utf-8")
    live_mode_config["confirmation_file"] = str(confirmation_path)
    live_mode_config["force_override"] = False
    _reset_confirmation_state(live_mode_config)

    monkeypatch.setattr(bot_config, "is_live", True, raising=False)

    assert require_live_confirmation()


def test_live_mode_blocks_without_confirmation(monkeypatch, live_mode_config, tmp_path):
    confirmation_path = tmp_path / "confirm_live"
    live_mode_config["confirmation_file"] = str(confirmation_path)
    live_mode_config["force_override"] = False
    _reset_confirmation_state(live_mode_config)

    monkeypatch.setattr(bot_config, "is_live", True, raising=False)

    with pytest.raises(ConfigurationError):
        require_live_confirmation()


def test_live_force_override_allows_without_file(monkeypatch, live_mode_config, tmp_path):
    confirmation_path = tmp_path / "confirm_live"
    live_mode_config["confirmation_file"] = str(confirmation_path)
    live_mode_config["force_override"] = True
    _reset_confirmation_state(live_mode_config)

    monkeypatch.setattr(bot_config, "is_live", True, raising=False)

    assert require_live_confirmation()
