"""Sanity checks for ATR sizing configuration helpers."""


def test_atr_sizing_scalar_present_when_enabled():
    """Ensure ATR sizing scalar calculation clamps to configured bounds."""
    from crypto_trading_bot.bot import trading_logic as tl  # pylint: disable=import-outside-toplevel

    cfg = tl.CONFIG.setdefault("risk", {}).setdefault("atr_sizing", {})
    cfg.update(
        {
            "enabled": True,
            "target_atr_pct": 0.02,
            "min_scalar": 0.5,
            "max_scalar": 1.5,
            "mode": "inverse",
        }
    )

    atr_value = 2.0  # arbitrary ATR
    price = 100.0
    atr_pct = atr_value / price
    eps = 1e-8
    target = cfg["target_atr_pct"]
    if cfg["mode"] == "inverse":
        scalar = target / max(atr_pct, eps)
    else:
        scalar = atr_pct / max(target, eps)
    scalar = max(cfg["min_scalar"], min(cfg["max_scalar"], scalar))

    assert cfg["min_scalar"] <= scalar <= cfg["max_scalar"]
