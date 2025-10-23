"""Tests for balance tolerance logic in reconciliation audit."""

import pytest

from scripts import audit_kraken_reconciliation as reconciliation


def test_balance_difference_within_tolerance():
    diff, within = reconciliation.balance_difference_within_tolerance(
        {"USDC": 1000.0},
        {"available_capital": 1000.005},
        tolerance=0.01,
    )
    assert diff == pytest.approx(-0.005)
    assert within is True


def test_balance_difference_exceeds_tolerance():
    diff, within = reconciliation.balance_difference_within_tolerance(
        {"USDC": 1000.0},
        {"available_capital": 999.0},
        tolerance=0.5,
    )
    assert diff == 1.0
    assert within is False
