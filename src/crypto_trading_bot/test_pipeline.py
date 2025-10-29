"""Import smoke tests for pipeline modules."""

import importlib.util
import sys
from importlib import import_module
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent

if str(SRC_ROOT) in sys.path:
    sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
    sys.path.append(str(PROJECT_ROOT))


@pytest.mark.smoke
def test_nightly_pipeline_importable():
    """Ensure the nightly pipeline script can be imported via the package."""
    base_spec = importlib.util.find_spec("crypto_trading_bot")
    scripts_spec = importlib.util.find_spec("crypto_trading_bot.scripts")
    spec = importlib.util.find_spec("crypto_trading_bot.scripts.nightly_pipeline")
    assert base_spec is not None, f"crypto_trading_bot missing; sys.path={sys.path}"
    assert scripts_spec is not None, f"crypto_trading_bot.scripts missing; sys.path={sys.path}"
    if spec is None:
        import pkgutil

        discovered = sorted(name for _, name, _ in pkgutil.iter_modules(scripts_spec.submodule_search_locations or []))
        pytest.fail(
            f"crypto_trading_bot.scripts.nightly_pipeline missing; discovered={discovered}; sys.path={sys.path}"
        )
    module = import_module("crypto_trading_bot.scripts.nightly_pipeline")
    assert hasattr(module, "run_nightly_pipeline")


@pytest.mark.skip(reason="Nightly pipeline requires external services; import check is sufficient.")
def test_nightly_pipeline_execution_placeholder():
    """Placeholder to document execution expectations without invoking external side effects."""
    module = import_module("crypto_trading_bot.scripts.nightly_pipeline")
    module.run_nightly_pipeline()
