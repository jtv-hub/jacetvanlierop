def pytest_sessionstart(session):
    import pkgutil

    import crypto_trading_bot

    pkg_file = getattr(crypto_trading_bot, "__file__", "<no file>")
    pkg_path = list(getattr(crypto_trading_bot, "__path__", []))
    kids = [m.name for m in pkgutil.iter_modules(crypto_trading_bot.__path__)]
    print("\n[pytest diag] crypto_trading_bot.__file__ =", pkg_file)
    print("[pytest diag] crypto_trading_bot.__path__ =", pkg_path)
    print("[pytest diag] children under package    =", kids)
