# jacetvanlierop

Crypto trading bot (paper/live-ready).  
Status: ✅ tests green, ✅ lint clean, ✅ pre-commit hooks installed.

## Quick Start
```bash
python -m venv venv && source venv/bin/activate
pip install -U pip
pip install -e .  # or: pip install -r requirements.txt
pytest -q
pre-commit install && pre-commit run -a
```
