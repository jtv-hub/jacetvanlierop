# PPO Agent Guide

## Config
- `PPO_ENABLED`: Enable PPO
- `PPO_MIN_CONFIDENCE`: 0.6
- `PPO_CONFIDENCE_TO_BUFFER`: {0.6:0.03, 0.7:0.06, 0.8:0.12, 0.9:0.18}
- `PPO_MIN_SHADOW_TRADES`: 75
- `PPO_SHADOW_MIN_WINRATE`: 0.50
- `FORCE_PPO_LIVE`: Admin override for execution

## Training
```bash
PYTHONPATH=src venv/bin/python scripts/train_ppo.py --timesteps 50000
```

## Inference
```bash
PYTHONPATH=src venv/bin/python scripts/test_ppo_inference.py
```

## Shadow Mode
- PPO logs every suggestion
- Executes only after:
  - 75+ shadow trades
  - 50%+ win rate
  - Paper mode or `FORCE_PPO_LIVE` enabled

### PPO Config Keys
- `ppo_n_steps`: int (default 2048)
- `ppo_batch_size`: int (default 64)
- `ppo_learning_rate`: float (default 3e-4)
- `ppo_n_epochs`: int (default 10)
- `ppo_train_timesteps`: int (default 50000)
