# Stock-trading-simulation-Gymnasium
# PPO Stock Trading Agent (Reinforcement Learning)

A reinforcement learning project that trains a **Proximal Policy Optimization (PPO)** agent to trade a single stock using [`gym-anytrading`](https://github.com/AminHP/gym-anytrading) and [`stable-baselines3`](https://stable-baselines3.readthedocs.io/), then evaluates the resulting strategy with buy/sell visualizations, a balance chart, and a full [QuantStats](https://github.com/ranaroussi/quantstats) tearsheet.

## Overview

The agent observes a rolling window of daily closing prices and chooses, at each step, to go **long (Buy)** or **flat/exit (Sell)**. The goal of the project is less "build a profitable trading bot" and more to explore — hands-on — how a standard RL algorithm behaves on real financial time series, and why naive setups like this one are hard to trust.

**Pipeline:**
1. Load historical daily price data (AAPL, 2009–2024).
2. Build a `stocks-v0` Gymnasium environment (`gym-anytrading`) with a 10-day lookback window.
3. Train a PPO agent (`stable-baselines3`) for 10,000 timesteps.
4. Run the trained policy through the full price history, tracking a simulated cash balance.
5. Visualize buy/sell actions against price, and balance over time.
6. Generate a QuantStats performance tearsheet (CAGR, Sharpe, Sortino, drawdown, etc.) from the strategy's returns.

## Data

- `AAPL.csv` — daily `Date, Close` prices for Apple Inc., 2009–2024.
- Loaded with `pd.read_csv(..., parse_dates=True, index_col='Date')`.

## Requirements

```bash
pip install gymnasium gym-anytrading stable-baselines3 quantstats
```

## Usage

```python
import pandas as pd
import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import Actions
from stable_baselines3 import PPO
import quantstats as qs

data = pd.read_csv("AAPL.csv", parse_dates=True, index_col='Date')

window_size = 10
env = gym.make('stocks-v0', df=data, window_size=window_size,
                frame_bound=(window_size, len(data)))

model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=10000)
```

See `notebook.ipynb` / `ppo_trading.py` for the full evaluation loop, charting code, and QuantStats report generation.

## Results

| Metric | Value |
|---|---|
| Starting balance | $100,000 |
| Final balance | $105,672 |
| Return | +5.7% |
| Buy actions | 137 |
| Sell actions | 3,626 |
| Underlying AAPL move (2009–2024) | ~40x |

**Chart 1 — Price with Buy/Sell markers:** the model's trades cluster around a handful of price levels rather than tracking the broader trend consistently.

**Chart 2 — Balance over time:** balance grows in discrete jumps, with most of the gain concentrated in a single stretch, and gives some of it back afterward.

**QuantStats tearsheet:** run on the strategy's daily returns; reports full risk-adjusted metrics (Sharpe, Sortino, max drawdown, CAGR, EOY returns) for direct comparison against a buy-and-hold benchmark.

## Key Findings

- The agent stayed flat/out of the market the large majority of the time (roughly 96% Sell vs. 4% Buy actions), which limited both its losses and its gains.
- The absolute return (+5.7%) is dwarfed by AAPL's actual ~40x appreciation over the same period — the model captured only a small fraction of the available upside.
- Results are highly sensitive to training run and random seed: a separate run of essentially the same setup produced a strategy with a **-99% cumulative return**, illustrating how unstable PPO's learned policy can be at only 10,000 timesteps with minimal (Close-price-only) features.
- Daily stock returns are close to a random walk, so a short lookback window and a small training budget give the agent very little exploitable signal.

## Limitations & Next Steps

- **Training budget:** 10,000 timesteps is very small for PPO on a financial time series; results vary widely across runs.
- **Features:** only closing price is used — no volume, volatility, or technical indicators.
- **Single asset, single period:** testing on one stock over one (strongly trending) historical window risks overfitting conclusions to that specific price path.
- Future work: longer training runs, richer observation features, multiple assets/time periods for out-of-sample validation, and a benchmark comparison against buy-and-hold.

## License

MIT .
