# Monte Carlo Option Pricing Simulator

A Python implementation of a **Monte Carlo option pricing simulator** that simulates 1,000+ stock price paths using **Geometric Brownian Motion (GBM)** to price European call and put options via probability.

---

## Features

| Feature | Detail |
|---|---|
| Stock path simulation | Geometric Brownian Motion with configurable paths & steps |
| Option types | European **call** and **put** |
| Pricing output | MC price · standard error · 95 % confidence interval · P(in-the-money) |
| Graphical output | Dark-theme plot of simulated price paths with strike/parameter overlay |
| Analytical validation | Black-Scholes closed-form price computed alongside every MC estimate |
| Reproducibility | Optional `random_seed` for deterministic results |

---

## Project structure

```
MonteCarlo/
├── monte_carlo.py          # Core simulator – GBM paths + MC pricer + Black-Scholes
├── main.py                 # Runnable demo with example output
├── requirements.txt        # Python dependencies (numpy, matplotlib)
└── tests/
    └── test_monte_carlo.py # 33 unit tests (pytest)
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the demo

```bash
python main.py
```

Example output:

```
==============================================================
  Monte Carlo Option Pricing Simulator
==============================================================
  Parameters
  ----------
  Spot price          : $100.00
  Strike price        : $105.00
  Risk-free rate      :   5.00 %
  Volatility          :  20.00 %
  Time to expiry      :   1.00 years
  Simulated paths     : 10,000
  Time steps / path   : 252
  ----------------------------------------------------------

  CALL option
  MC price            : $    8.02   (± 0.09 std-err)
  95 % CI             : [$   7.85 , $   8.19 ]
  Black-Scholes price : $    8.02
  P(in-the-money)     :  45.23 %
  ----------------------------------------------------------

  PUT option
  MC price            : $    7.90   (± 0.06 std-err)
  95 % CI             : [$   7.79 , $   8.02 ]
  Black-Scholes price : $    7.90
  P(in-the-money)     :  54.77 %
  ----------------------------------------------------------
```

### 3. Use as a library

```python
from monte_carlo import MonteCarloOptionPricer

pricer = MonteCarloOptionPricer(
    spot=100.0,
    strike=105.0,
    risk_free_rate=0.05,   # 5 %
    volatility=0.20,        # 20 %
    time_to_expiry=1.0,     # 1 year
    num_paths=10_000,       # ≥ 1 000 required
    num_steps=252,          # daily steps
    random_seed=42,         # optional – for reproducibility
)

call_result = pricer.price("call")
put_result  = pricer.price("put")

print(call_result)
# {
#   'price': 8.02,
#   'std_error': 0.09,
#   'confidence_interval_95': (7.85, 8.19),
#   'black_scholes_price': 8.02,
#   'num_paths': 10000,
#   'option_type': 'call',
#   'probability_in_the_money': 0.4523,
# }
```

---

## How it works

### Geometric Brownian Motion

Stock prices follow GBM under the risk-neutral measure:

```
S(t + dt) = S(t) · exp( (r − σ²/2)·dt + σ·√dt·Z )
```

where `Z ~ N(0,1)`, `r` is the risk-free rate, and `σ` is the volatility.

### Monte Carlo pricing

For each simulated path the terminal payoff is computed:

```
Call payoff = max(S_T − K, 0)
Put  payoff = max(K − S_T, 0)
```

The option price is the **discounted expected payoff** over all paths:

```
Price = exp(−r·T) · mean(payoffs)
```

### Validation against Black-Scholes

The analytical Black-Scholes price is computed alongside every Monte Carlo estimate so you can immediately see how closely the simulation converges.

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

All **33 tests** should pass.

---

## Parameters reference

| Parameter | Type | Description |
|---|---|---|
| `spot` | float | Current stock price S₀ (must be > 0) |
| `strike` | float | Option strike price K (must be > 0) |
| `risk_free_rate` | float | Annualised risk-free rate r (e.g. `0.05` = 5 %) |
| `volatility` | float | Annualised volatility σ (e.g. `0.20` = 20 %) |
| `time_to_expiry` | float | Time to expiry in years T |
| `num_paths` | int | Number of simulated paths (minimum **1,000**) |
| `num_steps` | int | Number of time steps per path (default 252 ≈ trading days/yr) |
| `random_seed` | int \| None | Optional seed for reproducible results |
