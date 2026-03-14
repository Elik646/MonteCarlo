# RetailOptions – Monte Carlo Option Pricing & Trading Platform

A full-stack Python/Flask web application built on a **Monte Carlo option pricing engine**.
It simulates thousands of stock price paths using **Geometric Brownian Motion (GBM)** and
exposes a dark-themed interactive UI for pricing, scenario analysis, strategy comparison,
Greeks, probability metrics, live market data, paper-trade demo, and an AI auto-trader.

---

## Features

| Feature | Detail |
|---|---|
| **Option Pricer** | MC + Black-Scholes pricing with confidence intervals and path visualisation |
| **Scenario Analysis** | Single-option payoff / P&L profile chart at expiry |
| **Strategy Comparison** | 8 strategies plotted side-by-side with summary table |
| **Greeks & Sensitivities** | Delta, Gamma, Theta, Vega, Rho curves; theta-decay visualisation |
| **Probability Lab** | MC probability of profit, VaR, CVaR, and P&L histogram |
| **Demo Trading** | Paper-trade with live quotes (yfinance); real-time P&L tracking |
| **AI Auto-Trader** | Q-learning agent that observes market state and selects strategies |
| **Live Market Data** | Real-time stock quotes, historical volatility, and candlestick charts |
| **CLI / Library mode** | `main.py` demo script and importable `MonteCarloOptionPricer` class |

---

## Project structure

```
MonteCarlo/
├── app.py              # Flask web application – API routes and server entry point
├── monte_carlo.py      # Core engine – GBM simulation, MC pricer, Black-Scholes, Greeks
├── strategies.py       # 8 option strategy payoff profiles (long call, straddle, spreads …)
├── market_data.py      # Live quotes via yfinance + in-memory paper-trade portfolio
├── ai_trader.py        # Q-learning AI trading engine (epsilon-greedy, Bellman update)
├── main.py             # Standalone CLI demo (no web server required)
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Bootstrap 5 + Plotly.js dark-themed single-page UI
└── tests/
    ├── test_monte_carlo.py  # Unit tests for the pricing engine
    └── test_web_app.py      # Integration tests for the Flask API
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the web application

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

For development with auto-reload:

```bash
FLASK_DEBUG=1 python app.py
```

### 3. Run the CLI demo (no web server)

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

### 4. Use the pricer as a Python library

```python
from monte_carlo import MonteCarloOptionPricer

pricer = MonteCarloOptionPricer(
    spot=100.0,
    strike=105.0,
    risk_free_rate=0.05,   # 5 %
    volatility=0.20,        # 20 %
    time_to_expiry=1.0,     # 1 year
    num_paths=10_000,       # 1 000 – 20 000
    num_steps=252,          # daily steps (≈ trading days per year)
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

## Web application – tab guide

Open the app at **http://localhost:5000** and explore the six tabs:

### Option Pricer
Enter spot price, strike, risk-free rate, volatility, and time to expiry.
Optionally type a ticker symbol (e.g. `AAPL`) to auto-fill live market data.
Click **Run Simulation** to see:
- Monte Carlo price, Black-Scholes price, standard error, and 95 % confidence interval for both call and put
- Interactive chart of simulated GBM price paths
- Terminal price distribution histogram

### Scenario Analysis
Analyse a single option's payoff and profit/loss across a range of stock prices at expiry.
Choose call or put, set the parameters, and click **Analyse** to render the P&L profile with breakeven annotation.

### Strategy Comparison
Compare up to eight strategies on the same chart:

| Strategy | Description |
|---|---|
| Long Call | Buy a call option |
| Long Put | Buy a put option |
| Bull Call Spread | Long call at lower strike + short call at higher strike |
| Bear Put Spread | Long put at higher strike + short put at lower strike |
| Long Straddle | Long call + long put at the same strike |
| Long Strangle | Long OTM call + long OTM put at different strikes |
| Covered Call | Long stock + short call |
| Protective Put | Long stock + long put |

Select strategies, enter market parameters, and click **Compare Strategies** to render combined P&L curves and a summary table (premium, max profit, max loss, breakevens).

### Greeks & Sensitivities
Compute the five Black-Scholes Greeks at any set of parameters:

| Greek | Meaning |
|---|---|
| Delta (Δ) | Change in option value per $1 move in the underlying |
| Gamma (Γ) | Rate of change of Delta |
| Theta (Θ) | Daily time-value decay |
| Vega (ν) | Sensitivity to a 1 % change in implied volatility |
| Rho (ρ) | Sensitivity to a 1 % change in the risk-free rate |

The tab also renders Delta/Gamma vs spot, Theta decay over time, and Vega vs spot charts.

### Probability Lab
Run a Monte Carlo simulation to measure the risk profile of any strategy:
- **Probability of profit** – fraction of simulated paths where P&L > 0
- **Expected P&L** and **Median P&L**
- **Value at Risk (VaR)** at 90 % and 95 % confidence levels
- **Conditional VaR (CVaR / Expected Shortfall)** – average loss in the worst 5 % of outcomes
- P&L histogram with breakeven and VaR annotations

### Demo Trading
Paper-trade real stocks with live market prices:
1. Enter a ticker symbol and click **Fetch Quote** to load the current price and historical volatility.
2. Choose a strategy, set the expiry and quantity, and click **Open Trade** to enter a position.
3. The portfolio table shows all open positions with live unrealised P&L updated from the market.
4. Click **Close** on any position to exit at the current market price.

A candlestick chart of the stock's recent price history is displayed alongside the ticker quote.

#### AI Auto-Trader
The Demo Trading tab includes a Q-learning AI that automates trade decisions:
- The AI observes a discrete market state derived from price trend, volatility level, and a Monte Carlo expected-return signal.
- It selects an action (a strategy or "no trade") via an **epsilon-greedy policy** and opens a paper position automatically.
- After closing, the reward (scaled P&L) is fed back to update the Q-table via a **Bellman one-step TD update**.
- The **Run AI Step** button triggers one decision cycle; **Reset AI Learning** reinitialises the Q-table and counters.

---

## API reference

All endpoints are served by the Flask app (`app.py`).

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the web UI |
| `POST` | `/api/price` | Monte Carlo + Black-Scholes pricing |
| `POST` | `/api/scenario` | Single-option payoff / P&L profile |
| `POST` | `/api/strategies` | Multi-strategy P&L comparison |
| `POST` | `/api/greeks` | Black-Scholes Greeks and sensitivity curves |
| `POST` | `/api/probability` | MC probability of profit and risk metrics |
| `GET` | `/api/quote` | Live stock quote (`?ticker=AAPL`) |
| `GET` | `/api/stock_chart` | OHLCV candlestick data (`?ticker=AAPL&period=6mo`) |
| `POST` | `/api/demo/trade` | Open a paper-trade position |
| `GET` | `/api/demo/portfolio` | List all demo positions with live P&L |
| `DELETE` | `/api/demo/trade/<id>` | Close a demo position |
| `GET` | `/api/ai/status` | AI Q-table snapshot and performance stats |
| `POST` | `/api/ai/trade` | Run one AI trading step for a ticker |
| `POST` | `/api/ai/close_trade/<id>` | Close an AI trade and update the Q-table |

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

### Q-learning AI

The AI maintains a Q-table over 27 discrete market states (3 price trends × 3 volatility levels × 3 MC signals) and 6 actions. Q-values are updated using the Bellman equation:

```
Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

Exploration decays over time via epsilon decay (`ε × 0.97` after each episode, floored at 0.05).

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Parameters reference

| Parameter | Type | Description |
|---|---|---|
| `spot` | float | Current stock price S₀ (must be > 0) |
| `strike` | float | Option strike price K (must be > 0) |
| `risk_free_rate` | float | Annualised risk-free rate r (e.g. `0.05` = 5 %) |
| `volatility` | float | Annualised volatility σ (e.g. `0.20` = 20 %) |
| `time_to_expiry` | float | Time to expiry in years T |
| `num_paths` | int | Number of simulated paths (1 000 – 20 000) |
| `num_steps` | int | Number of time steps per path (default 252 ≈ trading days/yr) |
| `random_seed` | int \| None | Optional seed for reproducible results |
