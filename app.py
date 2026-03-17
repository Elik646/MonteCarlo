"""
RetailOptions – Flask Web Application
======================================
Exposes the Monte Carlo option pricing engine and strategy analysis
as a JSON API consumed by the single-page web UI (templates/index.html).

Endpoints
---------
GET  /                       Serve the web UI.
POST /api/price              MC + BS pricing with probability metrics.
POST /api/scenario           Single-option payoff / P&L profile.
POST /api/strategies         Multi-strategy comparison (payoff profiles).
POST /api/greeks             Black-Scholes Greeks and sensitivity curves.
POST /api/probability        MC probability / risk metrics for a strategy.
GET  /api/quote              Real-time stock quote (ticker query param).
POST /api/demo/trade         Open a paper-trade position.
GET  /api/demo/portfolio     List all demo positions with live P&L.
DELETE /api/demo/trade/<id>  Close a demo position.

Run
---
    python app.py        (dev server on http://localhost:5000)
"""

from __future__ import annotations

import math

import numpy as np
from flask import Flask, jsonify, render_template, request

from monte_carlo import (
    MonteCarloOptionPricer,
    black_scholes_greeks,
    black_scholes_price,
    simulate_stock_paths,
)
from strategies import compute_strategy_profile
import market_data
import ai_trader

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Spot price range constants for P&L profile charts
# ---------------------------------------------------------------------------

#: Lower bound as a fraction of current spot (e.g. 0.30 → 30 % of spot)
SPOT_RANGE_MIN_FACTOR: float = 0.30
#: Absolute minimum spot price to avoid zero/negative values
SPOT_MIN_ABSOLUTE: float = 0.01
#: Upper bound as a fraction of current spot (e.g. 2.00 → 200 % of spot)
SPOT_RANGE_MAX_FACTOR: float = 2.00


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_float(data: dict, key: str) -> float:
    if key not in data:
        raise KeyError(key)
    return float(data[key])


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"'{name}' must be > 0 (got {value})")


def _validate_range(value: float, lo: float, hi: float, name: str) -> None:
    if not (lo <= value <= hi):
        raise ValueError(f"'{name}' must be between {lo} and {hi} (got {value})")


def _find_breakevens(spot_prices: np.ndarray, pnl: np.ndarray) -> list[float]:
    """Linear-interpolation zero-crossings."""
    result: list[float] = []
    for i in range(len(pnl) - 1):
        y0, y1 = float(pnl[i]), float(pnl[i + 1])
        if y0 * y1 <= 0 and y0 != y1:
            be = float(spot_prices[i]) - y0 * (
                float(spot_prices[i + 1]) - float(spot_prices[i])
            ) / (y1 - y0)
            result.append(round(be, 2))
    return result


def _err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


def _compute_strategy_tpsl(
    action: str,
    premium: float,
    max_profit: float | None,
    max_loss: float | None,
) -> tuple[float, float]:
    """
    Return (take_profit, stop_loss) P&L dollar targets calibrated to a
    strategy's actual max profit / max loss characteristics rather than a
    flat percentage of account balance.

    Premium convention: positive = debit paid; negative = credit received.
    Returns: take_profit (positive dollar gain), stop_loss (negative dollar loss)
    """
    abs_prem = abs(premium)
    abs_max_loss = abs(max_loss) if max_loss is not None else abs_prem

    if action in ("long_call", "long_put", "long_straddle", "long_strangle"):
        # Debit strategies: max loss = premium; TP at 2× premium, SL at 50% loss
        tp = round(abs_prem * 2.0, 2)
        sl = round(-abs_prem * 0.5, 2)

    elif action in ("bull_call_spread", "bear_put_spread"):
        # Debit spreads: capped profit; target 75% of max profit, stop at 70% of debit
        mp = max_profit if (max_profit is not None and max_profit > 0) else abs_prem * 1.5
        tp = round(mp * 0.75, 2)
        sl = round(-abs_prem * 0.7, 2)

    elif action in ("covered_call", "protective_put"):
        tp = round(abs_prem * 1.5, 2)
        sl = round(-abs_prem * 1.5, 2)

    elif action in ("short_call", "short_put"):
        # Credit: max profit = premium received; stop at 2× premium loss
        tp = round(abs_prem * 0.75, 2)
        sl = round(-abs_prem * 2.0, 2)

    elif action in ("bull_put_spread", "bear_call_spread"):
        # Credit spreads: TP at 75% of credit collected; SL at 50% of max loss
        tp = round(abs_prem * 0.75, 2)
        sl = round(-abs_max_loss * 0.5, 2)

    elif action == "iron_condor":
        # TP at 60% of net credit; SL at 50% of max loss per side
        tp = round(abs_prem * 0.6, 2)
        sl = round(-abs_max_loss * 0.5, 2)

    else:
        tp = round(abs_prem * 2.0, 2)
        sl = round(-abs_prem, 2)

    # Guard: ensure the stop-loss is at least -$1 (caps upward towards zero,
    # preventing trivially small risk on near-zero premiums) and TP is at
    # least +$1.
    sl = min(sl, -1.0)   # min keeps the more-negative value; ensures |SL| >= $1
    tp = max(tp, 1.0)
    return tp, sl


def _generate_ai_reasoning(
    action: str,
    state: list,
    quant: dict,
    risk: dict,
    mc_return: float,
    intraday_pct: float,
    ticker: str,
) -> str:
    """
    Generate a natural language explanation of the AI's trading decision.
    This is a deterministic, template-based reasoning engine that uses all
    available market signals to produce a coherent, paragraph-form analysis.
    """
    parts: list[str] = []

    price_trend = state[0] if len(state) > 0 else "neutral"
    vol_regime = risk.get("vol_regime", "moderate")
    rsi = float(quant.get("rsi", 50))
    above_sma20 = quant.get("above_sma20", True)
    above_sma50 = quant.get("above_sma50", True)
    mc_pct = mc_return * 100

    # Intraday trend
    if price_trend == "up":
        parts.append(
            f"{ticker} is exhibiting upward intraday momentum ({intraday_pct:+.2f}%), "
            "suggesting short-term bullish pressure and positive market sentiment."
        )
    elif price_trend == "down":
        parts.append(
            f"{ticker} is trending lower intraday ({intraday_pct:+.2f}%), "
            "indicating bearish short-term pressure and potential continuation risk."
        )
    else:
        parts.append(
            f"{ticker} shows neutral intraday movement ({intraday_pct:+.2f}%), "
            "with no dominant directional bias at this time."
        )

    # Volatility regime
    if vol_regime == "high":
        parts.append(
            "Historical volatility is elevated, making option premiums expensive. "
            "In high-IV environments, selling premium (credit strategies) or using "
            "spreads to reduce net debit is more capital-efficient."
        )
    elif vol_regime == "low":
        parts.append(
            "Volatility is subdued, which makes long options relatively cheap. "
            "This is a favourable environment for buying premium with defined risk, "
            "as IV expansion could amplify gains."
        )
    else:
        parts.append(
            "Volatility is at a moderate level, which supports a balanced approach. "
            "Both debit and credit strategies are viable depending on the directional view."
        )

    # Monte Carlo expected return
    if mc_pct > 3:
        parts.append(
            f"The Monte Carlo simulation (2,000 paths) projects a positive expected "
            f"return of {mc_pct:.2f}% by expiry, providing quantitative support for "
            "a bullish directional bias."
        )
    elif mc_pct < -3:
        parts.append(
            f"Monte Carlo simulation projects a negative expected return of {mc_pct:.2f}%, "
            "which supports a bearish or hedged stance."
        )
    else:
        parts.append(
            f"Monte Carlo projects a near-flat expected return ({mc_pct:.2f}%), "
            "suggesting range-bound price action is the most likely scenario."
        )

    # RSI
    if rsi >= 70:
        parts.append(
            f"RSI at {rsi:.1f} signals overbought conditions — buying pressure may "
            "be exhausted and a pullback or consolidation is possible."
        )
    elif rsi <= 30:
        parts.append(
            f"RSI at {rsi:.1f} is deep in oversold territory — the stock may be "
            "due for a mean-reversion bounce or stabilisation."
        )
    elif rsi > 55:
        parts.append(
            f"RSI at {rsi:.1f} leans bullish, consistent with constructive price momentum."
        )
    elif rsi < 45:
        parts.append(
            f"RSI at {rsi:.1f} leans bearish, consistent with softening price action."
        )
    else:
        parts.append(
            f"RSI at {rsi:.1f} is neutral, neither overbought nor oversold — "
            "no strong mean-reversion signal from this indicator."
        )

    # SMA trend
    if above_sma50 and above_sma20:
        parts.append(
            "Price is trading above both the 20-day and 50-day moving averages, "
            "confirming a healthy intermediate-term uptrend."
        )
    elif above_sma50 and not above_sma20:
        parts.append(
            "Price is above the 50-day SMA but below the 20-day SMA, suggesting "
            "some short-term weakness within a broader uptrend — a mixed signal."
        )
    elif not above_sma50 and above_sma20:
        parts.append(
            "Price is above the 20-day but below the 50-day SMA. "
            "Short-term momentum exists but the medium-term trend is still bearish."
        )
    else:
        parts.append(
            "Price is below both the 20-day and 50-day moving averages, "
            "confirming a downtrend across multiple timeframes."
        )

    # Strategy decision
    explanations = {
        "no_trade": (
            "Taking all signals together — including risk constraints and position "
            "concentration — the AI elected to stay flat and preserve capital. "
            "No clear edge justifies opening a new position at this time."
        ),
        "long_call": (
            "The combination of bullish momentum, positive MC outlook, and supportive "
            "RSI/SMA readings favours a Long Call. This captures upside participation "
            "with risk strictly limited to the premium paid."
        ),
        "long_put": (
            "The bearish signals across multiple dimensions — downward price trend, "
            "negative MC projection, and bearish technical indicators — favour a Long Put. "
            "Risk is limited to the premium paid while downside capture potential is significant."
        ),
        "bull_call_spread": (
            "A moderately bullish view combined with elevated volatility (expensive premiums) "
            "favours a Bull Call Spread over a naked long call. The spread reduces the net "
            "debit while capping profit at the upper strike — a more capital-efficient structure."
        ),
        "bear_put_spread": (
            "A moderately bearish outlook with elevated IV favours a Bear Put Spread. "
            "Selling the lower-strike put offsets the cost of the long put, reducing the "
            "net debit while retaining meaningful downside exposure."
        ),
        "long_straddle": (
            "Multiple conflicting signals point to high uncertainty with potential for a "
            "sharp move in either direction. A Long Straddle profits from volatility itself, "
            "regardless of direction — ideal when a big move is expected but direction is unclear."
        ),
        "short_call": (
            "Overbought RSI, stalling momentum, and proximity to technical resistance "
            "suggest limited near-term upside. Selling an OTM call collects premium "
            "immediately, with maximum profit if the stock stays below the strike at expiry."
        ),
        "short_put": (
            "Oversold RSI and nearby support suggest the stock is unlikely to fall much "
            "further. Selling an OTM put collects premium with maximum profit if the stock "
            "stabilises or recovers above the strike by expiry."
        ),
        "bull_put_spread": (
            "The moderately bullish or stable outlook combined with high volatility (rich "
            "premiums) favours a Bull Put Spread. Selling the higher-strike put and buying "
            "the lower-strike put collects net credit while capping downside risk."
        ),
        "bear_call_spread": (
            "The moderately bearish or neutral outlook favours a Bear Call Spread. "
            "Selling the lower-strike call and buying the higher-strike call collects net "
            "credit that is fully kept if the stock stays below the short strike."
        ),
        "iron_condor": (
            "Range-bound signals — neutral MC return, moderate RSI, no clear SMA trend "
            "divergence — make an Iron Condor the most appropriate play. Collecting "
            "premium from both the put spread and call spread profits if the stock "
            "stays within the defined range, with risk capped on both wings."
        ),
    }

    action_name = action.replace("_", " ").title()
    decision_text = explanations.get(
        action,
        f"The AI selected {action_name} based on the current market conditions and Q-table policy.",
    )
    parts.append(f"Strategy Decision — {action_name}: {decision_text}")

    # Risk summary
    risk_level = risk.get("risk_level", "moderate")
    max_risk_pct = float(risk.get("max_risk_pct", 0.01)) * 100
    parts.append(
        f"Risk Management ({risk_level.capitalize()}): position is sized to risk at most "
        f"{max_risk_pct:.1f}% of the portfolio. Take-profit and stop-loss levels are "
        "calibrated to the strategy's specific max-profit and max-loss profile rather "
        "than a flat dollar amount."
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/price", methods=["POST"])
def api_price():
    """
    Run Monte Carlo simulation and return pricing + probability metrics.

    Request JSON
    ------------
    spot, strike, rate, vol, expiry  : floats
    num_paths                        : int  (1 000 – 20 000, default 5 000)
    seed                             : int | null

    Response JSON
    -------------
    call, put          : pricing dicts (mc_price, bs_price, std_error, …)
    paths_sample       : list[list[float]] – up to 50 paths for the chart
    time_axis          : list[float]
    histogram          : {bins, counts}
    params             : echo of input parameters
    """
    try:
        data = request.get_json(force=True) or {}

        spot = _require_float(data, "spot")
        strike = _require_float(data, "strike")
        rate = _require_float(data, "rate")
        vol = _require_float(data, "vol")
        expiry = _require_float(data, "expiry")
        num_paths = int(data.get("num_paths", 5_000))
        seed_raw = data.get("seed")
        seed = int(seed_raw) if seed_raw is not None else None

        _validate_positive(spot, "spot")
        _validate_positive(strike, "strike")
        _validate_positive(vol, "vol")
        _validate_positive(expiry, "expiry")
        _validate_range(rate, -0.1, 0.5, "rate")
        _validate_range(vol, 0.001, 5.0, "vol")
        _validate_range(num_paths, 1_000, 20_000, "num_paths")

        pricer = MonteCarloOptionPricer(
            spot=spot,
            strike=strike,
            risk_free_rate=rate,
            volatility=vol,
            time_to_expiry=expiry,
            num_paths=num_paths,
            num_steps=252,
            random_seed=seed,
        )

        def _fmt(result: dict) -> dict:
            ci_lo, ci_hi = result["confidence_interval_95"]
            return {
                "mc_price": round(result["price"], 4),
                "bs_price": round(result["black_scholes_price"], 4),
                "std_error": round(result["std_error"], 4),
                "ci_lower": round(ci_lo, 4),
                "ci_upper": round(ci_hi, 4),
                "prob_itm": round(result["probability_in_the_money"], 4),
            }

        # Sample paths for simulation chart (up to 50 paths)
        n_plot = min(50, num_paths)
        paths_sample = pricer.paths[:n_plot].tolist()
        time_axis = np.linspace(0.0, expiry, pricer.paths.shape[1]).tolist()

        # Terminal price histogram (60 bins)
        counts, edges = np.histogram(pricer.terminal_prices, bins=60)
        bin_centers = ((edges[:-1] + edges[1:]) / 2).tolist()

        return jsonify({
            "call": _fmt(pricer.price("call")),
            "put": _fmt(pricer.price("put")),
            "paths_sample": paths_sample,
            "time_axis": time_axis,
            "histogram": {"bins": bin_centers, "counts": counts.tolist()},
            "params": {
                "spot": spot, "strike": strike, "rate": rate,
                "vol": vol, "expiry": expiry, "num_paths": num_paths,
            },
        })

    except KeyError as exc:
        return _err(f"Missing required field: {exc}")
    except (TypeError, ValueError) as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


@app.route("/api/scenario", methods=["POST"])
def api_scenario():
    """
    Compute a single option's payoff and P&L profile across spot prices.

    Request JSON
    ------------
    spot, strike, rate, vol, expiry : floats
    option_type                     : "call" | "put"
    premium                         : float | null  (override; default = BS price)

    Response JSON
    -------------
    spot_prices, payoff, pnl : parallel lists
    premium, breakevens, max_profit, max_loss, option_type, params
    """
    try:
        data = request.get_json(force=True) or {}

        spot = _require_float(data, "spot")
        strike = _require_float(data, "strike")
        rate = _require_float(data, "rate")
        vol = _require_float(data, "vol")
        expiry = _require_float(data, "expiry")
        option_type = str(data.get("option_type", "call")).lower()
        premium_override = data.get("premium")

        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

        _validate_positive(spot, "spot")
        _validate_positive(strike, "strike")
        _validate_positive(vol, "vol")
        _validate_positive(expiry, "expiry")

        premium = (
            float(premium_override)
            if premium_override is not None
            else black_scholes_price(spot, strike, rate, vol, expiry, option_type)
        )

        # Spot range: SPOT_RANGE_MIN_FACTOR to SPOT_RANGE_MAX_FACTOR of current spot
        s_min = max(spot * SPOT_RANGE_MIN_FACTOR, SPOT_MIN_ABSOLUTE)
        s_max = spot * SPOT_RANGE_MAX_FACTOR
        spot_prices = np.linspace(s_min, s_max, 300)

        if option_type == "call":
            payoff = np.maximum(spot_prices - strike, 0.0)
        else:
            payoff = np.maximum(strike - spot_prices, 0.0)

        pnl = payoff - premium

        max_profit = None if option_type == "call" else round(float(strike - premium), 4)

        return jsonify({
            "spot_prices": spot_prices.tolist(),
            "payoff": payoff.tolist(),
            "pnl": pnl.tolist(),
            "premium": round(float(premium), 4),
            "breakevens": _find_breakevens(spot_prices, pnl),
            "max_profit": max_profit,
            "max_loss": round(-float(premium), 4),
            "option_type": option_type,
            "params": {
                "spot": spot, "strike": strike, "rate": rate,
                "vol": vol, "expiry": expiry,
            },
        })

    except KeyError as exc:
        return _err(f"Missing required field: {exc}")
    except (TypeError, ValueError) as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


@app.route("/api/strategies", methods=["POST"])
def api_strategies():
    """
    Compare multiple option strategies via their P&L profiles.

    Request JSON
    ------------
    spot, rate, vol, expiry : floats
    strategies              : list of strategy config objects, each with:
                                "id"  : strategy identifier
                                + strike keys (strategy-dependent)

    Response JSON
    -------------
    spot_prices    : list[float]
    strategies     : list of profile dicts (id, name, color, pnl, …)
    current_spot   : float
    """
    try:
        data = request.get_json(force=True) or {}

        spot = _require_float(data, "spot")
        rate = _require_float(data, "rate")
        vol = _require_float(data, "vol")
        expiry = _require_float(data, "expiry")
        strategies_cfg = list(data.get("strategies", []))

        _validate_positive(spot, "spot")
        _validate_positive(vol, "vol")
        _validate_positive(expiry, "expiry")

        if not strategies_cfg:
            raise ValueError("At least one strategy must be provided")

        s_min = max(spot * SPOT_RANGE_MIN_FACTOR, SPOT_MIN_ABSOLUTE)
        s_max = spot * SPOT_RANGE_MAX_FACTOR
        spot_prices = np.linspace(s_min, s_max, 300)

        base = {"spot": spot, "rate": rate, "vol": vol, "expiry": expiry}
        profiles = []
        for cfg in strategies_cfg:
            sid = str(cfg.get("id", ""))
            merged = {**base, **cfg}
            profiles.append(compute_strategy_profile(sid, merged, spot_prices))

        return jsonify({
            "spot_prices": spot_prices.tolist(),
            "strategies": profiles,
            "current_spot": spot,
        })

    except KeyError as exc:
        return _err(f"Missing required field: {exc}")
    except (TypeError, ValueError) as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)

@app.route("/api/greeks", methods=["POST"])
def api_greeks():
    """
    Compute Black-Scholes Greeks and sensitivity curves.

    Request JSON
    ------------
    spot, strike, rate, vol, expiry : floats
    option_type                     : "call" | "put"

    Response JSON
    -------------
    greeks         : {delta, gamma, theta, vega, rho} at current params
    spot_prices    : list[float]  – x-axis for sensitivity curves
    delta_curve    : list[float]
    gamma_curve    : list[float]
    vega_curve     : list[float]
    decay_times    : list[float]  – time remaining (years), descending
    decay_values   : list[float]  – option value at each time point
    option_price   : float        – BS price at current params
    params         : echo of input parameters
    """
    try:
        data = request.get_json(force=True) or {}

        spot = _require_float(data, "spot")
        strike = _require_float(data, "strike")
        rate = _require_float(data, "rate")
        vol = _require_float(data, "vol")
        expiry = _require_float(data, "expiry")
        option_type = str(data.get("option_type", "call")).lower()

        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

        _validate_positive(spot, "spot")
        _validate_positive(strike, "strike")
        _validate_positive(vol, "vol")
        _validate_positive(expiry, "expiry")

        # Greeks at current parameters
        greeks = black_scholes_greeks(spot, strike, rate, vol, expiry, option_type)

        # Sensitivity curves over a spot-price range
        s_min = max(spot * SPOT_RANGE_MIN_FACTOR, SPOT_MIN_ABSOLUTE)
        s_max = spot * SPOT_RANGE_MAX_FACTOR
        spot_arr = np.linspace(s_min, s_max, 200)

        delta_curve = []
        gamma_curve = []
        vega_curve  = []
        for s in spot_arr:
            g = black_scholes_greeks(s, strike, rate, vol, expiry, option_type)
            delta_curve.append(g["delta"])
            gamma_curve.append(g["gamma"])
            vega_curve.append(g["vega"])

        # Theta-decay curve: option value from now until expiry
        # Evaluate at 60 evenly-spaced time-remaining points
        decay_times = np.linspace(expiry, 0.0, 60).tolist()
        decay_values = [
            round(black_scholes_price(spot, strike, rate, vol, max(t, 1e-9), option_type), 4)
            for t in decay_times
        ]

        option_price = black_scholes_price(spot, strike, rate, vol, expiry, option_type)

        return jsonify({
            "greeks": greeks,
            "spot_prices":  spot_arr.tolist(),
            "delta_curve":  delta_curve,
            "gamma_curve":  gamma_curve,
            "vega_curve":   vega_curve,
            "decay_times":  decay_times,
            "decay_values": decay_values,
            "option_price": round(option_price, 4),
            "params": {
                "spot": spot, "strike": strike, "rate": rate,
                "vol": vol, "expiry": expiry, "option_type": option_type,
            },
        })

    except KeyError as exc:
        return _err(f"Missing required field: {exc}")
    except (TypeError, ValueError) as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


@app.route("/api/probability", methods=["POST"])
def api_probability():
    """
    Run Monte Carlo simulation to compute probability of profit and risk
    metrics for a single option strategy.

    Request JSON
    ------------
    spot, rate, vol, expiry : floats
    strategy_id             : str  (same IDs as /api/strategies)
    num_paths               : int  (1 000 – 20 000, default 10 000)
    seed                    : int | null
    + strategy-specific strike keys (strike, strike_low, strike_high, …)

    Response JSON
    -------------
    prob_profit   : float  – fraction of paths with P&L > 0
    expected_pnl  : float  – mean P&L across all paths
    median_pnl    : float  – median P&L
    var_5         : float  – 5th-percentile P&L (Value at Risk, 95 % confidence)
    var_10        : float  – 10th-percentile P&L (VaR, 90 % confidence)
    cvar_5        : float  – mean P&L of the worst 5 % outcomes (CVaR / ES)
    max_profit    : float  – best simulated outcome
    max_loss      : float  – worst simulated outcome
    premium       : float  – net premium paid (positive) / received (negative)
    strategy_name : str
    histogram     : {bins, counts}
    params        : echo of input
    """
    try:
        data = request.get_json(force=True) or {}

        spot = _require_float(data, "spot")
        rate = _require_float(data, "rate")
        vol = _require_float(data, "vol")
        expiry = _require_float(data, "expiry")
        strategy_id = str(data.get("strategy_id", "long_call"))
        num_paths = int(data.get("num_paths", 10_000))
        seed_raw = data.get("seed")
        seed = int(seed_raw) if seed_raw is not None else None

        _validate_positive(spot, "spot")
        _validate_positive(vol, "vol")
        _validate_positive(expiry, "expiry")
        _validate_range(rate, -0.1, 0.5, "rate")
        _validate_range(vol, 0.001, 5.0, "vol")
        _validate_range(num_paths, 1_000, 20_000, "num_paths")

        # Simulate stock paths – use simulate_stock_paths directly so we do
        # not need a dummy strike for MonteCarloOptionPricer.
        paths = simulate_stock_paths(
            spot=spot,
            risk_free_rate=rate,
            volatility=vol,
            time_to_expiry=expiry,
            num_paths=num_paths,
            num_steps=252,
            random_seed=seed,
        )
        terminal_prices = paths[:, -1]  # shape (num_paths,)

        # Compute strategy P&L at each terminal price
        merged = {**data, "spot": spot, "rate": rate, "vol": vol, "expiry": expiry}
        profile = compute_strategy_profile(strategy_id, merged, terminal_prices)
        pnl_array = np.array(profile["pnl"])

        # ── Risk metrics ────────────────────────────────────────────────────
        prob_profit = float(np.mean(pnl_array > 0))
        expected_pnl = float(np.mean(pnl_array))
        median_pnl = float(np.median(pnl_array))
        var_5 = float(np.percentile(pnl_array, 5))
        var_10 = float(np.percentile(pnl_array, 10))
        # CVaR (Conditional VaR / Expected Shortfall) at 5 %
        cvar_mask = pnl_array <= var_5
        cvar_5 = float(np.mean(pnl_array[cvar_mask])) if cvar_mask.any() else var_5

        # ── Histogram (50 bins) ─────────────────────────────────────────────
        counts, edges = np.histogram(pnl_array, bins=50)
        bin_centers = ((edges[:-1] + edges[1:]) / 2).tolist()

        return jsonify({
            "prob_profit":   round(prob_profit, 4),
            "expected_pnl":  round(expected_pnl, 4),
            "median_pnl":    round(median_pnl, 4),
            "var_5":         round(var_5, 4),
            "var_10":        round(var_10, 4),
            "cvar_5":        round(cvar_5, 4),
            "max_profit":    round(float(np.max(pnl_array)), 4),
            "max_loss":      round(float(np.min(pnl_array)), 4),
            "premium":       profile["premium"],
            "strategy_name": profile["name"],
            "histogram": {
                "bins":   bin_centers,
                "counts": counts.tolist(),
            },
            "params": {
                "spot": spot, "rate": rate, "vol": vol,
                "expiry": expiry, "strategy_id": strategy_id,
                "num_paths": num_paths,
            },
        })

    except KeyError as exc:
        return _err(f"Missing required field: {exc}")
    except (TypeError, ValueError) as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


# ---------------------------------------------------------------------------
# Demo Trading Routes
# ---------------------------------------------------------------------------

@app.route("/api/quote", methods=["GET"])
def api_quote():
    """
    Fetch a real-time stock quote including historical volatility.

    Query Parameters
    ----------------
    ticker : str  – stock ticker symbol (e.g. AAPL)

    Response JSON
    -------------
    ticker, name, price, prev_close, change, change_pct,
    hist_vol, currency, market_cap, sector, timestamp
    """
    ticker = request.args.get("ticker", "").strip()
    if not ticker:
        return _err("Missing required query parameter: ticker")
    try:
        data = market_data.get_quote(ticker)
        return jsonify(data)
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


@app.route("/api/demo/trade", methods=["POST"])
def api_demo_open_trade():
    """
    Open a new paper-trade (demo) position.

    Request JSON
    ------------
    ticker      : str   – stock ticker
    strategy_id : str   – strategy identifier (same as /api/strategies)
    expiry      : float – time to expiry in years (e.g. 0.25 for 3 months)
    rate        : float – risk-free rate (fraction, e.g. 0.05)
    quantity    : int   – number of contracts (default 1)
    + strategy-specific strike keys (strike, strike_low, strike_high, …)

    The current market price and historical volatility are fetched
    automatically from the live quote for the given ticker.

    Response JSON
    -------------
    trade dict (id, ticker, strategy, entry details, premium, status, …)
    """
    try:
        data = request.get_json(force=True) or {}

        ticker = str(data.get("ticker", "")).strip().upper()
        if not ticker:
            raise KeyError("ticker")
        strategy_id = str(data.get("strategy_id", "")).strip()
        if not strategy_id:
            raise KeyError("strategy_id")
        expiry = _require_float(data, "expiry")
        rate = float(data.get("rate", 0.05))
        quantity = int(data.get("quantity", 1))

        _validate_positive(expiry, "expiry")
        if quantity <= 0:
            raise ValueError("quantity must be a positive integer")

        # Fetch live price and historical vol
        quote = market_data.get_quote(ticker)
        spot = quote["price"]
        hist_vol = quote["hist_vol"]

        # Optional strike overrides
        strike       = float(data["strike"])       if "strike"       in data else None
        strike_low   = float(data["strike_low"])   if "strike_low"   in data else None
        strike_high  = float(data["strike_high"])  if "strike_high"  in data else None
        strike_call  = float(data["strike_call"])  if "strike_call"  in data else None
        strike_put   = float(data["strike_put"])   if "strike_put"   in data else None

        # Optional TP/SL overrides from the request (manual trader can set these)
        take_profit = float(data["take_profit"]) if "take_profit" in data else None
        stop_loss   = float(data["stop_loss"])   if "stop_loss"   in data else None

        # Default strike to ATM when a single strike is needed and not provided
        if strike is None and strike_low is None and strike_call is None:
            if strategy_id in ("long_call", "long_put", "long_straddle",
                               "covered_call", "protective_put"):
                strike = round(spot)

        # If the caller didn't provide TP/SL, compute strategy-aware defaults
        if take_profit is None or stop_loss is None:
            _spot_arr = np.linspace(max(spot * 0.5, 0.01), spot * 1.5, 200)
            _params: dict = {
                "spot": spot, "rate": rate, "vol": hist_vol, "expiry": expiry,
            }
            for _k, _v in (
                ("strike", strike), ("strike_low", strike_low),
                ("strike_high", strike_high), ("strike_call", strike_call),
                ("strike_put", strike_put),
            ):
                if _v is not None:
                    _params[_k] = _v
            try:
                from strategies import compute_strategy_profile as _csp
                _profile = _csp(strategy_id, _params, _spot_arr)
                _tp, _sl = _compute_strategy_tpsl(
                    strategy_id,
                    _profile["premium"],
                    _profile["max_profit"],
                    _profile["max_loss"],
                )
                if take_profit is None:
                    take_profit = _tp
                if stop_loss is None:
                    stop_loss = _sl
            except (ValueError, KeyError, TypeError, AttributeError):
                pass  # leave as None if computation fails; open_trade handles defaults

        trade = market_data.open_trade(
            ticker=ticker,
            strategy_id=strategy_id,
            spot=spot,
            expiry=expiry,
            rate=rate,
            hist_vol=hist_vol,
            quantity=quantity,
            strike=strike,
            strike_low=strike_low,
            strike_high=strike_high,
            strike_call=strike_call,
            strike_put=strike_put,
            take_profit=take_profit,
            stop_loss=stop_loss,
        )
        return jsonify(trade), 201

    except KeyError as exc:
        return _err(f"Missing required field: {exc}")
    except (TypeError, ValueError) as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


@app.route("/api/demo/portfolio", methods=["GET"])
def api_demo_portfolio():
    """
    Return all demo positions with live P&L.

    Response JSON
    -------------
    { "trades": [ … ], "balance": float }
    Each trade dict contains current_spot, current_pnl, current_pnl_pct, etc.
    """
    try:
        trades = market_data.get_portfolio(refresh_prices=True)
        return jsonify({"trades": trades, "balance": round(market_data.get_balance(), 2)})
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


@app.route("/api/demo/balance", methods=["GET"])
def api_demo_balance():
    """
    Return the current demo account balance.

    Response JSON
    -------------
    { "balance": float, "initial": float }
    """
    return jsonify({
        "balance": round(market_data.get_balance(), 2),
        "initial": market_data.DEMO_BALANCE_INITIAL,
    })


@app.route("/api/demo/balance/reset", methods=["POST"])
def api_demo_balance_reset():
    """
    Reset the demo account balance to the initial value ($10 000) and close
    all open positions.

    Response JSON
    -------------
    { "balance": float, "closed": int }
    """
    try:
        # Close all open positions first
        open_trades = [t for t in market_data.get_portfolio(refresh_prices=False)
                       if t["status"] == "open"]
        closed_count = 0
        for t in open_trades:
            try:
                market_data.close_trade(t["id"])
                closed_count += 1
            except Exception:
                pass

        # Reset the balance
        balance = market_data.reset_balance()
        return jsonify({"balance": round(balance, 2), "closed": closed_count})
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


@app.route("/api/demo/trade/<trade_id>", methods=["DELETE"])
def api_demo_close_trade(trade_id: str):
    """
    Close (exit) an open demo position at the current market price.

    If the trade was opened by the AI (has an ``ai_state`` field), the Q-table
    is automatically updated from the resulting P&L reward so that learning
    happens without any extra manual step.

    Response JSON
    -------------
    Closed trade dict with exit_spot, exit_time, final P&L.
    """
    try:
        trade = market_data.close_trade(trade_id)

        # Auto-learn if this was an AI trade
        ai_state_raw = trade.get("ai_state")
        ai_action    = trade.get("ai_action")
        if ai_state_raw is not None and ai_action is not None:
            state  = tuple(ai_state_raw)
            pnl    = trade.get("current_pnl", 0.0)
            reward = ai_trader.compute_reward(pnl)

            # Best-effort: compute next state from a fresh quote
            next_state = None
            try:
                ticker   = trade["ticker"]
                quote    = market_data.get_quote(ticker)
                expiry   = trade.get("expiry_years", 0.25)
                rate     = trade.get("rate", 0.05)
                mc_ret   = ai_trader.compute_mc_expected_return(
                    spot=quote["price"], hist_vol=quote["hist_vol"],
                    rate=rate, expiry=expiry, num_paths=500,
                )
                rsi_next = market_data.get_technical_indicators(ticker).get("rsi", 50.0)
                next_state = ai_trader.get_state(
                    price_change_pct=quote["change_pct"],
                    hist_vol=quote["hist_vol"],
                    mc_expected_return=mc_ret,
                    rsi=rsi_next,
                )
            except Exception:  # noqa: BLE001
                pass  # next_state remains None (terminal); non-fatal

            ai_trader.record_reward(state, ai_action, reward, next_state)

        return jsonify(trade)
    except KeyError:
        return _err(f"Trade '{trade_id}' not found", 404)
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


# ---------------------------------------------------------------------------
# Stock Chart Route
# ---------------------------------------------------------------------------

@app.route("/api/stock_chart", methods=["GET"])
def api_stock_chart():
    """
    Return OHLCV historical data for a ticker as a candlestick-ready payload.

    Query Parameters
    ----------------
    ticker : str  – stock ticker symbol (e.g. AAPL)
    period : str  – history window: "1mo" | "3mo" | "6mo" | "1y" | "2y" (default "6mo")

    Response JSON
    -------------
    ticker, period, dates, opens, highs, lows, closes, volumes
    """
    ticker = request.args.get("ticker", "").strip()
    period = request.args.get("period", "6mo").strip()
    if not ticker:
        return _err("Missing required query parameter: ticker")
    try:
        data = market_data.get_history(ticker, period)
        return jsonify(data)
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


# ---------------------------------------------------------------------------
# AI Trading Routes
# ---------------------------------------------------------------------------

@app.route("/api/ai/status", methods=["GET"])
def api_ai_status():
    """
    Return the current AI trading status and Q-table snapshot.

    Response JSON
    -------------
    epsilon, total_trades, winning_trades, win_rate, total_reward,
    reward_history, trade_log, q_table, actions, states, best_actions,
    learning_rate, discount, epsilon_min, epsilon_decay
    """
    try:
        status = ai_trader.get_status()
        return jsonify(status)
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


@app.route("/api/ai/trade", methods=["POST"])
def api_ai_trade():
    """
    Run one AI trading step for a given ticker.

    The AI:
    1. Fetches the live quote (price, hist_vol, change_pct).
    2. Computes quantitative signals (RSI, momentum) from historical data.
    3. Runs a quick Monte Carlo simulation to estimate expected return.
    4. Derives the discrete market state (price_trend, vol_level, mc_signal, rsi_signal).
    5. Performs a risk assessment (position sizing, vol regime).
    6. Selects an action via epsilon-greedy policy.
    7. If the action is not "no_trade", opens a demo trade and returns it.

    Request JSON
    ------------
    ticker  : str   – stock ticker symbol
    expiry  : float – time to expiry in years (default 0.25)
    rate    : float – risk-free rate (fraction, default 0.05)

    Response JSON
    -------------
    action        : str   – selected action id
    state         : list  – [price_trend, vol_level, mc_signal, rsi_signal]
    mc_return     : float – MC expected fractional return
    quant_signals : dict  – RSI, momentum, SMA info
    risk          : dict  – risk assessment (recommended contracts, vol regime, etc.)
    trade         : dict | null  – opened trade dict (null for "no_trade")
    ai_status     : dict  – current AI status snapshot
    """
    try:
        data = request.get_json(force=True) or {}
        ticker = str(data.get("ticker", "")).strip().upper()
        if not ticker:
            raise KeyError("ticker")
        expiry = float(data.get("expiry", 0.25))
        rate   = float(data.get("rate",   0.05))

        _validate_positive(expiry, "expiry")

        # 1. Live quote (cache TTL is now 15 s for near-real-time data)
        quote    = market_data.get_quote(ticker)
        spot     = quote["price"]
        hist_vol = quote["hist_vol"]

        # Use intraday (1-minute interval) price change for the price-trend
        # signal so the state reflects the latest market movement rather than
        # the static day-over-day figure which stays constant all session.
        intraday_change_pct = market_data.get_intraday_change(ticker)

        # 2. Quantitative signals (RSI, momentum, moving averages)
        quant = market_data.get_technical_indicators(ticker)
        rsi   = quant["rsi"]

        # 3. Monte Carlo expected return (quick, 2 000 paths)
        mc_return = ai_trader.compute_mc_expected_return(
            spot=spot,
            hist_vol=hist_vol,
            rate=rate,
            expiry=expiry,
            num_paths=2000,
        )

        # 4. Discrete state – use intraday change for the price-trend dimension
        state = ai_trader.get_state(
            price_change_pct=intraday_change_pct,
            hist_vol=hist_vol,
            mc_expected_return=mc_return,
            rsi=rsi,
        )

        # 5. Risk assessment
        action_hint = ai_trader.decide(state)  # peek at likely action for sizing
        risk = ai_trader.assess_risk(
            spot=spot,
            hist_vol=hist_vol,
            strategy_id=action_hint if action_hint != "no_trade" else "long_call",
        )

        # 6. AI decision (final, epsilon-greedy)
        action = action_hint

        # 7. Check if balance allows a new trade (require at least $50 of buying power)
        balance = market_data.get_balance()
        skip_reason = None
        if balance < 50.0 and action != "no_trade":
            action = "no_trade"
            skip_reason = f"Skipped: insufficient demo balance (${balance:.2f})."

        # 8. Execute trade if action is not "no_trade"
        trade = None
        if action != "no_trade":
            # Determine default strikes for the chosen strategy
            strike       = None
            strike_low   = None
            strike_high  = None
            strike_call  = None
            strike_put   = None

            if action == "long_call":
                strike = round(spot * 1.02, 2)       # slightly OTM call
            elif action == "long_straddle":
                strike = round(spot, 2)               # ATM for straddle
            elif action == "long_put":
                strike = round(spot * 0.98, 2)        # slightly OTM put
            elif action == "bull_call_spread":
                strike_low  = round(spot * 1.00, 2)
                strike_high = round(spot * 1.08, 2)
            elif action == "bear_put_spread":
                strike_low  = round(spot * 0.92, 2)
                strike_high = round(spot * 1.00, 2)
            elif action == "short_call":
                strike = round(spot * 1.05, 2)        # OTM short call
            elif action == "short_put":
                strike = round(spot * 0.95, 2)        # OTM short put
            elif action == "bull_put_spread":
                strike_low  = round(spot * 0.90, 2)
                strike_high = round(spot * 0.97, 2)
            elif action == "bear_call_spread":
                strike_low  = round(spot * 1.03, 2)
                strike_high = round(spot * 1.10, 2)
            elif action == "iron_condor":
                # 4-leg: inner strikes ~3–5% OTM, outer wings ~8–10% OTM
                strike_put  = round(spot * 0.95, 2)   # inner put (sell)
                strike_call = round(spot * 1.05, 2)   # inner call (sell)
                strike_low  = round(spot * 0.90, 2)   # outer put wing (buy)
                strike_high = round(spot * 1.10, 2)   # outer call wing (buy)

            # Strategy-aware position sizing: compute the strategy profile to get
            # premium, max_profit, and max_loss, then calibrate TP/SL accordingly.
            _spot_arr = np.linspace(max(spot * 0.5, 0.01), spot * 1.5, 200)
            _profile_params: dict = {
                "spot": spot, "rate": rate, "vol": hist_vol, "expiry": expiry,
            }
            for _pk, _pv in (
                ("strike", strike), ("strike_low", strike_low),
                ("strike_high", strike_high), ("strike_call", strike_call),
                ("strike_put", strike_put),
            ):
                if _pv is not None:
                    _profile_params[_pk] = _pv
            try:
                _sp = compute_strategy_profile(action, _profile_params, _spot_arr)
                take_profit_dollars, stop_loss_dollars = _compute_strategy_tpsl(
                    action, _sp["premium"], _sp["max_profit"], _sp["max_loss"]
                )
            except Exception:
                # Fallback: 1% risk / 2% reward of current balance
                _fb_risk = round(balance * 0.01, 2)
                take_profit_dollars = round(_fb_risk * 2.0, 2)
                stop_loss_dollars   = -_fb_risk

            trade = market_data.open_trade(
                ticker=ticker,
                strategy_id=action,
                spot=spot,
                expiry=expiry,
                rate=rate,
                hist_vol=hist_vol,
                quantity=1,
                strike=strike,
                strike_low=strike_low,
                strike_high=strike_high,
                strike_call=strike_call,
                strike_put=strike_put,
                take_profit=take_profit_dollars,
                stop_loss=stop_loss_dollars,
            )
            # Persist AI metadata in the stored portfolio entry so close_trade
            # can retrieve it when the position is eventually closed.
            with market_data._portfolio_lock:
                stored = market_data._portfolio.get(trade["id"])
                if stored is not None:
                    stored["ai_state"]  = list(state)
                    stored["ai_action"] = action
            trade["ai_state"]  = list(state)
            trade["ai_action"] = action

        response: dict = {
            "action":             action,
            "state":              list(state),
            "intraday_change_pct": round(intraday_change_pct, 4),
            "mc_return":          round(mc_return, 6),
            "quant_signals":      quant,
            "risk":               risk,
            "trade":              trade,
            "balance":            round(market_data.get_balance(), 2),
            "skipped":            skip_reason is not None,
            "skip_reason":        skip_reason,
            "ai_status":          ai_trader.get_status(),
            "reasoning":          _generate_ai_reasoning(
                action, list(state), quant, risk,
                mc_return, intraday_change_pct, ticker,
            ),
        }
        return jsonify(response), 201 if trade else 200

    except KeyError as exc:
        return _err(f"Missing required field: {exc}")
    except (TypeError, ValueError) as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


@app.route("/api/ai/close_trade/<trade_id>", methods=["POST"])
def api_ai_close_trade(trade_id: str):
    """
    Close a demo trade opened by the AI, compute the reward, and update the
    Q-table.

    Optionally accepts a JSON body with:
        next_state_ticker : str  – ticker to compute the next state for
                                   (uses same ticker if omitted)

    Response JSON
    -------------
    trade, reward, ai_status
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        # Close the trade and get final P&L
        trade = market_data.close_trade(trade_id)

        # Retrieve the AI state & action that opened this trade
        ai_state_raw  = trade.get("ai_state")
        ai_action     = trade.get("ai_action", "no_trade")

        if ai_state_raw is None:
            return _err("This trade was not opened by the AI", 400)

        state  = tuple(ai_state_raw)

        # Compute reward: scaled, clipped P&L (positive = reward, negative = penalty)
        pnl    = trade.get("current_pnl", 0.0)
        reward = ai_trader.compute_reward(pnl)

        # Best-effort: compute next state from a fresh quote
        next_state = None
        try:
            ticker   = trade["ticker"]
            quote    = market_data.get_quote(ticker)
            expiry   = trade.get("expiry_years", 0.25)
            rate     = trade.get("rate", 0.05)
            hist_vol = quote["hist_vol"]
            mc_ret   = ai_trader.compute_mc_expected_return(
                spot=quote["price"], hist_vol=hist_vol,
                rate=rate, expiry=expiry, num_paths=1000,
            )
            rsi_next = market_data.get_technical_indicators(ticker).get("rsi", 50.0)
            next_state = ai_trader.get_state(
                price_change_pct=quote["change_pct"],
                hist_vol=hist_vol,
                mc_expected_return=mc_ret,
                rsi=rsi_next,
            )
        except Exception:
            pass

        # Update Q-table
        ai_trader.record_reward(state, ai_action, reward, next_state)

        return jsonify({
            "trade":     trade,
            "reward":    round(reward, 4),
            "ai_status": ai_trader.get_status(),
        })

    except KeyError as exc:
        if str(exc) == f"'{trade_id}'":
            return _err(f"Trade '{trade_id}' not found", 404)
        return _err(f"Missing field: {exc}")
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


@app.route("/api/ai/reset", methods=["POST"])
def api_ai_reset():
    """
    Reset the AI Q-table and all performance counters.

    Response JSON
    -------------
    { "message": "AI reset successful", "ai_status": … }
    """
    try:
        ai_trader.reset()
        return jsonify({"message": "AI reset successful", "ai_status": ai_trader.get_status()})
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


@app.route("/api/ai/close_all", methods=["POST"])
def api_ai_close_all():
    """
    Close all open AI trades at current market prices and learn from each one.

    This allows the agent to batch-learn from all outstanding positions in one
    click rather than closing them one-by-one.

    Response JSON
    -------------
    {
        "closed": <count>,
        "skipped": <count>,
        "results": [ { trade, reward } … ],
        "ai_status": …
    }
    """
    try:
        trades = market_data.get_portfolio(refresh_prices=True)
        ai_trades = [t for t in trades if t.get("status") == "open" and t.get("ai_action")]

        results  = []
        skipped  = 0
        skip_errors: list[str] = []
        for t in ai_trades:
            try:
                closed = market_data.close_trade(t["id"])
                state  = tuple(closed["ai_state"])
                pnl    = closed.get("current_pnl", 0.0)
                reward = ai_trader.compute_reward(pnl)

                next_state = None
                try:
                    ticker   = closed["ticker"]
                    quote    = market_data.get_quote(ticker)
                    mc_ret   = ai_trader.compute_mc_expected_return(
                        spot=quote["price"], hist_vol=quote["hist_vol"],
                        rate=closed.get("rate", 0.05),
                        expiry=closed.get("expiry_years", 0.25),
                        num_paths=500,
                    )
                    rsi_next = market_data.get_technical_indicators(ticker).get("rsi", 50.0)
                    next_state = ai_trader.get_state(
                        price_change_pct=quote["change_pct"],
                        hist_vol=quote["hist_vol"],
                        mc_expected_return=mc_ret,
                        rsi=rsi_next,
                    )
                except Exception:  # noqa: BLE001
                    pass  # next_state remains None (terminal); non-fatal

                ai_trader.record_reward(state, closed["ai_action"], reward, next_state)
                results.append({"trade": closed, "reward": round(reward, 4)})
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                skip_errors.append(str(exc))

        return jsonify({
            "closed":      len(results),
            "skipped":     skipped,
            "skip_errors": skip_errors,
            "results":     results,
            "ai_status":   ai_trader.get_status(),
        })
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


@app.route("/api/ai/reasoning", methods=["POST"])
def api_ai_reasoning():
    """
    Generate a natural language explanation of why the AI selected a strategy.

    Request JSON
    ------------
    action            : str   – selected action id
    state             : list  – [price_trend, vol_level, mc_signal, rsi_signal]
    mc_return         : float – MC expected fractional return
    intraday_change_pct: float – intraday price change percentage
    quant_signals     : dict  – RSI, momentum, SMA flags
    risk              : dict  – risk assessment dict
    ticker            : str   – stock ticker (used in narrative)

    Response JSON
    -------------
    { "reasoning": "<multi-paragraph text>", "action": "<action_id>" }
    """
    try:
        data = request.get_json(force=True) or {}
        action = str(data.get("action", "no_trade"))
        state = list(data.get("state", []))
        quant = dict(data.get("quant_signals", {}))
        risk = dict(data.get("risk", {}))
        mc_return = float(data.get("mc_return", 0.0))
        intraday_pct = float(data.get("intraday_change_pct", 0.0))
        ticker = str(data.get("ticker", "the stock")).upper()

        reasoning = _generate_ai_reasoning(
            action, state, quant, risk, mc_return, intraday_pct, ticker
        )
        return jsonify({"reasoning": reasoning, "action": action})
    except Exception as exc:  # noqa: BLE001
        return _err(f"Internal error: {exc}", 500)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # Enable debug mode only when explicitly requested via the FLASK_DEBUG
    # environment variable.  Never enable debug in production, as it exposes
    # an interactive debugger that allows arbitrary code execution.
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, port=5000)

