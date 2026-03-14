"""
RetailOptions – Flask Web Application
======================================
Exposes the Monte Carlo option pricing engine and strategy analysis
as a JSON API consumed by the single-page web UI (templates/index.html).

Endpoints
---------
GET  /                  Serve the web UI.
POST /api/price         MC + BS pricing with probability metrics.
POST /api/scenario      Single-option payoff / P&L profile.
POST /api/strategies    Multi-strategy comparison (payoff profiles).

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
)
from strategies import compute_strategy_profile

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

