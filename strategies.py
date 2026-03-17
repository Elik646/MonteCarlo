"""
Option strategy payoff profile computations for the RetailOptions web app.

Supported strategies
--------------------
long_call        – Buy one call option.
long_put         – Buy one put option.
bull_call_spread – Long call at K_low + short call at K_high.
bear_put_spread  – Long put at K_high + short put at K_low.
long_straddle    – Long call + long put at the same strike.
long_strangle    – Long OTM call + long OTM put (different strikes).
covered_call     – Long stock + short call.
protective_put   – Long stock + long put.
short_call       – Sell one call option (collect premium; profit if stock ≤ K).
short_put        – Sell one put option (collect premium; profit if stock ≥ K).
bull_put_spread  – Sell higher-strike put + buy lower-strike put (credit spread).
bear_call_spread – Sell lower-strike call + buy higher-strike call (credit spread).
iron_condor      – Bull put spread + bear call spread (profit in low-vol range).
"""

from __future__ import annotations

import numpy as np

from monte_carlo import black_scholes_price

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

STRATEGY_NAMES: dict[str, str] = {
    "long_call": "Long Call",
    "long_put": "Long Put",
    "bull_call_spread": "Bull Call Spread",
    "bear_put_spread": "Bear Put Spread",
    "long_straddle": "Long Straddle",
    "long_strangle": "Long Strangle",
    "covered_call": "Covered Call",
    "protective_put": "Protective Put",
    "short_call": "Short Call",
    "short_put": "Short Put",
    "bull_put_spread": "Bull Put Spread",
    "bear_call_spread": "Bear Call Spread",
    "iron_condor": "Iron Condor",
}

STRATEGY_COLORS: dict[str, str] = {
    "long_call": "#00bcd4",
    "long_put": "#ff6b6b",
    "bull_call_spread": "#4ecdc4",
    "bear_put_spread": "#ff9f43",
    "long_straddle": "#a29bfe",
    "long_strangle": "#fd79a8",
    "covered_call": "#55efc4",
    "protective_put": "#fdcb6e",
    "short_call": "#e17055",
    "short_put": "#74b9ff",
    "bull_put_spread": "#00cec9",
    "bear_call_spread": "#d63031",
    "iron_condor": "#6c5ce7",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bs(spot: float, strike: float, rate: float, vol: float,
        expiry: float, option_type: str) -> float:
    """Thin wrapper around black_scholes_price for brevity."""
    return black_scholes_price(
        spot=spot,
        strike=strike,
        risk_free_rate=rate,
        volatility=vol,
        time_to_expiry=expiry,
        option_type=option_type,
    )


def _find_breakevens(spot_prices: np.ndarray, pnl: np.ndarray) -> list[float]:
    """Return approximate breakeven prices via linear interpolation."""
    breakevens: list[float] = []
    for i in range(len(pnl) - 1):
        y0, y1 = float(pnl[i]), float(pnl[i + 1])
        if y0 * y1 <= 0 and y0 != y1:
            be = float(spot_prices[i]) - y0 * (
                float(spot_prices[i + 1]) - float(spot_prices[i])
            ) / (y1 - y0)
            breakevens.append(round(be, 2))
    return breakevens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_strategy_profile(
    strategy_id: str,
    params: dict,
    spot_prices: np.ndarray,
) -> dict:
    """
    Compute the P&L profile for an option strategy at expiry.

    Parameters
    ----------
    strategy_id : Identifier string (see module docstring for valid IDs).
    params      : Dict containing ``spot``, ``rate``, ``vol``, ``expiry``,
                  and strategy-specific strike keys.
    spot_prices : 1-D array of stock prices to evaluate at expiry.

    Returns
    -------
    dict with keys:
        id, name, color, pnl, premium, max_profit, max_loss, breakevens.
    """
    S = spot_prices.astype(float)
    s0 = float(params["spot"])
    r = float(params["rate"])
    v = float(params["vol"])
    T = float(params["expiry"])

    if strategy_id == "long_call":
        K = float(params["strike"])
        premium = _bs(s0, K, r, v, T, "call")
        pnl = np.maximum(S - K, 0.0) - premium

    elif strategy_id == "long_put":
        K = float(params["strike"])
        premium = _bs(s0, K, r, v, T, "put")
        pnl = np.maximum(K - S, 0.0) - premium

    elif strategy_id == "bull_call_spread":
        K1 = float(params["strike_low"])
        K2 = float(params["strike_high"])
        premium = _bs(s0, K1, r, v, T, "call") - _bs(s0, K2, r, v, T, "call")
        pnl = np.maximum(S - K1, 0.0) - np.maximum(S - K2, 0.0) - premium

    elif strategy_id == "bear_put_spread":
        K1 = float(params["strike_low"])
        K2 = float(params["strike_high"])
        premium = _bs(s0, K2, r, v, T, "put") - _bs(s0, K1, r, v, T, "put")
        pnl = np.maximum(K2 - S, 0.0) - np.maximum(K1 - S, 0.0) - premium

    elif strategy_id == "long_straddle":
        K = float(params["strike"])
        premium = _bs(s0, K, r, v, T, "call") + _bs(s0, K, r, v, T, "put")
        pnl = np.maximum(S - K, 0.0) + np.maximum(K - S, 0.0) - premium

    elif strategy_id == "long_strangle":
        K_call = float(params["strike_call"])
        K_put = float(params["strike_put"])
        premium = _bs(s0, K_call, r, v, T, "call") + _bs(s0, K_put, r, v, T, "put")
        pnl = np.maximum(S - K_call, 0.0) + np.maximum(K_put - S, 0.0) - premium

    elif strategy_id == "covered_call":
        K = float(params["strike"])
        call_p = _bs(s0, K, r, v, T, "call")
        # P&L = (S_T − S_0) − max(S_T − K, 0) + call_premium_received
        pnl = (S - s0) - np.maximum(S - K, 0.0) + call_p
        premium = -call_p  # net credit

    elif strategy_id == "protective_put":
        K = float(params["strike"])
        put_p = _bs(s0, K, r, v, T, "put")
        # P&L = (S_T − S_0) + max(K − S_T, 0) − put_premium_paid
        pnl = (S - s0) + np.maximum(K - S, 0.0) - put_p
        premium = put_p

    elif strategy_id == "short_call":
        K = float(params["strike"])
        call_p = _bs(s0, K, r, v, T, "call")
        # Sell a call: collect premium, pay out intrinsic at expiry
        pnl = call_p - np.maximum(S - K, 0.0)
        premium = -call_p  # negative premium = credit received

    elif strategy_id == "short_put":
        K = float(params["strike"])
        put_p = _bs(s0, K, r, v, T, "put")
        # Sell a put: collect premium, pay out intrinsic at expiry
        pnl = put_p - np.maximum(K - S, 0.0)
        premium = -put_p  # negative premium = credit received

    elif strategy_id == "bull_put_spread":
        # Credit put spread: sell K_high put, buy K_low put (K_low < K_high)
        K1 = float(params["strike_low"])
        K2 = float(params["strike_high"])
        credit = _bs(s0, K2, r, v, T, "put") - _bs(s0, K1, r, v, T, "put")
        pnl = credit - (np.maximum(K2 - S, 0.0) - np.maximum(K1 - S, 0.0))
        premium = -credit  # credit received (negative premium)

    elif strategy_id == "bear_call_spread":
        # Credit call spread: sell K_low call, buy K_high call (K_low < K_high)
        K1 = float(params["strike_low"])
        K2 = float(params["strike_high"])
        credit = _bs(s0, K1, r, v, T, "call") - _bs(s0, K2, r, v, T, "call")
        pnl = credit - (np.maximum(S - K1, 0.0) - np.maximum(S - K2, 0.0))
        premium = -credit  # credit received (negative premium)

    elif strategy_id == "iron_condor":
        # Bull put spread (sell K_put_high, buy K_put_low) +
        # Bear call spread (sell K_call_low, buy K_call_high)
        # Parameters: strike_low (put wing), strike_put (inner put),
        #             strike_call (inner call), strike_high (call wing)
        K_put_low  = float(params["strike_low"])
        K_put_high = float(params["strike_put"])
        K_call_low = float(params["strike_call"])
        K_call_high = float(params["strike_high"])
        # Credit from bull put spread
        put_credit  = _bs(s0, K_put_high, r, v, T, "put")  - _bs(s0, K_put_low, r, v, T, "put")
        # Credit from bear call spread
        call_credit = _bs(s0, K_call_low, r, v, T, "call") - _bs(s0, K_call_high, r, v, T, "call")
        total_credit = put_credit + call_credit
        # P&L = total_credit - debit from legs hitting intrinsic
        put_spread_loss  = np.maximum(K_put_high - S, 0.0)  - np.maximum(K_put_low - S, 0.0)
        call_spread_loss = np.maximum(S - K_call_low, 0.0) - np.maximum(S - K_call_high, 0.0)
        pnl = total_credit - put_spread_loss - call_spread_loss
        premium = -total_credit  # credit received (negative premium)

    else:
        raise ValueError(f"Unknown strategy: {strategy_id!r}")

    max_profit_val = float(np.max(pnl))
    max_loss_val = float(np.min(pnl))

    return {
        "id": strategy_id,
        "name": STRATEGY_NAMES.get(strategy_id, strategy_id),
        "color": STRATEGY_COLORS.get(strategy_id, "#ffffff"),
        "pnl": pnl.tolist(),
        "premium": round(float(premium), 4),
        "max_profit": None if np.isinf(max_profit_val) else round(max_profit_val, 4),
        "max_loss": round(max_loss_val, 4),
        "breakevens": _find_breakevens(S, pnl),
    }
