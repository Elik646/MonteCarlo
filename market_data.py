"""
market_data.py – Real market data helpers for RetailOptions demo trading.

Fetches live quotes and historical volatility via yfinance and provides a
lightweight in-memory portfolio store for paper-trade positions.

Functions
---------
get_quote(ticker)        → dict  – spot price, historical vol, company info.
                                   Results are cached for CACHE_TTL_SECONDS.

Portfolio management (module-level in-memory store)
---------------------------------------------------
open_trade(...)          → dict  – create a new demo position.
close_trade(trade_id)    → dict  – mark a trade as closed; returns it.
get_portfolio()          → list  – all open positions with live P&L.
get_trade(trade_id)      → dict  – single position (open or closed).
"""

from __future__ import annotations

import math
import time
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import yfinance as yf

from monte_carlo import black_scholes_price

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: How long (seconds) a cached quote is considered fresh (default: 60 s).
CACHE_TTL_SECONDS: int = 60

#: Annualised trading days used when converting daily σ to annual.
TRADING_DAYS_PER_YEAR: int = 252

#: Look-back window (calendar days) for historical-vol calculation.
HIST_VOL_LOOKBACK_DAYS: int = 365

# ---------------------------------------------------------------------------
# Quote cache (thread-safe)
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_quote_cache: dict[str, tuple[float, dict]] = {}  # ticker → (timestamp, data)


def get_quote(ticker: str) -> dict:
    """
    Return a dict with price, historical volatility, and company metadata for
    *ticker*.  Results are cached for ``CACHE_TTL_SECONDS``.

    Returns
    -------
    {
        ticker          : str,
        name            : str,
        price           : float,
        prev_close      : float,
        change          : float,   # absolute
        change_pct      : float,   # percentage
        hist_vol        : float,   # annualised volatility (fraction, e.g. 0.25)
        currency        : str,
        market_cap      : int | None,
        sector          : str,
        timestamp       : float,   # unix epoch of quote fetch
    }

    Raises
    ------
    ValueError  if the ticker is invalid or no price data is available.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("Ticker symbol must not be empty")

    now = time.time()
    with _cache_lock:
        if ticker in _quote_cache:
            ts, cached = _quote_cache[ticker]
            if now - ts < CACHE_TTL_SECONDS:
                return cached

    # Fetch from yfinance (outside the lock to avoid blocking other threads)
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info  # lightweight; avoids heavy scraping
        price = float(info.last_price or 0)
        prev_close = float(info.previous_close or 0)
    except Exception as exc:
        raise ValueError(f"Could not fetch quote for '{ticker}': {exc}") from exc

    if price <= 0:
        raise ValueError(f"No valid price data found for ticker '{ticker}'")

    # Historical volatility: annualised std of log-returns
    try:
        hist = t.history(period="1y")
        if len(hist) >= 20:
            closes = hist["Close"].values.astype(float)
            log_returns = np.diff(np.log(closes))
            daily_vol = float(np.std(log_returns, ddof=1))
            hist_vol = daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            hist_vol = 0.25  # fallback
    except Exception:
        hist_vol = 0.25  # fallback

    # Company metadata (best-effort)
    try:
        meta = t.info or {}
    except Exception:
        meta = {}

    change = price - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0.0

    data = {
        "ticker": ticker,
        "name": meta.get("longName") or meta.get("shortName") or ticker,
        "price": round(price, 4),
        "prev_close": round(prev_close, 4),
        "change": round(change, 4),
        "change_pct": round(change_pct, 4),
        "hist_vol": round(hist_vol, 4),
        "currency": meta.get("currency", "USD"),
        "market_cap": meta.get("marketCap"),
        "sector": meta.get("sector", ""),
        "timestamp": now,
    }

    with _cache_lock:
        _quote_cache[ticker] = (now, data)

    return data


# ---------------------------------------------------------------------------
# In-memory demo-trading portfolio
# ---------------------------------------------------------------------------

_portfolio_lock = threading.Lock()
_portfolio: dict[str, dict] = {}  # trade_id → trade dict


def open_trade(
    ticker: str,
    strategy_id: str,
    spot: float,
    expiry: float,
    rate: float,
    hist_vol: float,
    quantity: int = 1,
    strike: Optional[float] = None,
    strike_low: Optional[float] = None,
    strike_high: Optional[float] = None,
    strike_call: Optional[float] = None,
    strike_put: Optional[float] = None,
) -> dict:
    """
    Record a new paper-trade position and return it.

    The entry premium is the Black-Scholes fair value at the time of opening.
    """
    from strategies import compute_strategy_profile  # local import to avoid circular

    if quantity <= 0:
        raise ValueError("quantity must be a positive integer")
    if expiry <= 0:
        raise ValueError("expiry must be > 0")
    if spot <= 0:
        raise ValueError("spot must be > 0")

    # Build params dict for strategy profile computation
    params: dict = {
        "spot": spot,
        "rate": rate,
        "vol": hist_vol,
        "expiry": expiry,
    }
    # Add strike keys only when provided
    if strike is not None:
        params["strike"] = float(strike)
    if strike_low is not None:
        params["strike_low"] = float(strike_low)
    if strike_high is not None:
        params["strike_high"] = float(strike_high)
    if strike_call is not None:
        params["strike_call"] = float(strike_call)
    if strike_put is not None:
        params["strike_put"] = float(strike_put)

    # Validate by computing profile at entry spot (single-point array)
    entry_arr = np.array([spot])
    profile = compute_strategy_profile(strategy_id, params, entry_arr)

    trade_id = str(uuid.uuid4())
    trade: dict = {
        "id": trade_id,
        "ticker": ticker.upper(),
        "strategy_id": strategy_id,
        "strategy_name": profile["name"],
        "entry_spot": spot,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "expiry_years": expiry,
        "rate": rate,
        "vol": hist_vol,
        "quantity": quantity,
        "premium": profile["premium"],  # per-share premium (positive = paid, neg = received)
        "status": "open",
        # Strike keys — store whichever were provided
        "strike": strike,
        "strike_low": strike_low,
        "strike_high": strike_high,
        "strike_call": strike_call,
        "strike_put": strike_put,
        # Runtime fields (updated on portfolio fetch)
        "current_spot": spot,
        "current_pnl": 0.0,
        "current_pnl_pct": 0.0,
    }

    with _portfolio_lock:
        _portfolio[trade_id] = trade

    return trade


#: Minimum remaining time to expiry used in mark-to-market P&L (1 calendar day).
_MIN_REMAINING_YEARS: float = 1.0 / 365


def _compute_live_pnl(trade: dict, current_spot: float) -> float:
    """
    Compute current unrealised P&L for one trade using Black-Scholes
    mark-to-market (current option value minus entry premium).

    Using the at-expiry intrinsic-value formula would always show a loss for
    out-of-the-money options because their intrinsic value is zero while the
    entry premium is positive.  Instead we reprice the option legs with the
    same volatility but the *remaining* time to expiry, which correctly
    returns a P&L near zero when the position has just been opened and no
    large price move has occurred.
    """
    # Remaining time to expiry (years) — at least MIN_REMAINING_YEARS
    try:
        entry_time = datetime.fromisoformat(trade["entry_time"])
        now = datetime.now(timezone.utc)
        # Ensure both datetimes are timezone-aware for safe subtraction
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        elapsed_years = (now - entry_time).total_seconds() / (365.25 * 24 * 3600)
        remaining = max(trade["expiry_years"] - elapsed_years, _MIN_REMAINING_YEARS)
    except Exception:
        remaining = trade["expiry_years"]

    r = trade["rate"]
    v = trade["vol"]
    entry_premium = trade["premium"]
    strategy_id = trade["strategy_id"]
    entry_spot = trade["entry_spot"]

    def _call(spot: float, strike: float) -> float:
        return black_scholes_price(spot, strike, r, v, remaining, "call")

    def _put(spot: float, strike: float) -> float:
        return black_scholes_price(spot, strike, r, v, remaining, "put")

    if strategy_id == "long_call":
        K = trade["strike"]
        pnl_per_share = _call(current_spot, K) - entry_premium

    elif strategy_id == "long_put":
        K = trade["strike"]
        pnl_per_share = _put(current_spot, K) - entry_premium

    elif strategy_id == "bull_call_spread":
        K1, K2 = trade["strike_low"], trade["strike_high"]
        pnl_per_share = (_call(current_spot, K1) - _call(current_spot, K2)) - entry_premium

    elif strategy_id == "bear_put_spread":
        K1, K2 = trade["strike_low"], trade["strike_high"]
        pnl_per_share = (_put(current_spot, K2) - _put(current_spot, K1)) - entry_premium

    elif strategy_id == "long_straddle":
        K = trade["strike"]
        pnl_per_share = (_call(current_spot, K) + _put(current_spot, K)) - entry_premium

    elif strategy_id == "long_strangle":
        K_call = trade["strike_call"]
        K_put = trade["strike_put"]
        pnl_per_share = (
            _call(current_spot, K_call) + _put(current_spot, K_put)
        ) - entry_premium

    elif strategy_id == "covered_call":
        # entry_premium = -call_p (credit received when selling the call)
        K = trade["strike"]
        pnl_per_share = (
            (current_spot - entry_spot)          # stock leg
            - _call(current_spot, K)             # short call (reprice)
            - entry_premium                      # undo the credit we received at entry
        )

    elif strategy_id == "protective_put":
        # entry_premium = put_p (debit paid for the put)
        K = trade["strike"]
        pnl_per_share = (
            (current_spot - entry_spot)          # stock leg
            + _put(current_spot, K)              # long put (reprice)
            - entry_premium                      # subtract the cost paid at entry
        )

    else:
        # Fallback for any other strategy: reprice at remaining time
        from strategies import compute_strategy_profile  # avoid circular
        params = {
            "spot": entry_spot,
            "rate": r,
            "vol": v,
            "expiry": remaining,
        }
        for key in ("strike", "strike_low", "strike_high", "strike_call", "strike_put"):
            if trade.get(key) is not None:
                params[key] = trade[key]
        arr = np.array([current_spot])
        profile = compute_strategy_profile(strategy_id, params, arr)
        pnl_per_share = float(profile["pnl"][0])

    return pnl_per_share * trade["quantity"]


def get_portfolio(refresh_prices: bool = True) -> list[dict]:
    """
    Return all open (and recently closed) demo trades.

    When *refresh_prices* is True, live quotes are fetched for each unique
    ticker and current P&L is recalculated.
    """
    with _portfolio_lock:
        trades = [dict(t) for t in _portfolio.values()]

    if not trades or not refresh_prices:
        return trades

    # Fetch latest prices for open trades
    open_tickers = {t["ticker"] for t in trades if t["status"] == "open"}
    live_prices: dict[str, float] = {}
    for tkr in open_tickers:
        try:
            q = get_quote(tkr)
            live_prices[tkr] = q["price"]
        except Exception:
            pass  # keep last known price

    # Update P&L
    enriched = []
    for trade in trades:
        if trade["status"] == "open":
            spot = live_prices.get(trade["ticker"], trade["current_spot"])
            trade["current_spot"] = spot
            try:
                pnl = _compute_live_pnl(trade, spot)
            except Exception:
                pnl = trade.get("current_pnl", 0.0)
            trade["current_pnl"] = round(pnl, 4)
            # P&L percentage relative to premium paid (avoid div-by-zero)
            abs_prem = abs(trade["premium"]) * trade["quantity"]
            trade["current_pnl_pct"] = (
                round(pnl / abs_prem * 100, 2) if abs_prem else 0.0
            )
        enriched.append(trade)

    return enriched


def get_trade(trade_id: str) -> dict:
    """Return a single trade by ID (raises KeyError if not found)."""
    with _portfolio_lock:
        if trade_id not in _portfolio:
            raise KeyError(trade_id)
        return dict(_portfolio[trade_id])


def get_history(ticker: str, period: str = "6mo") -> dict:
    """
    Return OHLCV history for *ticker* as lists suitable for a candlestick chart.

    Parameters
    ----------
    ticker : str   – stock ticker symbol.
    period : str   – yfinance period string (e.g. "1mo", "6mo", "1y", "2y").

    Returns
    -------
    {
        dates   : list[str]    – ISO date strings
        opens   : list[float]
        highs   : list[float]
        lows    : list[float]
        closes  : list[float]
        volumes : list[int]
    }

    Raises
    ------
    ValueError  if the ticker is invalid or no data is available.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("Ticker symbol must not be empty")

    valid_periods = {"1mo", "3mo", "6mo", "1y", "2y", "5y"}
    if period not in valid_periods:
        period = "6mo"

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
    except Exception as exc:
        raise ValueError(f"Could not fetch history for '{ticker}': {exc}") from exc

    if hist.empty:
        raise ValueError(f"No historical data found for ticker '{ticker}'")

    # yfinance returns a DatetimeIndex; convert to plain date strings
    dates   = [str(d.date()) for d in hist.index]
    opens   = [round(float(v), 4) for v in hist["Open"]]
    highs   = [round(float(v), 4) for v in hist["High"]]
    lows    = [round(float(v), 4) for v in hist["Low"]]
    closes  = [round(float(v), 4) for v in hist["Close"]]
    volumes = [int(v) for v in hist["Volume"]]

    return {
        "ticker":  ticker,
        "period":  period,
        "dates":   dates,
        "opens":   opens,
        "highs":   highs,
        "lows":    lows,
        "closes":  closes,
        "volumes": volumes,
    }


def close_trade(trade_id: str) -> dict:
    """
    Mark a trade as closed at the current market price and return it.

    Raises KeyError if the trade does not exist.
    """
    with _portfolio_lock:
        if trade_id not in _portfolio:
            raise KeyError(trade_id)
        trade = _portfolio[trade_id]
        if trade["status"] != "open":
            raise ValueError(f"Trade {trade_id!r} is already closed")

    # Fetch live price outside the lock
    try:
        q = get_quote(trade["ticker"])
        exit_spot = q["price"]
    except Exception:
        exit_spot = trade["current_spot"]

    try:
        final_pnl = _compute_live_pnl(trade, exit_spot)
    except Exception:
        final_pnl = trade.get("current_pnl", 0.0)

    with _portfolio_lock:
        trade = _portfolio[trade_id]
        trade["status"] = "closed"
        trade["exit_spot"] = exit_spot
        trade["exit_time"] = datetime.now(timezone.utc).isoformat()
        trade["current_spot"] = exit_spot
        trade["current_pnl"] = round(final_pnl, 4)
        abs_prem = abs(trade["premium"]) * trade["quantity"]
        trade["current_pnl_pct"] = (
            round(final_pnl / abs_prem * 100, 2) if abs_prem else 0.0
        )
        return dict(trade)
