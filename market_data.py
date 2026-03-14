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

#: How long (seconds) a cached quote is considered fresh.
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


def _compute_live_pnl(trade: dict, current_spot: float) -> float:
    """Compute current unrealised P&L for one trade."""
    from strategies import compute_strategy_profile  # avoid circular

    params = {
        "spot": trade["entry_spot"],
        "rate": trade["rate"],
        "vol": trade["vol"],
        "expiry": trade["expiry_years"],
    }
    for key in ("strike", "strike_low", "strike_high", "strike_call", "strike_put"):
        if trade.get(key) is not None:
            params[key] = trade[key]

    arr = np.array([current_spot])
    profile = compute_strategy_profile(trade["strategy_id"], params, arr)
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
