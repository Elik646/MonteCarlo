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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: How long (seconds) a cached quote is considered fresh (default: 15 s).
CACHE_TTL_SECONDS: int = 15

#: How long (seconds) a cached intraday-change value is considered fresh.
INTRADAY_CACHE_TTL_SECONDS: int = 10

#: Annualised trading days used when converting daily σ to annual.
TRADING_DAYS_PER_YEAR: int = 252

#: Look-back window (calendar days) for historical-vol calculation.
HIST_VOL_LOOKBACK_DAYS: int = 365

# ---------------------------------------------------------------------------
# Quote cache (thread-safe)
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_quote_cache: dict[str, tuple[float, dict]] = {}  # ticker → (timestamp, data)

_intraday_cache_lock = threading.Lock()
_intraday_cache: dict[str, tuple[float, float]] = {}  # ticker → (timestamp, change_pct)


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


def get_intraday_change(ticker: str, lookback_minutes: int = 30) -> float:
    """
    Return the recent intraday price-change percentage for *ticker* computed
    from 1-minute interval data over the last *lookback_minutes* minutes.

    This gives the AI a genuinely real-time price-trend signal that changes
    throughout the trading day as the market moves, rather than the static
    day-over-day ``change_pct`` which is constant until the next session.

    Results are cached for ``INTRADAY_CACHE_TTL_SECONDS`` to avoid
    hammering the data provider.

    Parameters
    ----------
    ticker           : str – stock ticker symbol.
    lookback_minutes : int – how many 1-minute bars to look back (default 30).

    Returns
    -------
    float – percentage change over the look-back window (e.g. 0.75 for +0.75 %).
            Returns 0.0 on any error or when market data is unavailable.
    """
    ticker = ticker.upper().strip()
    now = time.time()

    with _intraday_cache_lock:
        if ticker in _intraday_cache:
            ts, cached_val = _intraday_cache[ticker]
            if now - ts < INTRADAY_CACHE_TTL_SECONDS:
                return cached_val

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d", interval="1m")
        if hist.empty or len(hist) < 2:
            change_pct = 0.0
        else:
            closes = hist["Close"].values.astype(float)
            # We need at least lookback_minutes + 1 bars for a valid comparison;
            # if fewer bars are available, fall back to 0.0 rather than silently
            # computing from the earliest available bar (which would cover a
            # shorter window than requested and could be misleading).
            if len(closes) < lookback_minutes + 1:
                change_pct = 0.0
            else:
                old_price = closes[-(lookback_minutes + 1)]
                new_price = closes[-1]
                if old_price <= 0:
                    change_pct = 0.0
                else:
                    change_pct = round((new_price - old_price) / old_price * 100.0, 4)
    except Exception:
        change_pct = 0.0

    with _intraday_cache_lock:
        _intraday_cache[ticker] = (now, change_pct)

    return change_pct


# ---------------------------------------------------------------------------
# In-memory demo-trading portfolio
# ---------------------------------------------------------------------------

_portfolio_lock = threading.Lock()
_portfolio: dict[str, dict] = {}  # trade_id → trade dict

# ---------------------------------------------------------------------------
# Demo account balance
# ---------------------------------------------------------------------------

#: Starting demo account balance in USD.
DEMO_BALANCE_INITIAL: float = 10_000.0

_balance_lock = threading.Lock()
_balance: float = DEMO_BALANCE_INITIAL


def get_balance() -> float:
    """Return the current demo account balance."""
    with _balance_lock:
        return _balance


def adjust_balance(delta: float) -> float:
    """Add *delta* to the balance and return the new balance."""
    global _balance
    with _balance_lock:
        _balance += delta
        return _balance


def reset_balance() -> float:
    """Reset the demo balance to the initial value and return it."""
    global _balance
    with _balance_lock:
        _balance = DEMO_BALANCE_INITIAL
        return _balance


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
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
) -> dict:
    """
    Record a new paper-trade position and return it.

    The entry premium is the Black-Scholes fair value at the time of opening.
    For debit strategies (premium > 0) the cost is deducted from the demo
    balance.  Credit received (premium < 0) is added to the balance.

    Parameters
    ----------
    take_profit : float | None – auto-close when P&L reaches this value.
    stop_loss   : float | None – auto-close when P&L falls below this value.
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

    premium = profile["premium"]  # positive = paid, negative = received
    total_cost = premium * quantity  # net cash outflow at open

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
        "premium": premium,       # per-unit premium
        "total_cost": total_cost, # net cash outflow (may be negative for credits)
        "status": "open",
        # Strike keys — store whichever were provided
        "strike": strike,
        "strike_low": strike_low,
        "strike_high": strike_high,
        "strike_call": strike_call,
        "strike_put": strike_put,
        # Risk management targets (in dollar P&L)
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        # Runtime fields (updated on portfolio fetch)
        "current_spot": spot,
        "current_pnl": 0.0,
        "current_pnl_pct": 0.0,
    }

    with _portfolio_lock:
        _portfolio[trade_id] = trade

    # Reflect opening cost in the demo balance
    if total_cost != 0.0:
        adjust_balance(-total_cost)

    return trade


def _compute_live_pnl(trade: dict, current_spot: float) -> float:
    """
    Compute the current mark-to-market (MTM) P&L for an open trade.

    Uses the Black-Scholes value of all option legs at the current spot and
    remaining time to expiry, rather than the at-expiry payoff.  This gives
    realistic, continuous P&L values that reflect both intrinsic and time value.

    Parameters
    ----------
    trade        : dict  – trade record as stored in the portfolio.
    current_spot : float – current underlying price.

    Returns
    -------
    float – unrealised P&L in dollars (negative = loss, positive = gain).
    """
    from strategies import compute_strategy_mtm  # avoid circular import

    # Minimum remaining time: 1 trading day (avoid near-zero expiry instabilities).
    _ONE_TRADING_DAY = 1.0 / 252

    # Calculate remaining time to expiry.
    # entry_time is stored as a UTC ISO string (e.g. "2025-01-01T12:00:00+00:00"),
    # so fromisoformat() returns a timezone-aware datetime which can be safely
    # subtracted from datetime.now(timezone.utc).
    entry_time = datetime.fromisoformat(trade["entry_time"])
    if entry_time.tzinfo is None:
        # Fallback: treat naive timestamps as UTC (shouldn't happen with current code)
        entry_time = entry_time.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    elapsed_years = (now - entry_time).total_seconds() / (365.25 * 24 * 3600)
    remaining_T = max(trade["expiry_years"] - elapsed_years, _ONE_TRADING_DAY)

    params: dict = {
        "spot":       current_spot,
        "entry_spot": trade["entry_spot"],  # needed for covered_call / protective_put
        "rate":       trade["rate"],
        "vol":        trade["vol"],
        "expiry":     remaining_T,
    }
    for key in ("strike", "strike_low", "strike_high", "strike_call", "strike_put"):
        if trade.get(key) is not None:
            params[key] = trade[key]

    current_value = compute_strategy_mtm(trade["strategy_id"], params)
    entry_premium = trade["premium"]  # positive = paid, negative = received

    # P&L = (current liquidation value) − (entry cost)
    pnl_per_unit = current_value - entry_premium
    return pnl_per_unit * trade["quantity"]



def get_portfolio(refresh_prices: bool = True) -> list[dict]:
    """
    Return all open (and recently closed) demo trades.

    When *refresh_prices* is True, live quotes are fetched for each unique
    ticker, current P&L is recalculated using mark-to-market pricing, and
    any take-profit / stop-loss targets that have been hit are auto-closed.
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

    # Update P&L and check TP/SL
    trades_to_close: list[str] = []
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
            # P&L percentage: use total cost as the base (avoids zero-division for
            # near-zero credit premiums); fall back to absolute premium if cost is 0.
            abs_cost = abs(trade.get("total_cost", trade["premium"] * trade["quantity"]))
            trade["current_pnl_pct"] = (
                round(pnl / abs_cost * 100, 2) if abs_cost else 0.0
            )

            # Check TP / SL targets and queue for auto-close
            tp = trade.get("take_profit")
            sl = trade.get("stop_loss")
            if tp is not None and pnl >= tp:
                trades_to_close.append(trade["id"])
            elif sl is not None and pnl <= sl:
                trades_to_close.append(trade["id"])

        enriched.append(trade)

    # Auto-close any TP/SL-triggered trades (outside the loop to avoid lock issues)
    for tid in trades_to_close:
        try:
            closed = close_trade(tid)
            # Update the enriched list entry with the closed trade details
            for i, t in enumerate(enriched):
                if t["id"] == tid:
                    enriched[i] = closed
                    break
        except Exception:
            pass  # non-fatal: trade may already be closed

    return enriched


def get_trade(trade_id: str) -> dict:
    """Return a single trade by ID (raises KeyError if not found)."""
    with _portfolio_lock:
        if trade_id not in _portfolio:
            raise KeyError(trade_id)
        return dict(_portfolio[trade_id])


def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """
    Compute the Relative Strength Index (RSI) for a price series.

    Parameters
    ----------
    closes : np.ndarray – closing prices (chronological order).
    period : int        – RSI look-back period (default 14).

    Returns
    -------
    float – RSI value in [0, 100].  Returns 50.0 if there is insufficient data.
    """
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes.astype(float))
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Initial averages
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    # Wilder's smoothed moving average for the rest
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def compute_momentum(closes: np.ndarray, period: int = 20) -> float:
    """
    Compute price momentum as the percentage change over *period* days.

    Parameters
    ----------
    closes : np.ndarray – closing prices (chronological order).
    period : int        – look-back window (default 20 trading days ≈ 1 month).

    Returns
    -------
    float – percentage change (e.g. 5.0 for +5 %).  Returns 0.0 if insufficient data.
    """
    closes = closes.astype(float)
    if len(closes) < period + 1:
        return 0.0
    old_price = closes[-(period + 1)]
    new_price = closes[-1]
    if old_price <= 0:
        return 0.0
    return round((new_price - old_price) / old_price * 100.0, 4)


def get_technical_indicators(ticker: str, period: str = "1y") -> dict:
    """
    Compute technical indicators for *ticker* using historical closing prices.

    Returns
    -------
    {
        rsi        : float  – 14-period RSI (0-100).
        momentum   : float  – 20-day price momentum (%).
        sma_20     : float  – 20-day simple moving average.
        sma_50     : float  – 50-day simple moving average.
        above_sma20: bool   – True if latest close > sma_20.
        above_sma50: bool   – True if latest close > sma_50.
    }

    Falls back to neutral defaults on any error.
    """
    defaults = {
        "rsi": 50.0,
        "momentum": 0.0,
        "sma_20": 0.0,
        "sma_50": 0.0,
        "above_sma20": True,
        "above_sma50": True,
    }
    try:
        t = yf.Ticker(ticker.upper().strip())
        hist = t.history(period=period)
        if hist.empty or len(hist) < 20:
            return defaults
        closes = hist["Close"].values.astype(float)
        sma_20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else closes[-1]
        sma_50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else sma_20
        last   = closes[-1]
        return {
            "rsi":         compute_rsi(closes),
            "momentum":    compute_momentum(closes),
            "sma_20":      round(sma_20, 4),
            "sma_50":      round(sma_50, 4),
            "above_sma20": bool(last > sma_20),
            "above_sma50": bool(last > sma_50),
        }
    except Exception:
        return defaults


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

    On close the original cost (premium) is returned to the balance and the
    final P&L is credited/debited, so the net effect is:
        balance += total_cost (return of capital) + final_pnl

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
        abs_cost = abs(trade.get("total_cost", trade["premium"] * trade["quantity"]))
        trade["current_pnl_pct"] = (
            round(final_pnl / abs_cost * 100, 2) if abs_cost else 0.0
        )
        closed = dict(trade)

    # Restore opening cost to balance then credit/debit the realised P&L.
    # For debit trades: total_cost > 0 → refund + gain/loss.
    # For credit trades: total_cost < 0 → reclaim the initial credit receipt + gain/loss.
    total_cost = closed.get("total_cost", closed["premium"] * closed["quantity"])
    adjust_balance(total_cost + final_pnl)

    return closed
