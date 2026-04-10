"""
bot_runner.py – Autonomous AI trading bot (terminal / background process).

Runs two concurrent loops without needing the Flask web server:

  Trade loop  (every --trade-interval seconds, default 300 = 5 min)
      For every watched ticker:
      1. Fetch live quote and intraday change.
      2. Compute quantitative signals (RSI, momentum, SMAs).
      3. Run a quick Monte Carlo simulation to estimate expected return.
      4. Derive discrete market state and run risk assessment.
      5. Let the AI select an action (epsilon-greedy Q-table policy).
      6. If action != "no_trade" and balance allows, open a demo trade with
         strategy-aware TP/SL and position sizing.

  Monitor loop  (every --monitor-interval seconds, default 60 = 1 min)
      For every open position:
      - Refresh mark-to-market P&L.
      - Auto-close if TP or SL target is hit.
      - Feed the closing reward back to the Q-table so the agent learns.

The Q-table is persisted to ai_trader_state.json after every update, so
learning accumulates across restarts.  Use --dry-run to observe decisions
without opening any real positions.

Usage
-----
    python bot_runner.py
    python bot_runner.py --tickers AAPL TSLA MSFT --trade-interval 600
    python bot_runner.py --dry-run --tickers SPY --log-file bot.log
    python bot_runner.py --help
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Lazy import guard – make sure we can find the project modules even when
# running the script from outside the repo root.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ai_trader
import market_data
from strategies import compute_strategy_profile


# ---------------------------------------------------------------------------
# Defaults (all overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULT_TICKERS:          list[str] = ["AAPL", "TSLA", "MSFT"]
DEFAULT_TRADE_INTERVAL:   int       = 300    # seconds between trade evaluations
DEFAULT_MONITOR_INTERVAL: int       = 60     # seconds between portfolio checks
DEFAULT_EXPIRY:           float     = 0.25   # 3-month options (years)
DEFAULT_RATE:             float     = 0.05   # 5% risk-free rate
DEFAULT_MC_PATHS:         int       = 2000   # paths for expected-return MC
DEFAULT_MIN_BALANCE:      float     = 50.0   # skip trade if balance < this


# ---------------------------------------------------------------------------
# TP/SL computation (mirrors _compute_strategy_tpsl in app.py)
# ---------------------------------------------------------------------------

def _compute_strategy_tpsl(
    action: str,
    premium: float,
    max_profit: Optional[float],
    max_loss: Optional[float],
) -> tuple[float, float]:
    """
    Return (take_profit, stop_loss) P&L dollar targets calibrated to the
    strategy's actual max-profit / max-loss characteristics.

    Mirrors the identical function in app.py so the bot uses the same
    risk-management logic as the web interface.
    """
    abs_prem     = abs(premium)
    abs_max_loss = abs(max_loss) if max_loss is not None else abs_prem

    if action in ("long_call", "long_put", "long_straddle", "long_strangle"):
        tp = round(abs_prem * 2.0, 2)
        sl = round(-abs_prem * 0.5, 2)
    elif action in ("bull_call_spread", "bear_put_spread"):
        mp = max_profit if (max_profit is not None and max_profit > 0) else abs_prem * 1.5
        tp = round(mp * 0.75, 2)
        sl = round(-abs_prem * 0.7, 2)
    elif action in ("covered_call", "protective_put"):
        tp = round(abs_prem * 1.5, 2)
        sl = round(-abs_prem * 1.5, 2)
    elif action in ("short_call", "short_put"):
        tp = round(abs_prem * 0.75, 2)
        sl = round(-abs_prem * 2.0, 2)
    elif action in ("bull_put_spread", "bear_call_spread"):
        tp = round(abs_prem * 0.75, 2)
        sl = round(-abs_max_loss * 0.5, 2)
    elif action == "iron_condor":
        tp = round(abs_prem * 0.6, 2)
        sl = round(-abs_max_loss * 0.5, 2)
    else:
        tp = round(abs_prem * 2.0, 2)
        sl = round(-abs_prem, 2)

    sl = min(sl, -1.0)
    tp = max(tp, 1.0)
    return tp, sl


# ---------------------------------------------------------------------------
# Strike selection (mirrors api_ai_trade logic in app.py)
# ---------------------------------------------------------------------------

def _compute_strikes(action: str, spot: float) -> dict:
    """Return the appropriate strike keys for *action* based on current *spot*."""
    strikes: dict = {}
    if action == "long_call":
        strikes["strike"] = round(spot * 1.02, 2)
    elif action == "long_straddle":
        strikes["strike"] = round(spot, 2)
    elif action == "long_put":
        strikes["strike"] = round(spot * 0.98, 2)
    elif action == "bull_call_spread":
        strikes["strike_low"]  = round(spot * 1.00, 2)
        strikes["strike_high"] = round(spot * 1.08, 2)
    elif action == "bear_put_spread":
        strikes["strike_low"]  = round(spot * 0.92, 2)
        strikes["strike_high"] = round(spot * 1.00, 2)
    elif action == "short_call":
        strikes["strike"] = round(spot * 1.05, 2)
    elif action == "short_put":
        strikes["strike"] = round(spot * 0.95, 2)
    elif action == "bull_put_spread":
        strikes["strike_low"]  = round(spot * 0.90, 2)
        strikes["strike_high"] = round(spot * 0.97, 2)
    elif action == "bear_call_spread":
        strikes["strike_low"]  = round(spot * 1.03, 2)
        strikes["strike_high"] = round(spot * 1.10, 2)
    elif action == "iron_condor":
        strikes["strike_put"]  = round(spot * 0.95, 2)
        strikes["strike_call"] = round(spot * 1.05, 2)
        strikes["strike_low"]  = round(spot * 0.90, 2)
        strikes["strike_high"] = round(spot * 1.10, 2)
    return strikes


# ---------------------------------------------------------------------------
# Core trade step
# ---------------------------------------------------------------------------

def run_trade_step(
    ticker: str,
    expiry: float,
    rate: float,
    dry_run: bool,
    log: logging.Logger,
) -> None:
    """
    Execute one complete AI trading evaluation for *ticker*.

    Fetches market data, derives state, selects action, and (if not dry-run)
    opens a demo trade with TP/SL and optimal position size.
    """
    try:
        log.info("[%s] ── Trade evaluation ──", ticker)

        # 1. Live quote
        quote    = market_data.get_quote(ticker)
        spot     = quote["price"]
        hist_vol = quote["hist_vol"]
        log.info("[%s] Price $%.2f  |  Hist vol %.1f%%", ticker, spot, hist_vol * 100)

        # 2. Intraday change (real-time price-trend signal)
        intraday_pct = market_data.get_intraday_change(ticker)

        # 3. Technical indicators
        quant = market_data.get_technical_indicators(ticker)
        rsi   = quant["rsi"]
        log.info(
            "[%s] RSI %.1f  |  Momentum %.2f%%  |  Above SMA20=%s SMA50=%s",
            ticker, rsi, quant["momentum"], quant["above_sma20"], quant["above_sma50"],
        )

        # 4. Monte Carlo expected return
        mc_return = ai_trader.compute_mc_expected_return(
            spot=spot,
            hist_vol=hist_vol,
            rate=rate,
            expiry=expiry,
            num_paths=DEFAULT_MC_PATHS,
        )
        log.info("[%s] MC expected return: %+.2f%%", ticker, mc_return * 100)

        # 5. Discrete market state
        state = ai_trader.get_state(
            price_change_pct=intraday_pct,
            hist_vol=hist_vol,
            mc_expected_return=mc_return,
            rsi=rsi,
        )
        log.info("[%s] State: %s", ticker, state)

        # 6. Risk assessment + action
        action = ai_trader.decide(state)
        risk   = ai_trader.assess_risk(
            spot=spot,
            hist_vol=hist_vol,
            strategy_id=action if action != "no_trade" else "long_call",
        )
        log.info(
            "[%s] Action: %-20s  |  Vol regime: %s  |  Recommended contracts: %d",
            ticker, action, risk["vol_regime"], risk["recommended_contracts"],
        )

        if action == "no_trade":
            log.info("[%s] No trade – holding.", ticker)
            return

        # 7. Balance check
        balance = market_data.get_balance()
        if balance < DEFAULT_MIN_BALANCE:
            log.warning(
                "[%s] Skipped: insufficient demo balance ($%.2f).", ticker, balance
            )
            return

        # 8. Position sizing & TP/SL
        strikes   = _compute_strikes(action, spot)
        spot_arr  = np.linspace(max(spot * 0.5, 0.01), spot * 1.5, 200)
        profile_params: dict = {
            "spot": spot, "rate": rate, "vol": hist_vol, "expiry": expiry,
            **strikes,
        }
        try:
            profile = compute_strategy_profile(action, profile_params, spot_arr)
            take_profit, stop_loss = _compute_strategy_tpsl(
                action, profile["premium"], profile["max_profit"], profile["max_loss"]
            )
        except Exception:
            fb_risk     = round(balance * 0.01, 2)
            take_profit = round(fb_risk * 2.0, 2)
            stop_loss   = -fb_risk

        quantity = max(1, risk["recommended_contracts"])
        log.info(
            "[%s] TP=$%.2f  SL=$%.2f  Qty=%d",
            ticker, take_profit, stop_loss, quantity,
        )

        if dry_run:
            log.info("[%s] DRY-RUN – trade would open: %s (qty %d).", ticker, action, quantity)
            return

        # 9. Open trade
        trade = market_data.open_trade(
            ticker=ticker,
            strategy_id=action,
            spot=spot,
            expiry=expiry,
            rate=rate,
            hist_vol=hist_vol,
            quantity=quantity,
            take_profit=take_profit,
            stop_loss=stop_loss,
            **{k: v for k, v in strikes.items()},
        )

        # Store AI metadata so the monitor loop can learn from the close
        with market_data._portfolio_lock:
            stored = market_data._portfolio.get(trade["id"])
            if stored is not None:
                stored["ai_state"]  = list(state)
                stored["ai_action"] = action

        log.info(
            "[%s] Opened trade %s  |  Strategy: %s  |  Premium $%.4f  |  Balance $%.2f",
            ticker, trade["id"][:8], action, trade["premium"],
            market_data.get_balance(),
        )

    except Exception as exc:
        log.error("[%s] Trade step error: %s", ticker, exc)


# ---------------------------------------------------------------------------
# Portfolio monitor step
# ---------------------------------------------------------------------------

def run_monitor_step(log: logging.Logger) -> None:
    """
    Refresh all open positions, auto-close TP/SL hits, and feed reward to AI.

    market_data.get_portfolio already triggers TP/SL closes internally; this
    function just needs to learn from those closed trades.
    """
    try:
        portfolio = market_data.get_portfolio(refresh_prices=True)
        open_trades  = [t for t in portfolio if t["status"] == "open"]
        closed_trades = [t for t in portfolio if t["status"] == "closed"]

        if open_trades:
            log.debug("Open positions: %d", len(open_trades))
            for t in open_trades:
                log.debug(
                    "  %s [%s] %s  P&L $%.2f (%.1f%%)",
                    t["id"][:8], t["ticker"], t["strategy_id"],
                    t.get("current_pnl", 0.0), t.get("current_pnl_pct", 0.0),
                )

        # Detect newly-closed AI trades and record rewards
        for t in closed_trades:
            ai_state_raw = t.get("ai_state")
            ai_action    = t.get("ai_action")
            # Only process trades that haven't had their reward recorded yet.
            # We mark processed trades by removing ai_state from the in-memory
            # record after feeding the reward.
            if ai_state_raw is None or ai_action is None:
                continue

            # Check whether this trade is still marked in _portfolio (not yet learned)
            with market_data._portfolio_lock:
                stored = market_data._portfolio.get(t["id"])
                if stored is None or stored.get("_reward_recorded"):
                    continue
                stored["_reward_recorded"] = True  # mark as processed

            pnl    = t.get("current_pnl", 0.0)
            reward = ai_trader.compute_reward(pnl)
            state  = tuple(ai_state_raw)

            # Compute next state from a fresh quote (best-effort)
            next_state = None
            try:
                q      = market_data.get_quote(t["ticker"])
                mc_ret = ai_trader.compute_mc_expected_return(
                    spot=q["price"], hist_vol=q["hist_vol"],
                    rate=t.get("rate", 0.05),
                    expiry=t.get("expiry_years", 0.25),
                    num_paths=500,
                )
                rsi_next = market_data.get_technical_indicators(t["ticker"]).get("rsi", 50.0)
                next_state = ai_trader.get_state(
                    price_change_pct=q["change_pct"],
                    hist_vol=q["hist_vol"],
                    mc_expected_return=mc_ret,
                    rsi=rsi_next,
                )
            except Exception:
                pass

            ai_trader.record_reward(state, ai_action, reward, next_state)

            outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "FLAT")
            log.info(
                "[%s] Closed trade %s  |  %s  |  P&L $%.2f  |  Reward %.4f  |  "
                "Total trades: %d  Win rate: %.1f%%",
                t["ticker"], t["id"][:8], outcome, pnl, reward,
                ai_trader._total_trades,
                ai_trader._winning_trades / max(ai_trader._total_trades, 1) * 100,
            )

    except Exception as exc:
        log.error("Monitor step error: %s", exc)


# ---------------------------------------------------------------------------
# Background thread loops
# ---------------------------------------------------------------------------

def _trade_loop(
    tickers: list[str],
    trade_interval: int,
    expiry: float,
    rate: float,
    dry_run: bool,
    stop_event: threading.Event,
    log: logging.Logger,
) -> None:
    log.info("Trade loop started (interval=%ds, tickers=%s)", trade_interval, tickers)
    while not stop_event.is_set():
        for ticker in tickers:
            if stop_event.is_set():
                break
            run_trade_step(ticker, expiry, rate, dry_run, log)
        # Sleep in short increments so we respond quickly to stop_event
        for _ in range(trade_interval):
            if stop_event.is_set():
                break
            time.sleep(1)
    log.info("Trade loop stopped.")


def _monitor_loop(
    monitor_interval: int,
    stop_event: threading.Event,
    log: logging.Logger,
) -> None:
    log.info("Monitor loop started (interval=%ds)", monitor_interval)
    while not stop_event.is_set():
        run_monitor_step(log)
        for _ in range(monitor_interval):
            if stop_event.is_set():
                break
            time.sleep(1)
    log.info("Monitor loop stopped.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Autonomous AI trading bot – runs in the terminal / background.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        metavar="TICKER",
        help="Space-separated list of stock tickers to watch.",
    )
    p.add_argument(
        "--trade-interval",
        type=int,
        default=DEFAULT_TRADE_INTERVAL,
        metavar="SECONDS",
        help="How often (seconds) the trade loop evaluates each ticker.",
    )
    p.add_argument(
        "--monitor-interval",
        type=int,
        default=DEFAULT_MONITOR_INTERVAL,
        metavar="SECONDS",
        help="How often (seconds) the monitor loop checks open positions.",
    )
    p.add_argument(
        "--expiry",
        type=float,
        default=DEFAULT_EXPIRY,
        metavar="YEARS",
        help="Option expiry in years (e.g. 0.25 = 3 months).",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE,
        metavar="FRACTION",
        help="Risk-free rate as a fraction (e.g. 0.05 = 5%%).",
    )
    p.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Optional path to a log file.  Logs go to stdout if omitted.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate trades and log decisions without opening any positions.",
    )
    p.add_argument(
        "--reset-ai",
        action="store_true",
        help="Reset the Q-table and all AI performance counters before starting.",
    )
    return p


def _setup_logging(log_file: Optional[str], log_level: str) -> logging.Logger:
    level   = getattr(logging, log_level.upper(), logging.INFO)
    fmt     = "%(asctime)s %(levelname)-8s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    return logging.getLogger("bot")


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    log = _setup_logging(args.log_file, args.log_level)

    log.info("=" * 60)
    log.info("RetailOptions Autonomous Trading Bot")
    log.info("=" * 60)
    log.info("Tickers:          %s", args.tickers)
    log.info("Trade interval:   %d s", args.trade_interval)
    log.info("Monitor interval: %d s", args.monitor_interval)
    log.info("Expiry:           %.2f years", args.expiry)
    log.info("Risk-free rate:   %.1f%%", args.rate * 100)
    log.info("Dry-run:          %s", args.dry_run)
    log.info("AI state path:    %s", ai_trader.AI_STATE_PATH)
    log.info("=" * 60)

    # Load persisted AI state (Q-table, epsilon, trade history)
    if args.reset_ai:
        ai_trader.reset()
        log.info("AI state reset - Q-table initialized with domain priors.")
    else:
        ai_trader.load()
        status = ai_trader.get_status()
        log.info(
            "AI loaded – total trades: %d  win rate: %.1f%%  epsilon: %.4f",
            status["total_trades"],
            status["win_rate"] * 100,
            status["epsilon"],
        )

    # Stop event shared between threads and signal handler
    stop_event = threading.Event()

    def _shutdown(signum, frame):  # noqa: ARG001
        log.info("Shutdown signal received - stopping loops...")
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start background threads
    trade_thread = threading.Thread(
        target=_trade_loop,
        args=(
            [t.upper().strip() for t in args.tickers],
            args.trade_interval,
            args.expiry,
            args.rate,
            args.dry_run,
            stop_event,
            log,
        ),
        daemon=True,
        name="trade-loop",
    )
    monitor_thread = threading.Thread(
        target=_monitor_loop,
        args=(args.monitor_interval, stop_event, log),
        daemon=True,
        name="monitor-loop",
    )

    monitor_thread.start()
    trade_thread.start()

    log.info("Bot running.  Press Ctrl+C to stop.")

    # Keep main thread alive until stop_event is set
    while not stop_event.is_set():
        time.sleep(1)

    trade_thread.join(timeout=10)
    monitor_thread.join(timeout=10)

    # Final AI snapshot
    status = ai_trader.get_status()
    log.info("=" * 60)
    log.info("Bot stopped.")
    log.info(
        "Final AI stats – trades: %d  wins: %d  win rate: %.1f%%  "
        "total reward: %.4f  epsilon: %.4f",
        status["total_trades"],
        status["winning_trades"],
        status["win_rate"] * 100,
        status["total_reward"],
        status["epsilon"],
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
