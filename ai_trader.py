"""
ai_trader.py – Q-learning AI trading engine for RetailOptions demo trading.

The AI observes a discrete market state derived from:
  * price_trend  – recent price movement (up / flat / down)
  * vol_level    – historical volatility bucket (low / medium / high)
  * mc_signal    – Monte Carlo expected return direction (bullish / neutral / bearish)

It selects an action (option strategy or "no trade") via an epsilon-greedy
policy and receives a reward/penalty when trades are closed:
  * reward > 0  – profitable trade (proportional to P&L)
  * reward < 0  – losing trade (proportional to loss)
  * reward = 0  – no-trade action

Q-values are updated with standard Bellman one-step TD:
    Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]

The Q-table is persisted to disk after every learning update so that knowledge
accumulates across server restarts.  Experience replay is used to stabilise
learning: a buffer of past experiences is maintained, and a mini-batch is
replayed after each new experience.

Public API
----------
get_state(price_change_pct, hist_vol, mc_expected_return) → tuple
decide(state) → str   action id ("no_trade" | strategy_id)
record_reward(state, action, reward, next_state) → None
get_status() → dict
save() → None
load() → None
reset() → None
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Learning rate (α)
LEARNING_RATE: float = 0.15
#: Discount factor (γ) – discount future rewards
DISCOUNT: float = 0.9
#: Initial exploration rate (ε)
EPSILON_START: float = 0.4
#: Minimum exploration rate
EPSILON_MIN: float = 0.05
#: Multiplicative decay applied after each trade episode
EPSILON_DECAY: float = 0.95

# Reward scaling: rewards are normalised P&L divided by this scale factor
REWARD_SCALE: float = 100.0

# Reward clipping: cap rewards to prevent extreme Q-value updates
REWARD_CLIP: float = 5.0

# ---------------------------------------------------------------------------
# Experience replay configuration
# ---------------------------------------------------------------------------

#: Maximum number of experiences stored in the replay buffer
REPLAY_BUFFER_SIZE: int = 500
#: Number of experiences sampled per replay pass
REPLAY_BATCH_SIZE: int = 32
#: Replay a mini-batch after every N new experiences
REPLAY_FREQUENCY: int = 4

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

#: Path to the JSON file used to persist Q-table and performance data.
#: Override by setting the AI_STATE_PATH environment variable.
AI_STATE_PATH: str = os.environ.get("AI_STATE_PATH", "ai_trader_state.json")

# ---------------------------------------------------------------------------
# State / Action definitions
# ---------------------------------------------------------------------------

# State dimensions
PRICE_TREND_LABELS  = ("up", "flat", "down")   # 3 values
VOL_LEVEL_LABELS    = ("low", "medium", "high") # 3 values
MC_SIGNAL_LABELS    = ("bullish", "neutral", "bearish")  # 3 values

# 27 unique discrete states
_ALL_STATES: list[tuple] = [
    (pt, vl, mc)
    for pt in PRICE_TREND_LABELS
    for vl in VOL_LEVEL_LABELS
    for mc in MC_SIGNAL_LABELS
]
_STATE_INDEX: dict[tuple, int] = {s: i for i, s in enumerate(_ALL_STATES)}

# Actions
ACTIONS: list[str] = [
    "no_trade",
    "long_call",
    "long_put",
    "bull_call_spread",
    "bear_put_spread",
    "long_straddle",
]
_ACTION_INDEX: dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}

N_STATES  = len(_ALL_STATES)  # 27
N_ACTIONS = len(ACTIONS)      # 6

# ---------------------------------------------------------------------------
# Module-level mutable state (thread-safe)
# ---------------------------------------------------------------------------

_lock = threading.Lock()

# Q-table: shape (N_STATES, N_ACTIONS), initialised with domain-knowledge
# priors so the agent starts with sensible bias (bullish signals → calls,
# bearish signals → puts, high vol → straddle, neutral/flat → no trade).
#: Base Q-value assigned to all state-action pairs before domain priors are applied.
_BASE_Q_VALUE: float = 0.05


def _make_initial_q_table() -> np.ndarray:
    """Return a Q-table initialised with option-trading domain knowledge."""
    q = np.full((N_STATES, N_ACTIONS), _BASE_Q_VALUE)
    for state, si in _STATE_INDEX.items():
        price_trend, vol_level, mc_signal = state

        # --- directional bias ---
        bullish = (mc_signal == "bullish") or (price_trend == "up")
        bearish = (mc_signal == "bearish") or (price_trend == "down")
        strong_bullish = mc_signal == "bullish" and price_trend == "up"
        strong_bearish = mc_signal == "bearish" and price_trend == "down"

        if strong_bullish:
            q[si, _ACTION_INDEX["long_call"]]        = 0.60
            q[si, _ACTION_INDEX["bull_call_spread"]] = 0.50
        elif bullish:
            q[si, _ACTION_INDEX["long_call"]]        = 0.35
            q[si, _ACTION_INDEX["bull_call_spread"]] = 0.30

        if strong_bearish:
            q[si, _ACTION_INDEX["long_put"]]         = 0.60
            q[si, _ACTION_INDEX["bear_put_spread"]]  = 0.50
        elif bearish:
            q[si, _ACTION_INDEX["long_put"]]         = 0.35
            q[si, _ACTION_INDEX["bear_put_spread"]]  = 0.30

        # --- volatility bias ---
        if vol_level == "high":
            # High vol → straddle benefits from large moves in either direction
            q[si, _ACTION_INDEX["long_straddle"]] = 0.45
        elif vol_level == "low" and not strong_bullish and not strong_bearish:
            # Low-vol, no strong signal → prefer not to trade
            q[si, _ACTION_INDEX["no_trade"]] = 0.35

        # --- flat/neutral → prefer no trade ---
        if mc_signal == "neutral" and price_trend == "flat":
            q[si, _ACTION_INDEX["no_trade"]] = 0.40

        # --- conflicting signals → straddle for protection ---
        if (price_trend == "up" and mc_signal == "bearish") or \
           (price_trend == "down" and mc_signal == "bullish"):
            q[si, _ACTION_INDEX["long_straddle"]] = max(
                q[si, _ACTION_INDEX["long_straddle"]], 0.30
            )

    return q


_q_table: np.ndarray = _make_initial_q_table()

# Epsilon (current exploration rate)
_epsilon: float = EPSILON_START

# Performance tracking
_total_trades:    int   = 0
_winning_trades:  int   = 0
_total_reward:    float = 0.0
_reward_history:  list[float] = []  # reward per closed trade
_trade_log:       list[dict]  = []  # lightweight record of AI decisions

# Experience replay buffer: list of (state_idx, action_idx, reward, next_state_idx | None)
_replay_buffer: list[tuple] = []
_replay_counter: int = 0


# ---------------------------------------------------------------------------
# State discretisation helpers
# ---------------------------------------------------------------------------

def _price_trend_bucket(price_change_pct: float) -> str:
    """Convert a percentage price change into a trend label."""
    if price_change_pct >= 1.5:
        return "up"
    if price_change_pct <= -1.5:
        return "down"
    return "flat"


def _vol_bucket(hist_vol: float) -> str:
    """Convert annualised historical volatility into a bucket label."""
    if hist_vol < 0.20:
        return "low"
    if hist_vol < 0.40:
        return "medium"
    return "high"


def _mc_signal_bucket(mc_expected_return: float) -> str:
    """
    Convert the Monte Carlo expected 1-year log-return into a directional signal.

    Parameters
    ----------
    mc_expected_return : float
        Expected fractional return under the risk-neutral measure derived from
        simulated terminal prices: (E[S_T] - S_0) / S_0.
    """
    if mc_expected_return >= 0.02:
        return "bullish"
    if mc_expected_return <= -0.02:
        return "bearish"
    return "neutral"


def get_state(
    price_change_pct: float,
    hist_vol: float,
    mc_expected_return: float,
) -> tuple:
    """
    Construct and return the discrete market state tuple.

    Parameters
    ----------
    price_change_pct    : float  – % change in price today (e.g. 1.5 for +1.5%)
    hist_vol            : float  – annualised historical volatility (e.g. 0.25)
    mc_expected_return  : float  – MC fractional expected return (e.g. 0.03)
    """
    return (
        _price_trend_bucket(price_change_pct),
        _vol_bucket(hist_vol),
        _mc_signal_bucket(mc_expected_return),
    )


# ---------------------------------------------------------------------------
# Decision & learning
# ---------------------------------------------------------------------------

def decide(state: tuple) -> str:
    """
    Choose an action via epsilon-greedy policy.

    Returns the action id string (one of ``ACTIONS``).
    """
    global _epsilon

    with _lock:
        eps = _epsilon
        si  = _STATE_INDEX[state]
        q_row = _q_table[si].copy()

    if random.random() < eps:
        # Explore: random action
        return random.choice(ACTIONS)

    # Exploit: greedy action (break ties randomly)
    best_q = float(np.max(q_row))
    best_actions = [ACTIONS[i] for i, q in enumerate(q_row) if q == best_q]
    return random.choice(best_actions)


def record_reward(
    state: tuple,
    action: str,
    reward: float,
    next_state: Optional[tuple] = None,
) -> None:
    """
    Update the Q-table from a completed trade episode.

    The reward is clipped to ``[-REWARD_CLIP, +REWARD_CLIP]`` to prevent
    extreme Q-value updates.  After updating, an experience replay pass
    is performed every ``REPLAY_FREQUENCY`` calls to improve stability.
    The Q-table is then persisted to disk.

    Parameters
    ----------
    state       : tuple  – state when the trade was opened.
    action      : str    – action taken (strategy id or "no_trade").
    reward      : float  – normalised P&L reward (+profit / -loss).
    next_state  : tuple | None  – state at trade close (None → terminal).
    """
    global _epsilon, _total_trades, _winning_trades, _total_reward
    global _replay_buffer, _replay_counter

    # Clip reward to avoid extreme updates
    reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

    si = _STATE_INDEX[state]
    ai = _ACTION_INDEX[action]
    ns_i = _STATE_INDEX[next_state] if (next_state is not None and next_state in _STATE_INDEX) else None

    with _lock:
        # Bellman update for the current experience
        _bellman_update(si, ai, reward, ns_i)

        # Store in experience replay buffer
        _replay_buffer.append((si, ai, reward, ns_i))
        if len(_replay_buffer) > REPLAY_BUFFER_SIZE:
            _replay_buffer.pop(0)

        _replay_counter += 1

        # Perform a mini-batch replay pass periodically
        if _replay_counter % REPLAY_FREQUENCY == 0 and len(_replay_buffer) >= REPLAY_BATCH_SIZE:
            batch = random.sample(_replay_buffer, REPLAY_BATCH_SIZE)
            for b_si, b_ai, b_reward, b_ns_i in batch:
                _bellman_update(b_si, b_ai, b_reward, b_ns_i)

        # Decay epsilon after each episode
        _epsilon = max(EPSILON_MIN, _epsilon * EPSILON_DECAY)

        # Track performance
        if action != "no_trade":
            _total_trades += 1
            if reward > 0:
                _winning_trades += 1
            _total_reward += reward
            _reward_history.append(round(reward, 4))

    _trade_log.append({
        "state":     list(state),
        "action":    action,
        "reward":    round(reward, 4),
        "epsilon":   round(_epsilon, 4),
        "timestamp": time.time(),
    })

    # Persist updated state to disk (non-blocking; errors are non-fatal)
    save()


def _bellman_update(si: int, ai: int, reward: float, ns_i: Optional[int]) -> None:
    """Apply one Bellman TD update to the Q-table (must be called under _lock)."""
    old_q = _q_table[si, ai]
    max_next_q = float(np.max(_q_table[ns_i])) if ns_i is not None else 0.0
    _q_table[si, ai] = old_q + LEARNING_RATE * (reward + DISCOUNT * max_next_q - old_q)


def compute_mc_expected_return(
    spot: float,
    hist_vol: float,
    rate: float,
    expiry: float,
    num_paths: int = 2000,
    seed: Optional[int] = None,
) -> float:
    """
    Run a quick Monte Carlo simulation and return the expected fractional
    return (E[S_T] - S_0) / S_0.
    """
    rng = np.random.default_rng(seed)
    # Vectorised GBM
    dt = expiry / 252
    steps = int(252 * expiry)
    if steps < 1:
        steps = 1
    z = rng.standard_normal((num_paths, steps))
    log_returns = (rate - 0.5 * hist_vol ** 2) * dt + hist_vol * math.sqrt(dt) * z
    terminal = spot * np.exp(log_returns.sum(axis=1))
    expected_terminal = float(np.mean(terminal))
    return (expected_terminal - spot) / spot


# ---------------------------------------------------------------------------
# Status & reset
# ---------------------------------------------------------------------------

def get_status() -> dict:
    """Return a snapshot of AI state for the API status endpoint."""
    with _lock:
        eps        = _epsilon
        total      = _total_trades
        wins       = _winning_trades
        tot_rew    = _total_reward
        history    = list(_reward_history[-50:])  # last 50 episodes
        log        = list(_trade_log[-20:])        # last 20 decisions
        q_snapshot = _q_table.tolist()

    win_rate = wins / total if total else 0.0

    # Best action per state for display
    best_actions = {}
    for state, si in _STATE_INDEX.items():
        best_ai = int(np.argmax(_q_table[si]))
        best_actions[str(state)] = ACTIONS[best_ai]

    return {
        "epsilon":        round(eps, 4),
        "total_trades":   total,
        "winning_trades": wins,
        "win_rate":       round(win_rate, 4),
        "total_reward":   round(tot_rew, 4),
        "reward_history": history,
        "trade_log":      log,
        "q_table":        q_snapshot,
        "actions":        ACTIONS,
        "states":         [list(s) for s in _ALL_STATES],
        "best_actions":   best_actions,
        "learning_rate":  LEARNING_RATE,
        "discount":       DISCOUNT,
        "epsilon_min":    EPSILON_MIN,
        "epsilon_decay":  EPSILON_DECAY,
    }


def reset() -> None:
    """
    Reset Q-table and all performance counters to their initial state.

    The save file on disk is removed (if it exists) so that the reset
    state becomes the new persistent baseline.
    """
    global _q_table, _epsilon, _total_trades, _winning_trades
    global _total_reward, _reward_history, _trade_log
    global _replay_buffer, _replay_counter

    with _lock:
        _q_table          = _make_initial_q_table()
        _epsilon          = EPSILON_START
        _total_trades     = 0
        _winning_trades   = 0
        _total_reward     = 0.0
        _reward_history   = []
        _trade_log        = []
        _replay_buffer    = []
        _replay_counter   = 0

    # Remove saved state so a fresh state persists next time save() is called
    try:
        if os.path.exists(AI_STATE_PATH):
            os.remove(AI_STATE_PATH)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save() -> None:
    """
    Persist Q-table and performance data to ``AI_STATE_PATH`` (JSON).

    Called automatically after every ``record_reward()`` invocation.  Errors
    are silently ignored so that failures never interrupt trading operations.
    """
    with _lock:
        state = {
            "q_table":        _q_table.tolist(),
            "epsilon":        float(_epsilon),
            "total_trades":   int(_total_trades),
            "winning_trades": int(_winning_trades),
            "total_reward":   float(_total_reward),
            "reward_history": list(_reward_history[-500:]),  # cap to last 500
            "trade_log":      list(_trade_log[-200:]),        # cap to last 200
        }
    try:
        tmp_path = AI_STATE_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f)
        os.replace(tmp_path, AI_STATE_PATH)   # atomic on most filesystems
    except OSError:
        pass


def load() -> None:
    """
    Load Q-table and performance data from ``AI_STATE_PATH`` if it exists.

    Called automatically at module initialisation.  Errors are silently
    ignored so that a missing or corrupt file simply uses fresh state.
    """
    global _q_table, _epsilon, _total_trades, _winning_trades
    global _total_reward, _reward_history, _trade_log

    if not os.path.exists(AI_STATE_PATH):
        return

    try:
        with open(AI_STATE_PATH) as f:
            data = json.load(f)

        q = np.array(data["q_table"], dtype=float)
        if q.shape != (_q_table.shape):
            return   # shape mismatch – stale file; discard

        with _lock:
            _q_table[:] = q
            _epsilon        = float(data.get("epsilon", EPSILON_START))
            _total_trades   = int(data.get("total_trades", 0))
            _winning_trades = int(data.get("winning_trades", 0))
            _total_reward   = float(data.get("total_reward", 0.0))
            _reward_history = list(data.get("reward_history", []))
            _trade_log      = list(data.get("trade_log", []))
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        pass  # any error → continue with default fresh state


# ---------------------------------------------------------------------------
# Module initialisation – restore persisted state
# ---------------------------------------------------------------------------
load()
