"""
Unit tests for the Monte Carlo Option Pricing Simulator.
"""

import math
import sys
import os

import numpy as np
import pytest
import matplotlib.pyplot as plt

# Ensure the repo root is on the path so we can import monte_carlo directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from monte_carlo import (
    MonteCarloOptionPricer,
    black_scholes_price,
    simulate_stock_paths,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 0
DEFAULT_PARAMS = dict(
    spot=100.0,
    strike=100.0,
    risk_free_rate=0.05,
    volatility=0.20,
    time_to_expiry=1.0,
)

# Parameters accepted by simulate_stock_paths (no 'strike')
PATH_PARAMS = {k: v for k, v in DEFAULT_PARAMS.items() if k != "strike"}


# ---------------------------------------------------------------------------
# black_scholes_price
# ---------------------------------------------------------------------------


class TestBlackScholesPrice:
    """Analytical Black-Scholes prices against known values."""

    def test_call_at_the_money(self):
        # ATM call: well-known BS result ~10.45 for S=K=100, r=5%, σ=20%, T=1
        price = black_scholes_price(**DEFAULT_PARAMS, option_type="call")
        assert abs(price - 10.45) < 0.1

    def test_put_at_the_money(self):
        # ATM put via put-call parity: P = C - S + K*exp(-rT)
        call = black_scholes_price(**DEFAULT_PARAMS, option_type="call")
        expected_put = (
            call
            - DEFAULT_PARAMS["spot"]
            + DEFAULT_PARAMS["strike"]
            * math.exp(-DEFAULT_PARAMS["risk_free_rate"] * DEFAULT_PARAMS["time_to_expiry"])
        )
        put = black_scholes_price(**DEFAULT_PARAMS, option_type="put")
        assert abs(put - expected_put) < 1e-8

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT) (put-call parity)."""
        call = black_scholes_price(**DEFAULT_PARAMS, option_type="call")
        put = black_scholes_price(**DEFAULT_PARAMS, option_type="put")
        lhs = call - put
        rhs = DEFAULT_PARAMS["spot"] - DEFAULT_PARAMS["strike"] * math.exp(
            -DEFAULT_PARAMS["risk_free_rate"] * DEFAULT_PARAMS["time_to_expiry"]
        )
        assert abs(lhs - rhs) < 1e-8

    def test_deep_itm_call_close_to_intrinsic(self):
        """A very deep in-the-money call approaches S - K*exp(-rT)."""
        price = black_scholes_price(
            spot=200.0, strike=100.0, risk_free_rate=0.05,
            volatility=0.20, time_to_expiry=1.0, option_type="call",
        )
        intrinsic = 200.0 - 100.0 * math.exp(-0.05 * 1.0)
        assert abs(price - intrinsic) < 1.0

    def test_zero_expiry_call(self):
        price = black_scholes_price(
            spot=110.0, strike=100.0, risk_free_rate=0.05,
            volatility=0.20, time_to_expiry=0.0, option_type="call",
        )
        assert price == pytest.approx(10.0)

    def test_zero_expiry_put_otm(self):
        price = black_scholes_price(
            spot=110.0, strike=100.0, risk_free_rate=0.05,
            volatility=0.20, time_to_expiry=0.0, option_type="put",
        )
        assert price == pytest.approx(0.0)

    def test_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            black_scholes_price(**DEFAULT_PARAMS, option_type="straddle")


# ---------------------------------------------------------------------------
# simulate_stock_paths
# ---------------------------------------------------------------------------


class TestSimulateStockPaths:
    def test_shape(self):
        paths = simulate_stock_paths(**PATH_PARAMS, num_paths=2000, num_steps=50, random_seed=SEED)
        assert paths.shape == (2000, 51)

    def test_starts_at_spot(self):
        spot = 150.0
        paths = simulate_stock_paths(
            spot=spot, risk_free_rate=0.05, volatility=0.20,
            time_to_expiry=1.0, num_paths=1000, num_steps=10, random_seed=SEED,
        )
        np.testing.assert_array_equal(paths[:, 0], spot)

    def test_positive_prices(self):
        paths = simulate_stock_paths(**PATH_PARAMS, num_paths=1000, num_steps=50, random_seed=SEED)
        assert np.all(paths > 0)

    def test_reproducible_with_seed(self):
        p1 = simulate_stock_paths(**PATH_PARAMS, num_paths=1000, num_steps=20, random_seed=99)
        p2 = simulate_stock_paths(**PATH_PARAMS, num_paths=1000, num_steps=20, random_seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        p1 = simulate_stock_paths(**PATH_PARAMS, num_paths=1000, num_steps=20, random_seed=1)
        p2 = simulate_stock_paths(**PATH_PARAMS, num_paths=1000, num_steps=20, random_seed=2)
        assert not np.array_equal(p1, p2)

    def test_invalid_num_paths(self):
        with pytest.raises(ValueError, match="num_paths"):
            simulate_stock_paths(**PATH_PARAMS, num_paths=0)

    def test_invalid_num_steps(self):
        with pytest.raises(ValueError, match="num_steps"):
            simulate_stock_paths(**PATH_PARAMS, num_steps=0)

    def test_invalid_time_to_expiry(self):
        with pytest.raises(ValueError, match="time_to_expiry"):
            simulate_stock_paths(
                spot=100.0, risk_free_rate=0.05, volatility=0.20, time_to_expiry=-1.0
            )

    def test_mean_terminal_price_close_to_forward(self):
        """E[S_T] = S0 * exp(r * T) under the risk-neutral measure."""
        params = dict(
            spot=100.0, risk_free_rate=0.05, volatility=0.30, time_to_expiry=1.0
        )
        paths = simulate_stock_paths(**params, num_paths=50_000, num_steps=50, random_seed=SEED)
        mean_terminal = np.mean(paths[:, -1])
        expected = params["spot"] * math.exp(params["risk_free_rate"] * params["time_to_expiry"])
        # Allow 1% tolerance given large path count.
        assert abs(mean_terminal - expected) / expected < 0.01


# ---------------------------------------------------------------------------
# MonteCarloOptionPricer
# ---------------------------------------------------------------------------


class TestMonteCarloOptionPricer:
    """Integration tests: MC prices should be close to Black-Scholes prices."""

    @pytest.fixture(scope="class")
    def pricer(self):
        return MonteCarloOptionPricer(**DEFAULT_PARAMS, num_paths=50_000, random_seed=SEED)

    def test_paths_shape(self, pricer):
        assert pricer.paths.shape == (50_000, 253)

    def test_terminal_prices_shape(self, pricer):
        assert pricer.terminal_prices.shape == (50_000,)

    def test_call_price_close_to_bs(self, pricer):
        result = pricer.price("call")
        bs = result["black_scholes_price"]
        mc = result["price"]
        # MC should be within 2 % of BS price
        assert abs(mc - bs) / bs < 0.02

    def test_put_price_close_to_bs(self, pricer):
        result = pricer.price("put")
        bs = result["black_scholes_price"]
        mc = result["price"]
        assert abs(mc - bs) / bs < 0.02

    def test_result_keys(self, pricer):
        result = pricer.price("call")
        expected_keys = {
            "price",
            "std_error",
            "confidence_interval_95",
            "black_scholes_price",
            "num_paths",
            "option_type",
            "probability_in_the_money",
        }
        assert set(result.keys()) == expected_keys

    def test_std_error_positive(self, pricer):
        result = pricer.price("call")
        assert result["std_error"] > 0

    def test_confidence_interval_contains_bs_price(self, pricer):
        result = pricer.price("call")
        lo, hi = result["confidence_interval_95"]
        bs = result["black_scholes_price"]
        # 95 % CI should contain the BS analytical price
        assert lo <= bs <= hi

    def test_probability_in_the_money_between_0_and_1(self, pricer):
        for opt_type in ("call", "put"):
            result = pricer.price(opt_type)
            p = result["probability_in_the_money"]
            assert 0.0 <= p <= 1.0

    def test_call_put_probabilities_sum_to_1(self, pricer):
        """P(call ITM) + P(put ITM) should equal 1 (no ties with continuous dist)."""
        call_p = pricer.price("call")["probability_in_the_money"]
        put_p = pricer.price("put")["probability_in_the_money"]
        assert abs(call_p + put_p - 1.0) < 1e-6

    def test_put_call_mc_parity(self, pricer):
        """MC put-call parity: C - P ≈ S - K*exp(-rT)."""
        call_price = pricer.price("call")["price"]
        put_price = pricer.price("put")["price"]
        lhs = call_price - put_price
        rhs = pricer.spot - pricer.strike * math.exp(
            -pricer.risk_free_rate * pricer.time_to_expiry
        )
        assert abs(lhs - rhs) < 0.20  # Small tolerance for MC noise

    def test_num_paths_recorded(self, pricer):
        result = pricer.price("call")
        assert result["num_paths"] == 50_000

    def test_invalid_option_type(self, pricer):
        with pytest.raises(ValueError, match="option_type"):
            pricer.price("binary")

    def test_invalid_num_paths_raises(self):
        with pytest.raises(ValueError, match="num_paths"):
            MonteCarloOptionPricer(**DEFAULT_PARAMS, num_paths=500)

    def test_invalid_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            MonteCarloOptionPricer(**{**DEFAULT_PARAMS, "spot": -1.0}, num_paths=1000)

    def test_invalid_strike_raises(self):
        with pytest.raises(ValueError, match="strike"):
            MonteCarloOptionPricer(**{**DEFAULT_PARAMS, "strike": 0.0}, num_paths=1000)

    def test_invalid_volatility_raises(self):
        with pytest.raises(ValueError, match="volatility"):
            MonteCarloOptionPricer(**{**DEFAULT_PARAMS, "volatility": 0.0}, num_paths=1000)

    def test_itm_call_price_greater_than_otm(self):
        """Deep ITM call should cost more than an OTM call, all else equal."""
        itm = MonteCarloOptionPricer(
            spot=120.0, strike=100.0, **{k: v for k, v in DEFAULT_PARAMS.items()
                                         if k not in ("spot", "strike")},
            num_paths=10_000, random_seed=SEED,
        ).price("call")["price"]
        otm = MonteCarloOptionPricer(
            spot=80.0, strike=100.0, **{k: v for k, v in DEFAULT_PARAMS.items()
                                        if k not in ("spot", "strike")},
            num_paths=10_000, random_seed=SEED,
        ).price("call")["price"]
        assert itm > otm

    def test_plot_simulation_returns_axis_with_expected_labels(self):
        pricer = MonteCarloOptionPricer(**DEFAULT_PARAMS, num_paths=1000, random_seed=SEED)
        fig, ax = pricer.plot_simulation(num_paths_to_plot=5, show=False)
        assert ax.get_title() == "Monte Carlo Stock Price Simulation"
        assert ax.get_xlabel() == "Time (years)"
        assert ax.get_ylabel() == "Stock Price"
        assert len(ax.lines) == 6  # 5 paths + strike reference line
        plt.close(fig)

    def test_plot_simulation_rejects_invalid_path_count(self):
        pricer = MonteCarloOptionPricer(**DEFAULT_PARAMS, num_paths=1000, random_seed=SEED)
        with pytest.raises(ValueError, match="num_paths_to_plot"):
            pricer.plot_simulation(num_paths_to_plot=0, show=False)
