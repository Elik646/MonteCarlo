"""
Unit tests for the strategies module and Flask web app API endpoints.
"""

import json
import sys
import os
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies import compute_strategy_profile, STRATEGY_NAMES, STRATEGY_COLORS
from monte_carlo import black_scholes_greeks
from app import app as flask_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SPOT = 100.0
BASE_PARAMS = {"spot": SPOT, "rate": 0.05, "vol": 0.20, "expiry": 1.0}
SPOT_PRICES = np.linspace(50, 200, 300)


@pytest.fixture(scope="module")
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# strategies.compute_strategy_profile
# ---------------------------------------------------------------------------


class TestComputeStrategyProfile:
    """Verify P&L profiles and key metrics for every supported strategy."""

    def _profile(self, strategy_id, extra=None):
        params = {**BASE_PARAMS, **(extra or {})}
        return compute_strategy_profile(strategy_id, params, SPOT_PRICES)

    def test_long_call_metadata(self):
        p = self._profile("long_call", {"strike": 105})
        assert p["id"] == "long_call"
        assert p["name"] == STRATEGY_NAMES["long_call"]
        assert p["color"] == STRATEGY_COLORS["long_call"]

    def test_long_call_pnl_length(self):
        p = self._profile("long_call", {"strike": 105})
        assert len(p["pnl"]) == len(SPOT_PRICES)

    def test_long_call_max_loss_is_premium(self):
        p = self._profile("long_call", {"strike": 105})
        assert abs(p["max_loss"] - (-p["premium"])) < 0.01

    def test_long_call_max_profit_is_unbounded(self):
        p = self._profile("long_call", {"strike": 105})
        # unlimited upside => max_profit is very large
        assert p["max_profit"] is None or p["max_profit"] > 50

    def test_long_call_has_one_breakeven(self):
        p = self._profile("long_call", {"strike": 105})
        assert len(p["breakevens"]) == 1

    def test_long_call_breakeven_above_strike(self):
        p = self._profile("long_call", {"strike": 105})
        assert p["breakevens"][0] > 105

    def test_long_put_max_loss_is_premium(self):
        p = self._profile("long_put", {"strike": 95})
        assert abs(p["max_loss"] - (-p["premium"])) < 0.01

    def test_long_put_breakeven_below_strike(self):
        p = self._profile("long_put", {"strike": 95})
        assert len(p["breakevens"]) == 1
        assert p["breakevens"][0] < 95

    def test_bull_call_spread_has_capped_profit(self):
        p = self._profile("bull_call_spread", {"strike_low": 100, "strike_high": 110})
        assert p["max_profit"] is not None
        # Max profit ≤ width of spread (K2 − K1)
        assert p["max_profit"] <= 10.0 + 0.01

    def test_bull_call_spread_max_loss_lte_zero(self):
        p = self._profile("bull_call_spread", {"strike_low": 100, "strike_high": 110})
        assert p["max_loss"] < 0

    def test_bear_put_spread_max_profit_positive(self):
        p = self._profile("bear_put_spread", {"strike_low": 90, "strike_high": 100})
        assert p["max_profit"] > 0

    def test_long_straddle_two_breakevens(self):
        p = self._profile("long_straddle", {"strike": 100})
        assert len(p["breakevens"]) == 2

    def test_long_straddle_breakevens_symmetric(self):
        p = self._profile("long_straddle", {"strike": 100})
        be_lo, be_hi = sorted(p["breakevens"])
        # Roughly symmetric around strike
        assert abs((be_hi - 100) - (100 - be_lo)) < 5

    def test_long_strangle_two_breakevens(self):
        p = self._profile("long_strangle", {"strike_call": 110, "strike_put": 90})
        assert len(p["breakevens"]) == 2

    def test_covered_call_premium_is_negative(self):
        # Credit strategy: net_premium < 0
        p = self._profile("covered_call", {"strike": 110})
        assert p["premium"] < 0

    def test_protective_put_premium_is_positive(self):
        p = self._profile("protective_put", {"strike": 95})
        assert p["premium"] > 0

    def test_short_call_premium_is_negative(self):
        """Short call receives premium (negative premium = credit)."""
        p = self._profile("short_call", {"strike": 105})
        assert p["premium"] < 0

    def test_short_call_max_profit_is_premium_received(self):
        p = self._profile("short_call", {"strike": 105})
        assert p["max_profit"] == pytest.approx(-p["premium"], abs=0.01)

    def test_short_put_premium_is_negative(self):
        """Short put receives premium (negative premium = credit)."""
        p = self._profile("short_put", {"strike": 95})
        assert p["premium"] < 0

    def test_short_put_max_profit_is_premium_received(self):
        p = self._profile("short_put", {"strike": 95})
        assert p["max_profit"] == pytest.approx(-p["premium"], abs=0.01)

    def test_bull_put_spread_max_profit_positive(self):
        """Bull put spread collects net credit (positive max profit)."""
        p = self._profile("bull_put_spread", {"strike_low": 90, "strike_high": 97})
        assert p["max_profit"] > 0
        assert p["premium"] < 0  # credit received

    def test_bear_call_spread_max_profit_positive(self):
        """Bear call spread collects net credit (positive max profit)."""
        p = self._profile("bear_call_spread", {"strike_low": 103, "strike_high": 110})
        assert p["max_profit"] > 0
        assert p["premium"] < 0  # credit received

    def test_iron_condor_max_profit_positive(self):
        """Iron condor collects net credit = max profit at entry."""
        p = self._profile("iron_condor", {
            "strike_low": 90, "strike_put": 95,
            "strike_call": 105, "strike_high": 110
        })
        assert p["max_profit"] > 0
        assert p["premium"] < 0  # credit received

    def test_iron_condor_profit_at_spot(self):
        """Iron condor should show max profit when stock stays near spot."""
        p = self._profile("iron_condor", {
            "strike_low": 90, "strike_put": 95,
            "strike_call": 105, "strike_high": 110
        })
        # At-the-money (spot=100) the payoff should equal the net credit
        at_spot_idx = np.argmin(np.abs(SPOT_PRICES - 100.0))
        assert p["pnl"][at_spot_idx] == pytest.approx(p["max_profit"], abs=0.01)

    def test_unknown_strategy_id_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            compute_strategy_profile("butterfly_spread", BASE_PARAMS, SPOT_PRICES)


# ---------------------------------------------------------------------------
# Flask /api/price
# ---------------------------------------------------------------------------

PRICE_BODY = {
    "spot": 100, "strike": 105, "rate": 0.05,
    "vol": 0.20, "expiry": 1.0, "num_paths": 2000, "seed": 42,
}


class TestApiPrice:
    def test_status_200(self, client):
        r = client.post("/api/price", json=PRICE_BODY)
        assert r.status_code == 200

    def test_response_has_call_and_put(self, client):
        d = client.post("/api/price", json=PRICE_BODY).get_json()
        assert "call" in d and "put" in d

    def test_call_keys(self, client):
        call = client.post("/api/price", json=PRICE_BODY).get_json()["call"]
        for key in ("mc_price", "bs_price", "std_error", "ci_lower", "ci_upper", "prob_itm"):
            assert key in call

    def test_histogram_shape(self, client):
        d = client.post("/api/price", json=PRICE_BODY).get_json()
        assert len(d["histogram"]["bins"]) == len(d["histogram"]["counts"])
        assert len(d["histogram"]["bins"]) > 0

    def test_paths_sample_count(self, client):
        d = client.post("/api/price", json=PRICE_BODY).get_json()
        assert len(d["paths_sample"]) <= 50

    def test_time_axis_length_matches_paths(self, client):
        d = client.post("/api/price", json=PRICE_BODY).get_json()
        assert len(d["time_axis"]) == len(d["paths_sample"][0])

    def test_prob_itm_in_range(self, client):
        d = client.post("/api/price", json=PRICE_BODY).get_json()
        assert 0 <= d["call"]["prob_itm"] <= 1
        assert 0 <= d["put"]["prob_itm"] <= 1

    def test_missing_field_returns_400(self, client):
        r = client.post("/api/price", json={"spot": 100, "strike": 105})
        assert r.status_code == 400
        assert "error" in r.get_json()

    def test_invalid_spot_returns_400(self, client):
        body = {**PRICE_BODY, "spot": -5}
        r = client.post("/api/price", json=body)
        assert r.status_code == 400

    def test_mc_price_close_to_bs(self, client):
        body = {**PRICE_BODY, "num_paths": 10000}
        d = client.post("/api/price", json=body).get_json()
        call = d["call"]
        # Expect MC within 5% of BS with 10k paths
        assert abs(call["mc_price"] - call["bs_price"]) / call["bs_price"] < 0.05


# ---------------------------------------------------------------------------
# Flask /api/scenario
# ---------------------------------------------------------------------------

SCENARIO_BODY = {
    "spot": 100, "strike": 105, "rate": 0.05,
    "vol": 0.20, "expiry": 1.0, "option_type": "call",
}


class TestApiScenario:
    def test_status_200(self, client):
        r = client.post("/api/scenario", json=SCENARIO_BODY)
        assert r.status_code == 200

    def test_response_keys(self, client):
        d = client.post("/api/scenario", json=SCENARIO_BODY).get_json()
        for key in ("spot_prices", "payoff", "pnl", "premium", "breakevens", "max_loss"):
            assert key in d

    def test_arrays_same_length(self, client):
        d = client.post("/api/scenario", json=SCENARIO_BODY).get_json()
        assert len(d["spot_prices"]) == len(d["payoff"]) == len(d["pnl"])

    def test_call_breakeven_above_strike(self, client):
        d = client.post("/api/scenario", json=SCENARIO_BODY).get_json()
        assert len(d["breakevens"]) >= 1
        assert d["breakevens"][0] > SCENARIO_BODY["strike"]

    def test_put_breakeven_below_strike(self, client):
        body = {**SCENARIO_BODY, "option_type": "put", "strike": 95}
        d = client.post("/api/scenario", json=body).get_json()
        assert len(d["breakevens"]) >= 1
        assert d["breakevens"][0] < 95

    def test_premium_override(self, client):
        body = {**SCENARIO_BODY, "premium": 5.0}
        d = client.post("/api/scenario", json=body).get_json()
        assert d["premium"] == pytest.approx(5.0)

    def test_max_loss_equals_negative_premium(self, client):
        d = client.post("/api/scenario", json=SCENARIO_BODY).get_json()
        assert d["max_loss"] == pytest.approx(-d["premium"], rel=1e-4)

    def test_invalid_option_type_returns_400(self, client):
        body = {**SCENARIO_BODY, "option_type": "straddle"}
        r = client.post("/api/scenario", json=body)
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Flask /api/strategies
# ---------------------------------------------------------------------------

STRATEGIES_BODY = {
    "spot": 100, "rate": 0.05, "vol": 0.20, "expiry": 1.0,
    "strategies": [
        {"id": "long_call", "strike": 105},
        {"id": "long_put", "strike": 95},
        {"id": "long_straddle", "strike": 100},
    ],
}


class TestApiStrategies:
    def test_status_200(self, client):
        r = client.post("/api/strategies", json=STRATEGIES_BODY)
        assert r.status_code == 200

    def test_response_has_spot_prices_and_strategies(self, client):
        d = client.post("/api/strategies", json=STRATEGIES_BODY).get_json()
        assert "spot_prices" in d
        assert "strategies" in d
        assert "current_spot" in d

    def test_number_of_strategies_matches_request(self, client):
        d = client.post("/api/strategies", json=STRATEGIES_BODY).get_json()
        assert len(d["strategies"]) == len(STRATEGIES_BODY["strategies"])

    def test_strategy_profile_keys(self, client):
        d = client.post("/api/strategies", json=STRATEGIES_BODY).get_json()
        for s in d["strategies"]:
            for key in ("id", "name", "color", "pnl", "premium", "max_loss", "breakevens"):
                assert key in s

    def test_pnl_length_matches_spot_prices(self, client):
        d = client.post("/api/strategies", json=STRATEGIES_BODY).get_json()
        n = len(d["spot_prices"])
        for s in d["strategies"]:
            assert len(s["pnl"]) == n

    def test_empty_strategies_returns_400(self, client):
        body = {**STRATEGIES_BODY, "strategies": []}
        r = client.post("/api/strategies", json=body)
        assert r.status_code == 400

    def test_unknown_strategy_returns_400(self, client):
        body = {**STRATEGIES_BODY,
                "strategies": [{"id": "iron_condor", "strike": 100}]}
        r = client.post("/api/strategies", json=body)
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# black_scholes_greeks (unit tests)
# ---------------------------------------------------------------------------

GREEKS_PARAMS = dict(
    spot=100.0, strike=100.0, risk_free_rate=0.05,
    volatility=0.20, time_to_expiry=1.0,
)


class TestBlackScholesGreeks:
    """Verify analytical Greeks against well-known BS properties."""

    def _call(self, **kw):
        p = {**GREEKS_PARAMS, **kw}
        return black_scholes_greeks(**p, option_type="call")

    def _put(self, **kw):
        p = {**GREEKS_PARAMS, **kw}
        return black_scholes_greeks(**p, option_type="put")

    def test_call_delta_between_0_and_1(self):
        assert 0 < self._call()["delta"] < 1

    def test_put_delta_between_minus1_and_0(self):
        assert -1 < self._put()["delta"] < 0

    def test_call_put_delta_sum_equals_1(self):
        # Call delta - Put delta = 1 (put-call delta parity: delta_call - delta_put = 1)
        c = self._call()["delta"]
        p = self._put()["delta"]
        assert abs(c - p - 1.0) < 1e-6

    def test_call_put_gamma_equal(self):
        assert abs(self._call()["gamma"] - self._put()["gamma"]) < 1e-8

    def test_gamma_is_positive(self):
        assert self._call()["gamma"] > 0

    def test_theta_is_negative(self):
        # Time value erodes — theta is negative for long options
        assert self._call()["theta"] < 0
        assert self._put()["theta"] < 0

    def test_vega_is_positive(self):
        assert self._call()["vega"] > 0
        assert self._put()["vega"] > 0

    def test_call_put_vega_equal(self):
        assert abs(self._call()["vega"] - self._put()["vega"]) < 1e-8

    def test_call_rho_is_positive(self):
        # Higher rates increase call value
        assert self._call()["rho"] > 0

    def test_put_rho_is_negative(self):
        # Higher rates decrease put value
        assert self._put()["rho"] < 0

    def test_itm_call_delta_above_0_5(self):
        # Deep ITM call → delta → 1
        g = black_scholes_greeks(
            spot=130.0, strike=100.0, risk_free_rate=0.05,
            volatility=0.20, time_to_expiry=1.0, option_type="call",
        )
        assert g["delta"] > 0.5

    def test_otm_call_delta_below_0_5(self):
        g = black_scholes_greeks(
            spot=80.0, strike=100.0, risk_free_rate=0.05,
            volatility=0.20, time_to_expiry=1.0, option_type="call",
        )
        assert g["delta"] < 0.5

    def test_atm_gamma_higher_than_deep_itm(self):
        atm_g = black_scholes_greeks(
            spot=100.0, strike=100.0, risk_free_rate=0.05,
            volatility=0.20, time_to_expiry=1.0, option_type="call",
        )["gamma"]
        deep_g = black_scholes_greeks(
            spot=200.0, strike=100.0, risk_free_rate=0.05,
            volatility=0.20, time_to_expiry=1.0, option_type="call",
        )["gamma"]
        assert atm_g > deep_g

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            black_scholes_greeks(**GREEKS_PARAMS, option_type="straddle")

    def test_zero_expiry_returns_zeros_for_gamma_theta_vega_rho(self):
        g = black_scholes_greeks(
            spot=100.0, strike=100.0, risk_free_rate=0.05,
            volatility=0.20, time_to_expiry=0.0, option_type="call",
        )
        assert g["gamma"] == 0.0
        assert g["theta"] == 0.0
        assert g["vega"]  == 0.0
        assert g["rho"]   == 0.0


# ---------------------------------------------------------------------------
# Flask /api/greeks
# ---------------------------------------------------------------------------

GREEKS_BODY = {
    "spot": 100, "strike": 105, "rate": 0.05,
    "vol": 0.20, "expiry": 1.0, "option_type": "call",
}


class TestApiGreeks:
    def test_status_200(self, client):
        r = client.post("/api/greeks", json=GREEKS_BODY)
        assert r.status_code == 200

    def test_response_has_greeks_key(self, client):
        d = client.post("/api/greeks", json=GREEKS_BODY).get_json()
        assert "greeks" in d

    def test_greeks_keys(self, client):
        g = client.post("/api/greeks", json=GREEKS_BODY).get_json()["greeks"]
        for key in ("delta", "gamma", "theta", "vega", "rho"):
            assert key in g

    def test_response_has_curves(self, client):
        d = client.post("/api/greeks", json=GREEKS_BODY).get_json()
        for key in ("spot_prices", "delta_curve", "gamma_curve", "vega_curve"):
            assert key in d
            assert len(d[key]) > 0

    def test_decay_curve_shape(self, client):
        d = client.post("/api/greeks", json=GREEKS_BODY).get_json()
        assert len(d["decay_times"]) == len(d["decay_values"])
        assert len(d["decay_times"]) > 0

    def test_curves_same_length(self, client):
        d = client.post("/api/greeks", json=GREEKS_BODY).get_json()
        n = len(d["spot_prices"])
        assert len(d["delta_curve"]) == n
        assert len(d["gamma_curve"]) == n
        assert len(d["vega_curve"])  == n

    def test_option_price_present(self, client):
        d = client.post("/api/greeks", json=GREEKS_BODY).get_json()
        assert "option_price" in d
        assert d["option_price"] > 0

    def test_call_delta_in_range(self, client):
        d = client.post("/api/greeks", json=GREEKS_BODY).get_json()
        assert 0 < d["greeks"]["delta"] < 1

    def test_put_delta_in_range(self, client):
        body = {**GREEKS_BODY, "option_type": "put"}
        d = client.post("/api/greeks", json=body).get_json()
        assert -1 < d["greeks"]["delta"] < 0

    def test_missing_field_returns_400(self, client):
        r = client.post("/api/greeks", json={"spot": 100})
        assert r.status_code == 400

    def test_invalid_option_type_returns_400(self, client):
        body = {**GREEKS_BODY, "option_type": "condor"}
        r = client.post("/api/greeks", json=body)
        assert r.status_code == 400

    def test_params_echoed(self, client):
        d = client.post("/api/greeks", json=GREEKS_BODY).get_json()
        assert "params" in d
        assert d["params"]["option_type"] == "call"


# ---------------------------------------------------------------------------
# Flask /api/probability
# ---------------------------------------------------------------------------

PROB_BODY = {
    "spot": 100, "rate": 0.05, "vol": 0.20, "expiry": 1.0,
    "strategy_id": "long_call", "strike": 105,
    "num_paths": 2000, "seed": 0,
}


class TestApiProbability:
    def test_status_200(self, client):
        r = client.post("/api/probability", json=PROB_BODY)
        assert r.status_code == 200

    def test_required_keys_present(self, client):
        d = client.post("/api/probability", json=PROB_BODY).get_json()
        for key in ("prob_profit", "expected_pnl", "median_pnl",
                    "var_5", "var_10", "cvar_5",
                    "max_profit", "max_loss",
                    "premium", "strategy_name", "histogram", "params"):
            assert key in d, f"Missing key: {key}"

    def test_prob_profit_in_unit_interval(self, client):
        d = client.post("/api/probability", json=PROB_BODY).get_json()
        assert 0.0 <= d["prob_profit"] <= 1.0

    def test_histogram_shape(self, client):
        d = client.post("/api/probability", json=PROB_BODY).get_json()
        h = d["histogram"]
        assert "bins" in h and "counts" in h
        assert len(h["bins"]) == len(h["counts"])
        assert len(h["bins"]) == 50

    def test_histogram_counts_sum_to_num_paths(self, client):
        d = client.post("/api/probability", json=PROB_BODY).get_json()
        assert sum(d["histogram"]["counts"]) == PROB_BODY["num_paths"]

    def test_var_5_leq_var_10(self, client):
        """VaR at 95% confidence must be <= VaR at 90% (worse tail)."""
        d = client.post("/api/probability", json=PROB_BODY).get_json()
        assert d["var_5"] <= d["var_10"]

    def test_cvar_leq_var_5(self, client):
        """CVaR (expected shortfall) must be <= VaR(5%)."""
        d = client.post("/api/probability", json=PROB_BODY).get_json()
        assert d["cvar_5"] <= d["var_5"]

    def test_max_profit_geq_expected_pnl(self, client):
        d = client.post("/api/probability", json=PROB_BODY).get_json()
        assert d["max_profit"] >= d["expected_pnl"]

    def test_max_loss_leq_expected_pnl(self, client):
        d = client.post("/api/probability", json=PROB_BODY).get_json()
        assert d["max_loss"] <= d["expected_pnl"]

    def test_strategy_name_returned(self, client):
        d = client.post("/api/probability", json=PROB_BODY).get_json()
        assert d["strategy_name"] == "Long Call"

    def test_params_echoed(self, client):
        d = client.post("/api/probability", json=PROB_BODY).get_json()
        assert d["params"]["strategy_id"] == "long_call"
        assert d["params"]["num_paths"] == PROB_BODY["num_paths"]

    def test_long_put_strategy(self, client):
        body = {**PROB_BODY, "strategy_id": "long_put", "strike": 95}
        d = client.post("/api/probability", json=body).get_json()
        assert d["strategy_name"] == "Long Put"
        assert 0.0 <= d["prob_profit"] <= 1.0

    def test_bull_call_spread(self, client):
        body = {**PROB_BODY, "strategy_id": "bull_call_spread",
                "strike_low": 100, "strike_high": 110}
        del body["strike"]
        r = client.post("/api/probability", json=body)
        assert r.status_code == 200
        d = r.get_json()
        assert d["strategy_name"] == "Bull Call Spread"

    def test_missing_field_returns_400(self, client):
        r = client.post("/api/probability", json={"spot": 100})
        assert r.status_code == 400

    def test_unknown_strategy_returns_400(self, client):
        body = {**PROB_BODY, "strategy_id": "iron_condor"}
        r = client.post("/api/probability", json=body)
        assert r.status_code == 400

    def test_invalid_num_paths_returns_400(self, client):
        body = {**PROB_BODY, "num_paths": 50}  # below minimum of 1000
        r = client.post("/api/probability", json=body)
        assert r.status_code == 400

    def test_seed_reproducibility(self, client):
        """Same seed should produce identical results."""
        d1 = client.post("/api/probability", json=PROB_BODY).get_json()
        d2 = client.post("/api/probability", json=PROB_BODY).get_json()
        assert d1["prob_profit"] == d2["prob_profit"]
        assert d1["expected_pnl"] == d2["expected_pnl"]


# ---------------------------------------------------------------------------
# Demo Trading endpoints (mocked market data)
# ---------------------------------------------------------------------------

import unittest.mock as mock
import market_data as md


# Fake quote returned by mocked get_quote
FAKE_QUOTE = {
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "price": 175.0,
    "prev_close": 173.0,
    "change": 2.0,
    "change_pct": 1.16,
    "hist_vol": 0.25,
    "currency": "USD",
    "market_cap": 2_700_000_000_000,
    "sector": "Technology",
    "timestamp": 1_700_000_000.0,
}


@pytest.fixture(autouse=False)
def mock_get_quote(monkeypatch):
    """Patch market_data.get_quote so tests never hit the network."""
    monkeypatch.setattr(md, "get_quote", lambda ticker: dict(FAKE_QUOTE, ticker=ticker.upper()))


@pytest.fixture(autouse=False)
def clean_portfolio():
    """Reset the in-memory portfolio before each test that uses it."""
    with md._portfolio_lock:
        md._portfolio.clear()
    yield
    with md._portfolio_lock:
        md._portfolio.clear()


class TestApiQuote:
    def test_status_200(self, client, mock_get_quote):
        r = client.get("/api/quote?ticker=AAPL")
        assert r.status_code == 200

    def test_response_keys(self, client, mock_get_quote):
        d = client.get("/api/quote?ticker=AAPL").get_json()
        for key in ("ticker", "name", "price", "hist_vol", "change_pct"):
            assert key in d

    def test_ticker_uppercased(self, client, mock_get_quote):
        d = client.get("/api/quote?ticker=aapl").get_json()
        assert d["ticker"] == "AAPL"

    def test_missing_ticker_returns_400(self, client):
        r = client.get("/api/quote")
        assert r.status_code == 400

    def test_invalid_ticker_returns_400(self, client, monkeypatch):
        monkeypatch.setattr(md, "get_quote", lambda t: (_ for _ in ()).throw(
            ValueError(f"No valid price data found for ticker '{t}'")))
        r = client.get("/api/quote?ticker=XXXXINVALID")
        assert r.status_code == 400


DEMO_OPEN_BODY = {
    "ticker": "AAPL",
    "strategy_id": "long_call",
    "expiry": 0.25,
    "rate": 0.05,
    "quantity": 1,
}


class TestApiDemoTrade:
    def test_open_trade_returns_201(self, client, mock_get_quote, clean_portfolio):
        r = client.post("/api/demo/trade", json=DEMO_OPEN_BODY)
        assert r.status_code == 201

    def test_open_trade_response_keys(self, client, mock_get_quote, clean_portfolio):
        d = client.post("/api/demo/trade", json=DEMO_OPEN_BODY).get_json()
        for key in ("id", "ticker", "strategy_id", "strategy_name",
                    "entry_spot", "premium", "status", "expiry_years"):
            assert key in d, f"Missing key: {key}"

    def test_open_trade_status_is_open(self, client, mock_get_quote, clean_portfolio):
        d = client.post("/api/demo/trade", json=DEMO_OPEN_BODY).get_json()
        assert d["status"] == "open"

    def test_open_trade_ticker_matches(self, client, mock_get_quote, clean_portfolio):
        d = client.post("/api/demo/trade", json=DEMO_OPEN_BODY).get_json()
        assert d["ticker"] == "AAPL"

    def test_open_trade_entry_spot_from_quote(self, client, mock_get_quote, clean_portfolio):
        d = client.post("/api/demo/trade", json=DEMO_OPEN_BODY).get_json()
        assert d["entry_spot"] == FAKE_QUOTE["price"]

    def test_open_trade_missing_ticker_returns_400(self, client, mock_get_quote, clean_portfolio):
        body = {k: v for k, v in DEMO_OPEN_BODY.items() if k != "ticker"}
        r = client.post("/api/demo/trade", json=body)
        assert r.status_code == 400

    def test_open_trade_missing_strategy_returns_400(self, client, mock_get_quote, clean_portfolio):
        body = {k: v for k, v in DEMO_OPEN_BODY.items() if k != "strategy_id"}
        r = client.post("/api/demo/trade", json=body)
        assert r.status_code == 400

    def test_open_trade_missing_expiry_returns_400(self, client, mock_get_quote, clean_portfolio):
        body = {k: v for k, v in DEMO_OPEN_BODY.items() if k != "expiry"}
        r = client.post("/api/demo/trade", json=body)
        assert r.status_code == 400

    def test_open_trade_unknown_strategy_returns_400(self, client, mock_get_quote, clean_portfolio):
        body = {**DEMO_OPEN_BODY, "strategy_id": "iron_condor"}
        r = client.post("/api/demo/trade", json=body)
        assert r.status_code == 400

    def test_open_trade_with_explicit_strike(self, client, mock_get_quote, clean_portfolio):
        body = {**DEMO_OPEN_BODY, "strike": 180.0}
        d = client.post("/api/demo/trade", json=body).get_json()
        assert d["strike"] == 180.0

    def test_open_spread_strategy(self, client, mock_get_quote, clean_portfolio):
        body = {
            "ticker": "AAPL",
            "strategy_id": "bull_call_spread",
            "expiry": 0.25,
            "rate": 0.05,
            "strike_low": 170.0,
            "strike_high": 180.0,
        }
        r = client.post("/api/demo/trade", json=body)
        assert r.status_code == 201
        d = r.get_json()
        assert d["strategy_id"] == "bull_call_spread"


class TestApiDemoPortfolio:
    def test_empty_portfolio(self, client, mock_get_quote, clean_portfolio):
        r = client.get("/api/demo/portfolio")
        assert r.status_code == 200
        assert r.get_json()["trades"] == []

    def test_portfolio_shows_open_trade(self, client, mock_get_quote, clean_portfolio):
        client.post("/api/demo/trade", json=DEMO_OPEN_BODY)
        d = client.get("/api/demo/portfolio").get_json()
        assert len(d["trades"]) == 1

    def test_portfolio_trade_has_pnl_fields(self, client, mock_get_quote, clean_portfolio):
        client.post("/api/demo/trade", json=DEMO_OPEN_BODY)
        trade = client.get("/api/demo/portfolio").get_json()["trades"][0]
        assert "current_pnl" in trade
        assert "current_pnl_pct" in trade
        assert "current_spot" in trade

    def test_long_call_pnl_negative_when_otm(self, client, clean_portfolio, monkeypatch):
        """P&L reflects at-expiry payoff: OTM long call shows negative (max-loss) P&L."""
        # Entry spot = 175, strike = 200 (OTM call, intrinsic = 0 at expiry)
        quote = dict(FAKE_QUOTE, price=175.0)
        monkeypatch.setattr(md, "get_quote", lambda t: dict(quote, ticker=t.upper()))
        body = {**DEMO_OPEN_BODY, "strike": 200.0}  # OTM call: max loss = premium
        client.post("/api/demo/trade", json=body)
        trade = client.get("/api/demo/portfolio").get_json()["trades"][0]
        # At-expiry with spot=175 < K=200: payoff=0, P&L = -premium
        assert trade["current_pnl"] < 0, "OTM long call should show max-loss (negative P&L)"
        assert trade["current_pnl"] != 0, "P&L must be non-zero"

    def test_long_call_pnl_positive_after_favorable_move(self, client, clean_portfolio, monkeypatch):
        """P&L reflects at-expiry payoff: long call gains when spot rises above breakeven."""
        entry_quote = dict(FAKE_QUOTE, price=175.0)
        # Simulate price jumping to 230 (well above strike 200) after opening
        up_quote = dict(FAKE_QUOTE, price=230.0)
        call_count = [0]

        def mock_quote(t):
            call_count[0] += 1
            # First call is for opening the trade; subsequent calls are for portfolio refresh
            q = entry_quote if call_count[0] == 1 else up_quote
            return dict(q, ticker=t.upper())

        monkeypatch.setattr(md, "get_quote", mock_quote)
        body = {**DEMO_OPEN_BODY, "strike": 200.0}  # OTM at entry
        client.post("/api/demo/trade", json=body)
        trade = client.get("/api/demo/portfolio").get_json()["trades"][0]
        # At-expiry with spot=230 > K=200: payoff=30, P&L should be positive
        assert trade["current_pnl"] > 0, "Long call is profitable when spot is well above strike"

    def test_pnl_reflects_quantity(self, client, clean_portfolio, monkeypatch):
        """P&L should scale linearly with quantity."""
        itm_quote = dict(FAKE_QUOTE, price=175.0)
        monkeypatch.setattr(md, "get_quote", lambda t: dict(itm_quote, ticker=t.upper()))
        body1 = {**DEMO_OPEN_BODY, "strike": 150.0, "quantity": 1}
        body2 = {**DEMO_OPEN_BODY, "strike": 150.0, "quantity": 3}
        client.post("/api/demo/trade", json=body1)
        client.post("/api/demo/trade", json=body2)
        trades = client.get("/api/demo/portfolio").get_json()["trades"]
        # Sort by quantity to identify which is which
        trades.sort(key=lambda t: t["quantity"])
        pnl1 = trades[0]["current_pnl"]
        pnl3 = trades[1]["current_pnl"]
        assert abs(pnl3 - 3 * pnl1) < 0.01, "P&L should scale with quantity"

    def test_multiple_trades_in_portfolio(self, client, mock_get_quote, clean_portfolio):
        client.post("/api/demo/trade", json=DEMO_OPEN_BODY)
        body2 = {**DEMO_OPEN_BODY, "strategy_id": "long_put"}
        client.post("/api/demo/trade", json=body2)
        d = client.get("/api/demo/portfolio").get_json()
        assert len(d["trades"]) == 2


class TestApiDemoCloseTrade:
    def test_close_trade_returns_200(self, client, mock_get_quote, clean_portfolio):
        trade = client.post("/api/demo/trade", json=DEMO_OPEN_BODY).get_json()
        r = client.delete(f"/api/demo/trade/{trade['id']}")
        assert r.status_code == 200

    def test_closed_trade_status(self, client, mock_get_quote, clean_portfolio):
        trade = client.post("/api/demo/trade", json=DEMO_OPEN_BODY).get_json()
        d = client.delete(f"/api/demo/trade/{trade['id']}").get_json()
        assert d["status"] == "closed"

    def test_closed_trade_has_exit_fields(self, client, mock_get_quote, clean_portfolio):
        trade = client.post("/api/demo/trade", json=DEMO_OPEN_BODY).get_json()
        d = client.delete(f"/api/demo/trade/{trade['id']}").get_json()
        assert "exit_spot" in d
        assert "exit_time" in d

    def test_close_nonexistent_trade_returns_404(self, client, mock_get_quote):
        r = client.delete("/api/demo/trade/nonexistent-id-xyz")
        assert r.status_code == 404

    def test_close_already_closed_trade_returns_400(self, client, mock_get_quote, clean_portfolio):
        trade = client.post("/api/demo/trade", json=DEMO_OPEN_BODY).get_json()
        client.delete(f"/api/demo/trade/{trade['id']}")
        r = client.delete(f"/api/demo/trade/{trade['id']}")
        assert r.status_code == 400

    def test_closed_trade_still_in_portfolio(self, client, mock_get_quote, clean_portfolio):
        trade = client.post("/api/demo/trade", json=DEMO_OPEN_BODY).get_json()
        client.delete(f"/api/demo/trade/{trade['id']}")
        portfolio = client.get("/api/demo/portfolio").get_json()["trades"]
        closed = [t for t in portfolio if t["id"] == trade["id"]]
        assert len(closed) == 1
        assert closed[0]["status"] == "closed"


# ---------------------------------------------------------------------------
# Stock Chart endpoint
# ---------------------------------------------------------------------------

FAKE_HISTORY = {
    "ticker": "AAPL",
    "period": "6mo",
    "dates":   ["2024-01-02", "2024-01-03", "2024-01-04"],
    "opens":   [185.0, 183.0, 182.5],
    "highs":   [187.0, 185.0, 184.0],
    "lows":    [183.5, 181.0, 181.5],
    "closes":  [185.5, 182.0, 183.0],
    "volumes": [50_000_000, 45_000_000, 48_000_000],
}


@pytest.fixture(autouse=False)
def mock_get_history(monkeypatch):
    """Patch market_data.get_history so tests never hit the network."""
    monkeypatch.setattr(md, "get_history",
                        lambda ticker, period="6mo": dict(FAKE_HISTORY, ticker=ticker.upper(), period=period))


class TestApiStockChart:
    def test_status_200(self, client, mock_get_history):
        r = client.get("/api/stock_chart?ticker=AAPL")
        assert r.status_code == 200

    def test_response_keys(self, client, mock_get_history):
        d = client.get("/api/stock_chart?ticker=AAPL").get_json()
        for key in ("ticker", "period", "dates", "opens", "highs", "lows", "closes", "volumes"):
            assert key in d, f"Missing key: {key}"

    def test_ticker_uppercased(self, client, mock_get_history):
        d = client.get("/api/stock_chart?ticker=aapl").get_json()
        assert d["ticker"] == "AAPL"

    def test_default_period_is_6mo(self, client, mock_get_history):
        d = client.get("/api/stock_chart?ticker=AAPL").get_json()
        assert d["period"] == "6mo"

    def test_custom_period(self, client, mock_get_history):
        d = client.get("/api/stock_chart?ticker=AAPL&period=1y").get_json()
        assert d["period"] == "1y"

    def test_missing_ticker_returns_400(self, client):
        r = client.get("/api/stock_chart")
        assert r.status_code == 400

    def test_invalid_ticker_returns_400(self, client, monkeypatch):
        def _raise(t, period="6mo"):
            raise ValueError(f"No historical data found for ticker '{t}'")
        monkeypatch.setattr(md, "get_history", _raise)
        r = client.get("/api/stock_chart?ticker=XXXXBAD")
        assert r.status_code == 400

    def test_ohlcv_lists_same_length(self, client, mock_get_history):
        d = client.get("/api/stock_chart?ticker=AAPL").get_json()
        n = len(d["dates"])
        assert len(d["opens"]) == n
        assert len(d["highs"]) == n
        assert len(d["lows"]) == n
        assert len(d["closes"]) == n
        assert len(d["volumes"]) == n


# ---------------------------------------------------------------------------
# AI Trading endpoints
# ---------------------------------------------------------------------------

import ai_trader


@pytest.fixture(autouse=False)
def clean_ai(monkeypatch):
    """Reset the AI trader state before/after each test."""
    ai_trader.reset()
    yield
    ai_trader.reset()


class TestApiAiStatus:
    def test_status_200(self, client, clean_ai):
        r = client.get("/api/ai/status")
        assert r.status_code == 200

    def test_response_keys(self, client, clean_ai):
        d = client.get("/api/ai/status").get_json()
        for key in ("epsilon", "total_trades", "winning_trades", "win_rate",
                    "total_reward", "reward_history", "q_table", "actions"):
            assert key in d, f"Missing key: {key}"

    def test_initial_epsilon(self, client, clean_ai):
        d = client.get("/api/ai/status").get_json()
        assert d["epsilon"] == pytest.approx(ai_trader.EPSILON_START, abs=0.001)

    def test_initial_total_trades_zero(self, client, clean_ai):
        d = client.get("/api/ai/status").get_json()
        assert d["total_trades"] == 0

    def test_q_table_shape(self, client, clean_ai):
        d = client.get("/api/ai/status").get_json()
        q = d["q_table"]
        assert len(q) == ai_trader.N_STATES
        assert all(len(row) == ai_trader.N_ACTIONS for row in q)


class TestApiAiTrade:
    def test_missing_ticker_returns_400(self, client, clean_ai):
        r = client.post("/api/ai/trade", json={"expiry": 0.25})
        assert r.status_code == 400

    def test_response_has_action(self, client, mock_get_quote, clean_ai):
        r = client.post("/api/ai/trade", json={"ticker": "AAPL", "expiry": 0.25})
        assert r.status_code in (200, 201)
        d = r.get_json()
        assert "action" in d
        assert d["action"] in ai_trader.ACTIONS

    def test_response_has_state(self, client, mock_get_quote, clean_ai):
        d = client.post("/api/ai/trade",
                        json={"ticker": "AAPL", "expiry": 0.25}).get_json()
        assert "state" in d
        assert len(d["state"]) == 4

    def test_response_has_mc_return(self, client, mock_get_quote, clean_ai):
        d = client.post("/api/ai/trade",
                        json={"ticker": "AAPL", "expiry": 0.25}).get_json()
        assert "mc_return" in d
        assert isinstance(d["mc_return"], float)

    def test_response_has_ai_status(self, client, mock_get_quote, clean_ai):
        d = client.post("/api/ai/trade",
                        json={"ticker": "AAPL", "expiry": 0.25}).get_json()
        assert "ai_status" in d

    def test_no_trade_action_returns_null_trade(self, client, mock_get_quote, clean_ai, monkeypatch):
        monkeypatch.setattr(ai_trader, "decide", lambda state: "no_trade")
        d = client.post("/api/ai/trade",
                        json={"ticker": "AAPL", "expiry": 0.25}).get_json()
        assert d["action"] == "no_trade"
        assert d["trade"] is None

    def test_trade_action_opens_a_trade(self, client, mock_get_quote, clean_ai, clean_portfolio, monkeypatch):
        monkeypatch.setattr(ai_trader, "decide", lambda state: "long_call")
        d = client.post("/api/ai/trade",
                        json={"ticker": "AAPL", "expiry": 0.25}).get_json()
        assert d["action"] == "long_call"
        assert d["trade"] is not None
        assert d["trade"]["strategy_id"] == "long_call"

    def test_ai_trade_has_ai_state_field(self, client, mock_get_quote, clean_ai, clean_portfolio, monkeypatch):
        monkeypatch.setattr(ai_trader, "decide", lambda state: "long_call")
        d = client.post("/api/ai/trade",
                        json={"ticker": "AAPL", "expiry": 0.25}).get_json()
        assert "ai_state" in d["trade"]
        assert len(d["trade"]["ai_state"]) == 4


class TestApiAiCloseTrade:
    def test_close_ai_trade_updates_q_table(self, client, mock_get_quote, clean_ai, clean_portfolio, monkeypatch):
        monkeypatch.setattr(ai_trader, "decide", lambda state: "long_call")
        open_resp = client.post("/api/ai/trade",
                                json={"ticker": "AAPL", "expiry": 0.25}).get_json()
        trade_id = open_resp["trade"]["id"]

        r = client.post(f"/api/ai/close_trade/{trade_id}")
        assert r.status_code == 200
        d = r.get_json()
        assert "reward" in d
        assert "ai_status" in d
        assert isinstance(d["reward"], float)

    def test_close_ai_trade_increments_total_trades(self, client, mock_get_quote, clean_ai, clean_portfolio, monkeypatch):
        monkeypatch.setattr(ai_trader, "decide", lambda state: "long_call")
        open_resp = client.post("/api/ai/trade",
                                json={"ticker": "AAPL", "expiry": 0.25}).get_json()
        trade_id = open_resp["trade"]["id"]
        client.post(f"/api/ai/close_trade/{trade_id}")
        d = client.get("/api/ai/status").get_json()
        assert d["total_trades"] == 1

    def test_close_nonexistent_ai_trade_returns_404(self, client, mock_get_quote, clean_ai):
        r = client.post("/api/ai/close_trade/nonexistent-xyz")
        assert r.status_code == 404

    def test_close_manual_trade_via_ai_endpoint_returns_400(self, client, mock_get_quote, clean_ai, clean_portfolio):
        # Manual trade opened without AI context should be rejected
        trade = client.post("/api/demo/trade", json=DEMO_OPEN_BODY).get_json()
        r = client.post(f"/api/ai/close_trade/{trade['id']}")
        assert r.status_code == 400


class TestApiAiReset:
    def test_reset_returns_200(self, client, clean_ai):
        r = client.post("/api/ai/reset")
        assert r.status_code == 200

    def test_reset_restores_epsilon(self, client, clean_ai):
        # Do some learning first
        ai_trader.record_reward(("up", "low", "bullish", "neutral"), "long_call", 0.5)
        d = client.post("/api/ai/reset").get_json()
        assert d["ai_status"]["epsilon"] == pytest.approx(ai_trader.EPSILON_START, abs=0.001)

    def test_reset_clears_trades(self, client, clean_ai):
        ai_trader.record_reward(("up", "low", "bullish", "neutral"), "long_call", 0.5)
        d = client.post("/api/ai/reset").get_json()
        assert d["ai_status"]["total_trades"] == 0

    def test_reset_message(self, client, clean_ai):
        d = client.post("/api/ai/reset").get_json()
        assert "message" in d


# ---------------------------------------------------------------------------
# ai_trader module unit tests
# ---------------------------------------------------------------------------

class TestAiTraderModule:
    def setup_method(self):
        ai_trader.reset()

    def test_get_state_returns_tuple(self):
        s = ai_trader.get_state(2.0, 0.25, 0.05)
        assert isinstance(s, tuple)
        assert len(s) == 4

    def test_get_state_price_trend(self):
        assert ai_trader.get_state(2.0, 0.25, 0.0)[0]  == "up"
        assert ai_trader.get_state(-2.0, 0.25, 0.0)[0] == "down"
        assert ai_trader.get_state(0.0, 0.25, 0.0)[0]  == "flat"

    def test_get_state_vol_bucket(self):
        assert ai_trader.get_state(0.0, 0.10, 0.0)[1] == "low"
        assert ai_trader.get_state(0.0, 0.30, 0.0)[1] == "medium"
        assert ai_trader.get_state(0.0, 0.50, 0.0)[1] == "high"

    def test_get_state_mc_signal(self):
        assert ai_trader.get_state(0.0, 0.25, 0.05)[2]  == "bullish"
        assert ai_trader.get_state(0.0, 0.25, -0.05)[2] == "bearish"
        assert ai_trader.get_state(0.0, 0.25, 0.0)[2]   == "neutral"

    def test_get_state_rsi_signal(self):
        assert ai_trader.get_state(0.0, 0.25, 0.0, rsi=75)[3] == "overbought"
        assert ai_trader.get_state(0.0, 0.25, 0.0, rsi=25)[3] == "oversold"
        assert ai_trader.get_state(0.0, 0.25, 0.0, rsi=50)[3] == "neutral"

    def test_decide_returns_valid_action(self):
        state = ai_trader.get_state(1.0, 0.25, 0.03)
        action = ai_trader.decide(state)
        assert action in ai_trader.ACTIONS

    def test_record_reward_updates_total_trades(self):
        state = ("up", "medium", "bullish", "neutral")
        ai_trader.record_reward(state, "long_call", 1.0)
        status = ai_trader.get_status()
        assert status["total_trades"] == 1

    def test_record_reward_positive_increments_wins(self):
        state = ("up", "medium", "bullish", "neutral")
        ai_trader.record_reward(state, "long_call", 2.0)
        status = ai_trader.get_status()
        assert status["winning_trades"] == 1

    def test_record_reward_negative_no_win(self):
        state = ("down", "high", "bearish", "overbought")
        ai_trader.record_reward(state, "long_put", -1.0)
        status = ai_trader.get_status()
        assert status["winning_trades"] == 0

    def test_no_trade_does_not_count_as_trade(self):
        state = ("flat", "low", "neutral", "neutral")
        ai_trader.record_reward(state, "no_trade", 0.0)
        status = ai_trader.get_status()
        assert status["total_trades"] == 0

    def test_epsilon_decays_after_reward(self):
        initial = ai_trader.get_status()["epsilon"]
        state = ("up", "medium", "bullish", "neutral")
        ai_trader.record_reward(state, "long_call", 1.0)
        new_eps = ai_trader.get_status()["epsilon"]
        assert new_eps < initial

    def test_epsilon_never_below_min(self):
        state = ("up", "medium", "bullish", "neutral")
        for _ in range(200):
            ai_trader.record_reward(state, "long_call", 1.0)
        status = ai_trader.get_status()
        assert status["epsilon"] >= ai_trader.EPSILON_MIN

    def test_reset_restores_initial_state(self):
        state = ("up", "medium", "bullish", "neutral")
        ai_trader.record_reward(state, "long_call", 5.0)
        ai_trader.reset()
        status = ai_trader.get_status()
        assert status["total_trades"] == 0
        assert status["epsilon"] == pytest.approx(ai_trader.EPSILON_START, abs=0.001)
        assert status["total_reward"] == 0.0
        assert status["reward_history"] == []

    def test_q_table_updates_after_reward(self):
        state = ("up", "medium", "bullish", "neutral")
        si = ai_trader._STATE_INDEX[state]
        ai = ai_trader._ACTION_INDEX["long_call"]
        old_q = float(ai_trader._q_table[si, ai])
        ai_trader.record_reward(state, "long_call", 10.0)
        new_q = float(ai_trader._q_table[si, ai])
        assert new_q > old_q

    def test_compute_mc_expected_return_is_float(self):
        result = ai_trader.compute_mc_expected_return(100.0, 0.25, 0.05, 0.25, num_paths=500, seed=42)
        assert isinstance(result, float)

    def test_win_rate_accuracy(self):
        state = ("flat", "medium", "neutral", "neutral")
        ai_trader.record_reward(state, "long_call", 1.0)
        ai_trader.record_reward(state, "long_put", -1.0)
        status = ai_trader.get_status()
        assert status["win_rate"] == pytest.approx(0.5, abs=0.01)


    def test_new_actions_in_action_list(self):
        """New credit/short strategies must appear in the ACTIONS list."""
        for action in ["short_call", "short_put", "bull_put_spread",
                       "bear_call_spread", "iron_condor"]:
            assert action in ai_trader.ACTIONS

    def test_n_states_is_81(self):
        """State space should be 81 (3^4) with the new RSI dimension."""
        assert ai_trader.N_STATES == 81

    def test_n_actions_is_11(self):
        assert ai_trader.N_ACTIONS == 11

    def test_assess_risk_returns_expected_keys(self):
        risk = ai_trader.assess_risk(100.0, 0.25, "iron_condor")
        assert "max_risk_dollars" in risk
        assert "recommended_contracts" in risk
        assert "vol_regime" in risk
        assert "risk_level" in risk
        assert "sharpe_ratio_est" in risk

    def test_assess_risk_credit_is_conservative(self):
        """Credit strategies should be classified as conservative risk."""
        for s in ["short_put", "short_call", "bull_put_spread",
                  "bear_call_spread", "iron_condor"]:
            risk = ai_trader.assess_risk(100.0, 0.25, s)
            assert risk["risk_level"] == "conservative", f"{s} should be conservative"

    def test_assess_risk_high_vol_reduces_size(self):
        low_vol  = ai_trader.assess_risk(100.0, 0.10, "long_call")
        high_vol = ai_trader.assess_risk(100.0, 0.60, "long_call")
        assert high_vol["max_risk_dollars"] < low_vol["max_risk_dollars"]

    def test_assess_risk_recommended_contracts_at_least_one(self):
        risk = ai_trader.assess_risk(100.0, 0.25, "long_call")
        assert risk["recommended_contracts"] >= 1


# ---------------------------------------------------------------------------
# market_data: RSI and momentum computation
# ---------------------------------------------------------------------------

class TestTechnicalIndicators:
    """Tests for compute_rsi and compute_momentum helper functions."""

    def test_rsi_uptrend_approaches_100(self):
        from market_data import compute_rsi
        prices = np.array([100.0 + i for i in range(20)])
        assert compute_rsi(prices) >= 90

    def test_rsi_downtrend_approaches_0(self):
        from market_data import compute_rsi
        prices = np.array([100.0 - i for i in range(20)])
        assert compute_rsi(prices) <= 10

    def test_rsi_default_is_50_when_insufficient_data(self):
        from market_data import compute_rsi
        prices = np.array([100.0, 101.0])
        assert compute_rsi(prices) == 50.0

    def test_rsi_overbought_threshold(self):
        from market_data import compute_rsi
        prices = np.array([100.0 + i * 2 for i in range(20)])
        assert compute_rsi(prices) > 70

    def test_momentum_positive_on_uptrend(self):
        from market_data import compute_momentum
        prices = np.array([100.0 + i * 0.5 for i in range(25)])
        assert compute_momentum(prices) > 0

    def test_momentum_negative_on_downtrend(self):
        from market_data import compute_momentum
        prices = np.array([100.0 - i * 0.5 for i in range(25)])
        assert compute_momentum(prices) < 0

    def test_momentum_returns_zero_on_insufficient_data(self):
        from market_data import compute_momentum
        prices = np.array([100.0, 101.0])
        assert compute_momentum(prices) == 0.0
