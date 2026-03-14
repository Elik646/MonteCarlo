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

    def test_unknown_strategy_id_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            compute_strategy_profile("iron_condor", BASE_PARAMS, SPOT_PRICES)


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
