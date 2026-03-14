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
