"""
Tests for Ch3 ICLV Nested Logit model.
Validates utility computation and probability calculations.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from config import CH3_PARAMS, THETA_DEFAULT
import ch3_model


def make_test_agents(n=100):
    """Create synthetic agent population for testing."""
    rng = np.random.RandomState(42)
    agents = {
        'sex': rng.randint(0, 2, n).astype(np.float32),
        'age1': (rng.random(n) < 0.2).astype(np.float32),
        'age2': (rng.random(n) < 0.3).astype(np.float32),
        'age3': (rng.random(n) < 0.3).astype(np.float32),
        'age4': (rng.random(n) < 0.2).astype(np.float32),
        'income1': (rng.random(n) < 0.3).astype(np.float32),
        'income2': (rng.random(n) < 0.4).astype(np.float32),
        'income3': (rng.random(n) < 0.3).astype(np.float32),
        'education': (rng.random(n) < 0.5).astype(np.float32),
        'occupy': (rng.random(n) < 0.3).astype(np.float32),
        'license': (rng.random(n) < 0.6).astype(np.float32),
        'have_car': (rng.random(n) < 0.4).astype(np.float32),
        'e_bike': (rng.random(n) < 0.3).astype(np.float32),
        'week_bus': (rng.random(n) < 0.4).astype(np.float32),
        'week_metro': (rng.random(n) < 0.5).astype(np.float32),
        'week_taxi': (rng.random(n) < 0.2).astype(np.float32),
        'week_ebike': (rng.random(n) < 0.15).astype(np.float32),
        'week_bike': (rng.random(n) < 0.2).astype(np.float32),
        'travel_num': (rng.random(n) < 0.5).astype(np.float32),
        'travel_distance_work': (rng.random(n) < 0.4).astype(np.float32),
        'travel_distance_weekend': (rng.random(n) < 0.3).astype(np.float32),
        'travel_aim': (rng.random(n) < 0.3).astype(np.float32),
        'cost': (rng.random(n) < 0.5).astype(np.float32),
        'c6': (rng.random(n) < 0.4).astype(np.float32),
        'c7': np.zeros(n, dtype=np.float32),
    }
    return agents


class TestBundlePrices:
    def test_default_prices(self):
        prices = ch3_model.compute_bundle_prices(THETA_DEFAULT)
        # Default: taxi_BF=0, ps_BF=1, taxi_MA=350, ps_MA=1, ps_VT=1, ps_UA=1
        assert abs(prices['BF'] - 80.0) < 0.01
        assert abs(prices['MA'] - 580.0) < 0.01  # (230+350)*1.0
        assert abs(prices['VT'] - 960.0) < 0.01
        assert abs(prices['UA'] - 1556.8) < 0.01

    def test_scaled_prices(self):
        theta = THETA_DEFAULT.copy()
        theta[1] = 0.8  # ps_BF = 0.8
        prices = ch3_model.compute_bundle_prices(theta)
        assert abs(prices['BF'] - 64.0) < 0.01  # 80 * 0.8


class TestFactorScores:
    def test_factor_shape(self):
        agents = make_test_agents(50)
        factors = ch3_model.compute_factor_scores(agents)
        assert factors['factor1'].shape == (50,)
        assert factors['factor6'].shape == (50,)

    def test_factor_range(self):
        agents = make_test_agents(1000)
        factors = ch3_model.compute_factor_scores(agents)
        # Factors should be in reasonable range (0-5 typically)
        for k, v in factors.items():
            assert v.min() >= -5, f"{k} min too low: {v.min()}"
            assert v.max() <= 10, f"{k} max too high: {v.max()}"


class TestUtilities:
    def test_utility_shape(self):
        agents = make_test_agents(50)
        factors = ch3_model.compute_factor_scores(agents)
        agents.update(factors)
        V = ch3_model.compute_utilities(agents, THETA_DEFAULT)
        assert V.shape == (50, 5)

    def test_v5_reference(self):
        """V5 should not depend on bundle attributes (theta[0:6])."""
        agents = make_test_agents(10)
        factors = ch3_model.compute_factor_scores(agents)
        agents.update(factors)

        theta1 = THETA_DEFAULT.copy()
        theta2 = THETA_DEFAULT.copy()
        theta2[0] = 100  # Change taxi_BF
        theta2[1] = 0.8  # Change ps_BF

        V1 = ch3_model.compute_utilities(agents, theta1)
        V2 = ch3_model.compute_utilities(agents, theta2)

        # V5 should be identical (no bundle attributes)
        np.testing.assert_array_almost_equal(V1[:, 4], V2[:, 4])
        # V1 (BF) should differ
        assert not np.allclose(V1[:, 0], V2[:, 0])


class TestNestedLogit:
    def test_probabilities_sum_to_one(self):
        agents = make_test_agents(100)
        factors = ch3_model.compute_factor_scores(agents)
        agents.update(factors)
        V = ch3_model.compute_utilities(agents, THETA_DEFAULT)
        P = ch3_model.nested_logit_probabilities(V)

        # Probabilities should sum to 1
        sums = P.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(100), decimal=5)

    def test_probabilities_positive(self):
        agents = make_test_agents(100)
        factors = ch3_model.compute_factor_scores(agents)
        agents.update(factors)
        V = ch3_model.compute_utilities(agents, THETA_DEFAULT)
        P = ch3_model.nested_logit_probabilities(V)

        assert np.all(P >= 0)
        assert np.all(P <= 1)

    def test_higher_utility_higher_prob(self):
        """Agent with much higher V1 should have higher P1."""
        V = np.array([[10, 0, 0, 0, 0],
                       [0, 10, 0, 0, 0]], dtype=np.float32)
        P = ch3_model.nested_logit_probabilities(V)
        assert P[0, 0] > P[0, 1]  # First agent prefers alt 1
        assert P[1, 1] > P[1, 0]  # Second agent prefers alt 2


class TestAddedValue:
    def test_av_computation(self):
        V = np.array([[3, 2, 1, 0, -1],
                       [-1, -2, -3, -4, 0]], dtype=np.float32)
        max_av, best_bundle = ch3_model.compute_added_value(V)

        assert max_av[0] == 4.0    # 3 - (-1)
        assert max_av[1] == -1.0   # -1 - 0
        assert best_bundle[0] == 0  # BF is best
        assert best_bundle[1] == 0  # BF is least bad


class TestEndToEnd:
    def test_full_pipeline(self):
        agents = make_test_agents(200)
        result = ch3_model.compute_subscription_probabilities(agents, THETA_DEFAULT)

        assert 'V' in result
        assert 'P' in result
        assert 'max_av' in result
        assert 'best_bundle' in result
        assert 'prices' in result

        assert result['V'].shape == (200, 5)
        assert result['P'].shape == (200, 5)
        assert result['max_av'].shape == (200,)

        # At least some agents should have positive AV
        assert np.any(result['max_av'] > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
