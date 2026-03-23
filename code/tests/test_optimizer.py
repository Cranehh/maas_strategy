"""
Tests for optimizer: end-to-end optimization flow.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from config import THETA_DEFAULT, THETA_LOWER, THETA_UPPER, N_THETA


class TestOptimizer:
    def test_import(self):
        from surrogate import AnalyticalSurrogate, GPResidualModel, SurrogateEvaluator
        from optimizer import MaaSSBOProblem

    def test_theta_bounds(self):
        assert len(THETA_LOWER) == N_THETA
        assert len(THETA_UPPER) == N_THETA
        assert np.all(THETA_LOWER <= THETA_UPPER)
        assert np.all(THETA_LOWER <= THETA_DEFAULT)
        assert np.all(THETA_DEFAULT <= THETA_UPPER)

    def test_constraints(self):
        """Test constraint: ps_BF <= ps_MA."""
        theta = THETA_DEFAULT.copy()
        # G0 = ps_BF - ps_MA <= 0
        g0 = theta[1] - theta[3]
        assert g0 <= 0, f"Default theta violates ps_BF <= ps_MA: {theta[1]} > {theta[3]}"

        # G1 = tau_low - tau_high < 0
        g1 = theta[7] - theta[6]
        assert g1 < 0, f"Default theta violates tau_low < tau_high: {theta[7]} >= {theta[6]}"

    def test_analytical_surrogate(self):
        from tests.test_ch3 import make_test_agents
        import ch3_model, ch1_model, ch2_model
        from surrogate import AnalyticalSurrogate

        agents = make_test_agents(200)
        surr = AnalyticalSurrogate(agents, ch3_model, ch1_model, ch2_model)
        F = surr.evaluate(THETA_DEFAULT)

        assert len(F) == 4
        # Adoption should be non-positive (negated for minimization)
        assert F[0] <= 0
        # Gini should be in [0, 1]
        assert 0 <= F[2] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
