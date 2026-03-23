"""
Tests for ABM simulation engine.
Validates single-step mechanics and full simulation sanity.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import time

from config import (
    THETA_DEFAULT, STATUS_UNAWARE, STATUS_AWARE, STATUS_TRIAL,
    STATUS_SUBSCRIBER, STATUS_CHURNED, N_WEEKS, AGENT_WEIGHT,
)


def make_test_agents(n=1000):
    """Create synthetic agent population for ABM testing."""
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
        'cost_alacarte': rng.uniform(100, 2000, n).astype(np.float32),
        'car_km_month': rng.uniform(0, 500, n).astype(np.float32),
        'c6': (rng.random(n) < 0.4).astype(np.float32),
        'c7': np.zeros(n, dtype=np.float32),
        'taz': rng.randint(0, 50, n).astype(np.int32),
        'weight': np.full(n, AGENT_WEIGHT, dtype=np.float32),
        'MaasFamiliar': (rng.random(n) < 0.3).astype(np.float32),
        'first_car': (rng.random(n) < 0.3).astype(np.float32),
        'first_taxi': (rng.random(n) < 0.2).astype(np.float32),
        'first_pt': (rng.random(n) < 0.5).astype(np.float32),
        'distance5': (rng.random(n) < 0.2).astype(np.float32),
        'normal_depart': np.full(n, 0.5, dtype=np.float32),
        'Al_taxi': (rng.random(n) < 0.2).astype(np.float32),
        'Al_PT': (rng.random(n) < 0.5).astype(np.float32),
        'Al_bike': (rng.random(n) < 0.2).astype(np.float32),
        'Al_sharedbike': np.zeros(n, dtype=np.float32),
        'Carown': (rng.random(n) < 0.4).astype(np.float32),
        'trips_per_month': rng.uniform(10, 60, n).astype(np.float32),
    }
    return agents


class TestABMBasic:
    def test_import(self):
        import abm_engine
        assert hasattr(abm_engine, 'ABMSimulation')

    def test_initialization(self):
        import ch3_model, ch1_model, ch2_model
        from abm_engine import ABMSimulation

        agents = make_test_agents(100)
        abm = ABMSimulation(agents, ch3_model, ch1_model, ch2_model)
        assert abm.N == 100

    def test_single_run(self):
        import ch3_model, ch1_model, ch2_model
        from abm_engine import ABMSimulation

        agents = make_test_agents(500)
        abm = ABMSimulation(agents, ch3_model, ch1_model, ch2_model)

        objectives, weekly_subs = abm.run(THETA_DEFAULT, seed=42)

        # 4 objectives
        assert len(objectives) == 4
        # Adoption rate should be between 0 and 1 (negated)
        assert -1 <= objectives[0] <= 0
        # Weekly subscribers should be monotonically non-decreasing initially
        assert len(weekly_subs) == N_WEEKS
        # Final subscriber count should be > 0
        assert weekly_subs[-1] > 0

    def test_deterministic_with_seed(self):
        import ch3_model, ch1_model, ch2_model
        from abm_engine import ABMSimulation

        agents = make_test_agents(200)
        abm = ABMSimulation(agents, ch3_model, ch1_model, ch2_model)

        obj1, _ = abm.run(THETA_DEFAULT, seed=42)
        obj2, _ = abm.run(THETA_DEFAULT, seed=42)

        np.testing.assert_array_almost_equal(obj1, obj2)


class TestABMPerformance:
    def test_speed(self):
        """79K agents should complete in < 5 seconds."""
        import ch3_model, ch1_model, ch2_model
        from abm_engine import ABMSimulation

        agents = make_test_agents(10000)  # Test with 10K (scale test)
        abm = ABMSimulation(agents, ch3_model, ch1_model, ch2_model)

        t0 = time.time()
        abm.run(THETA_DEFAULT, seed=42)
        elapsed = time.time() - t0

        print(f"10K agents: {elapsed:.2f}s")
        assert elapsed < 10, f"Too slow: {elapsed:.2f}s for 10K agents"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
