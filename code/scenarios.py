"""
Policy scenarios for the MaaS ABM optimization framework.

Each scenario returns a modifier function that adjusts agents, theta,
and/or Ch3 utility components before the ABM evaluation loop.

5 scenarios:
    S0_baseline              -- No modifications (control)
    S1_carbon_credit         -- Carbon credit benefit added to Ch3 added value
    S2_congestion_charge     -- Congestion surcharge on private car users
    S3_low_income_subsidy    -- 50% bundle price discount for low-income agents
    S4_spatial_differentiation -- District-level price differentiation (22-dim theta)
"""

import numpy as np
from copy import deepcopy

from config import (
    CAR_CO2_PER_KM,
    N_THETA, THETA_NAMES, THETA_LOWER, THETA_UPPER,
    PRICE_BASE_BF, PRICE_BASE_MA, PRICE_BASE_VT, PRICE_BASE_UA,
)


# ============================================================
# Scenario modifier functions
# ============================================================

def _modifier_baseline(agents, theta, ch3_utilities):
    """S0: No modifications -- baseline scenario."""
    return agents, theta, ch3_utilities


def _modifier_carbon_credit(agents, theta, ch3_utilities):
    """S1: Carbon credit benefit increases Ch3 added value for bundles.

    Agents who currently drive receive a carbon credit bonus proportional
    to their monthly car-km.  This bonus is added to the utility of
    bundle alternatives (V1--V4), making subscription more attractive
    relative to V5 (no-purchase).

    AV_bonus = car_km_month * CAR_CO2_PER_KM * carbon_price_per_kg

    Parameters
    ----------
    agents : dict[str, ndarray[N]]
    theta : ndarray
    ch3_utilities : dict
        Must contain keys 'V1'...'V5' as ndarray[N].

    Returns
    -------
    agents, theta, ch3_utilities (modified in-place copies)
    """
    CARBON_PRICE_PER_KG = 0.1  # yuan per kg CO2

    agents = _shallow_copy_agents(agents)
    ch3_utilities = dict(ch3_utilities)

    # Estimate monthly car km: travel_distance_work (km/day) * 22 workdays
    travel_dist = agents.get(
        'travel_distance_work',
        np.zeros(len(agents['sex']), dtype=np.float64),
    )
    have_car = agents.get(
        'have_car',
        np.zeros(len(agents['sex']), dtype=np.float64),
    )
    car_km_month = travel_dist * 22.0 * have_car  # only car owners benefit

    av_bonus = car_km_month * CAR_CO2_PER_KM * CARBON_PRICE_PER_KG

    # Add bonus to bundle utilities V1-V4 (not V5 = no-purchase)
    for vkey in ('V1', 'V2', 'V3', 'V4'):
        if vkey in ch3_utilities:
            ch3_utilities[vkey] = ch3_utilities[vkey] + av_bonus

    return agents, theta, ch3_utilities


def _modifier_congestion_charge(agents, theta, ch3_utilities):
    """S2: Congestion surcharge on private car users.

    Increases the monthly a-la-carte cost for agents who own a car,
    reflecting a congestion pricing policy of 15--25 yuan/day for
    22 working days/month.

    congestion_surcharge = 20 * 22 = 440 yuan/month (average)

    This makes subscription bundles relatively more attractive.
    """
    CONGESTION_SURCHARGE = 20.0 * 22.0  # 440 yuan/month

    agents = _shallow_copy_agents(agents)

    N = len(agents['sex'])
    have_car = agents.get(
        'have_car',
        np.zeros(N, dtype=np.float64),
    )

    # Increase a-la-carte cost for car owners
    cost_alacarte = np.array(
        agents.get('cost_alacarte', np.full(N, 300.0, dtype=np.float64)),
        dtype=np.float64,
        copy=True,
    )
    cost_alacarte += CONGESTION_SURCHARGE * (have_car > 0).astype(np.float64)
    agents['cost_alacarte'] = cost_alacarte

    return agents, theta, ch3_utilities


def _modifier_low_income_subsidy(agents, theta, ch3_utilities):
    """S3: 50% bundle price discount for low-income agents.

    For agents whose income1 == 1 (lowest income bracket), the
    effective bundle prices are halved.  This is implemented by
    adjusting the Ch3 bundle price attribute seen by these agents.
    """
    DISCOUNT_FACTOR = 0.5

    agents = _shallow_copy_agents(agents)
    ch3_utilities = dict(ch3_utilities)

    N = len(agents['sex'])
    income1 = agents.get(
        'income1',
        np.zeros(N, dtype=np.float64),
    )
    is_low_income = (income1 > 0.5).astype(np.float64)

    # Scale down the bundle price component in utilities for low-income agents.
    # The price enters Ch3 utility via B_BUN_PRICE * price.  A 50% discount
    # means the price effect is halved, i.e. we ADD back
    #   |B_BUN_PRICE| * price * 0.5 * is_low_income  to the utility.
    # Since B_BUN_PRICE is negative, the original contribution is negative;
    # adding back half recovers the discount benefit.
    from config import CH3_PARAMS
    b_price = abs(CH3_PARAMS.get('B_BUN_PRICE', 0.0475))

    bundle_prices = {
        'BF': PRICE_BASE_BF,
        'MA': PRICE_BASE_MA,
        'VT': PRICE_BASE_VT,
        'UA': PRICE_BASE_UA,
    }

    bundle_vkeys = {'BF': 'V1', 'MA': 'V2', 'VT': 'V3', 'UA': 'V4'}
    for bundle_name, vkey in bundle_vkeys.items():
        if vkey in ch3_utilities:
            price_relief = b_price * bundle_prices[bundle_name] * DISCOUNT_FACTOR
            ch3_utilities[vkey] = (
                ch3_utilities[vkey] + price_relief * is_low_income
            )

    # Also store a flag for downstream use (e.g. revenue calculation)
    agents['_subsidy_discount'] = DISCOUNT_FACTOR * is_low_income

    return agents, theta, ch3_utilities


def _modifier_spatial_differentiation(agents, theta, ch3_utilities):
    """S4: District-level price differentiation.

    Expands theta from 17-dim to 22-dim by appending 5 district-level
    price-scale parameters:

        theta[17] = ps_core      (core urban: Dongcheng, Xicheng)
        theta[18] = ps_inner     (inner ring: Chaoyang, Haidian, Fengtai, Shijingshan)
        theta[19] = ps_outer     (outer ring: Tongzhou, Shunyi, etc.)
        theta[20] = ps_suburban  (suburban: Changping, Daxing, etc.)
        theta[21] = ps_rural     (rural: Miyun, Yanqing, Huairou, etc.)

    If theta is only 17-dim (standard), the extra 5 parameters default
    to 1.0 (no district differentiation).
    """
    agents = _shallow_copy_agents(agents)
    ch3_utilities = dict(ch3_utilities)

    N = len(agents['sex'])

    # Ensure theta has 22 dimensions; pad with 1.0 if needed
    if len(theta) < 22:
        theta_ext = np.ones(22, dtype=np.float64)
        theta_ext[:len(theta)] = theta
        theta = theta_ext

    # District-group assignment: map district codes to group index (0-4)
    # District codes follow Beijing convention:
    #   Core (0):     Dongcheng(1), Xicheng(2)
    #   Inner (1):    Chaoyang(5), Haidian(8), Fengtai(6), Shijingshan(7)
    #   Outer (2):    Tongzhou(12), Shunyi(13), Fangshan(11)
    #   Suburban (3):  Changping(14), Daxing(15), Mentougou(9), Pinggu(17)
    #   Rural (4):     Miyun(16), Yanqing(18), Huairou(19)
    DISTRICT_TO_GROUP = {
        1: 0, 2: 0,                       # core
        5: 1, 8: 1, 6: 1, 7: 1,           # inner
        12: 2, 13: 2, 11: 2, 3: 2, 4: 2,  # outer (incl. Chongwen, Xuanwu legacy)
        14: 3, 15: 3, 9: 3, 17: 3, 10: 3, # suburban
        16: 4, 18: 4, 19: 4,              # rural
    }

    district = agents.get(
        'district_code',
        np.ones(N, dtype=np.int32),  # default to core
    )

    # Build per-agent district price scale
    district_ps = np.ones(N, dtype=np.float64)
    for dist_code, group_idx in DISTRICT_TO_GROUP.items():
        mask = (district == dist_code)
        district_ps[mask] = theta[17 + group_idx]

    # Apply district price scale to bundle utilities
    # Effect: multiply the price component by district_ps
    # Since B_BUN_PRICE is negative, a lower ps means LESS price disutility
    from config import CH3_PARAMS
    b_price = CH3_PARAMS.get('B_BUN_PRICE', -0.0475)

    bundle_prices = {
        'V1': PRICE_BASE_BF,
        'V2': PRICE_BASE_MA,
        'V3': PRICE_BASE_VT,
        'V4': PRICE_BASE_UA,
    }

    for vkey, base_price in bundle_prices.items():
        if vkey in ch3_utilities:
            # Original contribution: b_price * price * ps_bundle
            # New contribution:      b_price * price * ps_bundle * district_ps
            # Delta:                  b_price * price * ps_bundle * (district_ps - 1)
            # ps_bundle is already embedded in the utility, so we only
            # need to add the marginal district effect
            price_delta = b_price * base_price * (district_ps - 1.0)
            ch3_utilities[vkey] = ch3_utilities[vkey] + price_delta

    # Store district group for downstream analysis
    district_group = np.zeros(N, dtype=np.int32)
    for dist_code, group_idx in DISTRICT_TO_GROUP.items():
        district_group[district == dist_code] = group_idx
    agents['_district_group'] = district_group

    return agents, theta, ch3_utilities


# ============================================================
# Utility
# ============================================================

def _shallow_copy_agents(agents):
    """Shallow copy the agents dict so we don't mutate the original."""
    return {k: v for k, v in agents.items()}


# ============================================================
# Scenario registry
# ============================================================

SCENARIOS = {
    'S0_baseline':              ('S0 Baseline',              _modifier_baseline),
    'S1_carbon_credit':         ('S1 Carbon Credit',         _modifier_carbon_credit),
    'S2_congestion_charge':     ('S2 Congestion Charge',     _modifier_congestion_charge),
    'S3_low_income_subsidy':    ('S3 Low-Income Subsidy',    _modifier_low_income_subsidy),
    'S4_spatial_differentiation': ('S4 Spatial Differentiation', _modifier_spatial_differentiation),
}


def get_scenario(name):
    """Return ``(scenario_name, modifier_fn)`` for the given scenario key.

    Parameters
    ----------
    name : str
        One of the keys in :data:`SCENARIOS`.

    Returns
    -------
    scenario_name : str
        Human-readable scenario label.
    modifier_fn : callable
        ``modifier_fn(agents, theta, ch3_utilities)`` -> ``(agents, theta, ch3_utilities)``

    Raises
    ------
    KeyError
        If *name* is not a registered scenario.
    """
    if name not in SCENARIOS:
        available = ', '.join(sorted(SCENARIOS.keys()))
        raise KeyError(
            f"Unknown scenario '{name}'. Available: {available}"
        )
    return SCENARIOS[name]


# ============================================================
# Extended bounds for S4 (22-dim theta)
# ============================================================

def get_extended_bounds():
    """Return (lower, upper, names) for the 22-dim theta used in S4.

    The 5 extra parameters are district-level price scales in [0.5, 1.5].
    """
    district_names = [
        'ps_core', 'ps_inner', 'ps_outer', 'ps_suburban', 'ps_rural'
    ]
    ext_lower = np.concatenate([THETA_LOWER, np.full(5, 0.5)])
    ext_upper = np.concatenate([THETA_UPPER, np.full(5, 1.5)])
    ext_names = list(THETA_NAMES) + district_names
    return ext_lower, ext_upper, ext_names
