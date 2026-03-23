"""
Ch1 LC-HCM trial probability model (pure NumPy).

Computes the probability that each agent will try MaaS in a single
choice occasion, using a Latent-Class Hybrid Choice Model (LC-HCM):

    P_try = P(class1) * P(MaaS | class1) + P(class2) * P(MaaS | class2)

The monthly trial probability accounts for multiple trip occasions:

    P_try_month = 1 - (1 - P_try_single) ^ trips_per_month
"""

import numpy as np
from config import CH1_PARAMS


# ------------------------------------------------------------------ #
#  Population-average defaults for SP-specific attributes             #
# ------------------------------------------------------------------ #
_DEFAULTS = {
    'MaasFamiliar': 0.3,
    'normal_depart': 0.5,
    'trips_per_month': 20.0,
}

# Thresholds for deriving binary SP indicators from survey behaviour
_TAXI_HIGH_THRESHOLD = 3       # week_taxi >= 3 => first_taxi = 1
_PT_HIGH_THRESHOLD = 3         # week_bus + week_metro >= 3 => first_pt = 1
_DISTANCE_WEEKEND_THRESHOLD = 10.0  # km; distance_weekend > 10 => distance5 = 1


# ------------------------------------------------------------------ #
#  Helper: safe softmax that avoids overflow                          #
# ------------------------------------------------------------------ #
def _softmax_cols(V):
    """Row-wise softmax over a 2-D array (N, J) -> (N, J) probabilities."""
    V_max = V.max(axis=1, keepdims=True)
    expV = np.exp(V - V_max)
    return expV / expV.sum(axis=1, keepdims=True)


def _safe_get(agents, key, default=None):
    """Retrieve a field from the agents dict, falling back to a scalar default."""
    if key in agents:
        return agents[key]
    if default is not None:
        return default
    raise KeyError(f"agents dict missing required key '{key}' with no default")


# ------------------------------------------------------------------ #
#  1. FACTOR1 structural equation (Ch1 version)                       #
# ------------------------------------------------------------------ #
def compute_ch1_factor1(agents, params=None):
    """Compute the Ch1 latent variable FACTOR1 for every agent.

    Parameters
    ----------
    agents : dict[str, ndarray[N]]
        Agent attribute arrays.
    params : dict, optional
        Model parameters; defaults to ``CH1_PARAMS``.

    Returns
    -------
    factor1 : ndarray[N]
    """
    if params is None:
        params = CH1_PARAMS

    N = len(next(iter(agents.values())))
    factor1 = np.zeros(N, dtype=np.float64)

    # Gender (sex: 1=male, 2=female typically)
    factor1 += params['ch1_coef1_gender'] * _safe_get(agents, 'sex')

    # Age dummies (age1, age2, age3)
    factor1 += params['ch1_coef1_age1'] * _safe_get(agents, 'age1')
    factor1 += params['ch1_coef1_age2'] * _safe_get(agents, 'age2')
    factor1 += params['ch1_coef1_age3'] * _safe_get(agents, 'age3')

    # Occupation
    factor1 += params['ch1_coef1_job'] * _safe_get(agents, 'occupy')

    # Income dummies
    factor1 += params['ch1_coef1_income1'] * _safe_get(agents, 'income1')
    factor1 += params['ch1_coef1_income2'] * _safe_get(agents, 'income2')

    # Education
    factor1 += params['ch1_coef1_education'] * _safe_get(agents, 'education')

    # Travel characteristics
    factor1 += params['ch1_coef1_travel_num'] * _safe_get(agents, 'travel_num')
    factor1 += params['ch1_coef1_travel_distance_day'] * _safe_get(
        agents, 'travel_distance_work')

    # Attitude / mode-choice indicators (6d-6g from questionnaire)
    factor1 += params['ch1_coef1_6d'] * _safe_get(agents, 'Al_shareedcar',
                                                    default=np.zeros(N))
    factor1 += params['ch1_coef1_6e'] * _safe_get(agents, 'Al_bike',
                                                    default=np.zeros(N))
    factor1 += params['ch1_coef1_6f'] * _safe_get(agents, 'Al_sharedbike',
                                                    default=np.zeros(N))
    factor1 += params['ch1_coef1_6g'] * _safe_get(agents, 'Al_walk',
                                                    default=np.zeros(N))

    # Car ownership and transit usage
    factor1 += params['ch1_coef1_car_home'] * _safe_get(agents, 'have_car')
    factor1 += params['ch1_coef1_metro'] * _safe_get(agents, 'week_metro')
    factor1 += params['ch1_coef1_ebike'] * _safe_get(agents, 'week_ebike')

    return factor1


# ------------------------------------------------------------------ #
#  2. Class membership probability (binary logit)                     #
# ------------------------------------------------------------------ #
def compute_class_membership(agents, factor1, params=None):
    """Compute latent-class membership probabilities (2 classes).

    Class 2 is the reference class (V_class2 = 0).

    Parameters
    ----------
    agents : dict[str, ndarray[N]]
    factor1 : ndarray[N]
        Output of :func:`compute_ch1_factor1`.
    params : dict, optional

    Returns
    -------
    P1, P2 : ndarray[N], ndarray[N]
        Probability of belonging to class 1 and class 2.
    """
    if params is None:
        params = CH1_PARAMS

    N = len(factor1)

    MaasFamiliar = _safe_get(agents, 'MaasFamiliar',
                             default=np.full(N, _DEFAULTS['MaasFamiliar']))
    Al_taxi = _safe_get(agents, 'Al_taxi', default=np.zeros(N))
    Al_PT = _safe_get(agents, 'Al_PT', default=np.zeros(N))
    Al_bike = _safe_get(agents, 'Al_bike', default=np.zeros(N))
    Carown = _safe_get(agents, 'Carown', default=_safe_get(agents, 'have_car',
                                                            default=np.zeros(N)))
    travel_distance_weekend = _safe_get(agents, 'travel_distance_weekend',
                                        default=np.zeros(N))

    V1 = (params['C_ASC']
          + params['C_B_MAASFAMILAR'] * MaasFamiliar
          + params['C_B_TAXI'] * Al_taxi
          + params['C_B_PT'] * Al_PT
          + params['C_B_BIKE'] * Al_bike
          + params['C_B_HAVECAR'] * Carown
          + params['C_B_TRAVEL_DISTANCE_END'] * travel_distance_weekend
          + params['C_B_FACTOR1'] * factor1)

    # Numerically stable sigmoid: P1 = 1 / (1 + exp(-V1))
    P1 = 1.0 / (1.0 + np.exp(-np.clip(V1, -500, 500)))
    P2 = 1.0 - P1

    return P1, P2


# ------------------------------------------------------------------ #
#  3. Within-class MaaS choice probability                            #
# ------------------------------------------------------------------ #
def _derive_sp_indicators(agents, N):
    """Derive SP-experiment binary variables from survey travel data.

    For agents that already carry the SP fields (e.g. from SP survey merge)
    the existing values are returned.  Otherwise, heuristic mappings from
    the 2023 travel-survey attributes are used.
    """
    # first_car: 1 if the agent currently uses car as primary mode
    first_car = _safe_get(
        agents, 'first_car',
        default=((_safe_get(agents, 'have_car', default=np.zeros(N))) > 0).astype(np.float64))

    # first_taxi: 1 if agent is a frequent taxi user
    first_taxi = _safe_get(
        agents, 'first_taxi',
        default=((_safe_get(agents, 'week_taxi', default=np.zeros(N)))
                 >= _TAXI_HIGH_THRESHOLD).astype(np.float64))

    # first_pt: 1 if agent is a frequent public-transit user
    if 'first_pt' in agents:
        first_pt = agents['first_pt']
    else:
        week_bus = _safe_get(agents, 'week_bus', default=np.zeros(N))
        week_metro = _safe_get(agents, 'week_metro', default=np.zeros(N))
        first_pt = ((week_bus + week_metro) >= _PT_HIGH_THRESHOLD).astype(np.float64)

    # distance5: 1 if weekend travel distance exceeds threshold
    distance5 = _safe_get(
        agents, 'distance5',
        default=((_safe_get(agents, 'travel_distance_weekend', default=np.zeros(N)))
                 > _DISTANCE_WEEKEND_THRESHOLD).astype(np.float64))

    # normal_depart: dummy for "normal" departure-time window
    normal = _safe_get(agents, 'normal_depart',
                       default=np.full(N, _DEFAULTS['normal_depart']))

    return first_car, first_taxi, first_pt, distance5, normal


def _base_trip_times(N):
    """Return base rail_time and trip_time for each of the 4 MaaS modes.

    In a full implementation these come from the SP experiment design or
    network skim matrices.  Here we use representative Beijing averages
    (in minutes) so the model produces meaningful probabilities.

    Returns
    -------
    rail_times : dict  keys 'm1'..'m3' (M4 has no rail segment)
    trip_times : dict  keys 'm1'..'m4'
    """
    rail_times = {
        'm1': np.full(N, 25.0),   # Metro + bus
        'm2': np.full(N, 30.0),   # Metro + bike-share
        'm3': np.full(N, 20.0),   # Metro + taxi
    }
    trip_times = {
        'm1': np.full(N, 45.0),   # Metro + bus total
        'm2': np.full(N, 40.0),   # Metro + bike-share total
        'm3': np.full(N, 35.0),   # Metro + taxi total
        'm4': np.full(N, 30.0),   # Direct taxi / ride-hail
    }
    return rail_times, trip_times


def compute_within_class_maas_prob(agents, theta, params=None, class_id=1):
    """Compute P(choose any MaaS alternative | class k) for each agent.

    The choice set has 5 alternatives:
        1 = no-transfer / keep current mode (car)
        4 = MaaS mode M1 (metro + bus)
        5 = MaaS mode M2 (metro + bike-share)
        6 = MaaS mode M3 (metro + taxi)
        7 = MaaS mode M4 (direct taxi / ride-hail)

    MaaS trial probability is P(choose 4 or 5 or 6 or 7).

    Parameters
    ----------
    agents : dict[str, ndarray[N]]
    theta : ndarray[17]
        Strategy vector.  theta[15] = time_improvement, theta[16] = price_discount.
    params : dict, optional
    class_id : int
        1 or 2 (the latent class).

    Returns
    -------
    P_maas : ndarray[N]
        Probability of choosing any MaaS mode for each agent.
    """
    if params is None:
        params = CH1_PARAMS

    k = str(class_id)  # '1' or '2'
    N = len(next(iter(agents.values())))

    # ---- Derive / retrieve SP variables ----
    first_car, first_taxi, first_pt, distance5, normal = _derive_sp_indicators(agents, N)

    # ---- Service quality adjustments from theta ----
    time_improvement = theta[15]
    price_discount = theta[16]

    rail_times, trip_times = _base_trip_times(N)

    # Apply time improvement to MaaS modes
    adj_factor = 1.0 - time_improvement
    rail_time_m1 = rail_times['m1'] * adj_factor
    rail_time_m2 = rail_times['m2'] * adj_factor
    rail_time_m3 = rail_times['m3'] * adj_factor
    trip_time_m1 = trip_times['m1'] * adj_factor
    trip_time_m2 = trip_times['m2'] * adj_factor
    trip_time_m3 = trip_times['m3'] * adj_factor
    trip_time_m4 = trip_times['m4'] * adj_factor

    # Price discount: increases MaaS utility by reducing perceived cost.
    # Applied as a positive utility bonus proportional to the discount ratio.
    # Sensitivity coefficient: ~2.0 (moderate price elasticity in mode choice)
    price_bonus = 2.0 * price_discount

    # ---- Build utilities for 5 alternatives ----
    # Alternative 1: no-transfer (keep car / current mode)
    # V1 = 0 (reference), but we add same-choice and distance effects
    V1 = (params.get(f'B_SAME_CHOICE_CAR{k}', 0.0) * first_car
          + params.get(f'B_SAME_CHOICE_TAXI{k}', 0.0) * first_taxi
          + params.get(f'B_DISTANCE5{k}', 0.0) * distance5)

    # Alternative 4: MaaS M1 (metro + bus)
    V4 = (params[f'ASC_4{k}']
          + params[f'B_RAIL_TIME{k}'] * rail_time_m1
          + params[f'B_TRIP_TIME{k}'] * trip_time_m1
          + params[f'B_FIRST_PT{k}'] * first_pt
          + params[f'B_DEPARTTIME3{k}'] * normal
          + price_bonus)

    # Alternative 5: MaaS M2 (metro + bike-share)
    V5 = (params[f'ASC_5{k}']
          + params[f'B_RAIL_TIME{k}'] * rail_time_m2
          + params[f'B_TRIP_TIME{k}'] * trip_time_m2
          + params[f'B_FIRST_PT{k}'] * first_pt
          + params[f'B_DEPARTTIME3{k}'] * normal
          + price_bonus)

    # Alternative 6: MaaS M3 (metro + taxi)
    V6 = (params[f'ASC_6{k}']
          + params[f'B_RAIL_TIME{k}'] * rail_time_m3
          + params[f'B_TRIP_TIME{k}'] * trip_time_m3
          + params[f'B_FIRST_TAXI{k}'] * first_taxi
          + params[f'B_DEPARTTIME3{k}'] * normal
          + price_bonus)

    # Alternative 7: MaaS M4 (direct taxi / ride-hail)
    V7 = (params[f'ASC_7{k}']
          + params[f'B_TRIP_TIME{k}'] * trip_time_m4
          + params[f'B_FIRST_TAXI{k}'] * first_taxi
          + params[f'B_DISTANCE5{k}'] * distance5
          + price_bonus)

    # Stack into (N, 5) and apply softmax
    V = np.column_stack([V1, V4, V5, V6, V7])  # shape (N, 5)
    P = _softmax_cols(V)                          # shape (N, 5)

    # P(MaaS) = P(alt 4) + P(alt 5) + P(alt 6) + P(alt 7)
    P_maas = P[:, 1] + P[:, 2] + P[:, 3] + P[:, 4]

    return P_maas


# ------------------------------------------------------------------ #
#  4. Main entry point: trial probability                             #
# ------------------------------------------------------------------ #
def compute_trial_probability(agents, theta, params=None):
    """Compute the monthly MaaS trial probability for every agent.

    This is the main entry point used by the ABM simulation loop.

    Steps:
        1. Compute Ch1 FACTOR1 latent variable.
        2. Compute class-membership probabilities (binary logit).
        3. For each class, compute within-class P(MaaS) via MNL.
        4. Aggregate:  P_try_single = P1 * P_maas_1 + P2 * P_maas_2.
        5. Scale to monthly: P_try_month = 1 - (1 - P_try_single)^N_trips.

    Parameters
    ----------
    agents : dict[str, ndarray[N]]
        Agent attribute arrays. See module docstring for required keys.
    theta : ndarray[17]
        Strategy / policy vector.
    params : dict, optional
        Model parameters; defaults to ``CH1_PARAMS``.

    Returns
    -------
    P_try_month : ndarray[N]
        Monthly trial probability for each agent, in [0, 1].
    """
    if params is None:
        params = CH1_PARAMS

    N = len(next(iter(agents.values())))

    # Step 1: Latent variable
    factor1 = compute_ch1_factor1(agents, params)

    # Step 2: Class membership
    P1, P2 = compute_class_membership(agents, factor1, params)

    # Step 3: Within-class MaaS probabilities
    P_maas_1 = compute_within_class_maas_prob(agents, theta, params, class_id=1)
    P_maas_2 = compute_within_class_maas_prob(agents, theta, params, class_id=2)

    # Step 4: Mixture across classes
    P_try_single = P1 * P_maas_1 + P2 * P_maas_2

    # Step 5: Monthly aggregation over repeated trip occasions
    trips_per_month = _safe_get(agents, 'trips_per_month',
                                default=np.full(N, _DEFAULTS['trips_per_month']))
    trips_per_month = np.maximum(trips_per_month, 1.0)  # at least 1

    P_try_month = 1.0 - np.power(1.0 - np.clip(P_try_single, 0.0, 1.0),
                                  trips_per_month)

    return P_try_month
