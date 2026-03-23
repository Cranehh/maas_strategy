"""
Ch3 ICLV Nested Logit model — pure NumPy vectorized implementation.
Reproduces the utility functions and nested logit from the Biogeme source code:
MaaS_HCM - nofactor5 - mc-同时也去掉聚类 - 小汽车放选择.py
"""
import numpy as np
from config import CH3_PARAMS, PRICE_BASE_BF, PRICE_BASE_MA, PRICE_BASE_VT, PRICE_BASE_UA


def compute_bundle_prices(theta):
    """Compute absolute bundle prices from theta.

    Args:
        theta: ndarray[17] strategy parameters.
            [0] taxi_BF, [1] ps_BF, [2] taxi_MA, [3] ps_MA,
            [4] ps_VT, [5] ps_UA

    Returns:
        dict with keys 'BF','MA','VT','UA' -> float price (元/月)
    """
    taxi_BF, ps_BF = theta[0], theta[1]
    taxi_MA, ps_MA = theta[2], theta[3]
    ps_VT, ps_UA = theta[4], theta[5]

    price_BF = (PRICE_BASE_BF + taxi_BF) * ps_BF
    price_MA = (PRICE_BASE_MA + taxi_MA) * ps_MA
    price_VT = PRICE_BASE_VT * ps_VT
    price_UA = PRICE_BASE_UA * ps_UA

    return {
        'BF': price_BF,
        'MA': price_MA,
        'VT': price_VT,
        'UA': price_UA,
    }


def compute_factor_scores(agents, params=None):
    """Compute FACTOR1-4,6 using Ch3 structural equation coefficients.

    For ABM usage: deterministic mean (no random sigma*draw term).

    Args:
        agents: dict of numpy arrays with agent attributes
        params: dict of Ch3 parameters (defaults to CH3_PARAMS)

    Returns:
        dict with keys 'factor1','factor2','factor3','factor4','factor6'
              -> ndarray[N] for each factor
    """
    if params is None:
        params = CH3_PARAMS

    N = len(agents['age1'])
    factors = {}

    for k in [1, 2, 3, 4, 6]:
        prefix = f'coef{k}_'
        f = np.zeros(N, dtype=np.float32)

        # Age dummies (age1, age2, age3 — age4 not included in structural equation)
        f += params[f'{prefix}age1'] * agents['age1']
        f += params[f'{prefix}age2'] * agents['age2']
        f += params[f'{prefix}age3'] * agents['age3']

        # Job (occupy)
        f += params[f'{prefix}job'] * agents['occupy']

        # Income dummies
        f += params[f'{prefix}income1'] * agents['income1']
        f += params[f'{prefix}income2'] * agents['income2']
        f += params.get(f'{prefix}income3', 0.0) * agents.get('income3', np.zeros(N, dtype=np.float32))

        # Travel behavior
        f += params[f'{prefix}travel_num'] * agents['travel_num']
        f += params[f'{prefix}travel_distance_day'] * agents['travel_distance_work']
        f += params[f'{prefix}travel_aim'] * agents['travel_aim']

        # Mode usage
        f += params[f'{prefix}bus'] * agents['week_bus']
        f += params[f'{prefix}metro'] * agents['week_metro']
        f += params[f'{prefix}taxi'] * agents['week_taxi']
        f += params[f'{prefix}ebike'] * agents['week_ebike']
        f += params[f'{prefix}bike'] * agents['week_bike']

        factors[f'factor{k}'] = f

    return factors


def compute_utilities(agents, theta, params=None):
    """Compute V1-V5 utility for each agent, exactly matching Ch3 source code.

    Variable scaling rules (must match source code lines 96-109):
    - taxi variables: ÷100 (source line 102)
    - price variables: ÷100 (source lines 106-109)
    - price_scale variables: NOT scaled (raw 0.7-1.3)
    - ebike variables: ÷10 (source line 98) — NOT used in final model

    Args:
        agents: dict of numpy arrays (N agents)
        theta: ndarray[17] strategy parameters
        params: dict of model parameters (defaults to CH3_PARAMS)

    Returns:
        V: ndarray[N, 5] — utilities for [BF, MA, VT, UA, No-purchase]
    """
    if params is None:
        params = CH3_PARAMS

    N = len(agents['sex'])
    V = np.zeros((N, 5), dtype=np.float32)

    # Unpack theta for bundle attributes
    taxi_BF, ps_BF = theta[0], theta[1]
    taxi_MA, ps_MA = theta[2], theta[3]
    ps_VT, ps_UA = theta[4], theta[5]

    # Compute absolute prices
    prices = compute_bundle_prices(theta)
    price_BF = prices['BF']
    price_MA = prices['MA']
    price_VT = prices['VT']
    price_UA = prices['UA']

    # Get factor scores (pre-computed or compute on the fly)
    if all(f'factor{k}' in agents for k in [1, 2, 3, 4, 6]):
        f1 = agents['factor1']
        f2 = agents['factor2']
        f3 = agents['factor3']
        f4 = agents['factor4']
        f6 = agents['factor6']
    else:
        factors = compute_factor_scores(agents, params)
        f1 = factors['factor1']
        f2 = factors['factor2']
        f3 = factors['factor3']
        f4 = factors['factor4']
        f6 = factors['factor6']

    # Shorthand for params
    p = params

    # ======================================================
    # V1: BF (Basic Flexibility) — source lines 1187-1213
    # ======================================================
    V[:, 0] = (
        p['ASC_1']
        + p['B_BUN_TAXI'] * (taxi_BF / 100.0)          # taxi÷100
        + p['B_BUN_PRICERATIO'] * ps_BF                 # price_scale NOT ÷
        + p['B_BUN_PRICE'] * (price_BF / 100.0)         # price÷100
        + p['B_BUS_NUM'] * agents['week_bus']
        + p['B_EBIKE_HOME'] * agents['e_bike']
        + p['B_OCCUPY'] * agents['occupy']
        + p['B_SEX'] * agents['sex']
        + p['B_INCOME1'] * agents['income1']
        + p['B_AGE4'] * agents['age4']
        + p['B_FACTOR2'] * f2
        + p['B_FACTOR4'] * f4
        + p['B_FACTOR3'] * f3
        + p['B_FACTOR6'] * f6
    )

    # ======================================================
    # V2: MA (Mobility Advantage) — source lines 1215-1241
    # ======================================================
    V[:, 1] = (
        p['ASC_2']
        + p['B_BUN_TAXI'] * (taxi_MA / 100.0)           # taxi÷100 (shared taxi_12)
        + p['B_BUN_PRICERATIO'] * ps_MA                  # uses price_12 -> ps_MA
        + p['B_BUN_PRICE'] * (price_MA / 100.0)          # price÷100
        + p['B_TRAVEL_DISTANCE_WORK'] * agents['travel_distance_work']
        + p['B_METRO_NUM'] * agents['week_metro']
        + p['B_EBIKE_HOME'] * agents['e_bike']
        + p['B_OCCUPY'] * agents['occupy']
        + p['B_SEX'] * agents['sex']
        + p['B_COMBINE_SHAREBIKE'] * agents.get('c7', np.zeros(N, dtype=np.float32))
        + p['B_INCOME1'] * agents['income1']
        + p['B_AGE4'] * agents['age4']
        + p['B_FACTOR2'] * f2
        + p['B_FACTOR4'] * f4
        + p['B_FACTOR3'] * f3
        + p['B_FACTOR6'] * f6
    )

    # ======================================================
    # V3: VT (Value Traveler) — source lines 1243-1267
    # ======================================================
    V[:, 2] = (
        p['ASC_3']
        + p['B_BUN_PRICERATIO'] * ps_VT                  # price_3 -> ps_VT
        + p['B_BUN_PRICE'] * (price_VT / 100.0)          # price÷100
        + p['B_TRAVEL_DISTANCE_END'] * agents['travel_distance_weekend']
        + p['B_TAXI_NUM'] * agents['week_taxi']
        + p['B_AGE3'] * agents['age3']
        + p['B_INCOME2'] * agents['income2']
        + p['B_FACTOR2'] * f2
        + p['B_FACTOR3'] * f3
        + p['B_FACTOR6'] * f6
    )

    # ======================================================
    # V4: UA (Unlimited Access) — source lines 1269-1295
    # ======================================================
    V[:, 3] = (
        p['ASC_4']
        + p['B_BUN_PRICERATIO'] * ps_UA                  # price_4 -> ps_UA
        + p['B_BUN_PRICE'] * (price_UA / 100.0)          # price÷100
        + p['B_CAR'] * agents.get('c6', np.zeros(N, dtype=np.float32))
        + p['B_COST'] * agents['cost']
        + p['B_TAXI_NUM'] * agents['week_taxi']
        + p['B_AGE3'] * agents['age3']
        + p['B_FACTOR2'] * f2
        + p['B_FACTOR3'] * f3
        + p['B_FACTOR6'] * f6
    )

    # ======================================================
    # V5: No-purchase (reference) — source lines 1297-1313
    # ======================================================
    V[:, 4] = (
        0.0  # ASC_5 fixed to 0
        + p['B_COST'] * agents['cost']
        + p['B_LICENSE'] * agents['license']
        + p['B_HAVECAR'] * agents['have_car']
        + p['B_EDUCATION'] * agents['education']
        + p['B_FACTOR1'] * f1
    )

    return V


def nested_logit_probabilities(V, mu1=None, params=None):
    """Compute Nested Logit choice probabilities.

    Nest structure (source lines 1343-1347):
        PT   = MU1, [alt 1 (BF), alt 2 (MA)]
        TAXI = 1.0, [alt 3 (VT)]
        MORE = 1.0, [alt 4 (UA)]
        NO   = 1.0, [alt 5 (No-purchase)]

    Args:
        V: ndarray[N, 5] utilities
        mu1: nesting parameter for PT nest (default from params)
        params: parameter dict (defaults to CH3_PARAMS)

    Returns:
        P: ndarray[N, 5] choice probabilities
    """
    if params is None:
        params = CH3_PARAMS
    if mu1 is None:
        mu1 = params['MU1']

    N = V.shape[0]

    # For numerical stability, subtract max
    V_max = np.max(V, axis=1, keepdims=True)
    V_shifted = V - V_max

    # Compute nest log-sums
    # PT nest: alternatives 0, 1 with scale mu1
    exp_V_pt = np.exp(mu1 * V_shifted[:, :2])  # [N, 2]
    G_pt = np.sum(exp_V_pt, axis=1)             # [N]
    logsum_pt = np.log(G_pt + 1e-30) / mu1      # [N]

    # Single-alt nests: alternatives 2, 3, 4 with scale 1
    # Their log-sum equals their own utility
    logsum_taxi = V_shifted[:, 2]
    logsum_more = V_shifted[:, 3]
    logsum_no = V_shifted[:, 4]

    # Upper-level logit over nests
    exp_nests = np.column_stack([
        np.exp(logsum_pt),
        np.exp(logsum_taxi),
        np.exp(logsum_more),
        np.exp(logsum_no),
    ])  # [N, 4]
    nest_sum = np.sum(exp_nests, axis=1, keepdims=True)  # [N, 1]
    P_nest = exp_nests / (nest_sum + 1e-30)               # [N, 4]

    # Within-nest probabilities
    # PT nest: conditional on choosing PT nest
    P_within_pt = exp_V_pt / (G_pt[:, None] + 1e-30)      # [N, 2]

    # Final probabilities
    P = np.zeros((N, 5), dtype=np.float32)
    P[:, 0] = P_nest[:, 0] * P_within_pt[:, 0]  # BF
    P[:, 1] = P_nest[:, 0] * P_within_pt[:, 1]  # MA
    P[:, 2] = P_nest[:, 1]                        # VT
    P[:, 3] = P_nest[:, 2]                        # UA
    P[:, 4] = P_nest[:, 3]                        # No-purchase

    return P


def compute_added_value(V):
    """Compute maximum added value (AV) for each agent.

    AV = max(V1, V2, V3, V4) - V5
    Positive AV means the agent values at least one bundle over no-purchase.

    Args:
        V: ndarray[N, 5] utilities

    Returns:
        max_av: ndarray[N] — maximum added value
        best_bundle: ndarray[N] — index of best bundle (0-3)
    """
    V_bundles = V[:, :4]
    V_no = V[:, 4]

    best_bundle = np.argmax(V_bundles, axis=1).astype(np.int8)
    max_bundle_V = np.max(V_bundles, axis=1)
    max_av = max_bundle_V - V_no

    return max_av, best_bundle


def compute_subscription_probabilities(agents, theta, params=None):
    """Convenience function: compute all Ch3 outputs at once.

    Args:
        agents: dict of agent arrays
        theta: ndarray[17]
        params: Ch3 parameter dict

    Returns:
        dict with keys:
            'V': ndarray[N,5] utilities
            'P': ndarray[N,5] nested logit probabilities
            'max_av': ndarray[N] added value
            'best_bundle': ndarray[N] best bundle index
            'prices': dict of bundle prices
    """
    V = compute_utilities(agents, theta, params)
    P = nested_logit_probabilities(V, params=params)
    max_av, best_bundle = compute_added_value(V)
    prices = compute_bundle_prices(theta)

    return {
        'V': V,
        'P': P,
        'max_av': max_av,
        'best_bundle': best_bundle,
        'prices': prices,
    }
