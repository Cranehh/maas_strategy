"""
Ch2 Subscription Transition Probability Model.

Estimates the probability that an agent transitions from trial to subscription
using a two-stage logistic regression approach:

    P_subscribe = P(transfer_group) * P(sub | transfer_group)

Stage 1 - Transfer group probability:
    Based on socio-demographics and travel patterns, identifies agents who
    are psychologically open to committing to a subscription.

Stage 2 - Subscription probability (conditional):
    Given an agent is in the transfer group *and* has tried MaaS, the
    subscription probability depends on:
      - Added value experienced during trial (from Ch3 model)
      - Cost savings of the best bundle vs a-la-carte spending
      - Accumulated trial experience (trial_count)

All computation is vectorized NumPy; no agent-level loops.
"""

import numpy as np

from config import (
    MIN_TRIALS_FOR_SUBSCRIBE,
    PRICE_BASE_BF, PRICE_BASE_MA, PRICE_BASE_VT, PRICE_BASE_UA,
)

# ============================================================
# Default coefficients for the transfer-group logit (Stage 1)
# ============================================================
# Calibrated to produce P(transfer) in roughly [0.3, 0.7] for the
# Beijing survey population.  Signs follow behavioural intuition:
#   - younger adults (age1, age2) more open to new mobility
#   - moderate income (income2) more price-sensitive -> subscription value
#   - frequent PT users see subscription value
#   - car owners less likely to switch
#   - higher education -> more receptive to innovation

_DEFAULT_TRANSFER_PARAMS = {
    'b0':         -0.60,
    'b_age':       0.35,   # coefficient on (age1 + age2), younger adults
    'b_income':    0.30,   # coefficient on income2 (moderate income)
    'b_pt':        0.08,   # coefficient on week_metro
    'b_taxi':      0.12,   # coefficient on week_taxi
    'b_car':       0.40,   # penalty for having a car
    'b_education': 0.20,   # coefficient on education level
    'b_bus':       0.05,   # coefficient on week_bus
    'b_travel':    0.04,   # coefficient on travel_num
    'b_factor1':   0.15,   # latent attitude toward new mobility
}

# ============================================================
# Default coefficients for the subscription logit (Stage 2)
# ============================================================
_DEFAULT_SUBSCRIBE_PARAMS = {
    'av_scale':       3.0,    # logistic steepness for added-value signal
    'av_shift':       0.5,    # midpoint shift for AV sigmoid
    'savings_weight': 2.0,    # weight on savings ratio
    'trial_power':    1.0,    # exponent for trial-count ramp (1 = linear)
    'income_high_penalty': -0.3,  # high-income less price-motivated
    'income_low_bonus':     0.2,  # low-income attracted by bundled savings
    'freq_bonus':     0.06,   # per weekly-trip bonus (more trips -> more value)
    'base_rate':     -1.2,    # intercept (keeps overall prob moderate)
    'cap':            0.50,   # hard upper cap on P(sub | transfer)
}


def _sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


# ------------------------------------------------------------------
# Stage 1 : Transfer-group probability
# ------------------------------------------------------------------

def compute_transfer_probability(agents, params=None):
    """
    Probability that each agent belongs to the 'transfer group', i.e. is
    psychologically open to committing to a MaaS subscription.

    Parameters
    ----------
    agents : dict[str, ndarray[N]]
        Agent attribute arrays.  Expected keys include:
        age1, age2, income2, week_metro, week_taxi, have_car,
        education, week_bus, travel_num, factor1.
    params : dict, optional
        Override default coefficients (keys match _DEFAULT_TRANSFER_PARAMS).

    Returns
    -------
    ndarray[N]
        P(transfer_group) in (0, 1) for every agent.
    """
    p = dict(_DEFAULT_TRANSFER_PARAMS)
    if params is not None:
        p.update(params)

    age_young = agents.get('age1', 0.0) + agents.get('age2', 0.0)
    income2   = agents.get('income2', 0.0)
    metro     = agents.get('week_metro', 0.0)
    taxi      = agents.get('week_taxi', 0.0)
    bus       = agents.get('week_bus', 0.0)
    have_car  = agents.get('have_car', 0.0)
    education = agents.get('education', 0.0)
    travel_n  = agents.get('travel_num', 0.0)
    factor1   = agents.get('factor1', 0.0)

    z = (p['b0']
         + p['b_age']       * age_young
         + p['b_income']    * income2
         + p['b_pt']        * metro
         + p['b_taxi']      * taxi
         + p['b_bus']       * bus
         - p['b_car']       * have_car
         + p['b_education'] * education
         + p['b_travel']    * travel_n
         + p['b_factor1']   * factor1)

    return _sigmoid(z)


# ------------------------------------------------------------------
# Stage 2 : Subscription probability | transfer group
# ------------------------------------------------------------------

def compute_subscription_probability(agents, max_av, bundle_prices, theta,
                                     params=None):
    """
    Probability of subscribing, conditional on being in the transfer group
    and having tried MaaS at least once.

    Parameters
    ----------
    agents : dict[str, ndarray[N]]
        Agent attribute arrays.  Must include cost_alacarte (monthly
        a-la-carte spending) and trial_count.
    max_av : ndarray[N]
        Maximum added value from the Ch3 bundle-choice model (higher is
        better; typically in [0, ~5] utils).
    bundle_prices : dict
        Mapping from bundle code to monthly price, e.g.
        {'BF': 80.0, 'MA': 230.0, 'VT': 960.0, 'UA': 1556.8}.
    theta : ndarray[17]
        Strategy parameter vector (used here only for any price-scale
        adjustments already baked into bundle_prices; kept for API
        consistency).
    params : dict, optional
        Override default coefficients (keys match _DEFAULT_SUBSCRIBE_PARAMS).

    Returns
    -------
    ndarray[N]
        P(subscribe | transfer_group) in [0, cap] for every agent.
    """
    p = dict(_DEFAULT_SUBSCRIBE_PARAMS)
    if params is not None:
        p.update(params)

    N = max_av.shape[0]

    # ---- 1. Added-value signal (sigmoid of scaled AV) ----
    av_signal = _sigmoid(p['av_scale'] * (max_av - p['av_shift']))

    # ---- 2. Savings ratio ----
    # Find the cheapest bundle price for each agent to compare against
    # a-la-carte cost.  We use a simple heuristic: the best bundle is the
    # one with the lowest price that the agent could plausibly use.
    # For a population-level model we take the minimum bundle price.
    prices = np.array([
        bundle_prices.get('BF', PRICE_BASE_BF),
        bundle_prices.get('MA', PRICE_BASE_MA),
        bundle_prices.get('VT', PRICE_BASE_VT),
        bundle_prices.get('UA', PRICE_BASE_UA),
    ])
    min_price = prices.min()

    cost_alacarte = np.asarray(agents.get('cost_alacarte', np.full(N, 300.0)),
                               dtype=np.float64)
    # Avoid division by zero; agents with zero a-la-carte cost get 0 savings
    safe_cost = np.maximum(cost_alacarte, 1.0)
    savings_ratio = np.clip((cost_alacarte - min_price) / safe_cost, 0.0, 1.0)

    # ---- 3. Trial-count ramp ----
    # Default to MIN_TRIALS (agents have had enough trials) when called
    # from the surrogate; the ABM engine will override via trial_count array.
    default_tc = np.full(N, float(MIN_TRIALS_FOR_SUBSCRIBE), dtype=np.float64)
    trial_count = np.asarray(agents.get('trial_count', default_tc),
                             dtype=np.float64)
    min_trials = max(float(MIN_TRIALS_FOR_SUBSCRIBE), 1.0)
    trial_ramp = np.clip(
        (trial_count / min_trials) ** p['trial_power'], 0.0, 1.0
    )

    # ---- 4. Income adjustment ----
    income3 = np.asarray(agents.get('income3', np.zeros(N)), dtype=np.float64)
    income1 = np.asarray(agents.get('income1', np.zeros(N)), dtype=np.float64)
    income_adj = (p['income_high_penalty'] * income3
                  + p['income_low_bonus'] * income1)

    # ---- 5. Travel frequency bonus ----
    travel_num = np.asarray(agents.get('travel_num', np.zeros(N)),
                            dtype=np.float64)
    freq_term = p['freq_bonus'] * travel_num

    # ---- Combine into logit ----
    z = (p['base_rate']
         + p['av_scale'] * av_signal      # AV contribution (double-dip intentional:
         + p['savings_weight'] * savings_ratio  #   captures both utility & cost dimensions)
         + income_adj
         + freq_term)

    # Modulate by trial ramp (no trial experience -> probability near 0)
    prob_raw = _sigmoid(z) * trial_ramp

    # Hard cap to keep in [0, cap]
    return np.clip(prob_raw, 0.0, p['cap'])


# ------------------------------------------------------------------
# Combined entry point
# ------------------------------------------------------------------

def compute_subscribe_probability(agents, max_av, bundle_prices, theta,
                                  params=None):
    """
    Main entry point: combined probability that an agent subscribes.

        P_subscribe = P(transfer_group) * P(sub | transfer_group)

    Parameters
    ----------
    agents : dict[str, ndarray[N]]
        Agent attribute arrays (see individual functions for required keys).
    max_av : ndarray[N]
        Maximum added value from Ch3.
    bundle_prices : dict
        Bundle code -> monthly price.
    theta : ndarray[17]
        Strategy parameter vector.
    params : dict, optional
        Override dict.  Keys prefixed with 'transfer_' are routed to
        ``compute_transfer_probability``; keys prefixed with 'subscribe_'
        are routed to ``compute_subscription_probability``.  Un-prefixed
        keys are sent to both.

    Returns
    -------
    ndarray[N]
        P(subscribe) in [0, ~0.5] for every agent.
    """
    # Split params into stage-specific dicts
    transfer_params = {}
    subscribe_params = {}
    if params is not None:
        for k, v in params.items():
            if k.startswith('transfer_'):
                transfer_params[k[len('transfer_'):]] = v
            elif k.startswith('subscribe_'):
                subscribe_params[k[len('subscribe_'):]] = v
            else:
                transfer_params[k] = v
                subscribe_params[k] = v

    p_transfer = compute_transfer_probability(
        agents,
        params=transfer_params if transfer_params else None,
    )

    p_sub_given_transfer = compute_subscription_probability(
        agents, max_av, bundle_prices, theta,
        params=subscribe_params if subscribe_params else None,
    )

    return p_transfer * p_sub_given_transfer
