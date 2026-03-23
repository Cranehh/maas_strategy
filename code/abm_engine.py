"""
ABM simulation engine for MaaS adoption in Beijing.

Simulates 156 weeks (3 years) of MaaS adoption for ~79K agents
(each representing ~25 people via AGENT_WEIGHT).  Pure NumPy
implementation targeting <1 s per run on 79K agents.
"""

import numpy as np

from config import (
    N_WEEKS,
    P_INNOV, P_IMIT,
    BETA_LOCAL, BETA_REMOTE,
    AWARENESS_THRESHOLD,
    SATISFACTION_ALPHA, SATISFACTION_NEW_WEIGHT,
    CHURN_THRESHOLD, CHURN_CONSECUTIVE_WEEKS,
    COOLDOWN_WEEKS, MIN_TRIALS_FOR_SUBSCRIBE,
    STATUS_UNAWARE, STATUS_AWARE, STATUS_TRIAL,
    STATUS_SUBSCRIBER, STATUS_CHURNED,
    AGENT_WEIGHT,
    PRICE_BASE_BF, PRICE_BASE_MA, PRICE_BASE_VT, PRICE_BASE_UA,
    CAR_CO2_PER_KM,
)


# ---------------------------------------------------------------------------
# Helper: Gini coefficient (vectorised)
# ---------------------------------------------------------------------------
def _gini(values):
    """Compute the Gini coefficient of a 1-D array.

    Returns 0.0 when the array is empty or all-zero.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or v.sum() == 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    idx = np.arange(1, n + 1)
    return (2.0 * np.sum(idx * v) - (n + 1) * np.sum(v)) / (n * np.sum(v))


# ---------------------------------------------------------------------------
# Helper: Marketing allocation across TAZs
# ---------------------------------------------------------------------------
def _compute_marketing_intensity(
    taz_ids, weight, status, awareness, n_taz,
    B_total, gamma_potential, gamma_gap, c_conc,
):
    """Return per-TAZ marketing intensity (float32 array of length *n_taz*).

    Higher intensity is directed toward:
      - TAZs with high adoption *potential* (large population, favourable
        demographics) controlled by ``gamma_potential``.
      - TAZs with a large *awareness gap* (low current awareness vs. potential)
        controlled by ``gamma_gap``.

    ``c_conc`` governs how concentrated the budget is across TAZs.
    """
    # Population per TAZ
    taz_pop = np.bincount(taz_ids, weights=weight, minlength=n_taz).astype(np.float32)
    taz_pop_share = taz_pop / (taz_pop.sum() + 1e-12)

    # Mean awareness per TAZ (proxy for current penetration)
    taz_aware_sum = np.bincount(
        taz_ids,
        weights=awareness.astype(np.float64) * weight,
        minlength=n_taz,
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        taz_mean_aware = np.where(taz_pop > 0, taz_aware_sum / taz_pop, 0.0).astype(np.float32)

    # Potential score ~ population share
    potential = taz_pop_share ** c_conc

    # Gap score: complement of current awareness
    gap = np.clip(1.0 - taz_mean_aware, 0.0, 1.0)

    # Combined score
    score = gamma_potential * potential + gamma_gap * gap
    score_total = score.sum() + 1e-12
    intensity = (B_total * score / score_total).astype(np.float32)
    return intensity


# ===================================================================
# Main ABM class
# ===================================================================
class ABMSimulation:
    """Agent-based simulation of weekly MaaS adoption dynamics."""

    def __init__(self, agents, ch3_model, ch1_model, ch2_model):
        """
        Parameters
        ----------
        agents : dict[str, np.ndarray]
            Dictionary of aligned NumPy arrays, each of length *N*.  Expected
            keys include at minimum: ``'sex'``, ``'taz'``, ``'weight'``,
            ``'district'``, ``'car_km_month'``, plus whatever feature columns
            are needed by the three choice models.
        ch3_model : module
            Bundle-choice model.  Must expose:
              ``compute_subscription_probabilities(agents, theta) -> dict``
        ch1_model : module
            Trial / mode-shift model.  Must expose:
              ``compute_trial_probability(agents, theta) -> P_try_monthly``  (N,)
        ch2_model : module
            Subscription-probability model.  Must expose:
              ``compute_subscribe_probability(agents, max_av, prices_dict, theta) -> P_sub``  (N,)
        """
        self.agents = agents
        self.N = len(agents['sex'])
        self.ch3 = ch3_model
        self.ch1 = ch1_model
        self.ch2 = ch2_model

    # ---------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------
    def run(self, theta, seed=42):
        """Run the full 156-week simulation.

        Parameters
        ----------
        theta : array-like, shape (17,)
            Strategy parameter vector (see ``config.THETA_NAMES``).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        objectives : np.ndarray, shape (4,)
            ``[-adoption_rate, -net_revenue, gini_adoption, -carbon_reduction]``
            All are formulated for *minimisation* (negated where maximisation
            is desired, Gini is already a quantity to minimise).
        weekly_subscribers : np.ndarray, shape (N_WEEKS,)
            Weighted subscriber count at the end of each week (for S-curve
            analysis).
        """
        theta = np.asarray(theta, dtype=np.float64)
        rng = np.random.default_rng(seed)
        N = self.N
        agents = self.agents

        # --- Extract agent-level constants ----------------------------------
        weight = agents.get('weight', np.full(N, AGENT_WEIGHT, dtype=np.float32))
        taz_ids = agents['taz'].astype(np.int32)
        n_taz = int(taz_ids.max()) + 1
        district = agents.get('district', np.zeros(N, dtype=np.int32)).astype(np.int32)
        car_km_month = agents.get('car_km_month', np.zeros(N, dtype=np.float32))

        # --- Pre-compute choice-model outputs (constant for a given theta) --
        # Ch3: compute utilities, added value, and prices
        ch3_result = self.ch3.compute_subscription_probabilities(agents, theta)
        max_av = np.asarray(ch3_result['max_av'], dtype=np.float32)
        best_bundle = np.asarray(ch3_result['best_bundle'], dtype=np.int8)
        prices_dict = ch3_result['prices']
        # Per-agent bundle price based on best bundle choice
        price_arr = np.array([prices_dict['BF'], prices_dict['MA'],
                              prices_dict['VT'], prices_dict['UA']], dtype=np.float32)
        bundle_prices = price_arr[best_bundle.clip(0, 3)]

        # Ch1: monthly trial probability (adjusted by theta[15:17])
        P_try_monthly = np.asarray(
            self.ch1.compute_trial_probability(agents, theta), dtype=np.float32,
        )
        P_try_monthly = np.clip(P_try_monthly, 0.0, 1.0)
        # Convert monthly -> weekly:  P_w = 1 - (1-P_m)^(1/4.33)
        P_try_weekly = (1.0 - (1.0 - P_try_monthly) ** (1.0 / 4.33)).astype(np.float32)

        # Ch2: subscription probability (will be re-computed with trial_count)
        P_sub_base = np.asarray(
            self.ch2.compute_subscribe_probability(agents, max_av, prices_dict, theta),
            dtype=np.float32,
        )
        P_sub_base = np.clip(P_sub_base, 0.0, 1.0)

        # --- Estimate per-agent a-la-carte monthly cost (for satisfaction) ---
        #     Rough proxy: sum of individual mode costs the agent currently pays
        cost_alacarte = agents.get(
            'cost_alacarte',
            np.full(N, 300.0, dtype=np.float32),  # sensible fallback
        ).astype(np.float32)

        # --- Initialise state arrays ----------------------------------------
        status = np.full(N, STATUS_UNAWARE, dtype=np.int8)
        awareness = np.zeros(N, dtype=np.float32)
        satisfaction = np.full(N, 0.5, dtype=np.float32)
        bundle_choice = np.full(N, -1, dtype=np.int8)
        churn_weeks = np.zeros(N, dtype=np.int8)
        cooldown_weeks = np.zeros(N, dtype=np.int8)
        trial_count = np.zeros(N, dtype=np.int8)

        # --- Marketing intensity (initial allocation) -----------------------
        B_total = theta[11]
        gamma_potential = theta[12]
        gamma_gap = theta[13]
        c_conc = theta[14]
        marketing_intensity = _compute_marketing_intensity(
            taz_ids, weight, status, awareness, n_taz,
            B_total, gamma_potential, gamma_gap, c_conc,
        )

        # Quarterly adjustment parameters
        freq_adj = max(1, int(theta[10]))
        adj_interval = freq_adj * 13  # weeks per adjustment period

        # --- Output buffer --------------------------------------------------
        weekly_subscribers = np.empty(N_WEEKS, dtype=np.float64)

        # ===================================================================
        # Weekly simulation loop
        # ===================================================================
        for week in range(N_WEEKS):
            # ---------------------------------------------------------------
            # Step 7 (placed first so it takes effect at start of quarter):
            # Quarterly marketing re-allocation
            # ---------------------------------------------------------------
            if week > 0 and week % adj_interval == 0:
                marketing_intensity = _compute_marketing_intensity(
                    taz_ids, weight, status, awareness, n_taz,
                    B_total, gamma_potential, gamma_gap, c_conc,
                )
                # Optional: scale budget up/down based on adoption momentum
                # (delta_up / delta_down from theta[8], theta[9] reserved for
                # future adaptive scaling; the spatial re-allocation above is
                # the primary adjustment mechanism.)

            # ---------------------------------------------------------------
            # Step 1: Awareness Diffusion (Bass + spatial + marketing)
            # ---------------------------------------------------------------
            # City-wide subscriber fraction
            is_sub = (status == STATUS_SUBSCRIBER)
            weight_f64 = weight.astype(np.float64)
            city_sub_rate = np.dot(is_sub.astype(np.float64), weight_f64) / (weight_f64.sum() + 1e-12)

            # TAZ-level subscriber fraction
            taz_sub_count = np.bincount(
                taz_ids,
                weights=is_sub.astype(np.float64) * weight_f64,
                minlength=n_taz,
            )
            taz_total = np.bincount(taz_ids, weights=weight_f64, minlength=n_taz)
            with np.errstate(invalid='ignore', divide='ignore'):
                taz_sub_rate = np.where(taz_total > 0, taz_sub_count / taz_total, 0.0)

            local_rate = taz_sub_rate[taz_ids].astype(np.float32)

            p_aware = (
                P_INNOV
                + P_IMIT * (BETA_LOCAL * local_rate + BETA_REMOTE * city_sub_rate)
            ).astype(np.float32)

            # Marketing boost (intensity is in 万元 / month; convert to a
            # small probability increment per week)
            p_aware += marketing_intensity[taz_ids] * np.float32(0.01)

            unaware = (status == STATUS_UNAWARE)
            awareness[unaware] += p_aware[unaware]

            # Transition: UNAWARE -> AWARE
            newly_aware = unaware & (awareness >= AWARENESS_THRESHOLD)
            status[newly_aware] = STATUS_AWARE

            # ---------------------------------------------------------------
            # Step 2: Trial Decision
            # ---------------------------------------------------------------
            # Aware agents can start a trial
            aware_mask = (status == STATUS_AWARE)
            if np.any(aware_mask):
                rand_trial = rng.random(N, dtype=np.float32)
                new_trial = aware_mask & (rand_trial < P_try_weekly)
                status[new_trial] = STATUS_TRIAL
                trial_count[new_trial] = np.clip(
                    trial_count[new_trial].astype(np.int16) + 1, 0, 127,
                ).astype(np.int8)

            # Agents already in trial accumulate trial experience each week
            # (representing continued MaaS usage during trial period)
            trial_using = (status == STATUS_TRIAL)
            if np.any(trial_using):
                rand_use = rng.random(N, dtype=np.float32)
                using = trial_using & (rand_use < P_try_weekly)
                trial_count[using] = np.clip(
                    trial_count[using].astype(np.int16) + 1, 0, 127,
                ).astype(np.int8)

            # ---------------------------------------------------------------
            # Step 3: Subscription Decision
            # ---------------------------------------------------------------
            trial_mask = (status == STATUS_TRIAL) & (trial_count >= MIN_TRIALS_FOR_SUBSCRIBE)
            if np.any(trial_mask):
                av_positive = max_av > 0
                eligible = trial_mask & av_positive
                if np.any(eligible):
                    # Scale P_sub by trial experience ramp
                    trial_ramp = np.clip(trial_count.astype(np.float32) / float(MIN_TRIALS_FOR_SUBSCRIBE), 0.0, 1.0)
                    P_sub = P_sub_base * trial_ramp
                    rand_sub = rng.random(N, dtype=np.float32)
                    new_sub = eligible & (rand_sub < P_sub)
                    status[new_sub] = STATUS_SUBSCRIBER
                    bundle_choice[new_sub] = best_bundle[new_sub]

            # ---------------------------------------------------------------
            # Step 4: Satisfaction Update
            # ---------------------------------------------------------------
            sub_mask = (status == STATUS_SUBSCRIBER)
            if np.any(sub_mask):
                savings_ratio = np.clip(
                    (cost_alacarte - bundle_prices) / (cost_alacarte + 1e-6),
                    -1.0, 1.0,
                )
                signal = (
                    0.4 * np.clip(savings_ratio, 0.0, 1.0)
                    + 0.3 * np.clip(max_av / 5.0, 0.0, 1.0)
                    + 0.3 * 0.5
                ).astype(np.float32)
                satisfaction[sub_mask] = (
                    SATISFACTION_ALPHA * satisfaction[sub_mask]
                    + SATISFACTION_NEW_WEIGHT * signal[sub_mask]
                )

            # ---------------------------------------------------------------
            # Step 5: Churn
            # ---------------------------------------------------------------
            if np.any(sub_mask):
                low_sat = sub_mask & (satisfaction < CHURN_THRESHOLD)
                ok_sat = sub_mask & ~low_sat
                churn_weeks[low_sat] = np.clip(
                    churn_weeks[low_sat].astype(np.int16) + 1, 0, 127,
                ).astype(np.int8)
                churn_weeks[ok_sat] = 0

                churned = sub_mask & (churn_weeks >= CHURN_CONSECUTIVE_WEEKS)
                status[churned] = STATUS_CHURNED
                cooldown_weeks[churned] = 0
                bundle_choice[churned] = -1

            # ---------------------------------------------------------------
            # Step 6: Re-subscribe (from churned)
            # ---------------------------------------------------------------
            churned_mask = (status == STATUS_CHURNED)
            if np.any(churned_mask):
                cooldown_weeks[churned_mask] = np.clip(
                    cooldown_weeks[churned_mask].astype(np.int16) + 1, 0, 127,
                ).astype(np.int8)
                can_reenter = churned_mask & (cooldown_weeks >= COOLDOWN_WEEKS)
                status[can_reenter] = STATUS_AWARE
                cooldown_weeks[can_reenter] = 0
                satisfaction[can_reenter] = 0.5  # Reset satisfaction on re-entry

            # ---------------------------------------------------------------
            # Record weekly subscriber count (weighted)
            # ---------------------------------------------------------------
            weekly_subscribers[week] = np.dot(
                (status == STATUS_SUBSCRIBER).astype(np.float64), weight_f64,
            )

        # ===================================================================
        # Compute objectives after 156 weeks
        # ===================================================================
        total_weight = weight_f64.sum()
        final_sub = (status == STATUS_SUBSCRIBER)

        # 1. Adoption rate (weighted)
        adoption_rate = np.dot(final_sub.astype(np.float64), weight_f64) / total_weight

        # 2. Net revenue (元) over 36 months
        #    Revenue = subscribers * bundle_price * 36 months
        #    Cost    = B_total * 10000 * 36 months  (万元 -> 元)
        revenue = np.dot(
            (final_sub.astype(np.float64) * bundle_prices.astype(np.float64)),
            weight_f64,
        ) * 36.0
        cost = theta[11] * 10_000.0 * 36.0
        net_revenue = revenue - cost

        # 3. Gini coefficient of adoption across districts
        n_districts = int(district.max()) + 1
        district_sub = np.bincount(
            district, weights=final_sub.astype(np.float64) * weight_f64,
            minlength=n_districts,
        )
        district_pop = np.bincount(
            district, weights=weight_f64,
            minlength=n_districts,
        )
        with np.errstate(invalid='ignore', divide='ignore'):
            district_rate = np.where(
                district_pop > 0, district_sub / district_pop, 0.0,
            )
        gini_adoption = _gini(district_rate)

        # 4. Carbon reduction (kg CO2 / year)
        #    30% mode-shift assumption for subscribers
        carbon_reduction = np.dot(
            (final_sub.astype(np.float64) * car_km_month.astype(np.float64)),
            weight_f64,
        ) * CAR_CO2_PER_KM * 0.3 * 12.0  # monthly -> annual

        # All objectives formulated for minimisation
        objectives = np.array([
            -adoption_rate,
            -net_revenue,
            gini_adoption,
            -carbon_reduction,
        ], dtype=np.float64)

        return objectives, weekly_subscribers
