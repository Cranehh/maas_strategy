"""
Surrogate models for fast MaaS ABM objective evaluation.

Architecture:
    1. AnalyticalSurrogate       -- millisecond-level deterministic approximation
       derived from Ch3 (bundle choice), Ch1 (trial probability), and Ch2
       (subscription probability) closed-form models.
    2. QuadraticResidualModel    -- quadratic polynomial residual correction
    3. GPResidualModel           -- Gaussian-Process residual (fallback)
    4. NPSurrogateEvaluator      -- ANP + residual correction combined evaluator

Objective vector (4-dim, all formulated for minimisation):
    [-adoption_rate, -net_revenue, gini_coefficient, -carbon_reduction]
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

from config import (
    CH3_PARAMS,
    PRICE_BASE_BF, PRICE_BASE_MA, PRICE_BASE_VT, PRICE_BASE_UA,
    N_WEEKS, P_INNOV, P_IMIT,
    AGENT_WEIGHT,
    CAR_CO2_PER_KM, PT_CO2_PER_KM, TAXI_CO2_PER_KM,
    N_THETA, THETA_LOWER, THETA_UPPER,
)


# ================================================================== #
#  Utility helpers                                                     #
# ================================================================== #

def _softmax_cols(V):
    """Row-wise softmax over (N, J) -> (N, J) probabilities."""
    V_max = V.max(axis=1, keepdims=True)
    expV = np.exp(V - V_max)
    return expV / expV.sum(axis=1, keepdims=True)


def _gini(values):
    """Compute the Gini coefficient of a 1-D array of non-negative values."""
    values = np.asarray(values, dtype=np.float64)
    if values.sum() == 0:
        return 0.0
    sorted_v = np.sort(values)
    n = len(sorted_v)
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * sorted_v) / (n * sorted_v.sum())) - (n + 1) / n


def _bundle_prices_from_theta(theta):
    """Compute actual bundle monthly prices from theta price-scale factors.

    Returns
    -------
    dict  {'BF': float, 'MA': float, 'VT': float, 'UA': float}
    ndarray[4]  prices in order [BF, MA, VT, UA]
    """
    prices = {
        'BF': (PRICE_BASE_BF + theta[0]) * theta[1],   # (base + taxi_BF) * ps_BF
        'MA': (PRICE_BASE_MA + theta[2]) * theta[3],   # (base + taxi_MA) * ps_MA
        'VT': PRICE_BASE_VT * theta[4],                 # ps_VT
        'UA': PRICE_BASE_UA * theta[5],                  # ps_UA
    }
    prices_arr = np.array([prices['BF'], prices['MA'],
                           prices['VT'], prices['UA']])
    return prices, prices_arr


# ================================================================== #
#  1. Analytical Surrogate                                             #
# ================================================================== #

class AnalyticalSurrogate:
    """Fast analytical approximation of ABM objectives (millisecond-level).

    Uses the Ch3 ICLV model to compute bundle added values, and derives
    adoption, revenue, equity, and carbon objectives analytically by
    chaining Ch1 (trial) and Ch2 (subscription) probability functions.
    """

    def __init__(self, agents, ch3_model, ch1_model, ch2_model):
        """Pre-compute agent-fixed quantities that do not depend on theta.

        Parameters
        ----------
        agents : dict[str, ndarray[N]]
            Agent population attribute arrays.
        ch3_model : module
            Ch3 bundle-choice module (must expose ``compute_added_values``
            or equivalent callable).
        ch1_model : module
            Ch1 trial probability module (``compute_trial_probability``).
        ch2_model : module
            Ch2 subscription probability module
            (``compute_subscribe_probability``).
        """
        self.agents = agents
        self.ch3 = ch3_model
        self.ch1 = ch1_model
        self.ch2 = ch2_model

        # Population size
        self.N = len(next(iter(agents.values())))

        # -----------------------------------------------------------
        # Pre-compute agent-fixed quantities
        # -----------------------------------------------------------

        # Ch3 latent-variable structural equations (factors 1-4, 6)
        # These depend only on socio-demographics, not on theta.
        self._precompute_ch3_factors()

        # District assignments for equity (Gini) calculation
        self.district = agents.get('district', np.zeros(self.N, dtype=int))
        self.unique_districts = np.unique(self.district)

        # Travel distances for carbon calculation
        self.travel_distance_work = agents.get(
            'travel_distance_work', np.full(self.N, 10.0))
        self.travel_distance_day = agents.get(
            'travel_distance_day',
            agents.get('travel_distance_work', np.full(self.N, 10.0)))

        # Current mode share indicators for carbon baseline
        self.has_car = agents.get('have_car', np.zeros(self.N))
        self.week_taxi = agents.get('week_taxi', np.zeros(self.N))
        self.week_bus = agents.get('week_bus', np.zeros(self.N))
        self.week_metro = agents.get('week_metro', np.zeros(self.N))

    # -------------------------------------------------------------- #
    #  Pre-computation of Ch3 factors                                  #
    # -------------------------------------------------------------- #

    def _precompute_ch3_factors(self):
        """Compute Ch3 latent factors from socio-demographics (agent-fixed)."""
        p = CH3_PARAMS
        N = self.N
        agents = self.agents

        self.factors = {}
        for fid in [1, 2, 3, 4, 6]:
            pf = f'coef{fid}_'
            factor = np.zeros(N, dtype=np.float64)

            # Age dummies
            factor += p.get(f'{pf}age1', 0.0) * agents.get('age1', np.zeros(N))
            factor += p.get(f'{pf}age2', 0.0) * agents.get('age2', np.zeros(N))
            factor += p.get(f'{pf}age3', 0.0) * agents.get('age3', np.zeros(N))

            # Occupation
            factor += p.get(f'{pf}job', 0.0) * agents.get('occupy', np.zeros(N))

            # Income dummies
            factor += p.get(f'{pf}income1', 0.0) * agents.get('income1', np.zeros(N))
            factor += p.get(f'{pf}income2', 0.0) * agents.get('income2', np.zeros(N))
            factor += p.get(f'{pf}income3', 0.0) * agents.get('income3', np.zeros(N))

            # Travel characteristics
            factor += p.get(f'{pf}travel_num', 0.0) * agents.get(
                'travel_num', np.zeros(N))
            factor += p.get(f'{pf}travel_distance_day', 0.0) * agents.get(
                'travel_distance_work', np.zeros(N))
            factor += p.get(f'{pf}travel_aim', 0.0) * agents.get(
                'travel_aim', np.zeros(N))

            # Mode-use frequencies
            factor += p.get(f'{pf}bus', 0.0) * agents.get('week_bus', np.zeros(N))
            factor += p.get(f'{pf}metro', 0.0) * agents.get('week_metro', np.zeros(N))
            factor += p.get(f'{pf}taxi', 0.0) * agents.get('week_taxi', np.zeros(N))
            factor += p.get(f'{pf}ebike', 0.0) * agents.get('week_ebike', np.zeros(N))
            factor += p.get(f'{pf}bike', 0.0) * agents.get('week_bike', np.zeros(N))

            self.factors[fid] = factor

    # -------------------------------------------------------------- #
    #  Ch3 bundle utilities and added values                           #
    # -------------------------------------------------------------- #

    def _compute_ch3_utilities(self, theta):
        """Compute Ch3 nested-logit utilities using the authoritative ch3_model.

        Delegates to ch3_model module for exact utility computation and
        nested logit probabilities, ensuring perfect consistency.

        Parameters
        ----------
        theta : ndarray[17]

        Returns
        -------
        V : ndarray[N, 5]  -- utilities for [BF, MA, VT, UA, NoPurchase]
        P : ndarray[N, 5]  -- choice probabilities
        added_values : ndarray[N]  -- max added value across bundles
        max_av : ndarray[N]  -- max(V_bundle) - V_no_purchase
        """
        # Ensure factor scores are in agents dict
        agents = self.agents
        if 'factor1' not in agents:
            for fid in [1, 2, 3, 4, 6]:
                agents[f'factor{fid}'] = self.factors[fid].astype(np.float32)

        # Delegate to ch3_model for exact computation
        result = self.ch3.compute_subscription_probabilities(agents, theta)
        V = result['V'].astype(np.float64)
        P = result['P'].astype(np.float64)
        max_av = result['max_av'].astype(np.float64)

        # Logsum-based added value (for analytical approximation)
        added_values = max_av  # Use max bundle V - V5 as added value

        return V, P, added_values, max_av

    # -------------------------------------------------------------- #
    #  Steady-state awareness approximation (Bass diffusion)           #
    # -------------------------------------------------------------- #

    @staticmethod
    def _steady_state_awareness(tau_high, tau_low, B_total, N_weeks=N_WEEKS):
        """Approximate steady-state awareness fraction using Bass model.

        At time T, the Bass cumulative adoption fraction is approximately:
            F(T) = [1 - exp(-(p+q)*T)] / [1 + (q/p)*exp(-(p+q)*T)]

        We evaluate this at t = N_weeks to get the expected fraction
        of the population that has become aware.

        Parameters
        ----------
        tau_high, tau_low : float
            Awareness thresholds (from theta[6], theta[7]).
        B_total : float
            Total marketing budget (affects effective p).
        N_weeks : int
            Simulation horizon.

        Returns
        -------
        float
            Approximate awareness fraction in [0, 1].
        """
        # Scale innovation coefficient by budget (normalised to 200 baseline)
        p_eff = P_INNOV * (B_total / 200.0) ** 0.5
        q_eff = P_IMIT

        # Bass cumulative fraction at T
        pq = p_eff + q_eff
        if pq < 1e-12:
            return 0.0
        ratio = q_eff / max(p_eff, 1e-12)
        exp_term = np.exp(-pq * N_weeks)
        F_T = (1.0 - exp_term) / (1.0 + ratio * exp_term)

        # The awareness threshold modulates how many "aware" agents actually
        # reach the actionable awareness level
        effective_threshold = (tau_high + tau_low) / 2.0
        P_aware_steady = F_T * (1.0 - effective_threshold * 0.5)

        return float(np.clip(P_aware_steady, 0.0, 1.0))

    # -------------------------------------------------------------- #
    #  Main evaluation entry point                                     #
    # -------------------------------------------------------------- #

    def evaluate(self, theta):
        """Evaluate 4 objectives analytically for a given strategy vector.

        Parameters
        ----------
        theta : ndarray[17]
            Strategy parameter vector.

        Returns
        -------
        objectives : ndarray[4]
            [-adoption_rate, -net_revenue, gini_coefficient, -carbon_reduction]
            All formulated for minimisation.
        """
        theta = np.asarray(theta, dtype=np.float64)
        N = self.N

        # ---------------------------------------------------------- #
        # 1. Bundle prices from theta                                  #
        # ---------------------------------------------------------- #
        prices, prices_arr = _bundle_prices_from_theta(theta)

        # ---------------------------------------------------------- #
        # 2. Ch3 utilities, probabilities, added values                #
        # ---------------------------------------------------------- #
        V, P_bundle, added_values, max_av = self._compute_ch3_utilities(theta)

        # ---------------------------------------------------------- #
        # 3. Estimate adoption pipeline                                #
        #    P_adopt ~ P_aware_steady * P_try * P_subscribe            #
        # ---------------------------------------------------------- #

        # 3a. Steady-state awareness fraction
        P_aware = self._steady_state_awareness(
            tau_high=theta[6], tau_low=theta[7], B_total=theta[11])

        # 3b. Trial probability from Ch1 model
        P_try = self.ch1.compute_trial_probability(self.agents, theta)
        # P_try is per-agent ndarray[N]

        # 3c. Subscription probability from Ch2 model
        P_subscribe = self.ch2.compute_subscribe_probability(
            self.agents, max_av, prices, theta)
        # P_subscribe is per-agent ndarray[N]

        # 3d. Combined per-agent adoption probability
        P_adopt_agent = P_aware * P_try * P_subscribe  # (N,)

        # ---------------------------------------------------------- #
        # 4. Objective 1: Adoption rate (maximise -> negate)           #
        # ---------------------------------------------------------- #
        adoption_rate = P_adopt_agent.mean()

        # ---------------------------------------------------------- #
        # 5. Objective 2: Net revenue (maximise -> negate)             #
        # ---------------------------------------------------------- #
        # Expected revenue per agent = P_adopt * E[bundle_price]
        # E[bundle_price] for adopter i = sum_j P(j|purchase) * price_j
        # P(j|purchase) = P_bundle[:, j] / (1 - P_bundle[:, 4])
        P_purchase = 1.0 - P_bundle[:, 4]  # (N,)
        safe_P_purchase = np.maximum(P_purchase, 1e-12)
        P_cond = P_bundle[:, :4] / safe_P_purchase[:, np.newaxis]  # (N, 4)
        expected_price_per_adopter = (P_cond * prices_arr[np.newaxis, :]).sum(axis=1)

        # Gross revenue (scaled to population)
        gross_revenue_per_agent = P_adopt_agent * expected_price_per_adopter
        total_gross_revenue = gross_revenue_per_agent.sum() * AGENT_WEIGHT

        # Marketing cost (万元/月 -> 元/月: *10000, over N_weeks/4 months)
        marketing_cost = theta[11] * 10000.0 * (N_WEEKS / 4.0)

        net_revenue = total_gross_revenue * (N_WEEKS / 4.0) - marketing_cost

        # ---------------------------------------------------------- #
        # 6. Objective 3: Gini coefficient of adoption across districts#
        # ---------------------------------------------------------- #
        district_adoption = np.zeros(len(self.unique_districts))
        for idx, d in enumerate(self.unique_districts):
            mask = self.district == d
            if mask.sum() > 0:
                district_adoption[idx] = P_adopt_agent[mask].mean()

        gini = _gini(district_adoption) if len(district_adoption) > 1 else 0.0

        # ---------------------------------------------------------- #
        # 7. Objective 4: Carbon reduction (maximise -> negate)        #
        # ---------------------------------------------------------- #
        # Baseline: each agent's weekly car/taxi km * emission factor
        # MaaS subscribers shift some car trips to PT
        weekly_km = self.travel_distance_day * 5.0  # 5 working days

        # Current emissions per agent per week (kg CO2)
        car_share = np.clip(self.has_car, 0.0, 1.0)
        taxi_share = np.clip(self.week_taxi / np.maximum(
            self.week_bus + self.week_metro + self.week_taxi + 1e-6, 1.0), 0.0, 1.0)
        pt_share = 1.0 - car_share - taxi_share
        pt_share = np.clip(pt_share, 0.0, 1.0)

        baseline_co2 = weekly_km * (
            car_share * CAR_CO2_PER_KM
            + taxi_share * TAXI_CO2_PER_KM
            + pt_share * PT_CO2_PER_KM
        )

        # Post-MaaS: adopters shift toward PT-dominant modes
        # Assume adopters' car share drops by 60%, taxi drops by 30%
        time_improvement = theta[15]
        mode_shift_factor = 0.6 + 0.4 * time_improvement  # better service -> more shift

        post_car_share = car_share * (1.0 - mode_shift_factor * P_adopt_agent)
        post_taxi_share = taxi_share * (1.0 - 0.3 * mode_shift_factor * P_adopt_agent)
        post_pt_share = 1.0 - post_car_share - post_taxi_share
        post_pt_share = np.clip(post_pt_share, 0.0, 1.0)

        post_co2 = weekly_km * (
            post_car_share * CAR_CO2_PER_KM
            + post_taxi_share * TAXI_CO2_PER_KM
            + post_pt_share * PT_CO2_PER_KM
        )

        # Total annual carbon reduction (tonnes) across population
        weeks_per_year = 52.0
        carbon_reduction = (
            (baseline_co2 - post_co2).sum() * AGENT_WEIGHT * weeks_per_year / 1000.0
        )

        # ---------------------------------------------------------- #
        # 8. Assemble objective vector (minimisation convention)       #
        # ---------------------------------------------------------- #
        objectives = np.array([
            -adoption_rate,      # maximise adoption
            -net_revenue,        # maximise revenue
            gini,                # minimise inequality
            -carbon_reduction,   # maximise carbon reduction
        ], dtype=np.float64)

        return objectives

    def evaluate_with_intermediates(self, theta):
        """Evaluate 4 objectives + 9 intermediate physical quantities.

        Returns
        -------
        y : ndarray[13]
            [0] P_aware:            steady-state awareness fraction [0, 1]
            [1] P_try_mean:         population-mean trial probability [0, 1]
            [2] P_subscribe_mean:   population-mean subscription probability [0, 1]
            [3] P_purchase_mean:    population-mean purchase probability [0, 1]
            [4] max_av_mean:        population-mean max added value
            [5] E_price_cond:       population-mean expected bundle price
            [6] mode_shift_factor:  mode shift intensity [0.6, 0.72]
            [7] adoption_rate:      population adoption rate [0, 1]
            [8] gini_raw:           Gini coefficient [0, 1]
            [9:13] objectives:      [-adoption, -revenue, gini, -carbon]
        """
        theta = np.asarray(theta, dtype=np.float64)
        N = self.N

        # 1. Bundle prices
        prices, prices_arr = _bundle_prices_from_theta(theta)

        # 2. Ch3 utilities
        V, P_bundle, added_values, max_av = self._compute_ch3_utilities(theta)

        # 3. Awareness
        P_aware = self._steady_state_awareness(
            tau_high=theta[6], tau_low=theta[7], B_total=theta[11])

        # 4. Trial probability
        P_try = self.ch1.compute_trial_probability(self.agents, theta)

        # 5. Subscription probability
        P_subscribe = self.ch2.compute_subscribe_probability(
            self.agents, max_av, prices, theta)

        # 6. Adoption
        P_adopt_agent = P_aware * P_try * P_subscribe
        adoption_rate = P_adopt_agent.mean()

        # 7. Revenue
        P_purchase = 1.0 - P_bundle[:, 4]
        safe_P_purchase = np.maximum(P_purchase, 1e-12)
        P_cond = P_bundle[:, :4] / safe_P_purchase[:, np.newaxis]
        expected_price_per_adopter = (P_cond * prices_arr[np.newaxis, :]).sum(axis=1)
        gross_revenue_per_agent = P_adopt_agent * expected_price_per_adopter
        total_gross_revenue = gross_revenue_per_agent.sum() * AGENT_WEIGHT
        marketing_cost = theta[11] * 10000.0 * (N_WEEKS / 4.0)
        net_revenue = total_gross_revenue * (N_WEEKS / 4.0) - marketing_cost

        # 8. Gini
        district_adoption = np.zeros(len(self.unique_districts))
        for idx, d in enumerate(self.unique_districts):
            mask = self.district == d
            if mask.sum() > 0:
                district_adoption[idx] = P_adopt_agent[mask].mean()
        gini = _gini(district_adoption) if len(district_adoption) > 1 else 0.0

        # 9. Carbon
        weekly_km = self.travel_distance_day * 5.0
        car_share = np.clip(self.has_car, 0.0, 1.0)
        taxi_share = np.clip(self.week_taxi / np.maximum(
            self.week_bus + self.week_metro + self.week_taxi + 1e-6, 1.0), 0.0, 1.0)
        pt_share = np.clip(1.0 - car_share - taxi_share, 0.0, 1.0)
        baseline_co2 = weekly_km * (
            car_share * CAR_CO2_PER_KM + taxi_share * TAXI_CO2_PER_KM + pt_share * PT_CO2_PER_KM)
        time_improvement = theta[15]
        mode_shift_factor = 0.6 + 0.4 * time_improvement
        post_car_share = car_share * (1.0 - mode_shift_factor * P_adopt_agent)
        post_taxi_share = taxi_share * (1.0 - 0.3 * mode_shift_factor * P_adopt_agent)
        post_pt_share = np.clip(1.0 - post_car_share - post_taxi_share, 0.0, 1.0)
        post_co2 = weekly_km * (
            post_car_share * CAR_CO2_PER_KM + post_taxi_share * TAXI_CO2_PER_KM + post_pt_share * PT_CO2_PER_KM)
        carbon_reduction = (baseline_co2 - post_co2).sum() * AGENT_WEIGHT * 52.0 / 1000.0

        # Objectives (minimisation)
        objectives = np.array([
            -adoption_rate, -net_revenue, gini, -carbon_reduction
        ], dtype=np.float64)

        # Intermediates
        intermediates = np.array([
            float(P_aware),
            float(P_try.mean()),
            float(P_subscribe.mean()),
            float(P_purchase.mean()),
            float(max_av.mean()),
            float(expected_price_per_adopter.mean()),
            float(mode_shift_factor),
            float(adoption_rate),
            float(gini),
        ], dtype=np.float64)

        return np.concatenate([intermediates, objectives])


# ================================================================== #
#  2. Quadratic Residual Model                                         #
# ================================================================== #

class QuadraticResidualModel:
    """Quadratic polynomial residual model: delta(theta) = beta_0 + beta_1*theta + beta_2*(theta x theta).

    Uses sklearn Ridge regression with polynomial features (degree 2,
    including interaction terms).
    """

    def __init__(self, n_theta=N_THETA, n_objectives=4):
        self.n_theta = n_theta
        self.n_objectives = n_objectives
        self.poly = PolynomialFeatures(degree=2, include_bias=True)
        self.models = [Ridge(alpha=1.0) for _ in range(n_objectives)]
        self.fitted = False
        self.X_train = None
        self.y_train = None
        self._residual_std = np.zeros(n_objectives)

    def _build_features(self, X):
        """Build polynomial feature matrix [1, theta, theta_i*theta_j].

        Parameters
        ----------
        X : ndarray (n, 17)

        Returns
        -------
        X_poly : ndarray (n, n_features)
        """
        return self.poly.fit_transform(X) if not self.fitted else self.poly.transform(X)

    def fit(self, X, residuals):
        """Fit quadratic models on training data.

        Parameters
        ----------
        X : ndarray (n, 17)
        residuals : ndarray (n, 4)
            Residuals = ABM_true - NP_predicted.
        """
        X = np.asarray(X, dtype=np.float64)
        residuals = np.asarray(residuals, dtype=np.float64)

        self.X_train = X.copy()
        self.y_train = residuals.copy()

        X_poly = self.poly.fit_transform(X)

        for i, model in enumerate(self.models):
            model.fit(X_poly, residuals[:, i])

        # Estimate residual std from training data
        X_poly_pred = self.poly.transform(X)
        for i, model in enumerate(self.models):
            pred = model.predict(X_poly_pred)
            self._residual_std[i] = np.std(residuals[:, i] - pred)

        self.fitted = True

    def predict(self, X):
        """Predict residual mean and standard deviation.

        Parameters
        ----------
        X : ndarray (n, 17) or (17,)

        Returns
        -------
        mean : ndarray (n, 4)
        std : ndarray (n, 4)
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n = X.shape[0]

        if not self.fitted:
            return np.zeros((n, 4)), np.zeros((n, 4))

        X_poly = self.poly.transform(X)
        mean = np.zeros((n, 4), dtype=np.float64)
        for i, model in enumerate(self.models):
            mean[:, i] = model.predict(X_poly)

        # Constant std estimate from training residuals
        std = np.tile(self._residual_std, (n, 1))
        return mean, std

    def update(self, X_new, residuals_new):
        """Append new data and refit.

        Parameters
        ----------
        X_new : ndarray (m, 17)
        residuals_new : ndarray (m, 4)
        """
        X_new = np.atleast_2d(np.asarray(X_new, dtype=np.float64))
        residuals_new = np.atleast_2d(np.asarray(residuals_new, dtype=np.float64))

        if self.X_train is not None:
            self.X_train = np.vstack([self.X_train, X_new])
            self.y_train = np.vstack([self.y_train, residuals_new])
        else:
            self.X_train = X_new.copy()
            self.y_train = residuals_new.copy()

        self.fitted = False  # Reset so _build_features re-fits poly
        self.fit(self.X_train, self.y_train)

    def cross_validate_error(self):
        """5-fold cross-validation MAPE.

        Returns
        -------
        float
            Mean CV MAPE across objectives.
        """
        if self.X_train is None or self.X_train.shape[0] < 10:
            return 1.0  # Not enough data

        X_poly = self.poly.transform(self.X_train)
        cv_errors = []
        n_folds = min(5, self.X_train.shape[0])

        for i, model in enumerate(self.models):
            scores = cross_val_score(
                Ridge(alpha=1.0), X_poly, self.y_train[:, i],
                cv=n_folds, scoring='neg_mean_absolute_error')
            y_scale = np.abs(self.y_train[:, i]).mean()
            if y_scale > 1e-12:
                cv_errors.append(-scores.mean() / y_scale)
            else:
                cv_errors.append(0.0)

        return float(np.mean(cv_errors))


# ================================================================== #
#  3. GP Residual Model (fallback)                                     #
# ================================================================== #

class GPResidualModel:
    """Four independent Gaussian Process regressors for residual correction.

    The residual is defined as:
        residual = ABM_true_objectives - NP_predicted_objectives

    so that:
        corrected = NP_predicted + GP_predict(residual)

    Each of the 4 objectives gets its own GP with a Matern-5/2 kernel.
    """

    def __init__(self):
        self.gps = []
        for _ in range(4):
            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) \
                     * Matern(length_scale=np.ones(N_THETA), nu=2.5,
                              length_scale_bounds=(1e-2, 1e2))
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                alpha=1e-6,
                normalize_y=True,
            )
            self.gps.append(gp)

        self.fitted = False
        self.X_train = None
        self.y_train = None

    def fit(self, X, residuals):
        """Fit GPs on training data.

        Parameters
        ----------
        X : ndarray[n, 17]
            Strategy parameter vectors.
        residuals : ndarray[n, 4]
            Residuals (ABM_true - NP_predicted) for each objective.
        """
        X = np.asarray(X, dtype=np.float64)
        residuals = np.asarray(residuals, dtype=np.float64)

        self.X_train = X.copy()
        self.y_train = residuals.copy()

        for i, gp in enumerate(self.gps):
            gp.fit(X, residuals[:, i])

        self.fitted = True

    def predict(self, X):
        """Predict residual mean and standard deviation.

        Parameters
        ----------
        X : ndarray[n, 17] or ndarray[17]

        Returns
        -------
        mean : ndarray[n, 4]
        std : ndarray[n, 4]
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n = X.shape[0]
        mean = np.zeros((n, 4), dtype=np.float64)
        std = np.zeros((n, 4), dtype=np.float64)

        if not self.fitted:
            return mean, std

        for i, gp in enumerate(self.gps):
            m, s = gp.predict(X, return_std=True)
            mean[:, i] = m
            std[:, i] = s

        return mean, std

    def update(self, X_new, residuals_new):
        """Incrementally update GPs with new observations.

        Parameters
        ----------
        X_new : ndarray[m, 17]
        residuals_new : ndarray[m, 4]
        """
        X_new = np.asarray(X_new, dtype=np.float64)
        residuals_new = np.asarray(residuals_new, dtype=np.float64)

        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        if residuals_new.ndim == 1:
            residuals_new = residuals_new.reshape(1, -1)

        if self.X_train is not None:
            self.X_train = np.vstack([self.X_train, X_new])
            self.y_train = np.vstack([self.y_train, residuals_new])
        else:
            self.X_train = X_new.copy()
            self.y_train = residuals_new.copy()

        for i, gp in enumerate(self.gps):
            gp.fit(self.X_train, self.y_train[:, i])

        self.fitted = True

    @property
    def n_training(self):
        """Number of training samples currently stored."""
        return 0 if self.X_train is None else self.X_train.shape[0]


# ================================================================== #
#  4. NP Surrogate Evaluator                                           #
# ================================================================== #

class NPSurrogateEvaluator:
    """Combined evaluator: ANP + residual correction.

    Provides corrected objective estimates and uncertainty from the ANP.
    """

    def __init__(self, anp_predictor, residual_model, analytical=None):
        """
        Parameters
        ----------
        anp_predictor : ANPPredictor
            ANP inference interface.
        residual_model : QuadraticResidualModel or GPResidualModel
            Residual correction model.
        analytical : AnalyticalSurrogate, optional
            Only used in Phase 0 before ANP is available.
        """
        self.anp = anp_predictor
        self.residual = residual_model
        self.analytical = analytical

    def evaluate(self, theta):
        """Evaluate corrected objectives with uncertainty.

        Parameters
        ----------
        theta : ndarray[17]

        Returns
        -------
        objectives : ndarray[4]
            Corrected: mu_NP_obj + delta_residual.
        uncertainty : ndarray[4]
            sigma_NP_obj.
        """
        theta = np.asarray(theta, dtype=np.float64)

        # ANP prediction (objectives only, 4-dim)
        mu_obj, sigma_obj = self.anp.predict_objectives(theta.reshape(1, -1))
        mu_obj = mu_obj.flatten()        # (4,)
        sigma_obj = sigma_obj.flatten()  # (4,)

        # Residual correction
        delta_mean, _ = self.residual.predict(theta.reshape(1, -1))
        delta_mean = delta_mean.flatten()  # (4,)

        objectives = mu_obj + delta_mean
        uncertainty = sigma_obj

        return objectives, uncertainty

    def evaluate_batch(self, X):
        """Batch evaluation.

        Parameters
        ----------
        X : ndarray[n, 17]

        Returns
        -------
        F : ndarray[n, 4]
            Corrected objectives.
        U : ndarray[n, 4]
            Uncertainties.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))

        # ANP batch prediction (objectives only, 4-dim)
        mu_obj, sigma_obj = self.anp.predict_objectives(X)

        # Residual correction
        delta_mean, _ = self.residual.predict(X)

        F = mu_obj + delta_mean
        U = sigma_obj

        return F, U


# Backward-compatible alias
SurrogateEvaluator = NPSurrogateEvaluator
