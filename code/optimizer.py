"""
Multi-objective optimisation for the MaaS ABM framework.

Implements a Surrogate-Based Optimisation (SBO) pipeline:
    Phase A: Latin Hypercube Sampling for initial GP training
    Phase B: NSGA-II with infill callbacks for GP refinement
    Phase C: Pareto front verification via full ABM runs

Uses pymoo for NSGA-II with 17 decision variables, 4 objectives, and
2 inequality constraints.
"""

import logging
import time

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from config import (
    THETA_LOWER, THETA_UPPER, N_THETA,
    LHS_SAMPLES, NSGA2_POP_SIZE, NSGA2_N_GEN,
    INFILL_PER_GEN, PARETO_VERIFY_N, PARETO_VERIFY_SEEDS,
)
from surrogate import AnalyticalSurrogate, GPResidualModel, SurrogateEvaluator

logger = logging.getLogger(__name__)


# ================================================================== #
#  1. NSGA-II Problem Definition                                       #
# ================================================================== #

class MaaSSBOProblem(Problem):
    """NSGA-II problem wrapper for the MaaS surrogate evaluator.

    Decision variables (17):
        See config.THETA_NAMES for the full layout.

    Objectives (4, all minimised):
        0: -adoption_rate       (maximise adoption)
        1: -net_revenue         (maximise revenue)
        2: gini_coefficient     (minimise inequality)
        3: -carbon_reduction    (maximise carbon reduction)

    Inequality constraints (2):
        G[0] = ps_BF - ps_MA <= 0
            BF bundle must be cheaper than MA bundle (price-scale ordering).
        G[1] = tau_low - tau_high <= 0
            Low awareness threshold must not exceed high threshold.
    """

    def __init__(self, surrogate_evaluator):
        """
        Parameters
        ----------
        surrogate_evaluator : SurrogateEvaluator
            Combined analytical + GP surrogate evaluator.
        """
        super().__init__(
            n_var=N_THETA,
            n_obj=4,
            n_ieq_constr=2,
            xl=THETA_LOWER,
            xu=THETA_UPPER,
        )
        self.evaluator_model = surrogate_evaluator

    def _evaluate(self, X, out, *args, **kwargs):
        """Batch-evaluate the population.

        Parameters
        ----------
        X : ndarray[pop_size, 17]
            Decision variable matrix.
        out : dict
            pymoo output dictionary. Sets:
            - out["F"]: ndarray[pop_size, 4] objectives
            - out["G"]: ndarray[pop_size, 2] constraints
        """
        pop_size = X.shape[0]

        # Evaluate objectives via surrogate
        F, U = self.evaluator_model.evaluate_batch(X)

        # Inequality constraints: G[i] <= 0 is feasible
        G = np.zeros((pop_size, 2), dtype=np.float64)

        # Constraint 1: ps_BF <= ps_MA  =>  ps_BF - ps_MA <= 0
        # theta[1] = ps_BF, theta[3] = ps_MA
        G[:, 0] = X[:, 1] - X[:, 3]

        # Constraint 2: tau_low <= tau_high  =>  tau_low - tau_high <= 0
        # theta[7] = tau_low, theta[6] = tau_high
        G[:, 1] = X[:, 7] - X[:, 6]

        out["F"] = F
        out["G"] = G

        # Store uncertainty for potential use by infill callback
        if hasattr(self, '_last_uncertainty'):
            self._last_uncertainty = U
        else:
            self._last_uncertainty = U


# ================================================================== #
#  2. Infill Callback for GP Refinement                                #
# ================================================================== #

class InfillCallback(Callback):
    """Every generation, select top candidates for ABM evaluation and update GP.

    Selection strategy:
        1. From the current population, find solutions near the Pareto front.
        2. Among those, select the ones with highest GP uncertainty.
        3. Run the full ABM on these solutions to get true objectives.
        4. Compute residuals (true - analytical) and update the GP.
    """

    def __init__(self, abm, agents, analytical, gp_model, n_infill=INFILL_PER_GEN):
        """
        Parameters
        ----------
        abm : callable
            ABM evaluation function: abm(theta, agents) -> ndarray[4].
        agents : dict[str, ndarray[N]]
            Agent population.
        analytical : AnalyticalSurrogate
            Analytical surrogate for computing residuals.
        gp_model : GPResidualModel
            GP residual model to update.
        n_infill : int
            Number of ABM evaluations per generation.
        """
        super().__init__()
        self.abm = abm
        self.agents = agents
        self.analytical = analytical
        self.gp = gp_model
        self.n_infill = n_infill

        # Track cumulative ABM evaluations
        self.total_abm_evals = 0
        self.infill_history_X = []
        self.infill_history_F = []

    def notify(self, algorithm):
        """Called at the end of each generation.

        Selects high-uncertainty Pareto-near solutions for ABM evaluation
        and updates the GP residual model.
        """
        gen = algorithm.n_gen
        pop = algorithm.pop

        # Only run infill every few generations to amortise ABM cost
        if gen % 5 != 0 and gen > 1:
            return

        X_pop = pop.get("X")      # (pop_size, 17)
        F_pop = pop.get("F")      # (pop_size, 4)

        if X_pop is None or F_pop is None:
            return

        pop_size = X_pop.shape[0]

        # ---- Step 1: Compute uncertainty for each individual ----
        _, U = self.gp.predict(X_pop) if self.gp.fitted else (
            np.zeros((pop_size, 4)), np.ones((pop_size, 4)))

        # Aggregate uncertainty: sum of std across objectives
        total_uncertainty = U.sum(axis=1)  # (pop_size,)

        # ---- Step 2: Identify Pareto-near solutions ----
        # Use non-dominated rank (approximate: compare each to population)
        is_nondominated = np.ones(pop_size, dtype=bool)
        for i in range(pop_size):
            for j in range(pop_size):
                if i == j:
                    continue
                if np.all(F_pop[j] <= F_pop[i]) and np.any(F_pop[j] < F_pop[i]):
                    is_nondominated[i] = False
                    break

        # ---- Step 3: Score = uncertainty, with bonus for Pareto-near ----
        scores = total_uncertainty.copy()
        scores[is_nondominated] *= 2.0  # Double weight for Pareto front members

        # Select top-n_infill candidates
        n_select = min(self.n_infill, pop_size)
        selected_idx = np.argsort(scores)[-n_select:]

        X_selected = X_pop[selected_idx]

        # ---- Step 4: Run ABM on selected candidates ----
        F_true = np.zeros((n_select, 4), dtype=np.float64)
        F_analytical = np.zeros((n_select, 4), dtype=np.float64)

        for i in range(n_select):
            try:
                F_true[i] = self.abm(X_selected[i], self.agents)
                F_analytical[i] = self.analytical.evaluate(X_selected[i])
            except Exception as e:
                logger.warning(
                    "ABM evaluation failed for infill point %d at gen %d: %s",
                    i, gen, e)
                F_true[i] = F_analytical[i]  # fallback: zero residual

        residuals = F_true - F_analytical

        # ---- Step 5: Update GP ----
        self.gp.update(X_selected, residuals)

        # Track history
        self.total_abm_evals += n_select
        self.infill_history_X.append(X_selected)
        self.infill_history_F.append(F_true)

        logger.info(
            "Gen %d: infill %d points, total ABM evals = %d, "
            "GP training size = %d",
            gen, n_select, self.total_abm_evals, self.gp.n_training)


# ================================================================== #
#  3. LHS Initialisation                                               #
# ================================================================== #

def _generate_lhs_samples(n_samples=LHS_SAMPLES):
    """Generate Latin Hypercube samples in the theta parameter space.

    Returns
    -------
    X : ndarray[n_samples, 17]
        LHS samples within [THETA_LOWER, THETA_UPPER].
    """
    from scipy.stats.qmc import LatinHypercube

    sampler = LatinHypercube(d=N_THETA, seed=42)
    X_unit = sampler.random(n=n_samples)  # (n_samples, 17) in [0,1]

    # Scale to parameter bounds
    X = THETA_LOWER + X_unit * (THETA_UPPER - THETA_LOWER)

    return X


def _initial_gp_training(X_lhs, abm, agents, analytical):
    """Run ABM on LHS samples and train the initial GP.

    Parameters
    ----------
    X_lhs : ndarray[n, 17]
    abm : callable
        abm(theta, agents) -> ndarray[4]
    agents : dict
    analytical : AnalyticalSurrogate

    Returns
    -------
    gp_model : GPResidualModel
        Fitted GP residual model.
    F_true : ndarray[n, 4]
        True ABM objectives for LHS samples.
    F_analytical : ndarray[n, 4]
        Analytical surrogate objectives for LHS samples.
    """
    n = X_lhs.shape[0]
    F_true = np.zeros((n, 4), dtype=np.float64)
    F_analytical = np.zeros((n, 4), dtype=np.float64)

    logger.info("Phase A: Running ABM on %d LHS samples...", n)
    t0 = time.time()

    for i in range(n):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            logger.info("  LHS sample %d/%d  (elapsed: %.1fs, ETA: %.1fs)",
                        i + 1, n, elapsed, eta)
        try:
            F_true[i] = abm(X_lhs[i], agents)
        except Exception as e:
            logger.warning("ABM failed on LHS sample %d: %s", i, e)
            F_true[i] = np.zeros(4)

        F_analytical[i] = analytical.evaluate(X_lhs[i])

    residuals = F_true - F_analytical
    elapsed = time.time() - t0
    logger.info("Phase A complete: %d ABM evaluations in %.1fs", n, elapsed)

    # Fit GP
    gp_model = GPResidualModel()
    gp_model.fit(X_lhs, residuals)

    return gp_model, F_true, F_analytical


# ================================================================== #
#  4. Pareto Front Verification                                        #
# ================================================================== #

def _verify_pareto_front(pareto_X, abm, agents, n_seeds=PARETO_VERIFY_SEEDS):
    """Verify Pareto-optimal solutions by running full ABM with multiple seeds.

    Parameters
    ----------
    pareto_X : ndarray[m, 17]
        Pareto-optimal strategy vectors from NSGA-II.
    abm : callable
        abm(theta, agents, seed=int) -> ndarray[4]
    agents : dict
    n_seeds : int
        Number of random seeds per solution.

    Returns
    -------
    verified_X : ndarray[m, 17]
        Pareto solutions (same as input).
    verified_F_mean : ndarray[m, 4]
        Mean ABM objectives across seeds.
    verified_F_std : ndarray[m, 4]
        Std of ABM objectives across seeds.
    """
    m = pareto_X.shape[0]
    all_F = np.zeros((m, n_seeds, 4), dtype=np.float64)

    logger.info("Phase C: Verifying %d Pareto solutions with %d seeds each...",
                m, n_seeds)
    t0 = time.time()

    for i in range(m):
        for s in range(n_seeds):
            try:
                all_F[i, s] = abm(pareto_X[i], agents, seed=42 + s)
            except TypeError:
                # ABM may not accept seed kwarg; run without it
                all_F[i, s] = abm(pareto_X[i], agents)
            except Exception as e:
                logger.warning("ABM verification failed: solution %d, seed %d: %s",
                               i, s, e)
                all_F[i, s] = np.nan

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info("  Verified %d/%d solutions (%.1fs)", i + 1, m, elapsed)

    verified_F_mean = np.nanmean(all_F, axis=1)
    verified_F_std = np.nanstd(all_F, axis=1)

    elapsed = time.time() - t0
    logger.info("Phase C complete: %d solutions verified in %.1fs", m, elapsed)

    return pareto_X, verified_F_mean, verified_F_std


def _extract_pareto_front(result, max_solutions=PARETO_VERIFY_N):
    """Extract Pareto-optimal solutions from the pymoo result.

    Parameters
    ----------
    result : pymoo.core.result.Result
    max_solutions : int
        Maximum number of Pareto solutions to return.

    Returns
    -------
    pareto_X : ndarray[m, 17]
    pareto_F : ndarray[m, 4]
    """
    # Get non-dominated solutions
    F = result.F
    X = result.X

    if F is None or X is None:
        return np.empty((0, N_THETA)), np.empty((0, 4))

    # Non-dominated sorting (rank 0 only)
    n = F.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                is_pareto[i] = False
                break

    pareto_X = X[is_pareto]
    pareto_F = F[is_pareto]

    # Limit to max_solutions (uniform spacing along first objective)
    if pareto_X.shape[0] > max_solutions:
        indices = np.linspace(0, pareto_X.shape[0] - 1,
                              max_solutions, dtype=int)
        # Sort by first objective for uniform spacing
        sort_order = np.argsort(pareto_F[:, 0])
        pareto_X = pareto_X[sort_order][indices]
        pareto_F = pareto_F[sort_order][indices]

    return pareto_X, pareto_F


# ================================================================== #
#  5. Main Optimisation Pipeline                                       #
# ================================================================== #

def run_optimization(agents, abm, ch3_model, ch1_model, ch2_model,
                     scenario_modifier=None):
    """Full SBO pipeline: LHS init -> NSGA-II with infill -> Pareto verification.

    Parameters
    ----------
    agents : dict[str, ndarray[N]]
        Agent population attribute arrays.
    abm : callable
        Full ABM evaluation function.
        Signature: abm(theta, agents) -> ndarray[4]
        (or abm(theta, agents, seed=int) for verification phase).
    ch3_model : module
        Ch3 bundle-choice model.
    ch1_model : module
        Ch1 trial probability model.
    ch2_model : module
        Ch2 subscription probability model.
    scenario_modifier : callable, optional
        Function that modifies agents dict before evaluation,
        e.g. for policy scenarios. Signature: modifier(agents) -> agents.

    Returns
    -------
    result : pymoo.core.result.Result
        Raw NSGA-II result object.
    pareto_X : ndarray[m, 17]
        Verified Pareto-optimal strategy vectors.
    pareto_F : ndarray[m, 4]
        Verified mean objective values.
    history : dict
        Contains 'gp_model', 'lhs_X', 'lhs_F_true', 'lhs_F_analytical',
        'infill_callback', 'pareto_F_std', 'wall_time'.
    """
    wall_t0 = time.time()

    # Apply scenario modifier if provided (pre-processes agents only)
    if scenario_modifier is not None:
        agents, _, _ = scenario_modifier(agents, np.zeros(N_THETA), {})

    # ============================================================== #
    # Phase A: Build analytical surrogate and initialise GP via LHS   #
    # ============================================================== #
    logger.info("=" * 60)
    logger.info("Phase A: Initialisation")
    logger.info("=" * 60)

    analytical = AnalyticalSurrogate(agents, ch3_model, ch1_model, ch2_model)

    X_lhs = _generate_lhs_samples(n_samples=LHS_SAMPLES)
    gp_model, F_true_lhs, F_analytical_lhs = _initial_gp_training(
        X_lhs, abm, agents, analytical)

    surrogate = SurrogateEvaluator(analytical, gp_model)

    # ============================================================== #
    # Phase B: NSGA-II with surrogate + infill callbacks              #
    # ============================================================== #
    logger.info("=" * 60)
    logger.info("Phase B: NSGA-II Optimisation")
    logger.info("  Population size: %d", NSGA2_POP_SIZE)
    logger.info("  Generations: %d", NSGA2_N_GEN)
    logger.info("  Infill per trigger: %d", INFILL_PER_GEN)
    logger.info("=" * 60)

    problem = MaaSSBOProblem(surrogate)

    infill_cb = InfillCallback(
        abm=abm,
        agents=agents,
        analytical=analytical,
        gp_model=gp_model,
        n_infill=INFILL_PER_GEN,
    )

    algorithm = NSGA2(
        pop_size=NSGA2_POP_SIZE,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", NSGA2_N_GEN)

    result = minimize(
        problem,
        algorithm,
        termination,
        callback=infill_cb,
        seed=42,
        verbose=True,
    )

    logger.info("NSGA-II complete: %d evaluations, %d generations",
                result.algorithm.evaluator.n_eval, result.algorithm.n_gen)

    # ============================================================== #
    # Phase C: Pareto front extraction and ABM verification           #
    # ============================================================== #
    logger.info("=" * 60)
    logger.info("Phase C: Pareto Verification")
    logger.info("=" * 60)

    pareto_X_surrogate, pareto_F_surrogate = _extract_pareto_front(
        result, max_solutions=PARETO_VERIFY_N)

    if pareto_X_surrogate.shape[0] > 0:
        pareto_X, pareto_F, pareto_F_std = _verify_pareto_front(
            pareto_X_surrogate, abm, agents, n_seeds=PARETO_VERIFY_SEEDS)
    else:
        logger.warning("No Pareto solutions found; returning empty results.")
        pareto_X = np.empty((0, N_THETA))
        pareto_F = np.empty((0, 4))
        pareto_F_std = np.empty((0, 4))

    wall_time = time.time() - wall_t0
    logger.info("=" * 60)
    logger.info("Optimisation pipeline complete.")
    logger.info("  Total wall time: %.1f s (%.1f min)", wall_time, wall_time / 60)
    logger.info("  Pareto solutions verified: %d", pareto_X.shape[0])
    logger.info("  Total ABM evaluations: %d",
                LHS_SAMPLES + infill_cb.total_abm_evals
                + pareto_X.shape[0] * PARETO_VERIFY_SEEDS)
    logger.info("=" * 60)

    # ============================================================== #
    # Assemble history                                                 #
    # ============================================================== #
    history = {
        'gp_model': gp_model,
        'analytical': analytical,
        'lhs_X': X_lhs,
        'lhs_F_true': F_true_lhs,
        'lhs_F_analytical': F_analytical_lhs,
        'infill_callback': infill_cb,
        'pareto_F_surrogate': pareto_F_surrogate,
        'pareto_F_std': pareto_F_std,
        'wall_time': wall_time,
    }

    return result, pareto_X, pareto_F, history
