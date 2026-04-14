"""
TR-MOBO multi-objective optimisation for the MaaS ABM framework.

Implements:
    Phase 0: Analytical pretraining of ANP
    Phase A: ABM calibration (LHS + finetune + residual model)
    Phase B: TR-MOBO main loop (EHVI acquisition + trust region updates)
    Phase C: Pareto front verification via full ABM runs
"""

import logging
import time

import numpy as np
from scipy.stats.qmc import LatinHypercube

from config import (
    THETA_LOWER, THETA_UPPER, N_THETA,
    TRMOBO_INIT_SAMPLES, TRMOBO_MAX_ITERATIONS,
    TRMOBO_CONVERGENCE_PATIENCE, TRMOBO_CONVERGENCE_EPS,
    ANP_PRETRAIN_SAMPLES, ANP_PRETRAIN_EPOCHS, ANP_PRETRAIN_LR,
    ANP_FINETUNE_EPOCHS, ANP_FINETUNE_LR,
    ANP_CONTINUAL_FINETUNE_EVERY, ANP_CONTINUAL_FINETUNE_EPOCHS,
    RESIDUAL_METHOD, RESIDUAL_SWITCH_THRESHOLD,
    PARETO_VERIFY_N, PARETO_VERIFY_SEEDS,
    EHVI_REFERENCE_POINT,
    IDX_OBJ_START, N_INTERMEDIATES,
)
from surrogate import (
    AnalyticalSurrogate, QuadraticResidualModel, GPResidualModel,
    NPSurrogateEvaluator,
)
from neural_process import (
    AttentiveNeuralProcess, ANPTrainer, ANPPredictor,
    aggregate_condition_vector,
)
from trust_region import (
    TrustRegion, MultiTrustRegionManager,
    compute_hypervolume, compute_ehvi, maximize_ehvi_in_tr,
    compute_improvement_ratio,
)

logger = logging.getLogger(__name__)


# ================================================================== #
#  LHS Sampling                                                        #
# ================================================================== #

def _generate_lhs_samples(n_samples, bounds_lower=THETA_LOWER,
                           bounds_upper=THETA_UPPER):
    """Generate Latin Hypercube samples in the theta parameter space.

    Returns
    -------
    X : ndarray[n_samples, 17]
    """
    sampler = LatinHypercube(d=N_THETA, seed=42)
    X_unit = sampler.random(n=n_samples)
    X = bounds_lower + X_unit * (bounds_upper - bounds_lower)
    return X


# ================================================================== #
#  Pareto Front Utilities                                              #
# ================================================================== #

def _update_pareto_front(X_all, F_all):
    """Extract non-dominated solutions from all evaluated points.

    Parameters
    ----------
    X_all : ndarray (n, 17)
    F_all : ndarray (n, 4)

    Returns
    -------
    pareto_X : ndarray (m, 17)
    pareto_F : ndarray (m, 4)
    """
    n = F_all.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                is_pareto[i] = False
                break
    return X_all[is_pareto], F_all[is_pareto]


# ================================================================== #
#  Phase 0: Analytical Pretraining                                     #
# ================================================================== #

def _phase0_analytical_pretrain(agents, analytical, condition_vec):
    """Phase 0: Pretrain ANP on analytical surrogate data.

    Parameters
    ----------
    agents : dict
    analytical : AnalyticalSurrogate
    condition_vec : ndarray (96,)

    Returns
    -------
    anp_model : AttentiveNeuralProcess
    anp_trainer : ANPTrainer
    """
    logger.info("Phase 0: Analytical pretraining of ANP")
    logger.info("  Generating %d LHS samples for pretraining...",
                ANP_PRETRAIN_SAMPLES)

    # 1. LHS sampling
    X_pretrain = _generate_lhs_samples(ANP_PRETRAIN_SAMPLES)

    # 2. Evaluate with analytical surrogate (13-dim: 9 intermediates + 4 objectives)
    Y_pretrain = np.zeros((ANP_PRETRAIN_SAMPLES, IDX_OBJ_START + 4), dtype=np.float64)
    for i in range(ANP_PRETRAIN_SAMPLES):
        Y_pretrain[i] = analytical.evaluate_with_intermediates(X_pretrain[i])
        if (i + 1) % 500 == 0:
            logger.info("  Analytical eval: %d/%d", i + 1, ANP_PRETRAIN_SAMPLES)

    # 3. Create ANP model
    anp_model = AttentiveNeuralProcess()
    anp_trainer = ANPTrainer(anp_model, lr=ANP_PRETRAIN_LR)

    # 4. Pretrain on 13-dim data
    analytical_data = {'theta': X_pretrain, 'y': Y_pretrain}
    anp_trainer.pretrain(
        analytical_data, condition_vec,
        epochs=ANP_PRETRAIN_EPOCHS,
    )

    logger.info("Phase 0 complete: ANP pretrained on %d analytical samples",
                ANP_PRETRAIN_SAMPLES)
    return anp_model, anp_trainer


# ================================================================== #
#  Phase A: ABM Calibration                                            #
# ================================================================== #

def _phase_a_abm_calibration(anp_model, anp_trainer, analytical,
                              abm, agents, condition_vec):
    """Phase A: Calibrate ANP with ABM data and build residual model.

    Parameters
    ----------
    anp_model : AttentiveNeuralProcess
    anp_trainer : ANPTrainer
    analytical : AnalyticalSurrogate
    abm : callable
    agents : dict
    condition_vec : ndarray (96,)

    Returns
    -------
    surrogate : NPSurrogateEvaluator
    tr_manager : MultiTrustRegionManager
    context_data : dict {'theta': ndarray, 'y': ndarray}
    residual_model : QuadraticResidualModel or GPResidualModel
    X_abm : ndarray (n, 17)
    F_abm : ndarray (n, 4)
    """
    logger.info("=" * 60)
    logger.info("Phase A: ABM Calibration")
    logger.info("=" * 60)

    # 1. LHS sampling for ABM evaluation
    X_abm = _generate_lhs_samples(TRMOBO_INIT_SAMPLES)
    F_abm_obj = np.zeros((TRMOBO_INIT_SAMPLES, 4), dtype=np.float64)

    logger.info("  Running ABM on %d LHS samples...", TRMOBO_INIT_SAMPLES)
    t0 = time.time()
    for i in range(TRMOBO_INIT_SAMPLES):
        try:
            F_abm_obj[i] = abm(X_abm[i], agents)
        except Exception as e:
            logger.warning("ABM failed on sample %d: %s", i, e)
            F_abm_obj[i] = analytical.evaluate(X_abm[i])
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (TRMOBO_INIT_SAMPLES - i - 1)
            logger.info("  ABM sample %d/%d (elapsed: %.1fs, ETA: %.1fs)",
                        i + 1, TRMOBO_INIT_SAMPLES, elapsed, eta)

    elapsed = time.time() - t0
    logger.info("  ABM evaluation complete: %d samples in %.1fs",
                TRMOBO_INIT_SAMPLES, elapsed)

    # Build 13-dim data: intermediates from analytical + objectives from ABM
    Y_abm_13 = np.zeros((TRMOBO_INIT_SAMPLES, IDX_OBJ_START + 4), dtype=np.float64)
    for i in range(TRMOBO_INIT_SAMPLES):
        full_analytical = analytical.evaluate_with_intermediates(X_abm[i])
        Y_abm_13[i, :IDX_OBJ_START] = full_analytical[:IDX_OBJ_START]  # intermediates
        Y_abm_13[i, IDX_OBJ_START:] = F_abm_obj[i]  # ABM objectives

    # 2. Finetune ANP on 13-dim ABM data
    logger.info("  Finetuning ANP on ABM data (13-dim)...")
    abm_data = {'theta': X_abm, 'y': Y_abm_13}
    anp_trainer.finetune(
        abm_data, condition_vec,
        epochs=ANP_FINETUNE_EPOCHS,
    )

    # 3. Create predictor and compute residuals (on 4-dim objectives only)
    predictor = ANPPredictor(
        anp_model, X_abm, Y_abm_13, condition_vec,
        normalizer=anp_trainer.normalizer)
    mu_all, _ = predictor.predict(X_abm)
    F_predicted_obj = mu_all[:, IDX_OBJ_START:]  # only objective dims
    residuals = F_abm_obj - F_predicted_obj

    # 4. Fit residual model (on 4-dim residuals)
    if RESIDUAL_METHOD == 'quadratic':
        residual_model = QuadraticResidualModel()
        residual_model.fit(X_abm, residuals)
        cv_error = residual_model.cross_validate_error()
        logger.info("  Quadratic residual CV error: %.4f", cv_error)
        if cv_error > RESIDUAL_SWITCH_THRESHOLD:
            logger.info("  CV error > %.2f, switching to GP residual",
                        RESIDUAL_SWITCH_THRESHOLD)
            residual_model = GPResidualModel()
            residual_model.fit(X_abm, residuals)
    else:
        residual_model = GPResidualModel()
        residual_model.fit(X_abm, residuals)

    # 5. Create combined surrogate evaluator
    surrogate = NPSurrogateEvaluator(predictor, residual_model, analytical)

    # 6. Initialize trust regions from ABM objective data
    tr_manager = MultiTrustRegionManager()
    tr_manager.initialize_from_data(X_abm, F_abm_obj)
    logger.info("  Initialized %d trust regions", tr_manager.n_regions)

    context_data = {'theta': X_abm.copy(), 'y': Y_abm_13.copy()}

    logger.info("Phase A complete")
    return surrogate, tr_manager, context_data, residual_model, X_abm, F_abm_obj


# ================================================================== #
#  Phase B: TR-MOBO Main Loop                                         #
# ================================================================== #

def _phase_b_trmobo(surrogate, tr_manager, anp_trainer,
                     residual_model, abm, agents, condition_vec,
                     context_data, X_all, F_all):
    """Phase B: TR-MOBO iterative optimization.

    Parameters
    ----------
    surrogate : NPSurrogateEvaluator
    tr_manager : MultiTrustRegionManager
    anp_trainer : ANPTrainer
    residual_model : residual model instance
    abm : callable
    agents : dict
    condition_vec : ndarray (96,)
    context_data : dict
    X_all : ndarray, all evaluated X so far
    F_all : ndarray, all evaluated F so far

    Returns
    -------
    pareto_X : ndarray
    pareto_F : ndarray
    iteration_history : list[dict]
    X_all : ndarray (n, 17) -- all evaluated theta
    F_all : ndarray (n, 4) -- all evaluated objectives
    Y_all_13 : ndarray (n, 13) -- all 13-dim data
    """
    logger.info("=" * 60)
    logger.info("Phase B: TR-MOBO Optimization")
    logger.info("  Max iterations: %d", TRMOBO_MAX_ITERATIONS)
    logger.info("  Convergence patience: %d", TRMOBO_CONVERGENCE_PATIENCE)
    logger.info("=" * 60)

    ref_point = EHVI_REFERENCE_POINT
    pareto_X, pareto_F = _update_pareto_front(X_all, F_all)
    hv_current = compute_hypervolume(pareto_F, ref_point)

    # Build initial Y_all_13 from context_data
    Y_all_13 = context_data['y'].copy()  # (n_init, 13)

    iteration_history = []
    no_improve_count = 0

    for iteration in range(TRMOBO_MAX_ITERATIONS):
        t_iter = time.time()

        # 1. Select one candidate per trust region
        candidates = []
        predicted_ehvi = []
        tr_bounds_list = tr_manager.get_all_bounds()

        for r_idx, tr_bounds in enumerate(tr_bounds_list):
            candidate = maximize_ehvi_in_tr(
                surrogate.anp, tr_bounds, pareto_F, ref_point,
                n_candidates=100,  # Reduced for speed
                n_best=1,
            )
            candidates.append(candidate[0])  # (17,)

            # Compute predicted EHVI for this candidate (objectives only)
            mu_obj, sigma_obj = surrogate.anp.predict_objectives(candidate)
            ehvi_val = compute_ehvi(mu_obj[0], sigma_obj[0], pareto_F, ref_point)
            predicted_ehvi.append(ehvi_val)

        candidates = np.array(candidates)  # (n_regions, 17)

        # 2. ABM evaluation of candidates → build 13-dim context entries
        F_candidates_obj = np.zeros((len(candidates), 4), dtype=np.float64)
        Y_candidates_13 = np.zeros((len(candidates), IDX_OBJ_START + 4),
                                    dtype=np.float64)
        for i in range(len(candidates)):
            try:
                F_candidates_obj[i] = abm(candidates[i], agents)
            except Exception as e:
                logger.warning("ABM failed at iter %d, TR %d: %s",
                               iteration, i, e)
                F_candidates_obj[i], _ = surrogate.evaluate(candidates[i])
            # Build 13-dim: intermediates from analytical + objectives from ABM
            full_analytical = surrogate.analytical.evaluate_with_intermediates(
                candidates[i])
            Y_candidates_13[i, :IDX_OBJ_START] = full_analytical[:IDX_OBJ_START]
            Y_candidates_13[i, IDX_OBJ_START:] = F_candidates_obj[i]

        # 3. Update Pareto front and compute HV
        X_all = np.vstack([X_all, candidates])
        F_all = np.vstack([F_all, F_candidates_obj])
        Y_all_13 = np.vstack([Y_all_13, Y_candidates_13])
        pareto_X, pareto_F = _update_pareto_front(X_all, F_all)
        hv_new = compute_hypervolume(pareto_F, ref_point)
        hv_gain = hv_new - hv_current

        # 4. Update trust regions
        for r_idx in range(len(candidates)):
            actual_gain = hv_new - hv_current  # Shared gain
            rho = compute_improvement_ratio(
                predicted_ehvi[r_idx], actual_gain / max(len(candidates), 1))

            # Update center if this candidate is non-dominated
            new_center = None
            if hv_gain > 0:
                new_center = candidates[r_idx]
            tr_manager.update_regions(r_idx, rho, new_center)

            # Restart degenerate TRs
            if tr_manager.regions[r_idx].length <= tr_manager.regions[r_idx].length_min:
                tr_manager.restart_region(r_idx)
                logger.info("  Iter %d: Restarted TR %d", iteration, r_idx)

        hv_current = hv_new

        # 5. Add 13-dim context to ANP predictor
        surrogate.anp.add_context(candidates, Y_candidates_13)

        # 6. Periodic continual finetuning
        if (iteration + 1) % ANP_CONTINUAL_FINETUNE_EVERY == 0:
            logger.info("  Iter %d: Finetuning ANP (continual)...", iteration)
            # Use accumulated Y_all_13 directly (no reconstruction needed)
            all_data = {'theta': X_all, 'y': Y_all_13}
            anp_trainer.continual_finetune(
                all_data, condition_vec,
                epochs=ANP_CONTINUAL_FINETUNE_EPOCHS,
            )
            # Refit residual model from scratch
            mu_all, _ = surrogate.anp.predict(X_all)
            F_pred_obj = mu_all[:, IDX_OBJ_START:]
            residuals_all = F_all - F_pred_obj
            residual_model.fit(X_all, residuals_all)

        # 7. Record iteration history
        tr_status = tr_manager.get_status()
        iter_record = {
            'iteration': iteration,
            'hypervolume': hv_current,
            'hv_gain': hv_gain,
            'n_pareto': pareto_F.shape[0],
            'tr_lengths': tr_status['lengths'],
            'n_abm_evals': X_all.shape[0],
            'wall_time': time.time() - t_iter,
        }
        iteration_history.append(iter_record)

        if (iteration + 1) % 10 == 0 or iteration == 0:
            logger.info(
                "  Iter %d/%d: HV=%.6f (+%.6f), Pareto=%d, TRs=%s",
                iteration + 1, TRMOBO_MAX_ITERATIONS,
                hv_current, hv_gain, pareto_F.shape[0],
                [f"{l:.3f}" for l in tr_status['lengths']],
            )

        # 8. Convergence check
        if hv_gain < TRMOBO_CONVERGENCE_EPS:
            no_improve_count += 1
        else:
            no_improve_count = 0

        if no_improve_count >= TRMOBO_CONVERGENCE_PATIENCE:
            logger.info("  Converged at iteration %d (no improvement for %d rounds)",
                        iteration, TRMOBO_CONVERGENCE_PATIENCE)
            break

    logger.info("Phase B complete: %d iterations, final HV=%.6f, Pareto=%d",
                len(iteration_history), hv_current, pareto_F.shape[0])

    return pareto_X, pareto_F, iteration_history, X_all, F_all, Y_all_13


# ================================================================== #
#  Phase C: Pareto Verification                                        #
# ================================================================== #

def _verify_pareto_front(pareto_X, abm, agents, n_seeds=PARETO_VERIFY_SEEDS):
    """Verify Pareto-optimal solutions by running full ABM with multiple seeds.

    Parameters
    ----------
    pareto_X : ndarray[m, 17]
    abm : callable
    agents : dict
    n_seeds : int

    Returns
    -------
    verified_X : ndarray[m, 17]
    verified_F_mean : ndarray[m, 4]
    verified_F_std : ndarray[m, 4]
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


# ================================================================== #
#  Main Optimisation Pipeline                                          #
# ================================================================== #

def run_optimization(agents, abm, ch3_model, ch1_model, ch2_model,
                     scenario_modifier=None):
    """Full TR-MOBO pipeline: Phase 0 -> A -> B -> C.

    Parameters
    ----------
    agents : dict[str, ndarray[N]]
        Agent population attribute arrays.
    abm : callable
        Full ABM evaluation function.
        Signature: abm(theta, agents) -> ndarray[4]
        (or abm(theta, agents, seed=int) for verification phase).
    ch3_model : module
    ch1_model : module
    ch2_model : module
    scenario_modifier : callable, optional

    Returns
    -------
    result_dict : dict
        {'pareto_X', 'pareto_F', 'n_iterations', 'converged'}
    pareto_X : ndarray[m, 17]
    pareto_F : ndarray[m, 4]
    history : dict
        Contains 'iteration_history', 'hypervolume',
        'pareto_F_std', 'wall_time'.
    """
    wall_t0 = time.time()

    # Apply scenario modifier if provided
    if scenario_modifier is not None:
        agents, _, _ = scenario_modifier(agents, np.zeros(N_THETA), {})

    # ============================================================== #
    # Build analytical surrogate and condition vector                  #
    # ============================================================== #
    analytical = AnalyticalSurrogate(agents, ch3_model, ch1_model, ch2_model)
    condition_vec = aggregate_condition_vector(agents)

    # ============================================================== #
    # Phase 0: Analytical pretraining                                  #
    # ============================================================== #
    logger.info("=" * 60)
    logger.info("Phase 0: ANP Pretraining")
    logger.info("=" * 60)

    anp_model, anp_trainer = _phase0_analytical_pretrain(
        agents, analytical, condition_vec)

    # ============================================================== #
    # Phase A: ABM calibration                                         #
    # ============================================================== #
    surrogate, tr_manager, context_data, residual_model, X_abm, F_abm = \
        _phase_a_abm_calibration(
            anp_model, anp_trainer, analytical,
            abm, agents, condition_vec)

    # ============================================================== #
    # Phase B: TR-MOBO main loop                                       #
    # ============================================================== #
    pareto_X, pareto_F, iteration_history, X_all, F_all, Y_all_13 = \
        _phase_b_trmobo(
            surrogate, tr_manager, anp_trainer,
            residual_model, abm, agents, condition_vec,
            context_data, X_abm, F_abm)

    # Limit Pareto solutions for verification
    if pareto_X.shape[0] > PARETO_VERIFY_N:
        indices = np.linspace(0, pareto_X.shape[0] - 1,
                              PARETO_VERIFY_N, dtype=int)
        sort_order = np.argsort(pareto_F[:, 0])
        pareto_X = pareto_X[sort_order][indices]
        pareto_F = pareto_F[sort_order][indices]

    # ============================================================== #
    # Phase C: Pareto verification                                     #
    # ============================================================== #
    logger.info("=" * 60)
    logger.info("Phase C: Pareto Verification")
    logger.info("=" * 60)

    if pareto_X.shape[0] > 0:
        pareto_X, pareto_F, pareto_F_std = _verify_pareto_front(
            pareto_X, abm, agents, n_seeds=PARETO_VERIFY_SEEDS)
    else:
        logger.warning("No Pareto solutions found; returning empty results.")
        pareto_X = np.empty((0, N_THETA))
        pareto_F = np.empty((0, 4))
        pareto_F_std = np.empty((0, 4))

    wall_time = time.time() - wall_t0

    # ============================================================== #
    # Summary                                                          #
    # ============================================================== #
    n_iterations = len(iteration_history)
    converged = (n_iterations < TRMOBO_MAX_ITERATIONS)

    logger.info("=" * 60)
    logger.info("Optimisation pipeline complete.")
    logger.info("  Total wall time: %.1f s (%.1f min)", wall_time, wall_time / 60)
    logger.info("  Pareto solutions verified: %d", pareto_X.shape[0])
    logger.info("  Total iterations: %d, Converged: %s", n_iterations, converged)
    logger.info("=" * 60)

    # ============================================================== #
    # Assemble results                                                 #
    # ============================================================== #
    result_dict = {
        'pareto_X': pareto_X,
        'pareto_F': pareto_F,
        'n_iterations': n_iterations,
        'converged': converged,
    }

    # Build hypervolume history for visualization
    hypervolume_history = [rec['hypervolume'] for rec in iteration_history]

    history = {
        'iteration_history': iteration_history,
        'hypervolume': hypervolume_history,
        'analytical': analytical,
        'anp_model': anp_model,
        'anp_trainer': anp_trainer,
        'anp_normalizer': anp_trainer.normalizer,
        'tr_manager': tr_manager,
        'residual_model': residual_model,
        'pareto_F_std': pareto_F_std,
        'wall_time': wall_time,
        'X_all': X_all,
        'F_all': F_all,
        'Y_all_13': Y_all_13,
    }

    return result_dict, pareto_X, pareto_F, history
