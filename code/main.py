"""
MaaS ABM Optimization Framework - Main Entry Point

Usage:
    python main.py [--scenario S0_baseline] [--quick] [--seed 42]
    python main.py --all-scenarios --output results
    python main.py --method trmobo   (default: ANP + TR-MOBO)

Workflow:
    1. Load agent population from 2023 travel survey + SP survey data
    2. Initialize Ch3 (bundle choice), Ch1 (trial), Ch2 (subscription) models
    3. Create ABM engine with Bass-diffusion + spatial network effects
    4. Either run a quick single-evaluation test (--quick) or
       full TR-MOBO multi-objective optimisation
    5. Generate visualisations and save results
"""

import argparse
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    THETA_DEFAULT, THETA_NAMES, N_THETA,
    THETA_LOWER, THETA_UPPER,
    N_WEEKS, AGENT_WEIGHT,
    TRMOBO_MAX_ITERATIONS, TRMOBO_INIT_SAMPLES,
)
import data_loader
import ch3_model
import ch1_model
import ch2_model
from abm_engine import ABMSimulation
from scenarios import get_scenario, SCENARIOS

# Lazy imports for modules requiring torch / scikit-learn (not needed for --quick)
def _import_optimizer():
    from surrogate import AnalyticalSurrogate, NPSurrogateEvaluator
    from optimizer import run_optimization
    return run_optimization


def main():
    parser = argparse.ArgumentParser(
        description='MaaS ABM Optimization Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Scenarios:\n'
            '  S0_baseline              No policy intervention (control)\n'
            '  S1_carbon_credit         Carbon credit for mode shift\n'
            '  S2_congestion_charge     Congestion surcharge on cars\n'
            '  S3_low_income_subsidy    50%% bundle discount for low income\n'
            '  S4_spatial_differentiation  District-level pricing\n'
        ),
    )
    parser.add_argument(
        '--scenario', default='S0_baseline',
        choices=list(SCENARIOS.keys()),
        help='Scenario to run (default: S0_baseline)',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test: single ABM run with default theta',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    parser.add_argument(
        '--output', default='results',
        help='Output directory for results and figures (default: results)',
    )
    parser.add_argument(
        '--all-scenarios', action='store_true',
        help='Run all 5 scenarios sequentially',
    )
    parser.add_argument(
        '--method', default='trmobo', choices=['trmobo'],
        help='Optimization method (default: trmobo)',
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    print(f"MaaS ABM Optimization Framework")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {os.path.abspath(args.output)}")
    print()

    # ----------------------------------------------------------------
    # Step 1: Load data and create agent population
    # ----------------------------------------------------------------
    print("Step 1: Loading agent population...")
    t0 = time.time()
    agents = data_loader.create_agent_population()
    N = len(agents['sex'])
    print(f"  {N} agents loaded (each representing ~{AGENT_WEIGHT:.0f} people)")
    print(f"  Loading took {time.time() - t0:.2f}s")
    print()

    # ----------------------------------------------------------------
    # Step 2: Initialize choice models
    # ----------------------------------------------------------------
    print("Step 2: Initializing choice models...")
    print(f"  Ch3 bundle-choice: ICLV with {len(THETA_NAMES)}-dim theta")
    print(f"  Ch1 trial:         LC-HCM (2 latent classes)")
    print(f"  Ch2 subscription:  Two-stage logistic")
    print()

    # ----------------------------------------------------------------
    # Step 3: Create ABM engine
    # ----------------------------------------------------------------
    print("Step 3: Creating ABM simulation engine...")
    abm = ABMSimulation(agents, ch3_model, ch1_model, ch2_model)
    print(f"  {N_WEEKS}-week simulation ({N_WEEKS // 52} years)")
    print()

    # ----------------------------------------------------------------
    # Step 4: Quick test or full optimisation
    # ----------------------------------------------------------------
    if args.quick:
        _run_quick_test(abm, agents, args)
    else:
        _run_optimisation(abm, agents, args)

    print("\nDone!")


# ==================================================================
# Quick test mode
# ==================================================================

def _run_quick_test(abm, agents, args):
    """Single ABM run with default theta for sanity checking."""
    print("Step 4: Quick test with default theta...")
    print(f"  theta = {THETA_DEFAULT}")

    t0 = time.time()
    objectives, weekly_subs = abm.run(THETA_DEFAULT, seed=args.seed)
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.2f}s")
    print(f"  Objectives (minimisation form):")
    print(f"    Adoption rate:    {-objectives[0]:.4f}")
    print(f"    Net revenue:      {-objectives[1]:.0f} yuan/month")
    print(f"    Gini coefficient: {objectives[2]:.4f}")
    print(f"    Carbon reduction: {-objectives[3]:.1f} kg/month")

    # S-curve plot
    try:
        import visualization as viz
        scurve_path = os.path.join(args.output, 'quick_scurve.png')
        viz.plot_s_curve(
            weekly_subs,
            scenario_name='Quick Test',
            save_path=scurve_path,
        )
    except Exception as e:
        print(f"  Warning: Could not generate S-curve plot: {e}")

    # Save raw results
    results_path = os.path.join(args.output, 'quick_results.npz')
    np.savez(
        results_path,
        objectives=objectives,
        weekly_subscribers=weekly_subs,
        theta=THETA_DEFAULT,
    )
    print(f"  Results saved to {results_path}")


# ==================================================================
# Full optimisation mode
# ==================================================================

def _run_optimisation(abm, agents, args):
    """Run TR-MOBO optimisation for one or all scenarios."""
    import visualization as viz
    run_optimization = _import_optimizer()

    # Wrap ABMSimulation.run() into the callable signature expected by optimizer:
    #   abm_func(theta, agents) -> ndarray[4]
    #   abm_func(theta, agents, seed=int) -> ndarray[4]
    def abm_func(theta, agents, seed=None):
        kwargs = {'seed': seed} if seed is not None else {}
        objectives, _ = abm.run(theta, **kwargs)
        return objectives

    scenarios_to_run = (
        list(SCENARIOS.keys()) if args.all_scenarios else [args.scenario]
    )

    all_results = {}

    for scenario_name in scenarios_to_run:
        print(f"\n{'=' * 60}")
        print(f"Running scenario: {scenario_name}")
        print(f"{'=' * 60}")

        scenario_label, modifier = get_scenario(scenario_name)
        print(f"  Label: {scenario_label}")
        print(f"  Method: TR-MOBO (ANP + EHVI)")
        print(f"  Init samples: {TRMOBO_INIT_SAMPLES}")
        print(f"  Max iterations: {TRMOBO_MAX_ITERATIONS}")

        t0 = time.time()
        result_dict, pareto_X, pareto_F, history = run_optimization(
            agents, abm_func, ch3_model, ch1_model, ch2_model,
            scenario_modifier=modifier,
        )
        elapsed = time.time() - t0

        n_solutions = pareto_F.shape[0] if pareto_F.ndim == 2 else 1
        n_iterations = result_dict.get('n_iterations', 0)
        converged = result_dict.get('converged', False)
        print(f"  Optimisation completed in {elapsed:.1f}s")
        print(f"  Pareto solutions found: {n_solutions}")
        print(f"  Iterations: {n_iterations}, Converged: {converged}")

        all_results[scenario_name] = (pareto_X, pareto_F, history)

        # Save numerical results
        results_path = os.path.join(
            args.output, f'{scenario_name}_results.npz'
        )
        np.savez(
            results_path,
            pareto_X=pareto_X,
            pareto_F=pareto_F,
        )
        print(f"  Results saved to {results_path}")

        # Pareto front plot
        pareto_path = os.path.join(
            args.output, f'{scenario_name}_pareto.png'
        )
        viz.plot_pareto_front(
            pareto_F, scenario_label,
            save_path=pareto_path,
        )

        # Convergence plot (hypervolume history)
        if history is not None:
            conv_path = os.path.join(
                args.output, f'{scenario_name}_convergence.png'
            )
            viz.plot_convergence(
                history,
                save_path=conv_path,
            )

            # TR convergence plot (new visualization)
            if 'iteration_history' in history:
                try:
                    tr_conv_path = os.path.join(
                        args.output, f'{scenario_name}_tr_convergence.png'
                    )
                    viz.plot_tr_convergence(
                        history['iteration_history'],
                        save_path=tr_conv_path,
                    )
                except Exception as e:
                    print(f"  Warning: TR convergence plot failed: {e}")

        # Print summary of extreme solutions
        _print_pareto_summary(pareto_F, scenario_label)

    # ------------------------------------------------------------------
    # Cross-scenario comparison (only if more than one scenario ran)
    # ------------------------------------------------------------------
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("Generating cross-scenario comparison...")
        print(f"{'=' * 60}")

        all_F = {k: v[1] for k, v in all_results.items()}
        comparison_path = os.path.join(args.output, 'scenario_comparison.png')
        viz.plot_scenario_comparison(
            all_F, list(all_F.keys()),
            save_path=comparison_path,
        )


def _print_pareto_summary(pareto_F, scenario_label):
    """Print a brief summary of the Pareto front extremes."""
    if pareto_F.ndim < 2 or pareto_F.shape[0] == 0:
        return

    obj_names = ['Adoption', 'Revenue', 'Gini', 'Carbon']
    print(f"\n  Pareto summary for {scenario_label}:")
    print(f"  {'Objective':<12s} {'Best':>12s} {'Worst':>12s} {'Median':>12s}")
    print(f"  {'-' * 50}")
    for i, name in enumerate(obj_names):
        col = pareto_F[:, i]
        # For negated objectives (0,1,3) "best" is most negative
        if i in (0, 1, 3):
            best = -col.min()
            worst = -col.max()
            median = -np.median(col)
        else:
            best = col.min()
            worst = col.max()
            median = np.median(col)
        print(f"  {name:<12s} {best:>12.4f} {worst:>12.4f} {median:>12.4f}")


if __name__ == '__main__':
    main()
