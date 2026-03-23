"""
Visualization module for the MaaS ABM optimization framework.

Produces publication-quality figures for:
    - 4D Pareto front scatter plots
    - Adoption S-curves over 3 years
    - TAZ-level adoption heatmaps (geopandas)
    - Multi-scenario Pareto front comparison
    - Tornado sensitivity diagrams
    - NSGA-II convergence (hypervolume) plots

All functions accept an optional ``save_path``; when provided the figure
is saved to disk (PNG 300 dpi) instead of being shown interactively.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from config import N_WEEKS

# ============================================================
# Global style defaults
# ============================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colormap for multi-scenario comparison
_SCENARIO_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f']

# Objective labels (minimisation form as stored by NSGA-II)
_OBJ_LABELS = [
    'Neg. Adoption Rate',
    'Neg. Net Revenue (yuan)',
    'Gini Coefficient',
    'Neg. Carbon Reduction (kg)',
]
_OBJ_LABELS_DISPLAY = [
    'Adoption Rate',
    'Net Revenue (yuan)',
    'Equity (Gini, lower=better)',
    'Carbon Reduction (kg)',
]


# ============================================================
# 1. Pareto front (4-objective)
# ============================================================

def plot_pareto_front(pareto_F, scenario_name='S0', save_path=None):
    """4D Pareto front visualised as 2D scatter with colour and size.

    Parameters
    ----------
    pareto_F : ndarray, shape (n_solutions, 4)
        Objective values in minimisation form (negated where maximised).
    scenario_name : str
        Title annotation.
    save_path : str or None
        If provided, save figure to this path.
    """
    if pareto_F.shape[1] < 4:
        raise ValueError(
            f"Expected 4 objectives, got {pareto_F.shape[1]} columns"
        )

    F = np.array(pareto_F, dtype=np.float64)
    n = F.shape[0]

    # Convert back to natural scale for display
    adoption = -F[:, 0]
    revenue = -F[:, 1]
    gini = F[:, 2]
    carbon = -F[:, 3]

    # Normalise 3rd and 4th objectives to [0,1] for size/colour
    gini_norm = _normalise(gini)
    carbon_norm = _normalise(carbon)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        adoption, revenue,
        c=carbon_norm,
        s=30 + 200 * (1.0 - gini_norm),  # larger = more equitable
        cmap='viridis',
        alpha=0.75,
        edgecolors='k',
        linewidths=0.4,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Carbon Reduction (normalised)')

    ax.set_xlabel('Adoption Rate')
    ax.set_ylabel('Net Revenue (yuan/month)')
    ax.set_title(f'Pareto Front  [{scenario_name}]  ({n} solutions)\n'
                 f'Size ~ Equity (larger = lower Gini)')
    ax.grid(True, linestyle='--', alpha=0.4)

    _save_or_show(fig, save_path)


# ============================================================
# 2. Adoption S-curve
# ============================================================

def plot_s_curve(weekly_subscribers, scenario_name='S0', save_path=None):
    """Plot the 3-year adoption S-curve (weekly subscriber count).

    Parameters
    ----------
    weekly_subscribers : array-like, shape (n_weeks,)
        Number of subscribers at the end of each week.
    scenario_name : str
    save_path : str or None
    """
    weeks = np.arange(1, len(weekly_subscribers) + 1)
    months = weeks / 4.33

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(weeks, weekly_subscribers, color='#1f77b4', linewidth=2.0)
    ax1.fill_between(weeks, 0, weekly_subscribers, alpha=0.15, color='#1f77b4')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Subscribers', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')

    # Secondary x-axis: months
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim()[0] / 4.33, ax1.get_xlim()[1] / 4.33)
    ax2.set_xlabel('Month')

    # Mark year boundaries
    for yr in range(1, 4):
        wk = yr * 52
        if wk <= len(weekly_subscribers):
            ax1.axvline(wk, color='gray', linestyle=':', alpha=0.5)
            ax1.text(wk, ax1.get_ylim()[1] * 0.95, f'Year {yr}',
                     ha='center', fontsize=9, color='gray')

    ax1.set_title(f'MaaS Adoption S-Curve  [{scenario_name}]')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

    _save_or_show(fig, save_path)


# ============================================================
# 3. TAZ-level adoption heatmap
# ============================================================

def plot_taz_heatmap(agents, status, taz_gdf, save_path=None):
    """TAZ-level adoption rate heatmap using geopandas.

    Parameters
    ----------
    agents : dict[str, ndarray[N]]
        Must contain 'taz_code' (integer TAZ identifier per agent).
    status : ndarray[N]
        Agent status codes (STATUS_SUBSCRIBER == 3).
    taz_gdf : geopandas.GeoDataFrame
        Must contain a 'TAZ' or 'taz_code' column and geometry.
    save_path : str or None
    """
    try:
        import geopandas as gpd
    except ImportError:
        print("[visualization] geopandas not installed; skipping TAZ heatmap.")
        return

    from config import STATUS_SUBSCRIBER

    taz_codes = np.asarray(agents.get('taz_code', agents.get('TAZ', [])))
    is_subscriber = (status == STATUS_SUBSCRIBER).astype(np.float64)

    if len(taz_codes) == 0:
        print("[visualization] No TAZ codes found in agents; skipping heatmap.")
        return

    # Compute per-TAZ adoption rate
    unique_taz = np.unique(taz_codes)
    taz_adoption = {}
    for tz in unique_taz:
        mask = taz_codes == tz
        if mask.sum() > 0:
            taz_adoption[tz] = is_subscriber[mask].mean()

    # Identify the TAZ column in the GeoDataFrame
    taz_col = 'TAZ' if 'TAZ' in taz_gdf.columns else 'taz_code'
    gdf = taz_gdf.copy()
    gdf['adoption_rate'] = gdf[taz_col].map(taz_adoption).fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(
        column='adoption_rate',
        cmap='YlOrRd',
        linewidth=0.3,
        edgecolor='gray',
        legend=True,
        legend_kwds={'label': 'Adoption Rate', 'shrink': 0.6},
        ax=ax,
    )
    ax.set_title('TAZ-Level MaaS Adoption Rate')
    ax.set_axis_off()

    _save_or_show(fig, save_path)


# ============================================================
# 4. Scenario comparison
# ============================================================

def plot_scenario_comparison(all_pareto_F, scenario_names, save_path=None):
    """Overlay Pareto fronts from multiple scenarios.

    Parameters
    ----------
    all_pareto_F : dict[str, ndarray(n, 4)]
        Mapping from scenario name to Pareto objective array.
    scenario_names : list[str]
        Order in which scenarios appear in legend.
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Three 2D projections: (adoption, revenue), (adoption, gini), (adoption, carbon)
    proj_pairs = [(0, 1), (0, 2), (0, 3)]
    y_labels = [
        _OBJ_LABELS_DISPLAY[1],
        _OBJ_LABELS_DISPLAY[2],
        _OBJ_LABELS_DISPLAY[3],
    ]
    y_negate = [True, False, True]  # whether to negate for display

    for idx, (xi, yi) in enumerate(proj_pairs):
        ax = axes[idx]
        for si, sname in enumerate(scenario_names):
            if sname not in all_pareto_F:
                continue
            F = np.array(all_pareto_F[sname], dtype=np.float64)
            x_vals = -F[:, xi]  # adoption (negated -> positive)
            if y_negate[idx]:
                y_vals = -F[:, yi]
            else:
                y_vals = F[:, yi]
            color = _SCENARIO_COLORS[si % len(_SCENARIO_COLORS)]
            ax.scatter(x_vals, y_vals, s=20, alpha=0.6, color=color,
                       label=sname, edgecolors='none')
        ax.set_xlabel(_OBJ_LABELS_DISPLAY[0])
        ax.set_ylabel(y_labels[idx])
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle('Scenario Comparison: Pareto Front Projections', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    _save_or_show(fig, save_path)


# ============================================================
# 5. Tornado sensitivity diagram
# ============================================================

def plot_sensitivity_tornado(base_F, sensitivities, param_names,
                             save_path=None):
    """Tornado diagram for parameter sensitivity analysis.

    Parameters
    ----------
    base_F : ndarray, shape (4,)
        Objective values at the base-case theta.
    sensitivities : dict[str, tuple(ndarray, ndarray)]
        Mapping from parameter name to (F_low, F_high), each shape (4,).
    param_names : list[str]
        Parameter names in display order (top to bottom).
    save_path : str or None
    """
    n_params = len(param_names)
    fig, axes = plt.subplots(1, 4, figsize=(20, max(4, 0.45 * n_params)))

    for obj_idx in range(4):
        ax = axes[obj_idx]
        base_val = base_F[obj_idx]

        deltas_low = []
        deltas_high = []
        labels = []
        for pname in param_names:
            if pname not in sensitivities:
                continue
            f_low, f_high = sensitivities[pname]
            deltas_low.append(f_low[obj_idx] - base_val)
            deltas_high.append(f_high[obj_idx] - base_val)
            labels.append(pname)

        n = len(labels)
        if n == 0:
            ax.set_title(_OBJ_LABELS[obj_idx])
            continue

        deltas_low = np.array(deltas_low)
        deltas_high = np.array(deltas_high)

        # Sort by total range (largest at top)
        total_range = np.abs(deltas_high - deltas_low)
        order = np.argsort(total_range)  # ascending
        y_pos = np.arange(n)

        ax.barh(y_pos, deltas_high[order], align='center', height=0.6,
                color='#d62728', alpha=0.7, label='High')
        ax.barh(y_pos, deltas_low[order], align='center', height=0.6,
                color='#1f77b4', alpha=0.7, label='Low')
        ax.axvline(0, color='k', linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([labels[i] for i in order], fontsize=8)
        ax.set_title(_OBJ_LABELS[obj_idx], fontsize=10)
        if obj_idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ============================================================
# 6. Convergence plot (hypervolume)
# ============================================================

def plot_convergence(history, save_path=None):
    """NSGA-II convergence: hypervolume over generations.

    Parameters
    ----------
    history : dict or list
        If dict, expects key 'hypervolume' -> list/array of HV per generation.
        If list/array, treated directly as HV values.
    save_path : str or None
    """
    if isinstance(history, dict):
        hv = np.asarray(history.get('hypervolume', history.get('hv', [])))
    else:
        hv = np.asarray(history)

    if len(hv) == 0:
        print("[visualization] No hypervolume data; skipping convergence plot.")
        return

    gens = np.arange(1, len(hv) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, hv, color='#2ca02c', linewidth=2.0)
    ax.fill_between(gens, hv.min(), hv, alpha=0.12, color='#2ca02c')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Hypervolume')
    ax.set_title('NSGA-II Convergence')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Annotate final HV
    ax.annotate(f'HV = {hv[-1]:.4f}',
                xy=(gens[-1], hv[-1]),
                xytext=(-60, 15),
                textcoords='offset points',
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))

    _save_or_show(fig, save_path)


# ============================================================
# Internal helpers
# ============================================================

def _normalise(x):
    """Min-max normalise to [0, 1]; returns zeros if range is zero."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _save_or_show(fig, save_path):
    """Save figure to *save_path* or plt.show() if None."""
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  [viz] Saved: {save_path}")
    else:
        plt.show()
