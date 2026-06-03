"""
Sister Kinetochore Oscillations - Auto-Correlation Analysis
============================================================
Reads auto-correlation of kinetochore spindle-axis projections directly
from KiT .mat files (sisterMotionCoupling/Inlier/autocorr/indcell/projectionsSis1).

Auto-correlation starts at 1.0 at lag=0 by definition.
The depth of the first trough reflects oscillation REGULARITY (not amplitude):
a deeper trough = more regular/periodic oscillations.
Period is estimated by damped cosine fit to the auto-correlation curve.

Outputs:
  1. sister_KT_oscillations.pdf / .png  - publication figure
  2. sister_KT_regularity_prism.csv     - regularity data ready for Prism
  3. sister_KT_period_prism.csv         - period data ready for Prism

FILE NAMING: <condition>_<rep>_makiAnalysis__1.mat
USAGE:  python sister_KT_oscillations.py
"""

import os, re
import h5py
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

# =============================================================================
#  USER SETTINGS
# =============================================================================

DATA_DIR  = r"C:\Users\skendo\Desktop\maki_oscillations"

CONDITIONS = ["siCTRL", "siHAUS6", "siHURP", "siHAUS6siHURP"]

LABELS = {
    "siCTRL":        "siCTRL",
    "siHAUS6":       "siHAUS6",
    "siHURP":        "siHURP",
    "siHAUS6siHURP": "siHAUS6+siHURP",
}

DT        = 7.5          # acquisition interval (seconds)
N_LAGS    = 21           # number of lag points stored by KiT (0 to 20*DT)
X_MIN     = 0            # display range start (seconds)
X_MAX     = 100          # display range end (seconds)

OUTPUT_PDF     = "sister_KT_oscillations.pdf"
OUTPUT_PNG     = "sister_KT_oscillations.png"
OUTPUT_REG_CSV = "sister_KT_regularity_prism.csv"
OUTPUT_PER_CSV = "sister_KT_period_prism.csv"

# =============================================================================
#  DATA LOADING
# =============================================================================

lags = np.arange(N_LAGS) * DT          # [0, 7.5, 15, ..., 150] seconds
mask = (lags >= X_MIN) & (lags <= X_MAX)
xp   = lags[mask]


def load_autocorr(filepath):
    """
    Load per-cell auto-correlation of spindle-axis kinetochore projections
    from one KiT .mat file.

    Returns list of 1-D arrays (one per valid cell), each of length N_LAGS.
    Each array starts at 1.0 (lag=0) and reflects the regularity of
    kinetochore oscillations: deeper first trough = more regular oscillations.
    """
    key = 'analysisStruct/sisterMotionCoupling/Inlier/autocorr/indcell/projectionsSis1'
    with h5py.File(filepath, 'r') as f:
        ac = f[key][:]          # shape (n_cells, 2, N_LAGS); row 0 = mean per cell
    valid = []
    for i in range(ac.shape[0]):
        row = ac[i, 0, :]
        if not np.isnan(row[0]):
            valid.append(row)
    return valid


def get_all_data(condition, data_dir):
    """
    Load all replicates for one condition.

    Returns:
      regularities  - 1-D array of per-cell regularity values
                      (peak - trough of auto-correlation curve in display window)
      periods       - 1-D array of per-cell oscillation periods (s) from damped cosine fit
      rep_means     - array (n_reps, n_displayed_lags) of replicate-mean curves
    """
    pattern = re.compile(
        rf'^{re.escape(condition)}_\d+_.*\.mat$', re.IGNORECASE
    )
    files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if pattern.match(f)
    ])
    if not files:
        raise FileNotFoundError(
            f"No .mat files found for '{condition}' in {data_dir}.\n"
            f"Expected pattern: {condition}_1_makiAnalysis__1.mat"
        )

    all_reg     = []
    all_periods = []
    rep_means   = []

    for fp in files:
        cells = load_autocorr(fp)
        rep_cell_curves = []

        for ac_full in cells:
            # crop to display window
            yp = ac_full[mask]

            # --- regularity: peak (lag=0, always 1.0) minus first trough ---
            # search for first trough between 15s and 60s
            trough_mask = (xp >= 15) & (xp <= 60)
            if trough_mask.any():
                trough_val = np.nanmin(yp[trough_mask])
            else:
                trough_val = np.nanmin(yp)
            regularity = (yp[0] - trough_val) / 2.0    # normalized [0,1]: 1.0 = perfectly periodic
            all_reg.append(regularity)

            # --- period: damped cosine fit to auto-correlation curve ---
            try:
                def damped_cosine(t, A, tau, T, phi, c):
                    return A * np.exp(-t / tau) * np.cos(2 * np.pi * t / T + phi) + c

                popt, _ = curve_fit(
                    damped_cosine, xp, yp,
                    p0     = [0.3, 50, 60, 0, 0.0],
                    bounds = ([0,  5,   5, -np.pi, -1],
                              [1, 500, 200,  np.pi,  1]),
                    maxfev = 5000
                )
                all_periods.append(abs(popt[2]))
            except Exception:
                pass

            rep_cell_curves.append(yp)

        if rep_cell_curves:
            rep_means.append(np.nanmean(rep_cell_curves, axis=0))

    print(f"  {condition}: {len(files)} replicates, "
          f"{len(all_reg)} cells, {len(all_periods)} period fits")

    return (np.array(all_reg),
            np.array(all_periods),
            np.array(rep_means))      # shape (n_reps, n_displayed_lags)


# =============================================================================
#  STATISTICS  (per-cell values, Kruskal-Wallis + Dunn-Bonferroni)
# =============================================================================

def p_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def run_statistics(metric_dict, metric_name):
    groups = [metric_dict[c] for c in CONDITIONS]

    all_normal = all(
        stats.shapiro(g[:50]).pvalue > 0.05
        for g in groups if len(g) >= 3
    )

    posthoc = {}

    if all_normal:
        F, p_omni = stats.f_oneway(*groups)
        method    = "One-way ANOVA + Tukey HSD"
        n_tot     = sum(len(g) for g in groups)
        k         = len(groups)
        ms_w      = sum(np.sum((g - g.mean())**2)
                        for g in groups) / (n_tot - k)
        ctrl = groups[0]
        for i, cond in enumerate(CONDITIONS[1:]):
            g2   = groups[i + 1]
            diff = ctrl.mean() - g2.mean()
            se   = np.sqrt(ms_w * (1/len(ctrl) + 1/len(g2)) / 2)
            q    = abs(diff) / se if se > 0 else 0
            try:
                p = stats.studentized_range.sf(q, k, n_tot - k)
            except Exception:
                p = 1.0
            posthoc[cond] = p
    else:
        H, p_omni = stats.kruskal(*groups)
        method    = "Kruskal-Wallis + Dunn (Bonferroni)"
        n_comp    = len(CONDITIONS) - 1
        all_data  = np.concatenate(groups)
        n_tot     = len(all_data)
        ranks     = stats.rankdata(all_data)
        rd = {}; idx = 0
        for c, g in zip(CONDITIONS, groups):
            rd[c] = ranks[idx: idx + len(g)]
            idx  += len(g)
        ctrl_r = rd[CONDITIONS[0]]
        for cond in CONDITIONS[1:]:
            r2  = rd[cond]
            se  = np.sqrt((n_tot * (n_tot + 1) / 12) *
                          (1/len(ctrl_r) + 1/len(r2)))
            z   = abs(ctrl_r.mean() - r2.mean()) / se if se > 0 else 0
            p   = min(2 * stats.norm.sf(z) * n_comp, 1.0)
            posthoc[cond] = p

    print(f"\n  [{metric_name}] {method}  |  omnibus p = {p_omni:.6f}")
    for cond, p in posthoc.items():
        print(f"    siCTRL vs {cond}: p = {p:.6f}  {p_stars(p)}")

    return posthoc


# =============================================================================
#  CSV EXPORT FOR PRISM
# =============================================================================

def save_prism_csv(metric_dict, filepath):
    cols   = [metric_dict[c] for c in CONDITIONS]
    n_rows = max(len(c) for c in cols)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([LABELS[c] for c in CONDITIONS])
        for i in range(n_rows):
            row = [col[i] if i < len(col) else '' for col in cols]
            writer.writerow(row)
    print(f"  Saved: {filepath}")


# =============================================================================
#  FIGURE
# =============================================================================

def mean_ci95(arr):
    m  = np.nanmean(arr, axis=0)
    se = stats.sem(arr, axis=0, nan_policy='omit')
    t  = stats.t.ppf(0.975, df=max(arr.shape[0] - 1, 1))
    return m, t * se


def draw_boxplot(ax, position, values, color='#C0C0C0'):
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr      = q3 - q1
    lo_whisk = max(values.min(), q1 - 1.5 * iqr)
    hi_whisk = min(values.max(), q3 + 1.5 * iqr)
    bw       = 0.35

    rect = plt.Rectangle((position - bw, q1), 2*bw, iqr,
                          facecolor=color, edgecolor='black',
                          linewidth=1.2, zorder=2)
    ax.add_patch(rect)
    ax.plot([position - bw, position + bw], [med, med],
            color='black', lw=1.5, zorder=3)
    ax.plot([position, position], [q3, hi_whisk],
            color='black', lw=1.0, zorder=2)
    ax.plot([position, position], [q1, lo_whisk],
            color='black', lw=1.0, zorder=2)
    ax.plot([position - bw*0.4, position + bw*0.4], [hi_whisk, hi_whisk],
            color='black', lw=1.0, zorder=2)
    ax.plot([position - bw*0.4, position + bw*0.4], [lo_whisk, lo_whisk],
            color='black', lw=1.0, zorder=2)


def add_brackets(ax, x1, x2, y, label, gap=0.02):
    ax.plot([x1, x1, x2, x2], [y, y + gap, y + gap, y],
            color='black', lw=1.0)
    ax.text((x1 + x2) / 2,
            y + gap + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            label, ha='center', va='bottom', fontsize=9)


def make_figure(rep_means_dict, regs, pers, ph_reg, ph_per):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Sister-Kinetochore Oscillations",
                 fontsize=14, fontweight='bold', y=1.01)

    COLORS = {
        "siCTRL":        "#808080",
        "siHAUS6":       "#FFB3C1",
        "siHURP":        "#4CAF50",
        "siHAUS6siHURP": "#4A0072",
    }

    # ── Panel A: auto-correlation (replicate means ± 95% CI) ─────────────────
    ax = axes[0]
    for c in CONDITIONS:
        rm = rep_means_dict[c]
        if rm.shape[0] == 0:
            continue
        m, ci = mean_ci95(rm)
        col = COLORS[c]
        ax.fill_between(xp, m - ci, m + ci, alpha=0.18, color=col)
        ax.plot(xp, m, color=col, lw=2, label=LABELS[c])

    ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-0.6, 1.05)
    ax.set_xlabel("Time lag (s)", fontsize=9)
    ax.set_ylabel("Auto-correlation of KT motion", fontsize=9)
    ax.set_title("A  Auto-correlation\n(replicate means ± 95% CI)",
                 fontsize=10, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False)

    # ── Panel B: oscillation regularity (box + scatter) ───────────────────────
    ax = axes[1]
    positions = list(range(len(CONDITIONS)))
    all_vals  = np.concatenate([regs[c] for c in CONDITIONS])
    y_max     = np.nanmax(all_vals)
    y_min     = np.nanmin(all_vals)
    pad       = (y_max - y_min) * 0.05

    for i, c in enumerate(CONDITIONS):
        vals = regs[c]
        draw_boxplot(ax, i, vals, color='#C0C0C0')
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.25
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color='#404040', s=20, zorder=4,
                   edgecolors='black', linewidths=0.3, alpha=0.7)

    bracket_base = y_max + pad
    step         = (y_max - y_min) * 0.08
    for i, cond in enumerate(CONDITIONS[1:]):
        p  = ph_reg[cond]
        yb = bracket_base + i * step
        add_brackets(ax, 0, i + 1, yb, p_stars(p), gap=step * 0.15)

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS],
                       fontsize=8, rotation=20, ha='right')
    ax.set_ylabel("Oscillation regularity\n(normalized, 0\u20131)",
                  fontsize=9)
    ax.set_title("B  Oscillation regularity", fontsize=10, loc='left')
    ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    ax.set_ylim(y_min - pad,
                bracket_base + len(CONDITIONS) * step + step)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Panel C: oscillation period (box + scatter) ───────────────────────────
    ax = axes[2]
    all_per   = np.concatenate([pers[c] for c in CONDITIONS if len(pers[c]) > 0])
    y_max_per = np.nanmax(all_per)
    y_min_per = np.nanmin(all_per)
    pad_per   = (y_max_per - y_min_per) * 0.05

    for i, c in enumerate(CONDITIONS):
        vals = pers[c]
        if len(vals) == 0:
            continue
        draw_boxplot(ax, i, vals, color='#C0C0C0')
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.25
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color='#404040', s=20, zorder=4,
                   edgecolors='black', linewidths=0.3, alpha=0.7)

    bracket_base_per = y_max_per + pad_per
    step_per         = (y_max_per - y_min_per) * 0.08
    for i, cond in enumerate(CONDITIONS[1:]):
        p  = ph_per[cond]
        yb = bracket_base_per + i * step_per
        add_brackets(ax, 0, i + 1, yb, p_stars(p), gap=step_per * 0.15)

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS],
                       fontsize=8, rotation=20, ha='right')
    ax.set_ylabel("Oscillation period (s)", fontsize=9)
    ax.set_title("C  Oscillation period (damped cosine fit)", fontsize=10, loc='left')
    ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    ax.set_ylim(y_min_per - pad_per,
                bracket_base_per + len(CONDITIONS) * step_per + step_per)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for out, fmt in [(OUTPUT_PDF, 'pdf'), (OUTPUT_PNG, 'png')]:
        path = os.path.join(script_dir, out)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")

    plt.show()


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":

    print("Loading auto-correlation data from .mat files...")
    regs           = {}
    pers           = {}
    rep_means_dict = {}

    for c in CONDITIONS:
        r, p, rm        = get_all_data(c, DATA_DIR)
        regs[c]         = r
        pers[c]         = p
        rep_means_dict[c] = rm

    print("\n--- Statistics (per-cell values) ---")
    ph_reg = run_statistics(regs, "Oscillation regularity")
    ph_per = run_statistics(pers, "Oscillation period")

    print("\nSaving Prism CSV files...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_prism_csv(regs, os.path.join(script_dir, OUTPUT_REG_CSV))
    save_prism_csv(pers, os.path.join(script_dir, OUTPUT_PER_CSV))

    print("\nGenerating figure...")
    make_figure(rep_means_dict, regs, pers, ph_reg, ph_per)

    print("\nAll done!")
