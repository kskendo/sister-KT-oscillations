"""
Sister Kinetochore Oscillations - Cross-Correlation Analysis
=============================================================
Outputs:
  1. sister_KT_oscillations.pdf / .png  - publication figure (Prism-style)
  2. sister_KT_amplitude_prism.csv      - amplitude data ready for Prism
  3. sister_KT_period_prism.csv         - period data ready for Prism

FILE NAMING: <condition>_<rep>_<anything>.fig
USAGE:  py sister_KT_oscillations.py
"""

import os, re
import h5py
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')   # remove if you want an interactive window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

# =============================================================================
#  USER SETTINGS
# =============================================================================

DATA_DIR   = r"C:\Users\skendo\Desktop\sister-kt-oscillations"

CONDITIONS = ["siCTRL", "siHAUS6", "siHURP", "siHAUS6siHURP"]

LABELS = {
    "siCTRL":        "siCTRL",
    "siHAUS6":       "siHAUS6",
    "siHURP":        "siHURP",
    "siHAUS6siHURP": "siHAUS6+siHURP",
}

X_MIN, X_MAX = 0, 100    # x-axis display range (seconds)

OUTPUT_PDF      = "sister_KT_oscillations.pdf"
OUTPUT_PNG      = "sister_KT_oscillations.png"
OUTPUT_AMP_CSV  = "sister_KT_amplitude_prism.csv"
OUTPUT_PER_CSV  = "sister_KT_period_prism.csv"

# =============================================================================
#  DATA LOADING
# =============================================================================

common_x = np.linspace(-140, 140, 41)
mask     = (common_x >= X_MIN) & (common_x <= X_MAX)
xp       = common_x[mask]


def extract_curves(filepath):
    """Return all 41-point cross-correlation curves from one .fig file."""
    curves = []
    with h5py.File(filepath, 'r') as f:
        refs = f.get('#refs#')
        if refs is None:
            return curves
        for key in refs:
            grp = refs[key]
            if not isinstance(grp, h5py.Group):
                continue
            if 'XData' not in grp or 'YData' not in grp:
                continue
            x = grp['XData'][:].flatten()
            y = grp['YData'][:].flatten()
            if len(y) == 41 and not np.all(np.isnan(y)):
                curves.append((x, y))
    return curves


def get_all_pairs(condition, data_dir):
    """
    Extract every individual KT-pair curve for a condition.
    Returns (amplitudes, periods) as 1-D arrays.
    Also returns per-replicate means for plotting.
    """
    pattern = re.compile(
        rf'^{re.escape(condition)}_\d+_.*\.fig$', re.IGNORECASE
    )
    files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if pattern.match(f)
    ])
    if not files:
        raise FileNotFoundError(
            f"No files found for '{condition}' in {data_dir}.\n"
            f"Expected: {condition}_1_<anything>.fig"
        )

    all_amps    = []
    all_periods = []
    rep_means   = []   # for the cross-correlation plot

    for fp in files:
        rep_curves = []
        for x0, y0 in extract_curves(fp):
            yi = interp1d(x0, y0, bounds_error=False,
                          fill_value=np.nan)(common_x)
            yp = yi[mask]
            if np.any(np.isnan(yp)):
                continue

            rep_curves.append(yp)

            # --- amplitude ---
            idx0  = np.argmin(np.abs(xp))
            tmask = (xp >= 20) & (xp <= 60)
            amp   = yp[idx0] - np.nanmin(yp[tmask])
            all_amps.append(amp)

            # --- period (damped cosine) ---
            try:
                def dc(t, A, tau, T, phi, c):
                    return A * np.exp(-t / tau) * np.cos(2*np.pi*t/T + phi) + c
                popt, _ = curve_fit(
                    dc, xp, yp,
                    p0     = [0.3, 50, 30, 0, 0.1],
                    bounds = ([0, 5, 5, -np.pi, -1],
                              [1, 500, 200,  np.pi,  1]),
                    maxfev = 5000
                )
                all_periods.append(abs(popt[2]))
            except Exception:
                pass

        if rep_curves:
            rep_means.append(np.nanmean(rep_curves, axis=0))

    print(f"  {condition}: {len(files)} replicates, "
          f"{len(all_amps)} KT pairs")

    return (np.array(all_amps),
            np.array(all_periods),
            np.array(rep_means))   # shape (n_reps, n_timepoints_displayed)


# =============================================================================
#  STATISTICS  (individual KT pairs, Kruskal-Wallis + Dunn-Bonferroni)
# =============================================================================

def p_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def run_statistics(metric_dict, metric_name):
    """
    Shapiro-Wilk on first 50 of each group to choose test.
    Returns dict: condition -> adjusted p-value vs siCTRL
    """
    groups = [metric_dict[c] for c in CONDITIONS]

    # normality check (Shapiro limited to n < 5000)
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
        ctrl      = groups[0]
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
    """
    Saves data in Prism-friendly format:
    One column per condition, one row per KT pair (unequal lengths OK in Prism).
    """
    cols   = [metric_dict[c] for c in CONDITIONS]
    n_rows = max(len(c) for c in cols)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # header
        writer.writerow([LABELS[c] for c in CONDITIONS])
        # data rows
        for i in range(n_rows):
            row = []
            for col in cols:
                row.append(col[i] if i < len(col) else '')
            writer.writerow(row)

    print(f"  Saved: {filepath}")


# =============================================================================
#  FIGURE  (Prism-style: box + scatter, black outlines, grey boxes)
# =============================================================================

def mean_ci95(arr):
    m  = np.nanmean(arr, axis=0)
    se = stats.sem(arr, axis=0, nan_policy='omit')
    t  = stats.t.ppf(0.975, df=max(arr.shape[0] - 1, 1))
    return m, t * se


def draw_boxplot(ax, position, values, color='#C0C0C0'):
    """Draw a Prism-style box (IQR box, median line, whiskers, no fliers)."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr          = q3 - q1
    lo_whisk     = max(values.min(), q1 - 1.5 * iqr)
    hi_whisk     = min(values.max(), q3 + 1.5 * iqr)
    bw           = 0.35   # box half-width

    # box
    rect = plt.Rectangle((position - bw, q1), 2*bw, iqr,
                          facecolor=color, edgecolor='black',
                          linewidth=1.2, zorder=2)
    ax.add_patch(rect)
    # median
    ax.plot([position - bw, position + bw], [med, med],
            color='black', lw=1.5, zorder=3)
    # whiskers
    ax.plot([position, position], [q3, hi_whisk],
            color='black', lw=1.0, zorder=2)
    ax.plot([position, position], [q1, lo_whisk],
            color='black', lw=1.0, zorder=2)
    ax.plot([position - bw*0.4, position + bw*0.4], [hi_whisk, hi_whisk],
            color='black', lw=1.0, zorder=2)
    ax.plot([position - bw*0.4, position + bw*0.4], [lo_whisk, lo_whisk],
            color='black', lw=1.0, zorder=2)


def add_brackets(ax, x1, x2, y, label, gap=0.02):
    """Draw a significance bracket between positions x1 and x2."""
    ax.plot([x1, x1, x2, x2], [y, y + gap, y + gap, y],
            color='black', lw=1.0)
    ax.text((x1 + x2) / 2, y + gap + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            label, ha='center', va='bottom', fontsize=9)


def make_figure(rep_means_dict, amps, pers, ph_amp, ph_per):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Sister-Kinetochore Oscillations",
                 fontsize=14, fontweight='bold', y=1.01)

    # ── Panel A: cross-correlation (replicate means ± 95 CI) ─────────────────
    ax = axes[0]
    COLORS = {
        "siCTRL":        "#808080",
        "siHAUS6":       "#FFB3C1",
        "siHURP":        "#4CAF50",
        "siHAUS6siHURP": "#4A0072",
    }
    for c in CONDITIONS:
        rm = rep_means_dict[c]   # (n_reps, n_timepoints)
        if rm.shape[0] == 0:
            continue
        m, ci = mean_ci95(rm)
        col = COLORS[c]
        ax.fill_between(xp, m - ci, m + ci, alpha=0.18, color=col)
        ax.plot(xp, m, color=col, lw=2, label=LABELS[c])

    ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xlabel("Time lag (s)", fontsize=9)
    ax.set_ylabel("Autocorrelation of KT-sister motion", fontsize=9)
    ax.set_title("A  Cross-correlation\n(replicate means ± 95% CI)",
                 fontsize=10, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False)

    # ── Panel B: amplitude (Prism-style box + scatter) ────────────────────────
    ax = axes[1]
    positions = list(range(len(CONDITIONS)))
    all_vals  = np.concatenate([amps[c] for c in CONDITIONS])
    y_max     = np.nanmax(all_vals)
    y_min     = np.nanmin(all_vals)
    pad       = (y_max - y_min) * 0.05

    for i, c in enumerate(CONDITIONS):
        vals = amps[c]
        draw_boxplot(ax, i, vals, color='#C0C0C0')
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.25
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color='#404040', s=20, zorder=4,
                   edgecolors='black', linewidths=0.3, alpha=0.7)

    # significance brackets
    bracket_base = y_max + pad
    step         = (y_max - y_min) * 0.08
    for i, cond in enumerate(CONDITIONS[1:]):
        p  = ph_amp[cond]
        yb = bracket_base + i * step
        add_brackets(ax, 0, i + 1, yb, p_stars(p),
                     gap=step * 0.15)

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS],
                       fontsize=8, rotation=20, ha='right')
    ax.set_ylabel("Oscillation amplitude\n(peak − trough correlation)", fontsize=9)
    ax.set_title("B  Amplitude", fontsize=10, loc='left')
    ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    ax.set_ylim(y_min - pad,
                bracket_base + len(CONDITIONS) * step + step)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Panel C: period (Prism-style box + scatter) ───────────────────────────
    ax = axes[2]
    all_per   = np.concatenate([pers[c] for c in CONDITIONS
                                 if len(pers[c]) > 0])
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
        add_brackets(ax, 0, i + 1, yb, p_stars(p),
                     gap=step_per * 0.15)

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS],
                       fontsize=8, rotation=20, ha='right')
    ax.set_ylabel("Oscillation period (s)", fontsize=9)
    ax.set_title("C  Period (damped cosine fit)", fontsize=10, loc='left')
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

    print("Loading data and extracting KT pairs...")
    amps          = {}
    pers          = {}
    rep_means_dict = {}

    for c in CONDITIONS:
        a, p, rm        = get_all_pairs(c, DATA_DIR)
        amps[c]         = a
        pers[c]         = p
        rep_means_dict[c] = rm

    print("\n--- Statistics (individual KT pairs) ---")
    ph_amp = run_statistics(amps, "Amplitude")
    ph_per = run_statistics(pers, "Period")

    print("\nSaving Prism CSV files...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_prism_csv(amps, os.path.join(script_dir, OUTPUT_AMP_CSV))
    save_prism_csv(pers, os.path.join(script_dir, OUTPUT_PER_CSV))

    print("\nGenerating figure...")
    make_figure(rep_means_dict, amps, pers, ph_amp, ph_per)

    print("\nAll done!")
