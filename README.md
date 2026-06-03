[sister_KT_README.md](https://github.com/user-attachments/files/28541456/sister_KT_README.md)
# sister-KT-oscillations

**Repository:** https://github.com/kskendo/sister-KT-oscillations

Python pipeline for quantification of sister kinetochore oscillation regularity and periodicity from KiT/maki `.mat` files.

Developed for:
> Skendo et al. (2026) *Title to be confirmed upon publication.* (in preparation)

---

## Overview

Sister kinetochore oscillations are analyzed by auto-correlation of kinetochore spindle-axis displacement over time. The auto-correlation is computed by KiT (`makiSepDispSpaceTimeAnalysis`) and stored in the `.mat` output files. This pipeline reads those files directly and produces a publication-ready figure with three panels:

| Panel | Contents |
|-------|----------|
| A | Auto-correlation curves (replicate means ± 95% CI) |
| B | Oscillation regularity (depth of first trough of auto-correlation) |
| C | Oscillation period (damped cosine fit to auto-correlation curve) |

**Oscillation regularity** (Panel B) is defined as the peak minus the first trough of the auto-correlation curve. A deeper trough indicates more regular, periodic oscillations. This is not a measure of oscillation amplitude in µm.

**Oscillation period** (Panel C) is estimated by fitting a damped cosine function to each cell's auto-correlation curve.

---

## Requirements

- Python 3.8 or newer
- numpy, scipy, matplotlib, h5py

Install dependencies:
```
python -m pip install numpy scipy matplotlib h5py
```

---

## Usage

### Step 1 — Prepare input files

Run the KiT pipeline in MATLAB on your kinetochore tracking data:

```matlab
analysisStruct = makiCollectMovies('MERALDI');
analysisStruct = makiSisterMotionCoupling('MERALDI', analysisStruct, 1, 1, 1);
analysisStruct = makiSepDispSpaceTimeAnalysis('MERALDI', analysisStruct, 1, 1, 1);
analysisStruct = makiSisterConnectionAnalysis('MERALDI', analysisStruct, 1, 1, 1);
```

Rename each output `.mat` file using this pattern:

```
<condition>_<replicate>_makiAnalysis__1.mat
```

Example:
```
siCTRL_1_makiAnalysis__1.mat
siCTRL_2_makiAnalysis__1.mat
siHAUS6_1_makiAnalysis__1.mat
siHURP_1_makiAnalysis__1.mat
siHAUS6siHURP_1_makiAnalysis__1.mat
```

Place all files in one folder.

### Step 2 — Configure the script

Open `sister_KT_oscillations.py` and set `DATA_DIR` to your folder:

```python
DATA_DIR = r"C:\Users\skendo\Desktop\maki_oscillations"
```

### Step 3 — Run

```
cd C:\Users\skendo\Desktop\maki_oscillations
python sister_KT_oscillations.py
```

---

## Output files

| File | Contents |
|------|----------|
| `sister_KT_oscillations.pdf` | Publication figure (panels A, B, C) |
| `sister_KT_oscillations.png` | Same figure as PNG (300 dpi) |
| `sister_KT_regularity_prism.csv` | Oscillation regularity per cell, formatted for GraphPad Prism |
| `sister_KT_period_prism.csv` | Oscillation period per cell, formatted for GraphPad Prism |

---

## Notes on cell inclusion

Only kinetochore pairs classified as inliers by KiT (complete tracks meeting quality thresholds) are included in the analysis. Not all tracked cells will produce valid auto-correlation curves — cells with incomplete tracks are stored as NaN by KiT and are automatically excluded.

---

## Statistical analysis

Oscillation regularity and period are compared across conditions using Kruskal-Wallis test with Dunn-Bonferroni post-hoc correction (non-parametric), or one-way ANOVA with Tukey HSD if data are normally distributed (assessed by Shapiro-Wilk test). All pairwise comparisons are made against siCTRL.

---

## Citation

If you use this pipeline, please cite:

> Skendo K et al. (2026) Title to be confirmed. DOI: to be added upon publication.

and the KiT software:

> Armond JW et al. (2016) KiT: a MATLAB package for kinetochore tracking. *Bioinformatics* 32, 1917–1919.

---

## License

MIT License — free to use and modify with attribution. See `LICENSE` for details.

---

## Contact

Kristjana Skendo
Meraldi Lab, Department of Cell Physiology and Metabolism
University of Geneva
patrick.meraldi@unige.ch
