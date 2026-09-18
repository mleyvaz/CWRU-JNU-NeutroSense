"""
Neutrosophic Ensemble Classification for Uncertainty-Aware
Bearing Fault Detection in Industrial Drive Systems
Dataset: CWRU (Case Western Reserve University Bearing Data Center)
"""

import os
import sys
import urllib.request
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.io import loadmat
from scipy import stats
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(OUTPUT_DIR, "cwru_raw")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DOWNLOAD CWRU .mat FILES
# ---------------------------------------------------------------------------
# Format: (filename, label_int, label_str, fault_type)
# NOTE (fixed 2026-09-17, external review): the previous mapping used files
# 105-108 as "Ball" and 109-112 as "Inner", both WRONG per the official CWRU
# Bearing Data Center catalog (engineering.case.edu/bearingdatacenter):
#   105-108 = Inner Race 0.007", 12 kHz drive-end  (same fault as old "Inner")
#   109-112 = Inner Race 0.007", 48 kHz drive-end  (SAME fault, different
#             sampling rate -- not a distinct class at all)
#   118-121 = Ball 0.007", 12 kHz drive-end        (the real Ball fault,
#             never previously included)
# The old "Ball" and "Inner" classes were therefore the same physical fault
# (inner race defect) acquired at two different sampling rates (12kHz vs
# 48kHz), and true ball-fault data was never used. Corrected below to use
# 118-121 for Ball and 105-108 for Inner, both at the consistent 12 kHz
# drive-end rate; the 48kHz duplicate (109-112) is dropped entirely.
CWRU_FILES = [
    # Normal baseline – 4 load conditions
    ("97.mat",  0, "Normal", "normal"),
    ("98.mat",  0, "Normal", "normal"),
    ("99.mat",  0, "Normal", "normal"),
    ("100.mat", 0, "Normal", "normal"),
    # Ball fault 0.007", 12 kHz drive-end – 4 loads (corrected: was 105-108)
    ("118.mat", 1, "Ball",   "ball"),
    ("119.mat", 1, "Ball",   "ball"),
    ("120.mat", 1, "Ball",   "ball"),
    ("121.mat", 1, "Ball",   "ball"),
    # Inner race 0.007", 12 kHz drive-end – 4 loads (corrected: was 109-112)
    ("105.mat", 2, "Inner",  "inner"),
    ("106.mat", 2, "Inner",  "inner"),
    ("107.mat", 2, "Inner",  "inner"),
    ("108.mat", 2, "Inner",  "inner"),
    # Outer race 0.007" @6h, 12 kHz drive-end – 4 loads (unchanged, was already correct)
    ("130.mat", 3, "Outer",  "outer"),
    ("131.mat", 3, "Outer",  "outer"),
    ("132.mat", 3, "Outer",  "outer"),
    ("133.mat", 3, "Outer",  "outer"),
]
BASE_URL = "https://engineering.case.edu/sites/default/files/"

def download_files():
    print("=== Downloading CWRU files ===")
    for fname, *_ in CWRU_FILES:
        dest = os.path.join(DATA_DIR, fname)
        if os.path.exists(dest):
            print(f"  {fname} already exists, skipping")
            continue
        url = BASE_URL + fname
        print(f"  Downloading {url} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            print("OK")
        except Exception as e:
            print(f"FAILED ({e})")

# ---------------------------------------------------------------------------
# 2. FEATURE EXTRACTION FROM VIBRATION WINDOWS
# ---------------------------------------------------------------------------
WINDOW = 1024
STEP   = 512

def extract_features(signal):
    signal = signal.flatten().astype(np.float64)
    rms    = np.sqrt(np.mean(signal**2))
    peak   = np.max(np.abs(signal))
    mean_a = np.mean(np.abs(signal))
    return np.array([
        np.mean(signal),
        np.std(signal),
        rms,
        peak,
        peak / (rms + 1e-10),                        # crest factor
        rms / (mean_a + 1e-10),                       # shape factor
        peak / (mean_a + 1e-10),                      # impulse factor
        stats.kurtosis(signal),
        stats.skew(signal),
        np.max(signal) - np.min(signal),              # peak-to-peak
        np.var(signal),
        np.sum(signal**2),                            # energy
    ])

FEATURE_NAMES = [
    "mean", "std", "rms", "peak",
    "crest_factor", "shape_factor", "impulse_factor",
    "kurtosis", "skewness", "peak_to_peak", "variance", "energy"
]

# NOTE (fixed 2026-09-17, external review): CWRU's "Normal Baseline Data"
# files (97-100) are recorded at 48 kHz, while the 12k Drive End fault files
# used here (105-108, 118-121, 130-133) are recorded at 12 kHz -- confirmed
# both by sample-count arithmetic (97-100 have ~4x the samples of a
# comparable ~10 s fault recording at the same nominal duration) and by the
# published literature describing a distinct "48k normal-baseline" category
# of this dataset. Applying the same fixed-length window (1,024 samples) to
# both without resampling would give Normal windows 1/4 the physical
# duration of fault windows -- a sampling-rate confound between the Normal
# class and every fault class, independent of the Ball/Inner fix above. We
# resample the four Normal files from 48 kHz to 12 kHz (decimate by 4, with
# the anti-aliasing low-pass filtering scipy.signal.decimate applies) before
# windowing, so every class is windowed from a consistent 12 kHz signal.
NORMAL_48K_FILES = {"97.mat", "98.mat", "99.mat", "100.mat"}


def mat_to_de_signal(fpath):
    """Extract the DE (drive-end) signal matching THIS file's own id.
    Some CWRU .mat files (e.g. 99.mat) bundle an extra channel from an
    adjacent file (X098_DE_time appears inside 99.mat alongside its own
    X099_DE_time) -- a known artifact of the public dataset. Selecting all
    keys containing "DE_time" would silently concatenate that foreign
    channel, duplicating another file's signal. We instead match the key
    whose numeric id equals the file's own basename."""
    mat = loadmat(fpath)
    file_id = os.path.splitext(os.path.basename(fpath))[0]
    own_key = next((k for k in mat.keys() if "DE_time" in k and file_id in k), None)
    fname = os.path.basename(fpath)
    if own_key is not None:
        sig = mat[own_key].flatten()
        if fname in NORMAL_48K_FILES:
            from scipy.signal import decimate
            sig = decimate(sig, 4, ftype="iir", zero_phase=True)
        return sig
    de_keys = [k for k in mat.keys() if "DE_time" in k]
    if de_keys:
        sig = mat[de_keys[0]].flatten()
        if fname in NORMAL_48K_FILES:
            from scipy.signal import decimate
            sig = decimate(sig, 4, ftype="iir", zero_phase=True)
        return sig
    keys = [k for k in mat.keys() if not k.startswith("_")]
    return mat[keys[0]].flatten()

def build_dataset():
    print("\n=== Extracting features ===")
    rows, labels = [], []
    for fname, label, lname, _ in CWRU_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  Missing {fname}, skipping")
            continue
        sig = mat_to_de_signal(fpath)
        n   = (len(sig) - WINDOW) // STEP + 1
        for i in range(n):
            w = sig[i*STEP : i*STEP + WINDOW]
            rows.append(extract_features(w))
            labels.append(label)
        print(f"  {fname}: {n} windows, class={lname}")
    X = np.array(rows)
    y = np.array(labels)
    print(f"\nTotal: {len(y)} windows | Classes: {np.bincount(y)}")
    return X, y

# ---------------------------------------------------------------------------
# 3. NEUTROSOPHIC DECOMPOSITION
# ---------------------------------------------------------------------------
def neutrosophic_decomposition(P):
    """
    P: (n_samples, n_classes) soft-voted probability matrix
    Returns T, F, I_entropy, I_margin, I_gini arrays (length n_samples)
    """
    T       = P.max(axis=1)
    F       = 1.0 - T
    eps     = 1e-12
    p_safe  = np.clip(P, eps, 1)
    I_ent   = -np.sum(p_safe * np.log(p_safe), axis=1) / np.log(P.shape[1])
    sorted_P = np.sort(P, axis=1)[:, ::-1]
    I_mar   = 1.0 - (sorted_P[:, 0] - sorted_P[:, 1])
    I_gin   = 1.0 - np.sum(P**2, axis=1)
    return T, F, I_ent, I_mar, I_gin

def soft_vote(proba_list):
    return np.mean(proba_list, axis=0)

# ---------------------------------------------------------------------------
# 4. STATISTICAL TESTS (same protocol as NeutroSense)
# ---------------------------------------------------------------------------
def run_statistical_tests(F_arr, I_ent, I_mar, I_gin, errors):
    print("\n=== Statistical Tests ===")

    # Pearson correlations
    for name, arr in [("F", F_arr), ("I_entropy", I_ent),
                      ("I_margin", I_mar), ("I_gini", I_gin)]:
        r, p = pearsonr(arr, errors)
        d    = np.mean(arr[errors==1]) - np.mean(arr[errors==0])
        pooled_std = np.sqrt((np.std(arr[errors==1])**2 + np.std(arr[errors==0])**2) / 2)
        cohens_d = d / (pooled_std + 1e-12)
        print(f"  {name:12s}  r={r:.3f}  p={p:.2e}  Cohen's d={cohens_d:.3f}")

    # Band-stratified analysis
    print("\n  Band-stratified: r(I_margin, error) within F bands")
    bands = [(0.0,0.05),(0.05,0.1),(0.1,0.2),(0.2,0.4),(0.4,0.6),(0.6,1.0)]
    for lo, hi in bands:
        mask = (F_arr >= lo) & (F_arr < hi)
        if mask.sum() < 10:
            continue
        if errors[mask].std() == 0:
            print(f"    F in [{lo:.2f},{hi:.2f}]: N={mask.sum():5d}  no variance (all correct)")
            continue
        r, p = pearsonr(I_mar[mask], errors[mask])
        print(f"    F in [{lo:.2f},{hi:.2f}]: N={mask.sum():5d}  r={r:.3f}  p={p:.2e}")

    # Median-split in critical zone F in [0.05, 0.3]
    crit = (F_arr >= 0.05) & (F_arr < 0.3)
    if crit.sum() > 30:
        median_I = np.median(I_mar[crit])
        hi_mask  = crit & (I_mar >= median_I)
        lo_mask  = crit & (I_mar <  median_I)
        err_hi   = errors[hi_mask].mean()
        err_lo   = errors[lo_mask].mean()
        stat, p  = mannwhitneyu(I_mar[hi_mask], I_mar[lo_mask], alternative="two-sided")
        print(f"\n  Median-split F in [0.05,0.3]: N={crit.sum()}")
        print(f"    High I_margin: error={err_hi:.3f} | Low I_margin: error={err_lo:.3f}")
        print(f"    Diff={err_hi-err_lo:.3f} pp | Mann-Whitney p={p:.2e}")

    # Partial correlation (linear conditioning on F)
    res_I = I_mar - np.polyval(np.polyfit(F_arr, I_mar, 1), F_arr)
    res_e = errors - np.polyval(np.polyfit(F_arr, errors.astype(float), 1), F_arr)
    r_part, p_part = pearsonr(res_I, res_e)
    print(f"\n  Partial r(I_margin, error | F) = {r_part:.3f}  p={p_part:.2e}")

    return {"r_partial": r_part, "p_partial": p_part}

# ---------------------------------------------------------------------------
# 5. FIGURES
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Normal", "Ball", "Inner", "Outer"]
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

def fig1_architecture():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        (0.05, "Raw\nVibration\nSignal", "#E3F2FD"),
        (0.22, "Feature\nExtraction\n(12 features)", "#FFF9C4"),
        (0.39, "Ensemble\nRF + XGB\n+ LR", "#E8F5E9"),
        (0.56, "Soft\nVoting\nP̄", "#F3E5F5"),
        (0.73, "Neutrosophic\nDecomposition\n(T, F, I)", "#FBE9E7"),
        (0.90, "Fault\nClassification\n+ Abstention", "#E0F2F1"),
    ]
    for x, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, 0.2), 0.14, 0.6,
                                   color=color, ec="#666", lw=1.2, transform=ax.transAxes))
        ax.text(x+0.07, 0.5, label, ha="center", va="center",
                fontsize=8.5, transform=ax.transAxes)
        if x < 0.90:
            ax.annotate("", xy=(x+0.17, 0.5), xytext=(x+0.14, 0.5),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))
    ax.set_title("Figure 1. Neutrosophic ensemble pipeline for CWRU bearing fault detection",
                 fontsize=10, pad=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "Figure1_Architecture.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 1 saved.")

def fig2_class_distribution(y_train, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, y, title in zip(axes, [y_train, y_test], ["Training set", "Test set"]):
        counts = [np.sum(y == i) for i in range(4)]
        bars = ax.bar(CLASS_NAMES, counts, color=COLORS, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Instances")
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    str(c), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Figure 2. Class distribution after SMOTE balancing", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "Figure2_ClassDistribution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 2 saved.")

def fig3_confusion(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Figure 3. Confusion matrix — neutrosophic ensemble on CWRU test set", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "Figure3_ConfusionMatrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 3 saved.")

def fig4_F_bands(F_arr, I_mar, errors):
    bands = [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),
             (0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]
    mid_points, err_rates, r_vals = [], [], []
    for lo, hi in bands:
        mask = (F_arr >= lo) & (F_arr < hi)
        if mask.sum() < 20:
            continue
        mid_points.append((lo+hi)/2)
        err_rates.append(errors[mask].mean())
        r, _ = pearsonr(I_mar[mask], errors[mask]) if mask.sum() > 5 else (0, 1)
        r_vals.append(r)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.bar(mid_points, err_rates, width=0.08, color="#2196F3", alpha=0.7, label="Error rate")
    ax2.plot(mid_points, r_vals, "o-", color="#FF5722", lw=2, ms=6, label="r(I_margin, error)")
    ax1.set_xlabel("F (Falsity) band midpoint")
    ax1.set_ylabel("Error rate", color="#2196F3")
    ax2.set_ylabel("Pearson r within band", color="#FF5722")
    ax1.set_title("Figure 4. Error rate and I_margin correlation within F bands", fontsize=9)
    lines1, _ = ax1.get_legend_handles_labels()
    lines2, _ = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, ["Error rate", "r(I_margin, error)"], loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "Figure4_FBands.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 4 saved.")

def fig5_decile_errorrate(I_mar, errors):
    deciles = np.array_split(np.argsort(I_mar), 10)
    err_per_decile = [errors[d].mean() for d in deciles]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(1, 11), err_per_decile, color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 10)),
                  edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Indeterminacy decile (D1 = most confident)")
    ax.set_ylabel("Error rate")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
    for bar, v in zip(bars, err_per_decile):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_title("Figure 5. Error rate by I_margin decile (D1 = most confident, D10 = most uncertain)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "Figure5_DecileError.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 5 saved.")

def fig6_abstention_curve(I_mar, errors):
    thresholds = np.linspace(0.0, 0.99, 100)
    acc_vals, cov_vals, util_vals = [], [], []
    for tau in thresholds:
        mask = I_mar <= tau
        if mask.sum() == 0:
            acc_vals.append(np.nan); cov_vals.append(0.0); util_vals.append(0.0)
            continue
        acc = 1 - errors[mask].mean()
        cov = mask.mean()
        acc_vals.append(acc)
        cov_vals.append(cov)
        util_vals.append(acc * cov)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, acc_vals, label="Accuracy",  color="#2196F3", lw=2)
    ax.plot(thresholds, cov_vals, label="Coverage",  color="#FF9800", lw=2, linestyle="--")
    ax.plot(thresholds, util_vals,label="Utility (Acc×Cov)", color="#4CAF50", lw=2, linestyle=":")
    best_tau = thresholds[np.nanargmax(util_vals)]
    ax.axvline(best_tau, color="#F44336", linestyle="-.", lw=1.5,
               label=f"tau* = {best_tau:.2f} (max utility)")
    ax.set_xlabel("Abstention threshold (tau)")
    ax.set_ylabel("Score")
    ax.set_title("Figure 6. Accuracy, coverage, and utility as a function of abstention threshold (tau)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "Figure6_AbstentionCurve.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 6 saved. (tau* = {best_tau:.2f})")
    return best_tau, thresholds, np.array(acc_vals), np.array(cov_vals)

# ---------------------------------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    # DEPRECATED for reported paper results (as of 2026-09-17): this main()
    # uses a random stratified train_test_split, which mixes operating
    # conditions and allows overlapping windows from the same file to leak
    # across train/test. All numbers reported in the manuscript come from
    # pipeline_v3_grouped.py's leave-one-condition-out split instead. This
    # function/module is kept only for its shared helpers (download_files,
    # extract_features, mat_to_de_signal, CWRU_FILES, CLASS_NAMES, WINDOW,
    # STEP, DATA_DIR), which pipeline_v3_grouped.py imports and reuses.
    # --- Download ---
    download_files()

    # --- Build dataset ---
    X, y = build_dataset()
    if len(X) == 0:
        print("ERROR: No data loaded. Check download.")
        sys.exit(1)

    # --- Train/test split (stratified, before SMOTE) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # --- SMOTE only on train ---
    smote = SMOTE(random_state=42)
    X_train_s, y_train_s = smote.fit_resample(X_train, y_train)

    # --- Scaling ---
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_s)
    X_test_sc  = scaler.transform(X_test)

    print(f"\nTrain (after SMOTE): {X_train_sc.shape[0]} | Test: {X_test_sc.shape[0]}")

    # --- Train models ---
    print("\n=== Training models ===")
    rf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, random_state=42,
                        use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
    lr  = LogisticRegression(max_iter=1000, random_state=42, C=1.0)

    rf.fit(X_train_sc, y_train_s);  print("  RF  trained")
    xgb.fit(X_train_sc, y_train_s); print("  XGB trained")
    lr.fit(X_train_sc, y_train_s);  print("  LR  trained")

    # --- Soft voting on test set ---
    P_rf  = rf.predict_proba(X_test_sc)
    P_xgb = xgb.predict_proba(X_test_sc)
    P_lr  = lr.predict_proba(X_test_sc)
    P_avg = soft_vote([P_rf, P_xgb, P_lr])
    y_pred = np.argmax(P_avg, axis=1)

    # --- Accuracy ---
    acc = accuracy_score(y_test, y_pred)
    n_correct = np.sum(y_pred == y_test)
    n_total   = len(y_test)
    print(f"\n=== Ensemble Results ===")
    print(f"  Overall accuracy: {acc*100:.1f}%  ({n_correct}/{n_total})")
    print("\n" + classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # --- Neutrosophic decomposition ---
    T, F, I_ent, I_mar, I_gin = neutrosophic_decomposition(P_avg)
    errors = (y_pred != y_test).astype(int)

    # --- Statistical tests ---
    stats_res = run_statistical_tests(F, I_ent, I_mar, I_gin, errors)

    # --- Individual model accuracies ---
    print("\n=== Individual model accuracies ===")
    for name, mod in [("RF", rf), ("XGB", xgb), ("LR", lr)]:
        pred = mod.predict(X_test_sc)
        print(f"  {name}: {accuracy_score(y_test, pred)*100:.1f}%")

    # --- Abstention results ---
    thresholds = np.linspace(0.0, 0.99, 100)
    tau04_mask = I_mar <= 0.4
    if tau04_mask.sum() > 0:
        acc04 = 1 - errors[tau04_mask].mean()
        cov04 = tau04_mask.mean()
        print(f"\n  Abstention tau=0.4: acc={acc04*100:.1f}%  coverage={cov04*100:.1f}%")

    # --- Figures ---
    print("\n=== Generating figures ===")
    fig1_architecture()
    fig2_class_distribution(y_train_s, y_test)
    fig3_confusion(y_test, y_pred)
    fig4_F_bands(F, I_mar, errors)
    fig5_decile_errorrate(I_mar, errors)
    best_tau, _, acc_arr, cov_arr = fig6_abstention_curve(I_mar, errors)

    # --- Summary table for paper ---
    print("\n=== PAPER NUMBERS SUMMARY ===")
    print(f"  Dataset:         CWRU Bearing (DE signal, 12kHz)")
    print(f"  Windows:         {n_total} test  |  {X_train_sc.shape[0]} train (post-SMOTE)")
    print(f"  Features:        {X.shape[1]} time-domain features")
    print(f"  Classes:         {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)})")
    print(f"  Ensemble acc:    {acc*100:.1f}%")
    print(f"  Best tau*:        {best_tau:.2f}")
    idx_t04 = np.argmin(np.abs(thresholds - 0.4))
    if not np.isnan(acc_arr[idx_t04]):
        print(f"  Acc @ tau=0.4:   {acc_arr[idx_t04]*100:.1f}%  coverage={cov_arr[idx_t04]*100:.1f}%")
    print(f"  Partial r:       {stats_res['r_partial']:.3f}  p={stats_res['p_partial']:.2e}")
    print("\nDone. All figures saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
