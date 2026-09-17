# -*- coding: utf-8 -*-
"""
JNU Bearing Dataset — Neutrosophic Ensemble Pipeline
Variable speed (600/800/1000 rpm), 50kHz, 4 classes
Jiangnan University bearing fault dataset
"""
import os, sys, urllib.request, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT, "jnu_raw")
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/ClarkGableWang/JNU-Bearing-Dataset/main/"

# (filename, label_int, label_str)
JNU_FILES = [
    ("n600_3_2.csv",   0, "Normal"),
    ("n800_3_2.csv",   0, "Normal"),
    ("n1000_3_2.csv",  0, "Normal"),
    ("ib600_2.csv",    1, "Inner"),
    ("ib800_2.csv",    1, "Inner"),
    ("ib1000_2.csv",   1, "Inner"),
    ("ob600_2.csv",    2, "Outer"),
    ("ob800_2.csv",    2, "Outer"),
    ("ob1000_2.csv",   2, "Outer"),
    ("tb600_2.csv",    3, "Ball"),
    ("tb800_2.csv",    3, "Ball"),
    ("tb1000_2.csv",   3, "Ball"),
]

WINDOW = 1024
STEP   = 512
CLASS_NAMES = ["Normal", "Inner", "Outer", "Ball"]

# ---------------------------------------------------------------------------
# 1. DOWNLOAD
# ---------------------------------------------------------------------------
def download_files():
    print("=== Downloading JNU files ===")
    for fname, _, _ in JNU_FILES:
        dest = os.path.join(DATA_DIR, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 100000:
            print(f"  {fname} OK ({os.path.getsize(dest)//1024}KB)")
            continue
        url = BASE_URL + fname
        print(f"  Downloading {fname}...", end=" ", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            print(f"OK ({os.path.getsize(dest)//1024}KB)")
        except Exception as e:
            print(f"FAIL: {e}")

# ---------------------------------------------------------------------------
# 2. FEATURE EXTRACTION
# ---------------------------------------------------------------------------
def extract_features(sig):
    sig  = sig.flatten().astype(np.float64)
    rms  = np.sqrt(np.mean(sig**2))
    peak = np.max(np.abs(sig))
    ma   = np.mean(np.abs(sig))
    return np.array([
        np.mean(sig), np.std(sig), rms, peak,
        peak / (rms + 1e-10),
        rms  / (ma  + 1e-10),
        peak / (ma  + 1e-10),
        stats.kurtosis(sig),
        stats.skew(sig),
        np.max(sig) - np.min(sig),
        np.var(sig),
        np.sum(sig**2),
    ])

def build_dataset():
    print("\n=== Extracting features ===")
    rows, labels = [], []
    for fname, label, lname in JNU_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  Missing {fname}, skipping")
            continue
        sig = pd.read_csv(fpath, header=None).values.flatten()
        n   = (len(sig) - WINDOW) // STEP + 1
        for i in range(n):
            rows.append(extract_features(sig[i*STEP : i*STEP + WINDOW]))
            labels.append(label)
        print(f"  {fname}: {n} windows, class={lname}")
    X = np.array(rows)
    y = np.array(labels)
    counts = np.bincount(y)
    print(f"\nTotal: {len(y)} windows | "
          f"Normal={counts[0]} Inner={counts[1]} Outer={counts[2]} Ball={counts[3]}")
    return X, y

# ---------------------------------------------------------------------------
# 3. NEUTROSOPHIC DECOMPOSITION
# ---------------------------------------------------------------------------
def neutrosophic(P):
    T     = P.max(axis=1)
    F     = 1.0 - T
    eps   = 1e-12
    p_s   = np.clip(P, eps, 1)
    I_ent = -np.sum(p_s * np.log(p_s), axis=1) / np.log(P.shape[1])
    sP    = np.sort(P, axis=1)[:, ::-1]
    I_mar = 1.0 - (sP[:, 0] - sP[:, 1])
    I_gin = 1.0 - np.sum(P**2, axis=1)
    return T, F, I_ent, I_mar, I_gin

# ---------------------------------------------------------------------------
# 4. STATISTICAL TESTS
# ---------------------------------------------------------------------------
def run_tests(F_arr, I_ent, I_mar, I_gin, errors):
    print("\n=== Statistical Tests ===")
    results = {}
    for name, arr in [("F", F_arr), ("I_entropy", I_ent),
                      ("I_margin", I_mar), ("I_gini", I_gin)]:
        r, p = pearsonr(arr, errors)
        d    = np.mean(arr[errors==1]) - np.mean(arr[errors==0])
        ps   = np.sqrt((np.std(arr[errors==1])**2 + np.std(arr[errors==0])**2) / 2)
        cd   = d / (ps + 1e-12)
        print(f"  {name:12s}  r={r:.3f}  p={p:.2e}  Cohen's d={cd:.3f}")
        results[name] = {"r": r, "p": p, "d": cd}

    # Band-stratified
    print("\n  Band-stratified: r(I_margin, error) within F bands")
    bands = [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.5),(0.5,0.7),(0.7,1.0)]
    for lo, hi in bands:
        mask = (F_arr >= lo) & (F_arr < hi)
        if mask.sum() < 20 or errors[mask].std() == 0:
            if mask.sum() >= 20:
                print(f"    F in [{lo:.1f},{hi:.1f}]: N={mask.sum():5d}  no variance")
            continue
        r, p = pearsonr(I_mar[mask], errors[mask])
        print(f"    F in [{lo:.1f},{hi:.1f}]: N={mask.sum():5d}  r={r:.3f}  p={p:.2e}")

    # Median-split in critical zone
    for lo, hi in [(0.1,0.4),(0.2,0.5),(0.3,0.6)]:
        crit = (F_arr >= lo) & (F_arr < hi)
        if crit.sum() >= 30:
            med  = np.median(I_mar[crit])
            hmask = crit & (I_mar >= med)
            lmask = crit & (I_mar <  med)
            if errors[hmask].std() + errors[lmask].std() > 0:
                _, p_mw = mannwhitneyu(I_mar[hmask], I_mar[lmask], alternative="two-sided")
                print(f"\n  Median-split F in [{lo},{hi}]: N={crit.sum()}")
                print(f"    High I_margin: error={errors[hmask].mean():.3f} | "
                      f"Low I_margin: error={errors[lmask].mean():.3f}")
                print(f"    Diff={errors[hmask].mean()-errors[lmask].mean():.3f} pp | "
                      f"Mann-Whitney p={p_mw:.2e}")
                results["median_split"] = {
                    "zone": f"[{lo},{hi}]", "N": crit.sum(),
                    "err_hi": errors[hmask].mean(), "err_lo": errors[lmask].mean(),
                    "p_mw": p_mw
                }
                break

    # Partial correlation
    res_I = I_mar - np.polyval(np.polyfit(F_arr, I_mar, 1), F_arr)
    res_e = errors.astype(float) - np.polyval(np.polyfit(F_arr, errors.astype(float), 1), F_arr)
    r_p, p_p = pearsonr(res_I, res_e)
    print(f"\n  Partial r(I_margin, error | F) = {r_p:.3f}  p={p_p:.2e}")
    results["partial_r"] = r_p
    results["partial_p"] = p_p
    return results

# ---------------------------------------------------------------------------
# 5. FIGURES
# ---------------------------------------------------------------------------
COLORS = ["#2196F3","#4CAF50","#FF9800","#F44336"]

def fig_confusion(y_test, y_pred, prefix="JNU"):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Figure. Confusion matrix — {prefix} test set", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"JNU_ConfusionMatrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  JNU_ConfusionMatrix.png saved.")

def fig_fbands(F_arr, I_mar, errors, prefix="JNU"):
    bands = [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),
             (0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]
    mids, errs, rs = [], [], []
    for lo, hi in bands:
        mask = (F_arr >= lo) & (F_arr < hi)
        if mask.sum() < 10:
            continue
        mids.append((lo+hi)/2)
        errs.append(errors[mask].mean())
        if errors[mask].std() > 0:
            r, _ = pearsonr(I_mar[mask], errors[mask])
        else:
            r = 0
        rs.append(r)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.bar(mids, errs, width=0.08, color="#2196F3", alpha=0.7, label="Error rate")
    ax2.plot(mids, rs, "o-", color="#FF5722", lw=2, ms=6, label="r(I_margin, error)")
    ax1.set_xlabel("F (Falsity) band midpoint")
    ax1.set_ylabel("Error rate", color="#2196F3")
    ax2.set_ylabel("Pearson r within band", color="#FF5722")
    ax1.set_title(f"Figure. Error rate and I_margin correlation within F bands — {prefix}", fontsize=9)
    lines = [plt.Line2D([0],[0],color="#2196F3",linewidth=3,label="Error rate"),
             plt.Line2D([0],[0],color="#FF5722",linewidth=2,label="r(I_margin, error)")]
    ax1.legend(handles=lines, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "JNU_FBands.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  JNU_FBands.png saved.")

def fig_decile(I_mar, errors, prefix="JNU"):
    deciles = np.array_split(np.argsort(I_mar), 10)
    err_d   = [errors[d].mean() for d in deciles]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(1,11), err_d,
                  color=plt.cm.RdYlGn_r(np.linspace(0.1,0.9,10)),
                  edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Indeterminacy decile (D1 = most confident)")
    ax.set_ylabel("Error rate")
    ax.set_xticks(range(1,11))
    ax.set_xticklabels([f"D{i}" for i in range(1,11)])
    for bar, v in zip(bars, err_d):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_title(f"Figure. Error rate by I_margin decile — {prefix}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "JNU_DecileError.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  JNU_DecileError.png saved.")

def fig_abstention(I_mar, errors, prefix="JNU"):
    thresholds = np.linspace(0.0, 0.99, 100)
    accs, covs, utils = [], [], []
    for tau in thresholds:
        mask = I_mar <= tau
        if mask.sum() == 0:
            accs.append(np.nan); covs.append(0); utils.append(0); continue
        a = 1 - errors[mask].mean()
        c = mask.mean()
        accs.append(a); covs.append(c); utils.append(a*c)
    best_tau = thresholds[np.nanargmax(utils)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, accs,  label="Accuracy",        color="#2196F3", lw=2)
    ax.plot(thresholds, covs,  label="Coverage",        color="#FF9800", lw=2, linestyle="--")
    ax.plot(thresholds, utils, label="Utility (AxC)",   color="#4CAF50", lw=2, linestyle=":")
    ax.axvline(best_tau, color="#F44336", linestyle="-.", lw=1.5,
               label=f"tau*={best_tau:.2f}")
    ax.set_xlabel("Abstention threshold (tau)")
    ax.set_ylabel("Score")
    ax.set_title(f"Figure. Abstention curve — {prefix}", fontsize=9)
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "JNU_AbstentionCurve.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  JNU_AbstentionCurve.png saved. (tau*={best_tau:.2f})")
    return best_tau, np.array(accs), np.array(covs)

def fig_comparison(cwru_stats, jnu_stats):
    """Bar chart comparing CWRU vs JNU partial r and accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    datasets = ["CWRU\n(lab, easy)", "JNU\n(variable speed, hard)"]

    # Partial r comparison
    ax = axes[0]
    vals = [cwru_stats["partial_r"], jnu_stats["partial_r"]]
    bars = ax.bar(datasets, vals, color=["#2196F3","#F44336"], width=0.4, edgecolor="white")
    ax.set_ylabel("Partial r  (I_margin, error | F)")
    ax.set_title("Indeterminacy contribution\nbeyond Falsity", fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(vals)*1.4)

    # Accuracy comparison
    ax2 = axes[1]
    vals2 = [cwru_stats["accuracy"], jnu_stats["accuracy"]]
    bars2 = ax2.bar(datasets, vals2, color=["#2196F3","#F44336"], width=0.4, edgecolor="white")
    ax2.set_ylabel("Ensemble Accuracy (%)")
    ax2.set_title("Classification accuracy\n(ensemble)", fontsize=9)
    for bar, v in zip(bars2, vals2):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylim(min(vals2)-5, 101)

    fig.suptitle("Figure. CWRU vs JNU: accuracy and indeterminacy contribution", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Comparison_CWRU_JNU.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Comparison_CWRU_JNU.png saved.")

# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    # DEPRECATED for reported paper results (as of 2026-09-17): this main()
    # uses a random stratified train_test_split, which mixes operating
    # conditions and allows overlapping windows from the same file to leak
    # across train/test. All numbers reported in the manuscript come from
    # pipeline_v3_grouped.py's leave-one-condition-out split instead. This
    # function/module is kept only for its shared helpers (download_files,
    # extract_features, JNU_FILES, CLASS_NAMES, WINDOW, STEP, DATA_DIR),
    # which pipeline_v3_grouped.py imports and reuses.
    download_files()
    X, y = build_dataset()
    if len(X) == 0:
        print("ERROR: No data loaded."); sys.exit(1)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    sm = SMOTE(random_state=42)
    Xs, ys = sm.fit_resample(X_tr, y_tr)

    sc = StandardScaler()
    Xs_sc = sc.fit_transform(Xs)
    Xte_sc = sc.transform(X_te)

    print(f"\nTrain (post-SMOTE): {Xs_sc.shape[0]} | Test: {Xte_sc.shape[0]}")

    print("\n=== Training models ===")
    rf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, random_state=42,
                        eval_metric="mlogloss", verbosity=0)
    lr  = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    rf.fit(Xs_sc, ys);  print("  RF  trained")
    xgb.fit(Xs_sc, ys); print("  XGB trained")
    lr.fit(Xs_sc, ys);  print("  LR  trained")

    P     = np.mean([rf.predict_proba(Xte_sc),
                     xgb.predict_proba(Xte_sc),
                     lr.predict_proba(Xte_sc)], axis=0)
    y_pred = np.argmax(P, axis=1)
    errors = (y_pred != y_te).astype(int)
    acc    = accuracy_score(y_te, y_pred)

    print(f"\n=== Results ===")
    print(f"  Accuracy: {acc*100:.1f}%  ({np.sum(y_pred==y_te)}/{len(y_te)})")
    print("\n" + classification_report(y_te, y_pred, target_names=CLASS_NAMES))

    print("=== Individual models ===")
    for name, mod in [("RF",rf),("XGB",xgb),("LR",lr)]:
        print(f"  {name}: {accuracy_score(y_te, mod.predict(Xte_sc))*100:.1f}%")

    T, F, I_ent, I_mar, I_gin = neutrosophic(P)
    stats_res = run_tests(F, I_ent, I_mar, I_gin, errors)

    tau04 = I_mar <= 0.4
    if tau04.sum() > 0:
        print(f"\n  Abstention tau=0.4: acc={1-errors[tau04].mean():.3f}  "
              f"cov={tau04.mean()*100:.1f}%")

    print("\n=== Generating figures ===")
    fig_confusion(y_te, y_pred)
    fig_fbands(F, I_mar, errors)
    fig_decile(I_mar, errors)
    best_tau, accs, covs = fig_abstention(I_mar, errors)

    jnu_stats = {
        "accuracy": acc*100,
        "partial_r": stats_res["partial_r"],
        "partial_p": stats_res["partial_p"],
        "n_test": len(y_te),
        "n_errors": int(errors.sum()),
        "best_tau": best_tau,
    }

    # CWRU known stats (from previous run)
    cwru_stats = {"accuracy": 99.8, "partial_r": 0.189}

    fig_comparison(cwru_stats, jnu_stats)

    print("\n=== JNU PAPER NUMBERS ===")
    print(f"  Dataset:       JNU Bearing, 50kHz, 600/800/1000 rpm")
    print(f"  Total windows: {len(y)}")
    print(f"  Test / Train:  {len(y_te)} / {Xs_sc.shape[0]} (post-SMOTE)")
    print(f"  Accuracy:      {acc*100:.1f}%  ({np.sum(y_pred==y_te)}/{len(y_te)})")
    print(f"  Errors:        {errors.sum()} ({errors.mean()*100:.1f}%)")
    print(f"  Partial r:     {stats_res['partial_r']:.3f}  p={stats_res['partial_p']:.2e}")
    print(f"  Best tau*:     {best_tau:.2f}")
    if tau04.sum() > 0:
        idx04 = np.argmin(np.abs(np.linspace(0,0.99,100)-0.4))
        print(f"  tau=0.4:       acc={1-errors[tau04].mean()*100:.1f}%  "
              f"cov={tau04.mean()*100:.1f}%")
    print(f"\nDone. Figures saved to: {OUT}")

if __name__ == "__main__":
    main()
