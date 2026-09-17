# -*- coding: utf-8 -*-
"""
Leakage-free pipeline: group (file/condition)-level holdout instead of
window-level shuffle. Windows from the same source file/recording NEVER
appear in both train and test (the original v1/v2 pipelines shuffled
overlapping windows globally, letting adjacent, 50%-overlapping windows
leak across the split -- confirmed by adversarial review, 2026-09-17).

Design: leave-one-condition-out.
  CWRU: hold out load = 3 HP (the 4th file listed per class: 100, 108, 112, 133)
        for TEST; train on loads 0/1/2 HP (files 97-99, 105-107, 109-111, 130-132).
  JNU:  hold out speed = 1000 rpm for TEST; train on 600/800 rpm.

This is a harder, more realistic generalization test (unseen operating
condition) and is leakage-free by construction: no file contributes windows
to both splits.
"""
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# CWRU: reuse feature extraction / file list from the original module,
# but split by file (load condition), not by shuffled window.
# ---------------------------------------------------------------------------
import cwru_neutro_pipeline as cw

# load = 3 HP, one per class. Updated for the corrected file mapping in
# cwru_neutro_pipeline.py (2026-09-17): Ball 3HP is 121.mat (was wrongly
# 108.mat), Inner 3HP is 108.mat (was wrongly 112.mat, which no longer
# appears in CWRU_FILES at all -- see that module's header comment).
CWRU_TEST_FILES = {"100.mat", "121.mat", "108.mat", "133.mat"}


def build_cwru_grouped():
    cw.download_files()
    rows_tr, y_tr, rows_te, y_te = [], [], [], []
    for fname, label, lname, _ in cw.CWRU_FILES:
        fpath = os.path.join(cw.DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  Missing {fname}, skipping")
            continue
        sig = cw.mat_to_de_signal(fpath)
        n = (len(sig) - cw.WINDOW) // cw.STEP + 1
        dest_rows, dest_y = (rows_te, y_te) if fname in CWRU_TEST_FILES else (rows_tr, y_tr)
        for i in range(n):
            w = sig[i * cw.STEP: i * cw.STEP + cw.WINDOW]
            dest_rows.append(cw.extract_features(w))
            dest_y.append(label)
        split = "TEST" if fname in CWRU_TEST_FILES else "train"
        print(f"  {fname}: {n} windows, class={lname} -> {split}")
    return (np.array(rows_tr), np.array(y_tr), np.array(rows_te), np.array(y_te))


# ---------------------------------------------------------------------------
# JNU: hold out 1000 rpm entirely for test.
# ---------------------------------------------------------------------------
import jnu_neutro_pipeline as jn

JNU_TEST_FILES = {"n1000_3_2.csv", "ib1000_2.csv", "ob1000_2.csv", "tb1000_2.csv"}  # 1000 rpm


def build_jnu_grouped():
    jn.download_files()
    rows_tr, y_tr, rows_te, y_te = [], [], [], []
    for fname, label, lname in jn.JNU_FILES:
        fpath = os.path.join(jn.DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  Missing {fname}, skipping")
            continue
        import pandas as pd
        sig = pd.read_csv(fpath, header=None).values.flatten()
        n = (len(sig) - jn.WINDOW) // jn.STEP + 1
        dest_rows, dest_y = (rows_te, y_te) if fname in JNU_TEST_FILES else (rows_tr, y_tr)
        for i in range(n):
            dest_rows.append(jn.extract_features(sig[i * jn.STEP: i * jn.STEP + jn.WINDOW]))
            dest_y.append(label)
        split = "TEST" if fname in JNU_TEST_FILES else "train"
        print(f"  {fname}: {n} windows, class={lname} -> {split}")
    return (np.array(rows_tr), np.array(y_tr), np.array(rows_te), np.array(y_te))


# ---------------------------------------------------------------------------
# Shared: train ensemble, compute refined decomposition + full stat protocol
# ---------------------------------------------------------------------------
# NOTE (fixed after adversarial review, round 3): partial-correlation
# significance must use df = n - k - 2 (k = number of control variables),
# not the df = n - 2 that a plain pearsonr() on the residuals implicitly
# assumes (residualizing against k controls removes k additional degrees of
# freedom). The numerical effect is negligible at these sample sizes but the
# formula was wrong; we now compute the p-value explicitly with the correct df.
def _partial_r_pvalue(r, n, k):
    from scipy.stats import t as _t
    df = n - k - 2
    if abs(r) >= 1.0 or df <= 0:
        return 0.0
    tstat = r * np.sqrt(df / (1 - r ** 2))
    return 2 * _t.sf(np.abs(tstat), df)


def multivariate_partial_r(target, controls, y):
    X = np.column_stack(controls + [np.ones_like(target)])
    beta_t, *_ = np.linalg.lstsq(X, target, rcond=None)
    res_t = target - X @ beta_t
    beta_y, *_ = np.linalg.lstsq(X, y.astype(float), rcond=None)
    res_y = y.astype(float) - X @ beta_y
    r, _ = pearsonr(res_t, res_y)
    p = _partial_r_pvalue(r, len(target), k=len(controls))
    return r, p


def univariate_partial_r(target, control, y):
    res_t = target - np.polyval(np.polyfit(control, target, 1), control)
    res_y = y.astype(float) - np.polyval(np.polyfit(control, y.astype(float), 1), control)
    r, _ = pearsonr(res_t, res_y)
    p = _partial_r_pvalue(r, len(target), k=1)
    return r, p


def run_full(name, X_tr, y_tr, X_te, y_te, class_names):
    print(f"\n{'='*72}\n  {name} -- GROUPED (leakage-free) SPLIT\n{'='*72}")
    print(f"  Train windows: {len(y_tr)}  (classes: {np.bincount(y_tr)})")
    print(f"  Test  windows: {len(y_te)}  (classes: {np.bincount(y_te)})")

    # NOTE (fixed after adversarial review, round 3): SMOTE previously ran on
    # RAW, unscaled features before StandardScaler. Because SMOTE synthesizes
    # samples using Euclidean nearest neighbors, and the 12 features differ by
    # orders of magnitude in scale (e.g. std(energy) >> std(mean)), this let
    # high-variance features dominate neighbor selection. We now fit the
    # scaler on the raw training data first, transform train/test, and apply
    # SMOTE only on the already-scaled training features.
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_tr)
    Xte_sc = scaler.transform(X_te)
    smote = SMOTE(random_state=42)
    Xs_sc, ys = smote.fit_resample(Xtr_sc, y_tr)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, random_state=42,
                         use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    rf.fit(Xs_sc, ys); xgb.fit(Xs_sc, ys); lr.fit(Xs_sc, ys)

    P_rf = rf.predict_proba(Xte_sc)
    P_xgb = xgb.predict_proba(Xte_sc)
    P_lr = lr.predict_proba(Xte_sc)
    P_avg = (P_rf + P_xgb + P_lr) / 3.0
    y_pred = np.argmax(P_avg, axis=1)
    errors = (y_pred != y_te).astype(int)
    acc = accuracy_score(y_te, y_pred)

    print(f"\n  ENSEMBLE ACCURACY (leakage-free): {acc*100:.2f}%  "
          f"({(y_pred==y_te).sum()}/{len(y_te)})")
    for nm, mod in [("RF", rf), ("XGB", xgb), ("LR", lr)]:
        print(f"    {nm}: {accuracy_score(y_te, mod.predict(Xte_sc))*100:.2f}%")
    print()
    print(classification_report(y_te, y_pred, target_names=class_names, zero_division=0))

    # --- refined decomposition ---
    sorted_idx = np.argsort(-P_avg, axis=1)
    P1 = np.take_along_axis(P_avg, sorted_idx[:, [0]], axis=1).ravel()
    P2 = np.take_along_axis(P_avg, sorted_idx[:, [1]], axis=1).ravel()
    T_hat = P1
    F_hat = P2
    # I1_hat = normalized Shannon entropy of the full P_avg distribution.
    # NOTE (fixed after adversarial review, round 2): the previous definition
    # I1_hat = 1-(P1-P2) = 1-T_hat+F_hat is an EXACT algebraic identity with
    # T_hat and F_hat (T_hat + I1_hat - F_hat = 1 always), so it carried no
    # information independent of T_hat/F_hat and any "partial correlation
    # controlling for F_hat" was mathematically just -T_hat's contribution.
    # Normalized entropy uses the full K-class distribution (not just the
    # top two probabilities), so it is not a deterministic function of
    # T_hat/F_hat alone and can genuinely add independent information.
    K = P_avg.shape[1]
    eps = 1e-12
    entropy = -np.sum(P_avg * np.log(P_avg + eps), axis=1)
    I1_hat = entropy / np.log(K)

    r_TF, p_TF = pearsonr(T_hat, F_hat)
    r_I1_raw, p_I1_raw = pearsonr(I1_hat, errors)
    # NOTE (fixed after adversarial review, round 3, finding N3): I1_hat's
    # independent contribution must be tested controlling for BOTH T_hat and
    # F_hat jointly, not F_hat alone -- this manuscript's own framework
    # treats T_hat and F_hat as two separate, non-redundant axes (Section
    # 3.5), so a claim that I1_hat adds information "beyond" them should
    # control for both simultaneously. Controlling for F_hat alone
    # overstated the JNU effect (r=+0.41-0.47 depending on other fixes)
    # relative to the jointly-controlled estimate below.
    r_I1_part, p_I1_part = multivariate_partial_r(I1_hat, [T_hat, F_hat], errors)

    votes = np.stack([np.argmax(P_rf, axis=1), np.argmax(P_xgb, axis=1),
                       np.argmax(P_lr, axis=1)], axis=1)
    agree = (votes == y_pred[:, None]).sum(axis=1)
    I2_hat = 1.0 - agree / 3.0
    r_I2_raw, p_I2_raw = pearsonr(I2_hat, errors)
    r_I2_part, p_I2_part = multivariate_partial_r(I2_hat, [T_hat, F_hat, I1_hat], errors)

    print(f"\n  r(T_hat, F_hat) = {r_TF:+.4f}  p={p_TF:.2e}")
    print(f"  I1_hat raw r={r_I1_raw:+.4f} p={p_I1_raw:.2e}  |  partial r|T,F={r_I1_part:+.4f} p={p_I1_part:.2e}")
    print(f"  I2_hat raw r={r_I2_raw:+.4f} p={p_I2_raw:.2e}  |  partial r|T,F,I1={r_I2_part:+.4f} p={p_I2_part:.2e}")

    for a, label in [(3, "unanimous (3/3)"), (2, "2/3 agree"), (1, "1/3 agree"), (0, "0/3 agree")]:
        mask = agree == a
        if mask.sum() > 0:
            print(f"    agree={a} ({label}): N={mask.sum()}  error_rate={errors[mask].mean()*100:.1f}%")

    # hidden risk zone
    med_I1 = np.median(I1_hat)
    low_i1 = I1_hat <= med_I1
    safe = low_i1 & (agree == 3)
    risky = low_i1 & (agree < 3)
    print(f"\n  Hidden-risk (I1<=median & disagreement): "
          f"safe N={safe.sum()} err={errors[safe].mean()*100 if safe.sum() else float('nan'):.1f}% | "
          f"risky N={risky.sum()} err={errors[risky].mean()*100 if risky.sum() else float('nan'):.1f}%")

    # median-split gap: compare ERROR RATES between hi/lo I1_hat groups.
    # NOTE (fixed after adversarial review, round 2): the previous version
    # ran Mann-Whitney U on I1_hat[hi] vs I1_hat[lo] -- i.e. on the very
    # variable used to define the two groups, which is tautologically
    # significant by construction and says nothing about error rates. We
    # instead test whether the ERROR RATE differs between the two groups
    # (a 2x2 contingency test), which is what the "gap" claim requires.
    # NOTE (fixed after adversarial review, round 3): the <=/>= median
    # convention now matches the hidden-risk block above exactly (low_i1 =
    # I1_hat <= median), rather than using a different inequality here.
    median_I1 = np.median(I1_hat)
    lo = low_i1
    hi = ~low_i1
    if hi.sum() > 0 and lo.sum() > 0:
        gap = errors[hi].mean() - errors[lo].mean()
        table = [[errors[hi].sum(), hi.sum() - errors[hi].sum()],
                 [errors[lo].sum(), lo.sum() - errors[lo].sum()]]
        # NOTE (fixed after CWRU dataset correction, 2026-09-17): with the
        # corrected, unconfounded CWRU classes the ensemble makes zero
        # errors, so the error/correct contingency table has an all-zero
        # row and chi2_contingency raises rather than returning a p-value.
        # Guard this degenerate case explicitly instead of crashing.
        if errors.sum() == 0:
            print(f"  Median-split I1_hat gap: {gap*100:+.2f} pp  (high={errors[hi].mean()*100:.1f}%, "
                  f"low={errors[lo].mean()*100:.1f}%)  chi2 test: undefined (zero errors overall)")
        else:
            chi2, p_gap, dof, _ = chi2_contingency(table)
            print(f"  Median-split I1_hat gap: {gap*100:+.2f} pp  (high={errors[hi].mean()*100:.1f}%, "
                  f"low={errors[lo].mean()*100:.1f}%)  chi2 test on error rate p={p_gap:.2e}")

    # --- Cohen's d for all four indicators ---
    # NOTE (fixed after adversarial review, round 2): pooled SD is now
    # weighted by group size (standard Cohen's d), not an unweighted mean
    # of the two group variances -- the groups are highly imbalanced here
    # (e.g. CWRU: 46 errors vs 2325 correct).
    def cohend(arr):
        x1, x0 = arr[errors == 1], arr[errors == 0]
        n1, n0 = len(x1), len(x0)
        if n1 < 2 or n0 < 2:
            return float("nan")
        v1, v0 = np.var(x1, ddof=1), np.var(x0, ddof=1)
        pooled = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2))
        return (np.mean(x1) - np.mean(x0)) / (pooled + 1e-12)

    print("\n  Cohen's d (error vs correct):")
    for nm, arr in [("T_hat", T_hat), ("F_hat", F_hat), ("I1_hat", I1_hat), ("I2_hat", I2_hat)]:
        r_, p_ = pearsonr(arr, errors)
        print(f"    {nm}: d={cohend(arr):+.3f}  r={r_:+.3f}  p={p_:.2e}")

    # --- band-stratified: r(I1_hat, error) within F_hat bands ---
    print("\n  Band-stratified r(I1_hat, error) within F_hat bands:")
    bands = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)]
    for lo, hi_ in bands:
        mask = (F_hat >= lo) & (F_hat < hi_)
        if mask.sum() < 10:
            continue
        if errors[mask].std() == 0:
            print(f"    F_hat in [{lo:.2f},{hi_:.2f}]: N={mask.sum():5d}  no variance (all {'correct' if errors[mask].mean()==0 else 'error'})")
            continue
        r_, p_ = pearsonr(I1_hat[mask], errors[mask])
        print(f"    F_hat in [{lo:.2f},{hi_:.2f}]: N={mask.sum():5d}  r={r_:+.3f}  p={p_:.2e}")

    # --- Figures ---
    class_names_local = class_names
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_te, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names_local).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix -- {name} (leakage-free, grouped holdout)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"Fig_Confusion_{name}_v3grouped.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    mid, errb, rvals = [], [], []
    for lo, hi_ in bands:
        mask = (F_hat >= lo) & (F_hat < hi_)
        if mask.sum() < 20:
            continue
        mid.append((lo + hi_) / 2)
        errb.append(errors[mask].mean())
        r_, _ = pearsonr(I1_hat[mask], errors[mask]) if errors[mask].std() > 0 else (0, 1)
        rvals.append(r_)
    ax1.bar(mid, errb, width=0.08, color="#2196F3", alpha=0.7)
    ax2.plot(mid, rvals, "o-", color="#FF5722", lw=2, ms=6)
    ax1.set_xlabel("F_hat band midpoint"); ax1.set_ylabel("Error rate", color="#2196F3")
    ax2.set_ylabel("Pearson r within band", color="#FF5722")
    ax1.set_title(f"Error rate and I1_hat correlation within F_hat bands -- {name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"Fig_FBands_{name}_v3grouped.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    order = np.argsort(I1_hat)
    deciles = np.array_split(order, 10)
    err_dec = [errors[d].mean() for d in deciles]
    print("\n  Error rate by I1_hat decile (D1=lowest):")
    print("   ", "  ".join(f"D{i+1}={e:.3f}" for i, e in enumerate(err_dec)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, 11), err_dec, color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 10)))
    ax.set_xticks(range(1, 11)); ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
    ax.set_xlabel("I1_hat decile (D1=lowest)"); ax.set_ylabel("Error rate")
    ax.set_title(f"Error rate by I1_hat decile -- {name}", fontsize=9)
    for i, v in enumerate(err_dec):
        ax.text(i + 1, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"Fig_Decile_{name}_v3grouped.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # NOTE (redesigned after adversarial review, round 3): the previous
    # "utility" = accuracy * coverage is degenerate here -- because the
    # accepted set is nested (raising tau only ever adds instances), this
    # product reduces algebraically to (correct-and-accepted)/N, which is
    # monotonically non-decreasing in tau by construction. Maximizing it
    # therefore always converges to ~100% coverage regardless of the data,
    # so it cannot represent any real accuracy/coverage trade-off, and we no
    # longer report a "tau*" derived from it. We instead report the standard
    # selective-classification risk-coverage curve (Geifman & El-Yaniv,
    # ref [10]): accuracy on the accepted set as a function of threshold tau
    # (accept if I1_hat <= tau), at a fixed, pre-specified grid of tau
    # values, with no optimization over the test set.
    thresholds = np.linspace(0.0, 0.99, 100)
    acc_v, cov_v = [], []
    for tau in thresholds:
        m = I1_hat <= tau
        if m.sum() == 0:
            acc_v.append(np.nan); cov_v.append(0.0); continue
        acc_v.append(1 - errors[m].mean()); cov_v.append(m.mean())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, acc_v, label="Accuracy (selective)", color="#2196F3", lw=2)
    ax.plot(thresholds, cov_v, label="Coverage", color="#FF9800", lw=2, linestyle="--")
    ax.set_xlabel("Abstention threshold tau (accept if I1_hat <= tau)"); ax.set_ylabel("Score")
    ax.set_title(f"Selective risk-coverage curve -- {name}", fontsize=9)
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"Fig_Abstention_{name}_v3grouped.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Selective risk-coverage (fixed tau grid, no test-set optimization):")
    tau_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    tau_report = {}
    for tau in tau_grid:
        idx = np.argmin(np.abs(thresholds - tau))
        print(f"    tau={tau:.1f}: acc={acc_v[idx]*100:.1f}%  cov={cov_v[idx]*100:.1f}%")
        tau_report[tau] = (acc_v[idx], cov_v[idx])
    idx04 = np.argmin(np.abs(thresholds - 0.4))

    return dict(acc=acc, r_TF=r_TF, r_I1_part=r_I1_part, p_I1_part=p_I1_part,
                r_I2_part=r_I2_part, p_I2_part=p_I2_part, n_test=len(y_te), n_err=int(errors.sum()),
                tau_report=tau_report, acc_tau04=acc_v[idx04], cov_tau04=cov_v[idx04])


if __name__ == "__main__":
    print("Building CWRU (grouped, leave-load-3HP-out)...")
    Xtr, ytr, Xte, yte = build_cwru_grouped()
    res_cwru = run_full("CWRU", Xtr, ytr, Xte, yte, cw.CLASS_NAMES)

    print("\n\nBuilding JNU (grouped, leave-1000rpm-out)...")
    Xtr, ytr, Xte, yte = build_jnu_grouped()
    res_jnu = run_full("JNU", Xtr, ytr, Xte, yte, jn.CLASS_NAMES)

    print(f"\n{'='*72}\n  FINAL SUMMARY (leakage-free)\n{'='*72}")
    print(f"  CWRU: acc={res_cwru['acc']*100:.2f}%  N_test={res_cwru['n_test']}  errors={res_cwru['n_err']}")
    print(f"  JNU:  acc={res_jnu['acc']*100:.2f}%  N_test={res_jnu['n_test']}  errors={res_jnu['n_err']}")
