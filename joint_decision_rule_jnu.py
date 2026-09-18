# -*- coding: utf-8 -*-
"""
Tests whether a genuinely JOINT decision rule over (I1_hat, I2_hat) -- accept
only if I1_hat <= tau1 AND I2_hat <= tau2 -- outperforms any single-score
selector (Section 4.4's baselines: T_hat, margin, I1_hat alone, I2_hat
alone, and the linear combination I1_hat+I2_hat). This directly tests
whether treating aleatoric and epistemic uncertainty as two SEPARATE axes
(rather than collapsing them into one score) has a demonstrable selective-
classification advantage -- the specific, previously untested claim needed
to argue for a genuinely multi-component ("neutrosophic") decision rule
rather than a single scalar uncertainty score under a different name.

REVISION NOTE (7th adversarial round): the first version of this script
built the joint rule's frontier from a COARSE grid (100 quantile-based tau1
values x 4 tau2 values, then an envelope over a separate 200-point coverage
grid), while the single-score baselines in baseline_comparison_jnu.py use
the EXACT per-instance tie-corrected curve (up to N=5859 points). An
external reviewer showed this made the two AURC numbers incomparable: on a
synthetic check with I2_hat held constant (so the joint family is
mathematically identical to the univariate one), the coarse procedure gave
a WORSE AURC than the exact univariate one purely from grid resolution,
which would have wrongly looked like "no improvement" even when the two
families are identical by construction. This version instead builds the
joint rule's frontier EXACTLY: for each discrete tau2 value, the achievable
(coverage, risk) pairs are computed at every one of that subset's own data
points using the same expected-value-under-random-tie-breaking method as
aurc(), expressed on the SAME global coverage grid (k/N_total, k=1..N_total)
as the univariate baselines, and the envelope is the pointwise minimum
across the four tau2 branches at every one of the N_total canonical
coverage points -- no interpolation or coarse binning at any step.

Method: for each of I2_hat's four discrete values tau2, restrict to the
subset with I2_hat<=tau2, sort by I1_hat ascending, and compute the exact
tie-corrected cumulative-risk curve on that subset (same logic as
baseline_comparison_jnu.aurc()). Express each subset's k-th point at global
coverage k/N_total (since coverage is always relative to the full test
set). The joint rule's own frontier is, at each global coverage level, the
minimum risk achievable by any of the four branches that can reach that
coverage. Compare its AURC against the single-score curves computed by the
same aurc() function, so both sides of the comparison use an identical
computational method.
"""
import os
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

from pipeline_v3_grouped import build_jnu_grouped
from baseline_comparison_jnu import aurc

OUT = os.path.dirname(os.path.abspath(__file__))


def tie_corrected_local_risk(scores, errs):
    """Exact tie-corrected cumulative risk curve on the given subset, indexed
    by local rank k=1..len(scores). Same expected-value-under-uniform-random
    -tie-breaking method as baseline_comparison_jnu.aurc()."""
    order = np.argsort(scores, kind="stable")
    s_sorted = np.asarray(scores)[order]
    e_sorted = np.asarray(errs)[order]
    n = len(e_sorted)
    cum_err_expected = np.empty(n, dtype=float)
    before_err = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        m = j - i
        e = e_sorted[i:j].sum()
        for k in range(1, m + 1):
            cum_err_expected[i + k - 1] = before_err + k * e / m
        before_err += e
        i = j
    k_arr = np.arange(1, n + 1)
    local_risk = cum_err_expected / k_arr
    return local_risk  # local_risk[k-1] = risk when accepting the k lowest-score instances of THIS subset


if __name__ == "__main__":
    print("Building JNU (grouped, leave-1000rpm-out)...")
    Xtr, ytr, Xte, yte = build_jnu_grouped()

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xte_sc = scaler.transform(Xte)
    smote = SMOTE(random_state=42)
    Xs_sc, ys = smote.fit_resample(Xtr_sc, ytr)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, random_state=42,
                         use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    rf.fit(Xs_sc, ys); xgb.fit(Xs_sc, ys); lr.fit(Xs_sc, ys)

    P_rf = rf.predict_proba(Xte_sc); P_xgb = xgb.predict_proba(Xte_sc); P_lr = lr.predict_proba(Xte_sc)
    P_avg = (P_rf + P_xgb + P_lr) / 3.0
    y_pred = np.argmax(P_avg, axis=1)
    errors = (y_pred != yte).astype(int)
    n_total = len(errors)

    sorted_idx = np.argsort(-P_avg, axis=1)
    P1 = np.take_along_axis(P_avg, sorted_idx[:, [0]], axis=1).ravel()
    P2 = np.take_along_axis(P_avg, sorted_idx[:, [1]], axis=1).ravel()
    entropy = -np.sum(P_avg * np.log(P_avg + 1e-12), axis=1) / np.log(P_avg.shape[1])
    votes = np.stack([np.argmax(P_rf, axis=1), np.argmax(P_xgb, axis=1), np.argmax(P_lr, axis=1)], axis=1)
    agree = (votes == y_pred[:, None]).sum(axis=1)
    vote_dis = 1.0 - agree / 3.0

    tau2_grid = np.unique(vote_dis)  # 0, 1/3, 2/3, 1 (4 discrete values)

    # For each tau2 branch, exact tie-corrected local risk curve, indexed by
    # k=1..n_t2, which corresponds to GLOBAL coverage k/n_total (k accepted
    # instances out of the whole test set).
    branch_risk_by_k = {}  # tau2 -> array of length n_total (up to n_t2), risk at global coverage k/n_total
    for t2 in tau2_grid:
        mask_t2 = vote_dis <= t2
        n_t2 = int(mask_t2.sum())
        local_risk = tie_corrected_local_risk(entropy[mask_t2], errors[mask_t2])
        branch_risk_by_k[t2] = local_risk  # length n_t2

    # Envelope at every canonical global coverage point k/n_total, k=1..n_total.
    env_risk = np.full(n_total, np.inf)
    for t2, local_risk in branch_risk_by_k.items():
        n_t2 = len(local_risk)
        # branch's risk at global k is local_risk[min(k, n_t2) - 1] for k=1..n_total,
        # i.e. once k exceeds n_t2 the branch cannot admit more instances -- it stays
        # capped at its own maximum coverage's risk (the branch's coverage cannot
        # exceed n_t2/n_total, so for k > n_t2 we do NOT let this branch contribute
        # at those larger k -- it is simply infeasible there).
        env_risk[:n_t2] = np.minimum(env_risk[:n_t2], local_risk)

    k_arr = np.arange(1, n_total + 1)
    coverage = k_arr / n_total
    aurc_joint = np.trapezoid(env_risk, coverage) if hasattr(np, "trapezoid") else np.trapz(env_risk, coverage)
    idx50 = int(n_total * 0.5) - 1
    acc50_joint = 1 - env_risk[idx50]

    print(f"\nJoint AND-rule (I1_hat<=tau1 AND I2_hat<=tau2) EXACT lower-envelope frontier:")
    print(f"  AURC = {aurc_joint:.4f}   Acc@50% coverage = {acc50_joint*100:.2f}%   (coverage grid point = {coverage[idx50]*100:.4f}%)")

    print(f"\nFor comparison (single-score selectors, identical exact tie-corrected AURC method):")
    ref_results = {}
    for name, score in [
        ("I1_hat alone", entropy),
        ("I2_hat alone", vote_dis),
        ("Max confidence (T_hat)", 1 - P1),
        ("Margin", 1 - (P1 - P2)),
    ]:
        area, acc50 = aurc(score, errors)
        ref_results[name] = area
        print(f"  {name:28s} AURC={area:.4f}  Acc@50%cov={acc50*100:.2f}%")

    ref_area = ref_results["I1_hat alone"]
    if aurc_joint < ref_area - 1e-9:
        print(f"\n=> Joint rule IMPROVES on I1_hat alone: {aurc_joint:.4f} < {ref_area:.4f}")
    elif abs(aurc_joint - ref_area) <= 1e-9:
        print(f"\n=> Joint rule is IDENTICAL to I1_hat alone: {aurc_joint:.4f} == {ref_area:.4f} "
              f"(expected: tau2=max makes the AND-rule degenerate to I1_hat alone, and the "
              f"envelope never needs a smaller tau2 to do better)")
    else:
        print(f"\n=> Joint rule does NOT improve on I1_hat alone: {aurc_joint:.4f} >= {ref_area:.4f}")

    # Sanity check requested by the 7th-round review: with I2_hat held constant
    # (single branch, tau2 = its only value), the joint family collapses exactly
    # to the univariate I1_hat family, so both methods must agree to numerical
    # precision. This is the check that exposed the original grid-mismatch bug.
    print("\n--- Sanity check: single-branch (I2_hat constant) must exactly match I1_hat-alone AURC ---")
    fake_vote_dis = np.zeros(n_total)
    fake_tau2_grid = np.unique(fake_vote_dis)
    fake_env = np.full(n_total, np.inf)
    for t2 in fake_tau2_grid:
        mask_t2 = fake_vote_dis <= t2
        local_risk = tie_corrected_local_risk(entropy[mask_t2], errors[mask_t2])
        fake_env[:len(local_risk)] = np.minimum(fake_env[:len(local_risk)], local_risk)
    fake_aurc = np.trapezoid(fake_env, coverage) if hasattr(np, "trapezoid") else np.trapz(fake_env, coverage)
    print(f"  Single-branch joint-procedure AURC = {fake_aurc:.6f}")
    print(f"  I1_hat-alone aurc() AURC           = {ref_area:.6f}")
    print(f"  Difference = {abs(fake_aurc-ref_area):.2e} (should be ~0; a nonzero gap here would "
          f"indicate the two procedures are still not using an identical method)")
