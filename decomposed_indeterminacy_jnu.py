# -*- coding: utf-8 -*-
"""
Extension requested after Section 4.6: re-operationalize I1_hat and I2_hat
using the EXACT algebraic decomposition already derived (but never used as
the actual working indicators) in Section 3.5:

    H(P_avg) = mean_m[H(P_m)]  +  mean_m[KL(P_m || P_avg)]

The current pipeline uses I1_hat = H(P_avg) (total predictive entropy of the
AVERAGED distribution) and I2_hat = discrete base-learner vote-disagreement
fraction (a coarse, argmax-only proxy). This script instead uses the two
ADDITIVE terms of the identity above directly as the two indeterminacy
indicators:

    I1_NEW (aleatoric-like) = mean_m[H(P_m)]      -- average of each base
        learner's OWN entropy: high when individual models are each
        internally uncertain about the class, regardless of whether they
        agree with each other.
    I2_NEW (epistemic-like) = mean_m[KL(P_m || P_avg)] -- the mutual-
        information-style term: high when base learners' FULL probability
        distributions diverge from the ensemble average, using the whole
        distribution (not just the argmax vote), so it is continuous, not
        a coarse 4-level (or 9-level) discrete count.

This is the standard aleatoric/epistemic decomposition used in the
Bayesian deep learning literature (Kendall & Gal 2017, already cited as
ref [16], and Depeweg et al.), applied here to a non-Bayesian soft-voting
ensemble. We verify the algebraic identity holds exactly on this data (a
sanity check the original I1_hat/I2_hat pair does not admit, since vote
disagreement is not an additive component of H(P_avg)), then re-run the
same statistical battery as Sections 4.3-4.4 with the new indicators.
"""
import os
import warnings
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

from pipeline_v3_grouped import build_jnu_grouped, multivariate_partial_r
from baseline_comparison_jnu import aurc

OUT = os.path.dirname(os.path.abspath(__file__))


def entropy_norm(P, n_classes):
    return -np.sum(P * np.log(P + 1e-12), axis=1) / np.log(n_classes)


def kl_div(P, Q):
    """KL(P || Q), row-wise, with clipping to avoid log(0)."""
    P = np.clip(P, 1e-12, 1.0)
    Q = np.clip(Q, 1e-12, 1.0)
    return np.sum(P * np.log(P / Q), axis=1)


if __name__ == "__main__":
    print("Building JNU (grouped, leave-1000rpm-out)...")
    Xtr, ytr, Xte, yte = build_jnu_grouped()

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xte_sc = scaler.transform(Xte)
    smote = SMOTE(random_state=42)
    Xs_sc, ys = smote.fit_resample(Xtr_sc, ytr)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    rf.fit(Xs_sc, ys); xgb.fit(Xs_sc, ys); lr.fit(Xs_sc, ys)

    P_rf = rf.predict_proba(Xte_sc); P_xgb = xgb.predict_proba(Xte_sc); P_lr = lr.predict_proba(Xte_sc)
    P_avg = (P_rf + P_xgb + P_lr) / 3.0
    y_pred = np.argmax(P_avg, axis=1)
    errors = (y_pred != yte).astype(int)
    n_classes = P_avg.shape[1]

    # --- Original indicators (for reference / side-by-side comparison) ---
    entropy_orig = entropy_norm(P_avg, n_classes)
    votes = np.stack([np.argmax(P_rf, axis=1), np.argmax(P_xgb, axis=1), np.argmax(P_lr, axis=1)], axis=1)
    agree = (votes == y_pred[:, None]).sum(axis=1)
    vote_dis_orig = 1.0 - agree / 3.0

    sorted_idx = np.argsort(-P_avg, axis=1)
    T_hat = np.take_along_axis(P_avg, sorted_idx[:, [0]], axis=1).ravel()
    F_hat = np.take_along_axis(P_avg, sorted_idx[:, [1]], axis=1).ravel()

    # --- New indicators: exact additive decomposition of H(P_avg) ---
    H_avg = entropy_norm(P_avg, n_classes)  # total predictive entropy (same as I1_hat original)
    H_rf = entropy_norm(P_rf, n_classes); H_xgb = entropy_norm(P_xgb, n_classes); H_lr = entropy_norm(P_lr, n_classes)
    I1_new = (H_rf + H_xgb + H_lr) / 3.0  # mean_m[H(P_m)] -- aleatoric-like

    kl_rf = kl_div(P_rf, P_avg) / np.log(n_classes)
    kl_xgb = kl_div(P_xgb, P_avg) / np.log(n_classes)
    kl_lr = kl_div(P_lr, P_avg) / np.log(n_classes)
    I2_new = (kl_rf + kl_xgb + kl_lr) / 3.0  # mean_m[KL(P_m||P_avg)] -- epistemic-like (normalized by log(n_classes) for scale)

    # Sanity check: H(P_avg) should equal I1_new + I2_new exactly (up to float error),
    # since this is an algebraic identity, not an approximation. This is a check the
    # ORIGINAL (entropy, vote-disagreement) pair cannot pass, because vote-disagreement
    # is not derived from this decomposition at all.
    identity_gap = np.abs(H_avg - (I1_new + I2_new))
    print(f"\nSanity check -- H(P_avg) = I1_new + I2_new identity: max |gap| = {identity_gap.max():.2e}, "
          f"mean |gap| = {identity_gap.mean():.2e} (should be ~0)")

    print(f"\nOriginal I1_hat (H(P_avg)):        mean={entropy_orig.mean():.4f}  std={entropy_orig.std():.4f}")
    print(f"New I1_new (mean_m[H(P_m)]):       mean={I1_new.mean():.4f}  std={I1_new.std():.4f}")
    print(f"Original I2_hat (vote disagreement, {len(np.unique(vote_dis_orig))} levels): "
          f"mean={vote_dis_orig.mean():.4f}  std={vote_dis_orig.std():.4f}")
    print(f"New I2_new (mean_m[KL(P_m||P_avg)], continuous): mean={I2_new.mean():.4f}  std={I2_new.std():.4f}  "
          f"unique values={len(np.unique(np.round(I2_new, 6)))}")

    r_i1_i2_new, _ = pearsonr(I1_new, I2_new)
    r_i1_i2_orig, _ = pearsonr(entropy_orig, vote_dis_orig)
    print(f"\nCorrelation I1/I2, ORIGINAL pair: r={r_i1_i2_orig:.4f}")
    print(f"Correlation I1/I2, NEW pair:      r={r_i1_i2_new:.4f}")

    # --- Partial correlations with error, controlling for T_hat, F_hat ---
    print("\n=== Partial correlations with error (controlling for T_hat, F_hat) ===")
    r_i1_new, p_i1_new = multivariate_partial_r(I1_new, [T_hat, F_hat], errors)
    r_i2_new, p_i2_new = multivariate_partial_r(I2_new, [T_hat, F_hat], errors)
    r_i2_new_full, p_i2_new_full = multivariate_partial_r(I2_new, [T_hat, F_hat, I1_new], errors)
    print(f"I1_new | T,F:        partial r={r_i1_new:.4f}  p={p_i1_new:.3g}")
    print(f"I2_new | T,F:        partial r={r_i2_new:.4f}  p={p_i2_new:.3g}")
    print(f"I2_new | T,F,I1_new: partial r={r_i2_new_full:.4f}  p={p_i2_new_full:.3g}")
    print(f"(For reference, original pipeline: I1_hat|T,F partial r=+0.217; "
          f"I2_hat|T,F,I1_hat partial r=+0.017, p=0.20, n.s.)")

    # --- AURC baseline comparison ---
    print("\n=== Selective-classification AURC comparison ===")
    area_i1_new, acc50_i1_new = aurc(I1_new, errors)
    area_i2_new, acc50_i2_new = aurc(I2_new, errors)
    area_i1_orig, acc50_i1_orig = aurc(entropy_orig, errors)
    print(f"I1_new (mean model entropy):     AURC={area_i1_new:.4f}  Acc@50%cov={acc50_i1_new*100:.2f}%")
    print(f"I2_new (mean KL to average):     AURC={area_i2_new:.4f}  Acc@50%cov={acc50_i2_new*100:.2f}%")
    print(f"I1_hat original (H(P_avg)):      AURC={area_i1_orig:.4f}  Acc@50%cov={acc50_i1_orig*100:.2f}%  "
          f"(reference, matches Table 3b's 0.3915)")

    # Linear combination and exact oracle joint rule with the NEW indicators.
    combo_new = (I1_new - I1_new.mean()) / I1_new.std() + (I2_new - I2_new.mean()) / I2_new.std()
    area_combo_new, acc50_combo_new = aurc(combo_new, errors)
    print(f"I1_new + I2_new (linear combination): AURC={area_combo_new:.4f}  Acc@50%cov={acc50_combo_new*100:.2f}%")

    from joint_decision_rule_jnu import tie_corrected_local_risk
    n_total = len(errors)
    tau2_grid_new = np.unique(np.round(I2_new, 6))
    if len(tau2_grid_new) > 200:
        tau2_grid_new = np.quantile(I2_new, np.linspace(0, 1, 200))
    branch_risk_by_k = {}
    for t2 in tau2_grid_new:
        mask_t2 = I2_new <= t2
        if mask_t2.sum() == 0:
            continue
        branch_risk_by_k[t2] = tie_corrected_local_risk(I1_new[mask_t2], errors[mask_t2])
    env_risk = np.full(n_total, np.inf)
    for t2, local_risk in branch_risk_by_k.items():
        env_risk[:len(local_risk)] = np.minimum(env_risk[:len(local_risk)], local_risk)
    coverage = np.arange(1, n_total + 1) / n_total
    aurc_joint_new = np.trapezoid(env_risk, coverage) if hasattr(np, "trapezoid") else np.trapz(env_risk, coverage)
    print(f"\nExact oracle joint rule (I1_new<=tau1 AND I2_new<=tau2): AURC={aurc_joint_new:.6f}")
    print(f"I1_new alone (reference, same method):                    AURC={area_i1_new:.6f}")
    diff_new = area_i1_new - aurc_joint_new
    if diff_new > 1e-9:
        print(f"=> Joint rule (NEW indicators) IMPROVES on I1_new alone by {diff_new:.6f}")
    else:
        print(f"=> Joint rule (NEW indicators) does NOT improve on I1_new alone (diff={diff_new:.6f})")

    # --- Hidden-risk zone with new indicators ---
    print("\n=== Hidden-risk zone (low I1_new, split by I2_new median) ===")
    tau1_med = np.median(I1_new)
    low_i1 = I1_new <= tau1_med
    tau2_med_within = np.median(I2_new[low_i1])
    low_i2_within = I2_new[low_i1] <= tau2_med_within
    err_within = errors[low_i1]
    print(f"Among low-I1_new instances (N={int(low_i1.sum())}): "
          f"low-I2_new (N={int(low_i2_within.sum())}) error={err_within[low_i2_within].mean()*100:.1f}%  "
          f"high-I2_new (N={int((~low_i2_within).sum())}) error={err_within[~low_i2_within].mean()*100:.1f}%")
    print(f"(For reference, original pipeline hidden-risk zone: 28.2% (agree) vs 63.7% (disagree))")
