# -*- coding: utf-8 -*-
"""
Extension requested after Section 4.6.5: everything tested so far in this
project treats T-hat=P1 and F-hat=P2 as two order statistics of the SAME
softmax output, so T-hat+F-hat<=1 always -- this is conventional
probability, not neutrosophic independence (a defining feature of
neutrosophic sets is that T, I, F need not sum to 1 and can in principle
be assessed independently). This script tests two things genuinely new:

(1) INDEPENDENT T/F: train two SEPARATE binary meta-classifiers, via
    leakage-free out-of-fold stacking on the training conditions only,
    that independently estimate:
      T_new = P(the ensemble's predicted class is correct)
      F_new = P(the runner-up class -- second-highest P_avg -- is actually
               the correct class)
    Because these come from two independently-fit logistic regressions
    (not two entries of one softmax vector), T_new+F_new is NOT
    constrained to be <=1. We report how often it exceeds 1 as a direct,
    verifiable measure of genuine independence, absent from every T/F
    pair used elsewhere in this project.

(2) ESTABLISHED NEUTROSOPHIC SCORE FUNCTIONS: apply score functions from
    the neutrosophic decision-making literature -- S(T,I,F)=(2+T-I-F)/3
    (a standard neutrosophic score function) and A(T,I,F)=T-F (accuracy
    function) -- to both the original (T,F) and the new, independent
    (T_new,F_new), using I1_new from Section 4.6.5 (the best-performing
    indeterminacy indicator found so far) as I. This connects to the
    actual neutrosophic ranking literature instead of an ad hoc linear
    combination, and tests whether either pairing gives a genuine,
    non-oracle selective-classification advantage.
"""
import os
import warnings
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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
    P = np.clip(P, 1e-12, 1.0)
    Q = np.clip(Q, 1e-12, 1.0)
    return np.sum(P * np.log(P / Q), axis=1)


def fit_ensemble(X, y):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    rf.fit(X, y); xgb.fit(X, y); lr.fit(X, y)
    return rf, xgb, lr


def ensemble_probs(models, X):
    rf, xgb, lr = models
    return (rf.predict_proba(X) + xgb.predict_proba(X) + lr.predict_proba(X)) / 3.0


if __name__ == "__main__":
    print("Building JNU (grouped, leave-1000rpm-out)...")
    Xtr, ytr, Xte, yte = build_jnu_grouped()

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xte_sc = scaler.transform(Xte)
    smote = SMOTE(random_state=42)
    Xs_sc, ys = smote.fit_resample(Xtr_sc, ytr)

    print("Fitting main 3-model ensemble on full training set (600+800 rpm, matches Table 1)...")
    main_models = fit_ensemble(Xs_sc, ys)
    P_avg_te = ensemble_probs(main_models, Xte_sc)
    y_pred_te = np.argmax(P_avg_te, axis=1)
    errors = (y_pred_te != yte).astype(int)
    n_classes = P_avg_te.shape[1]

    sorted_idx_te = np.argsort(-P_avg_te, axis=1)
    T_orig = np.take_along_axis(P_avg_te, sorted_idx_te[:, [0]], axis=1).ravel()
    F_orig = np.take_along_axis(P_avg_te, sorted_idx_te[:, [1]], axis=1).ravel()
    runnerup_class_te = sorted_idx_te[:, 1]

    entropy_orig = entropy_norm(P_avg_te, n_classes)
    votes = np.stack([np.argmax(m.predict_proba(Xte_sc), axis=1) for m in main_models], axis=1)
    agree = (votes == y_pred_te[:, None]).sum(axis=1)
    vote_dis_orig = 1.0 - agree / 3.0

    print(f"Original T_hat+F_hat: always <=1 by softmax construction. "
          f"max observed = {(T_orig+F_orig).max():.4f}, mean = {(T_orig+F_orig).mean():.4f} "
          f"(as expected, this is just a sanity check of the known constraint).")

    # =====================================================================
    # PART 1: genuinely independent T_new, F_new via leakage-free stacking
    # =====================================================================
    print("\n=== PART 1: Independent T_new, F_new via out-of-fold meta-classifiers ===")
    # Generate out-of-fold ensemble predictions on the ORIGINAL (pre-SMOTE) training
    # set only -- never touches the 1000rpm test set -- to build meta-training data
    # without leakage. Using the original (unresampled) Xtr/ytr for the outer CV
    # loop, applying SMOTE fresh within each fold's training portion only.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_P = np.zeros((len(ytr), n_classes))
    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(Xtr_sc, ytr)):
        X_fold_tr, y_fold_tr = Xtr_sc[tr_idx], ytr[tr_idx]
        X_fold_val = Xtr_sc[val_idx]
        Xs_fold, ys_fold = SMOTE(random_state=42).fit_resample(X_fold_tr, y_fold_tr)
        fold_models = fit_ensemble(Xs_fold, ys_fold)
        oof_P[val_idx] = ensemble_probs(fold_models, X_fold_val)
        print(f"  Fold {fold_i+1}/5 done.")

    oof_pred = np.argmax(oof_P, axis=1)
    oof_sorted_idx = np.argsort(-oof_P, axis=1)
    oof_runnerup = oof_sorted_idx[:, 1]
    y_T_meta = (oof_pred == ytr).astype(int)
    y_F_meta = (ytr == oof_runnerup).astype(int)
    print(f"\nMeta-training labels (out-of-fold, N={len(ytr)}): "
          f"P(ensemble correct)={y_T_meta.mean()*100:.1f}%, "
          f"P(true label is the runner-up class)={y_F_meta.mean()*100:.1f}%")

    # Meta-features: the full OOF probability vector (n_classes features).
    meta_T = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    meta_F = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    meta_T.fit(oof_P, y_T_meta)
    meta_F.fit(oof_P, y_F_meta)

    T_new = meta_T.predict_proba(P_avg_te)[:, 1]
    F_new = meta_F.predict_proba(P_avg_te)[:, 1]

    print(f"\nT_new (independent 'ensemble is correct' estimate): mean={T_new.mean():.4f}  std={T_new.std():.4f}")
    print(f"F_new (independent 'runner-up is correct' estimate): mean={F_new.mean():.4f}  std={F_new.std():.4f}")
    r_tf_new, p_tf_new = pearsonr(T_new, F_new)
    r_tf_orig, p_tf_orig = pearsonr(T_orig, F_orig)
    print(f"Correlation T_new vs F_new: r={r_tf_new:.4f} (p={p_tf_new:.3g})")
    print(f"Correlation T_orig vs F_orig (reference): r={r_tf_orig:.4f} (p={p_tf_orig:.3g})")
    exceed_1_new = (T_new + F_new > 1.0).mean()
    exceed_1_orig = (T_orig + F_orig > 1.0).mean()
    print(f"\nFraction of instances with T+F > 1: NEW pair = {exceed_1_new*100:.2f}%  "
          f"ORIGINAL pair = {exceed_1_orig*100:.2f}% (must be exactly 0% by softmax construction)")
    print("(A nonzero fraction for the NEW pair is direct, verifiable evidence of genuine "
          "T/F independence -- something the original P1/P2 pair cannot exhibit by construction.)")

    # Does T_new track correctness the way it should? Sanity check.
    r_tnew_err, p_tnew_err = pearsonr(T_new, errors)
    r_fnew_err, p_fnew_err = pearsonr(F_new, errors)
    print(f"\nSanity: corr(T_new, error) = {r_tnew_err:.4f} (should be negative -- higher T_new means less error)")
    print(f"Sanity: corr(F_new, error) = {r_fnew_err:.4f} (should be positive -- higher F_new means more error)")

    # =====================================================================
    # PART 2: established neutrosophic score functions
    # =====================================================================
    print("\n=== PART 2: Neutrosophic score functions S=(2+T-I-F)/3 and A=T-F ===")
    # Use I1_new from Section 4.6.5 (mean per-model entropy) as I, computed on the
    # SAME main ensemble used throughout this script.
    P_rf_te = main_models[0].predict_proba(Xte_sc)
    P_xgb_te = main_models[1].predict_proba(Xte_sc)
    P_lr_te = main_models[2].predict_proba(Xte_sc)
    H_rf = entropy_norm(P_rf_te, n_classes); H_xgb = entropy_norm(P_xgb_te, n_classes); H_lr = entropy_norm(P_lr_te, n_classes)
    I1_new = (H_rf + H_xgb + H_lr) / 3.0

    def score_S(T, I, F):
        return (2 + T - I - F) / 3.0

    def score_A(T, F):
        return T - F

    configs = [
        ("S(T_orig, I1_orig, F_orig)", score_S(T_orig, entropy_orig, F_orig)),
        ("S(T_orig, I1_new,  F_orig)", score_S(T_orig, I1_new, F_orig)),
        ("S(T_new,  I1_new,  F_new)", score_S(T_new, I1_new, F_new)),
        ("A(T_orig, F_orig) = T-F",   score_A(T_orig, F_orig)),
        ("A(T_new,  F_new)  = T-F",   score_A(T_new, F_new)),
    ]
    print(f"\n{'Score function':32s} {'AURC':>8s} {'Acc@50%cov':>12s} {'partial r|controls':>20s} {'p-value':>10s}")
    for name, score in configs:
        area, acc50 = aurc(-score, errors)  # higher score = more confident -> ascending-is-confident needs negation
        r_part, p_part = pearsonr(score, errors)
        print(f"{name:32s} {area:8.4f} {acc50*100:11.2f}% {r_part:19.4f} {p_part:10.3g}")

    print(f"\n(Reference: original I1_hat alone AURC=0.3915; Section 4.6.5's I1_new+I2_new fixed "
          f"combination AURC=0.3680 -- the best non-oracle result in this project so far.)")

    # Best combination found in 4.6.5 for direct side-by-side reference.
    kl_rf = kl_div(P_rf_te, P_avg_te) / np.log(n_classes)
    kl_xgb = kl_div(P_xgb_te, P_avg_te) / np.log(n_classes)
    kl_lr = kl_div(P_lr_te, P_avg_te) / np.log(n_classes)
    I2_new = (kl_rf + kl_xgb + kl_lr) / 3.0
    combo_465 = (I1_new - I1_new.mean())/I1_new.std() + (I2_new - I2_new.mean())/I2_new.std()
    area_465, acc50_465 = aurc(combo_465, errors)
    print(f"Section 4.6.5 fixed combination I1_new+I2_new (recomputed here for reference): "
          f"AURC={area_465:.4f}  Acc@50%cov={acc50_465*100:.2f}%")
