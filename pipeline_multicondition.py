# -*- coding: utf-8 -*-
"""
Multi-condition robustness check (added 2026-09-17, external review point 4):
the main manuscript results hold out a SINGLE operating condition per dataset
(CWRU: 3 HP load; JNU: 1000 rpm). This script repeats the leave-one-
condition-out protocol for EVERY available condition in turn (CWRU: 0/1/2/3
HP; JNU: 600/800/1000 rpm) and reports ensemble accuracy for each fold, to
show whether the single-fold result generalizes across held-out conditions
or is an outlier. Reuses the exact same feature extraction, scaler-then-
SMOTE order, and RF+XGB+LR ensemble as pipeline_v3_grouped.py; only which
files are held out changes per fold.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

import cwru_neutro_pipeline as cw
import jnu_neutro_pipeline as jn

OUT = os.path.dirname(os.path.abspath(__file__))


def run_fold(X_tr, y_tr, X_te, y_te):
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

    acc = accuracy_score(y_te, y_pred)
    bacc = balanced_accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro")
    acc_rf = accuracy_score(y_te, rf.predict(Xte_sc))
    acc_xgb = accuracy_score(y_te, xgb.predict(Xte_sc))
    acc_lr = accuracy_score(y_te, lr.predict(Xte_sc))
    return dict(acc=acc, bacc=bacc, f1_macro=f1m, acc_rf=acc_rf, acc_xgb=acc_xgb, acc_lr=acc_lr,
                n_test=len(y_te))


# ---------------------------------------------------------------------------
# CWRU: 4 folds, one per load (0/1/2/3 HP)
# ---------------------------------------------------------------------------
CWRU_LOAD_OF_FILE = {
    "97.mat": 0, "98.mat": 1, "99.mat": 2, "100.mat": 3,      # Normal
    "118.mat": 0, "119.mat": 1, "120.mat": 2, "121.mat": 3,   # Ball
    "105.mat": 0, "106.mat": 1, "107.mat": 2, "108.mat": 3,   # Inner
    "130.mat": 0, "131.mat": 1, "132.mat": 2, "133.mat": 3,   # Outer
}


def build_cwru_fold(held_out_load):
    cw.download_files()
    rows_tr, y_tr, rows_te, y_te = [], [], [], []
    for fname, label, lname, _ in cw.CWRU_FILES:
        fpath = os.path.join(cw.DATA_DIR, fname)
        sig = cw.mat_to_de_signal(fpath)
        n = (len(sig) - cw.WINDOW) // cw.STEP + 1
        is_test = CWRU_LOAD_OF_FILE[fname] == held_out_load
        dest_rows, dest_y = (rows_te, y_te) if is_test else (rows_tr, y_tr)
        for i in range(n):
            w = sig[i * cw.STEP: i * cw.STEP + cw.WINDOW]
            dest_rows.append(cw.extract_features(w))
            dest_y.append(label)
    return (np.array(rows_tr), np.array(y_tr), np.array(rows_te), np.array(y_te))


# ---------------------------------------------------------------------------
# JNU: 3 folds, one per speed (600/800/1000 rpm)
# ---------------------------------------------------------------------------
def build_jnu_fold(held_out_speed):
    jn.download_files()
    rows_tr, y_tr, rows_te, y_te = [], [], [], []
    for fname, label, lname in jn.JNU_FILES:
        fpath = os.path.join(jn.DATA_DIR, fname)
        sig = pd.read_csv(fpath, header=None).values.flatten()
        n = (len(sig) - jn.WINDOW) // jn.STEP + 1
        is_test = str(held_out_speed) in fname
        dest_rows, dest_y = (rows_te, y_te) if is_test else (rows_tr, y_tr)
        for i in range(n):
            dest_rows.append(jn.extract_features(sig[i * jn.STEP: i * jn.STEP + jn.WINDOW]))
            dest_y.append(label)
    return (np.array(rows_tr), np.array(y_tr), np.array(rows_te), np.array(y_te))


if __name__ == "__main__":
    print("="*72)
    print("  CWRU multi-condition robustness check (4 folds: hold out each load)")
    print("="*72)
    for load in [0, 1, 2, 3]:
        Xtr, ytr, Xte, yte = build_cwru_fold(load)
        res = run_fold(Xtr, ytr, Xte, yte)
        print(f"  Held out load={load}HP: N_test={res['n_test']}  "
              f"ensemble acc={res['acc']*100:.2f}%  bal.acc={res['bacc']*100:.2f}%  "
              f"macroF1={res['f1_macro']*100:.2f}%  "
              f"(RF={res['acc_rf']*100:.2f}% XGB={res['acc_xgb']*100:.2f}% LR={res['acc_lr']*100:.2f}%)")

    print()
    print("="*72)
    print("  JNU multi-condition robustness check (3 folds: hold out each speed)")
    print("="*72)
    for speed in [600, 800, 1000]:
        Xtr, ytr, Xte, yte = build_jnu_fold(speed)
        res = run_fold(Xtr, ytr, Xte, yte)
        print(f"  Held out speed={speed}rpm: N_test={res['n_test']}  "
              f"ensemble acc={res['acc']*100:.2f}%  bal.acc={res['bacc']*100:.2f}%  "
              f"macroF1={res['f1_macro']*100:.2f}%  "
              f"(RF={res['acc_rf']*100:.2f}% XGB={res['acc_xgb']*100:.2f}% LR={res['acc_lr']*100:.2f}%)")
