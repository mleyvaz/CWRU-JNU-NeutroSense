# -*- coding: utf-8 -*-
"""
Tests whether using I2_hat (decision disagreement) to route already-flagged
instances between a CHEAP, less-reliable intervention (AUDIT) and an
EXPENSIVE, fully-reliable one (REVIEW) reduces total expected cost compared
to (a) routing everyone to REVIEW, or (b) routing to AUDIT/REVIEW in the
same proportion but RANDOMLY rather than using I2_hat -- i.e., isolating
whether I2_hat specifically, not just "sometimes using a cheaper option",
drives any savings.

REVISION NOTE (6th round): a first version of this experiment defined AUDIT
as a flat, always-successful cheap action, which made "audit everything"
the trivial global optimum whenever C_audit < C_review, regardless of
I1_hat or I2_hat -- a degenerate result that did not test the framework at
all. That version instead tied AUDIT's SUCCESS PROBABILITY to I2_hat, this
time correctly acknowledging it as a stated, illustrative modeling
assumption, not a measured quantity.

REVISION NOTE (7th round, terminology fix): an external review correctly
pointed out that the median split used below (I2_hat <= median WITHIN the
flagged, high-entropy subset) is NOT the same as full 3-of-3 base-learner
UNANIMITY, and that calling it "models agree" vs. "models disagree" was
imprecise. Verified directly against the data: of 2,929 flagged instances,
only 43 are truly unanimous (I2_hat=0); the "lower-disagreement half"
(1,997 instances) is 43 unanimous plus 1,954 with exactly 2-of-3 agreement,
and the "higher-disagreement half" (932 instances) is 907 with 1-of-3
agreement plus 25 with 0-of-3 (full disagreement). All labels and comments
below now say "lower/higher-disagreement half", not "agree"/"disagree".

Population: instances FLAGGED for intervention by I1_hat alone (I1_hat >
tau1, the same "ambiguous signal" criterion a single-score policy would
use) are not accepted automatically in any of the three policies compared;
only how they are split between AUDIT and REVIEW differs. The accept-zone
(I1_hat <= tau1) is identical and unaffected across all three policies, so
comparisons isolate the value of I2_hat-based routing specifically.

Policies compared, all applied to the SAME flagged set (I1_hat > tau1):
  ALL-REVIEW      : every flagged instance -> REVIEW (cost C_review, always
                    resolves the case, i.e. incurs no residual error cost).
  RANDOM-SPLIT     : flagged instances split into AUDIT/REVIEW in the same
                    global proportion as I2-SPLIT below, but the assignment
                    is a random permutation (does not use I2_hat) -- control
                    condition isolating whether I2_hat's specific groupings
                    matter, not just "using cheaper audit sometimes".
  I2-SPLIT         : flagged instances with I2_hat <= tau2 (lower-disagreement
                    half, mostly 2-of-3 agreement) -> AUDIT (cost C_audit,
                    succeeds with probability p_audit_low); flagged instances
                    with I2_hat > tau2 (higher-disagreement half, 1-of-3 or
                    0-of-3 agreement) -> REVIEW (cost C_review, always
                    resolves).
For AUDIT, if it does not succeed (probability 1-p_audit_*), the case is
NOT corrected: it still costs C_audit (the audit attempt itself), plus
C_FN if the underlying ensemble prediction for that instance was actually
wrong (an undetected error slips through the failed audit).
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

OUT = os.path.dirname(os.path.abspath(__file__))


def expected_cost_all_review(n_flagged, c_review):
    return n_flagged * c_review


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

    entropy = -np.sum(P_avg * np.log(P_avg + 1e-12), axis=1) / np.log(P_avg.shape[1])
    votes = np.stack([np.argmax(P_rf, axis=1), np.argmax(P_xgb, axis=1), np.argmax(P_lr, axis=1)], axis=1)
    agree = (votes == y_pred[:, None]).sum(axis=1)
    vote_dis = 1.0 - agree / 3.0

    # Flagged set: same I1_hat-based criterion any single-score policy would use.
    # Use the median split from the manuscript's own Section 4.3 protocol.
    tau1 = np.median(entropy)
    flagged = entropy > tau1
    n_flagged = flagged.sum()
    errors_flagged = errors[flagged]
    vote_dis_flagged = vote_dis[flagged]
    print(f"\nFlagged (I1_hat > median): N={n_flagged} of {len(entropy)} "
          f"({n_flagged/len(entropy)*100:.1f}%), error rate among flagged = {errors_flagged.mean()*100:.1f}%")

    is_low_i2 = vote_dis_flagged <= np.median(vote_dis_flagged)
    # IMPORTANT (fixed after 7th adversarial round): "I2_hat <= median WITHIN the
    # already-flagged, high-entropy subset" is NOT the same as full 3-of-3 base-learner
    # unanimity. Verified directly (not just asserted): of the n_flagged instances,
    # only 43 are truly unanimous; the "low I2" bucket below is 1,997 instances, of
    # which 43 are unanimous (3/3) and 1,954 have exactly 2-of-3 agreement. The "high
    # I2" bucket (932) is 907 with 1-of-3 agreement and 25 with 0-of-3 (full disagreement).
    # For reference, over the WHOLE test set (not just flagged), true unanimity (3/3) is
    # 1,655/5,859 = 28.2% of instances with a 28.7% error rate, versus 71.4% error rate
    # among the 4,204 non-unanimous instances -- consistent with Section 4.5 of the
    # manuscript. Do not describe the flagged-set median split below as "agree" vs.
    # "disagree"; it is a relative low-vs-high split on a continuous-looking but
    # actually 4-valued score, mostly contrasting 2-of-3 agreement against 1-of-3/0-of-3.
    n_unanimous_flagged = (vote_dis_flagged == 0).sum()
    print(f"Within flagged set: true unanimity (I2_hat=0, 3/3 agree) = {n_unanimous_flagged} of {n_flagged} "
          f"({n_unanimous_flagged/n_flagged*100:.1f}%) -- most flagged instances are NOT unanimous either way.")
    print(f"Within flagged set: {is_low_i2.sum()} have I2_hat <= median (lower-disagreement half: "
          f"mostly 2-of-3 agreement, only {n_unanimous_flagged} of these are truly unanimous), "
          f"{(~is_low_i2).sum()} have I2_hat > median (higher-disagreement half: 1-of-3 or 0-of-3 agreement)")
    print(f"  Error rate | lower-disagreement half (candidates for cheap audit): {errors_flagged[is_low_i2].mean()*100:.1f}%")
    print(f"  Error rate | higher-disagreement half (routed to full review):     {errors_flagged[~is_low_i2].mean()*100:.1f}%")
    print(f"  For reference, over the FULL test set: unanimous (3/3) error rate = "
          f"{errors[vote_dis==0].mean()*100:.1f}% (N={int((vote_dis==0).sum())}), "
          f"non-unanimous error rate = {errors[vote_dis>0].mean()*100:.1f}% (N={int((vote_dis>0).sum())})")

    scenarios = [
        # (name, C_review, C_audit, C_FN, p_audit_success_low_I2, p_audit_success_high_I2)
        ("S1: audit equally reliable in both I2 halves",     5, 1, 15, 0.90, 0.90),
        ("S2: audit much less reliable in high-I2 half",     5, 1, 15, 0.90, 0.30),
        ("S3: same as S2, MORE reliable in low-I2 half too", 5, 1, 15, 0.95, 0.30),
        ("S4: no reliability gap (control)",                 5, 1, 15, 0.70, 0.70),
    ]

    erg_low = errors_flagged[is_low_i2].sum()
    erg_high = errors_flagged[~is_low_i2].sum()
    n_audit_target = int(is_low_i2.sum())

    print(f"\n{'Scenario':46s} {'ALL-REVIEW':>12s} {'AUDIT-ALL(opt)':>15s} {'AUDIT-ALL(pes)':>15s} {'RANDOM-SPLIT':>14s} {'I2-SPLIT':>10s} {'Sav.vs review':>14s} {'Sav.vs random':>14s}")
    for name, c_review, c_audit, c_fn, p_low, p_high in scenarios:
        cost_all_review = expected_cost_all_review(n_flagged, c_review)

        # I2-SPLIT: audit success probability depends on which I2 group (this IS the mechanism being tested)
        cost_i2split_audit = is_low_i2.sum() * c_audit + (1 - p_low) * erg_low * c_fn
        cost_i2split_review = (~is_low_i2).sum() * c_review
        cost_i2split = cost_i2split_audit + cost_i2split_review

        # AUDIT-ALL reference (7th round, requested): route every flagged instance to
        # AUDIT (never REVIEW), under the scenario's own two possible reliabilities --
        # optimistic (uniformly p_low) and pessimistic (uniformly p_high) -- to show
        # where the cheapest possible blanket policy sits relative to I2-SPLIT.
        cost_audit_all_opt = n_flagged * c_audit + (1 - p_low) * errors_flagged.sum() * c_fn
        cost_audit_all_pes = n_flagged * c_audit + (1 - p_high) * errors_flagged.sum() * c_fn

        # RANDOM-SPLIT control: same audit/review split SIZE as I2-SPLIT, but WHICH
        # instances go to audit is a uniform random draw (ignores I2_hat). Computed as
        # an EXACT expectation (linearity of expectation over a hypergeometric draw),
        # not a Monte Carlo average: each instance has probability n_audit_target/n_flagged
        # of being the one audited, independent of its true I2 group or error status.
        p_audit_draw = n_audit_target / n_flagged
        expected_audit_fail_cost = c_fn * p_audit_draw * ((1 - p_low) * erg_low + (1 - p_high) * erg_high)
        cost_random = n_audit_target * c_audit + (n_flagged - n_audit_target) * c_review + expected_audit_fail_cost

        sav_vs_review = (cost_all_review - cost_i2split) / cost_all_review * 100
        sav_vs_random = (cost_random - cost_i2split) / cost_random * 100
        print(f"{name:46s} {cost_all_review:12.1f} {cost_audit_all_opt:15.1f} {cost_audit_all_pes:15.1f} "
              f"{cost_random:14.1f} {cost_i2split:10.1f} {sav_vs_review:13.2f}% {sav_vs_random:13.2f}%")

    print(f"\n(All costs are expected total cost over the {n_flagged} flagged JNU test instances, "
          f"arbitrary relative units. RANDOM-SPLIT and AUDIT-ALL are exact expectations, not "
          f"simulations. AUDIT-ALL(opt) applies the scenario's low-disagreement-half reliability "
          f"p_low to every flagged instance uniformly; AUDIT-ALL(pes) applies p_high uniformly -- "
          f"neither uses I2_hat at all, so together they bracket what a policy that ignores I2_hat "
          f"entirely, but is optimistic or pessimistic about audit reliability, would cost.)")
