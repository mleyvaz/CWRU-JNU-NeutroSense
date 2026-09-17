# CWRU + JNU NeutroSense: Leakage-Free Bearing Fault Detection Pipeline

Reproducible pipeline for the manuscript "Neutrosophic Ensemble Classification
for Uncertainty-Aware Bearing Fault Detection: Evidence from Laboratory and
Variable-Speed Industrial Benchmarks" (Leyva-Vazquez, Rumbaut Rangel,
Cevallos-Torres, Matheu Perez, Smarandache).

## What this is

An RF + XGBoost + Logistic Regression ensemble for bearing fault
classification on two public benchmarks (CWRU, JNU), with a refined
neutrosophic decomposition of the ensemble's soft-voted output into four
operational indicators:

- **T-hat, F-hat**: top-class and best-competitor evidence (not `F = 1-T`)
- **I1-hat (predictive entropy)**: normalized Shannon entropy of the
  ensemble's averaged class-probability distribution
- **I2-hat (decision disagreement)**: fraction of base learners whose own
  prediction disagrees with the ensemble's final vote

Evaluated under a **leave-one-condition-out** protocol: an entire operating
condition (load for CWRU, speed for JNU) is held out per class for testing,
so no window from a held-out condition ever appears in training.

## Key files

| File | Purpose |
|---|---|
| `cwru_neutro_pipeline.py` | CWRU data loading, feature extraction, correct file-to-class mapping |
| `jnu_neutro_pipeline.py` | JNU data loading, feature extraction |
| `pipeline_v3_grouped.py` | Main leave-one-condition-out pipeline (single fold per dataset), full statistical battery |
| `pipeline_multicondition.py` | Robustness check: repeats the protocol holding out every available condition in turn |
| `baseline_comparison_jnu.py` | Selective-classification baseline comparison (T-hat, margin, I1-hat, I2-hat, standalone LR) |
| `generate_paper_v7.py` | Manuscript generator (python-docx) |

## Notable corrections made during development

1. **Window-level data leakage**: an earlier random stratified split let
   overlapping windows from the same recording appear in both train and
   test. Fixed with the leave-one-condition-out design above.
2. **CWRU file-to-class mapping error**: files 105-108 were labeled "Ball"
   and 109-112 "Inner"; per the official CWRU Bearing Data Center catalog,
   both are actually **Inner Race** data (12 kHz and 48 kHz respectively) --
   the same physical fault at two sampling rates, not two classes. The real
   Ball fault (12 kHz) is files 118-121, never previously used. Corrected in
   `cwru_neutro_pipeline.py`.
3. **I1-hat definitional flaw**: an earlier `I1-hat = 1-(P1-P2)` was an exact
   algebraic function of T-hat and F-hat, so it could carry no independent
   information by construction. Replaced with normalized Shannon entropy of
   the full distribution.
4. **SMOTE/scaling order**: SMOTE was applied before feature scaling;
   corrected to scale first, then resample on scaled features only.
5. Several statistical fixes: pooled Cohen's d, correct partial-correlation
   degrees of freedom, controlling for T-hat and F-hat jointly (not F-hat
   alone), and a properly non-degenerate selective risk-coverage report
   (the original accuracy x coverage "utility" was monotonic by construction).

## Running

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn scipy matplotlib
python pipeline_v3_grouped.py          # main results (single fold per dataset)
python pipeline_multicondition.py      # robustness check (all folds)
python baseline_comparison_jnu.py      # selective-classification baselines
```

Raw data (`cwru_raw/`, `jnu_raw/`) is downloaded automatically from the
official CWRU Bearing Data Center and the JNU-Bearing-Dataset GitHub
repository on first run.

## Data sources

- CWRU Bearing Dataset: https://engineering.case.edu/bearingdatacenter
- JNU Bearing Dataset: https://github.com/ClarkGableWang/JNU-Bearing-Dataset

## License

Code released for research reproducibility. See manuscript for citation.
