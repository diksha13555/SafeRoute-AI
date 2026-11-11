# SafeRoute-AI

SafeRoute-AI is an educational injury-severity classifier. Given a recorded UK road accident, the deployed V3 model estimates the conditional probability that the injury is **KSI** (killed or seriously injured) rather than **Slight**.

It does not estimate whether a crash will happen, and it is not an emergency warning system.

## Features

- Scenario form for road, weather, lighting, junction, time, place, and the ego vehicle
- Decision threshold selector
  - **Standard:** 0.175
  - **Safety-focused:** 0.145
- Animated fictional incident for the classified result, with a separate collision-free journey
- SHAP explanation of the current scenario on the deployed model
- Comparison of an alternative weather and road surface

The probability stays the same when the threshold changes. Only the KSI or Slight classification changes.

## Entry point

Streamlit Community Cloud should use this repository root and this main file:

```text
app.py
```

Files required to run the dashboard:

| File | Role |
|---|---|
| `app.py` | Streamlit application |
| `v3_predict.py` | Feature row used by the saved model |
| `v3_explain.py` | SHAP contributions for the current row |
| `models/saferoute_v3_precrash_2026.pkl` | Deployed V3 checkpoint (2.7 MB) |
| `simulator.html` | Fictional road animation |
| `hero.svg` | Page header image |
| `requirements.txt` | Python dependencies |
| `.python-version` | Python 3.13 |

Paths inside `app.py` are resolved from the project directory, so the same files work on Linux and Windows.

## Machine-learning methodology

V3 is a LightGBM binary classifier with 35 pre-crash features. Inputs are limited to information that can be known before travel: road and junction attributes, weather, lighting, time, location, and the ego vehicle and driver profile. Engineered flags such as high speed, dark rural roads, adverse weather, and a wet surface are derived from those inputs.

| Split | Years | Use |
|---|---|---|
| Training | 2009–2013 | Fit the model |
| Validation | 2014–2015 | Report metrics and choose 0.175 by KSI F1 |
| Holdout | 2016–2017 | Not used to choose features, hyperparameters, or either dashboard threshold |

The positive class is KSI. On the 286,378 validation rows, 43,988 accidents are KSI (prevalence 0.1536). Predicting Slight for every row would score 84.64% accuracy and detect no KSI cases.

## Verified V3 validation metrics

These figures are from the saved V3 checkpoint on 2014–2015. The dashboard rounds the Standard-mode rates to two decimal places.

### Standard mode, threshold 0.175

| Metric | Value |
|---|---|
| Accuracy | 0.720349 (72.03%) |
| KSI precision | 0.266011 (26.60%) |
| KSI recall | 0.466468 (46.65%) |
| KSI F1 | 0.338810 |
| ROC-AUC | 0.671150 |
| PR-AUC | 0.295021 |

Confusion matrix: TN 185,773 · FP 56,617 · FN 23,469 · TP 20,519.

### Safety-focused mode, threshold 0.145

Measured on the same validation rows and the same model scores. This cutoff was the highest-precision point on a 0.05–0.30 grid that still reached at least 60% KSI recall.

| Metric | Value |
|---|---|
| Accuracy | 0.625844 (62.58%) |
| KSI precision | 0.230795 (23.08%) |
| KSI recall | 0.615509 (61.55%) |
| KSI F1 | 0.335710 |

Confusion matrix: TN 152,153 · FP 90,237 · FN 16,913 · TP 27,075.

Safety-focused mode detects more KSI cases and raises the false-alarm count. About three in four accidents it flags are still slight injuries. ROC-AUC and PR-AUC do not change, because the threshold does not change the model score.

The original model threshold remains 0.175.

## SHAP explainability

The section **Understanding the AI Prediction** uses `shap.TreeExplainer` on the loaded V3 booster and the exact scenario row. Contributions are in the model's raw score. For this binary LightGBM model, that raw score is a log-odds. The displayed probability is the logistic function of the raw score.

The chart shows the five features with the largest absolute contribution. Amber bars increase the raw KSI score. Teal bars decrease it. The values are local contributions for the current inputs. They are not percentage-point changes, and they are not proven accident causes. Changing the decision threshold does not change them.

## Local installation

Python 3.13 is the version used to pin the requirements.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file to `app.py` and the Python version to **3.13**.
4. The dependency file is `requirements.txt`. No secrets and no extra system packages are required.

The V3 checkpoint is included. V2 and V4 checkpoints are not.

## Limitations

- The estimate is conditional on an accident already being recorded.
- Validation KSI recall at the original threshold is 46.65%. Standard mode misses more than half of KSI cases.
- Safety-focused mode raises KSI recall to 61.55% and lowers accuracy to 62.58%.
- Ranking quality is modest: validation ROC-AUC is 0.6712 and PR-AUC is 0.2950.
- Latitude and longitude can receive SHAP contribution because the trees split on coordinates. That is a model contribution, not a claim that a coordinate caused a crash.
- The fictional collision is a teaching illustration built from the selected scenario. The model does not identify an accident cause.
- Training archives are not in this repository. `train_saferoute_v3.py` looks for `Accident_Information.csv.zip` and `Vehicle_Information.csv.zip` in the parent of the project directory.

## Educational use

SafeRoute-AI is a research prototype for teaching injury severity among historical UK accident records. Do not use it for routing, enforcement, clinical decisions, or real-time safety warnings.
