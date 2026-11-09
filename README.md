# SafeRoute-AI

A machine-learning research prototype for
predicting UK road accident severity.

## Features

- Road-only accident severity prediction
- Road + vehicle accident severity prediction
- Multiple vehicle inputs
- Missing vehicle information support
- SHAP-based global model explanations

## Technology

Python, Streamlit, LightGBM, Pandas, SHAP.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important limitation

The models predict severity conditional on an
accident having occurred. They do not predict
whether an accident will happen.

Predictions and weighted scores are not
validated for safety-critical decisions.
