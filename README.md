# NG NTB Stop Rate Predictor

A lightweight Streamlit app for internal treasury scenario testing around Nigerian Treasury Bill auction stop rates.

## What is in this repo

- `app.py` — current deployed app
- `app_v2.py` — improved app preview with better resilience and export support
- `requirements.txt` — current dependency file
- `requirements_v2.txt` — recommended dependency file aligned to the saved model environment
- `runtime.txt` — Python runtime for deployment
- `gti_ntb_v5_91D.pkl`, `gti_ntb_v5_182D.pkl`, `gti_ntb_v5_364D.pkl` — tenor-specific trained model files

## What the app does

The app predicts stop rates for:
- 91-day NTB
- 182-day NTB
- 364-day NTB

It uses a small set of engineered pre-auction features, including:
- recent stop-rate lags
- moving average of prior stop rates
- auction offer size and change in supply
- prior bid cover
- secondary market rate and 5-day change
- system liquidity
- MPR
- inflation

## Recommended next deployment path

1. Replace `app.py` with `app_v2.py`
2. Replace `requirements.txt` with `requirements_v2.txt`
3. Redeploy the Streamlit app
4. Confirm all three model pickle files are in the project root

## Why `requirements_v2.txt` matters

The saved model files appear to have been built with a newer scikit-learn version than the one pinned in the current `requirements.txt`.

Recommended package line:
- `scikit-learn==1.6.1`

This should reduce model loading and prediction compatibility issues.

## Improvements added in `app_v2.py`

- app no longer stops completely when one model file is missing
- clearer environment diagnostics in the sidebar
- reset-to-default button
- input validation before prediction
- CSV export for prediction results
- cleaner deployment notes

## Local run

```bash
pip install -r requirements_v2.txt
streamlit run app_v2.py
```

## Deployment note

When promoting the improved version, rename:
- `app_v2.py` -> `app.py`
- `requirements_v2.txt` -> `requirements.txt`

## Important usage note

This tool is for internal treasury decision support only. It should be used alongside market colour, liquidity assessment, auction supply context, and desk judgement.
