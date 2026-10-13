# Heart Disease Prediction System

A Streamlit demonstration app that predicts the `target` label using a Random Forest model.

## Files required

- `app.py` — user interface and prediction logic
- `heart.csv` — dataset with the 13 expected feature columns and a `target` column
- `train_model.py` — trains and saves the model and preprocessing files
- `heart_model.pkl`, `scaler.pkl`, `selector.pkl` — generated model artifacts
- `requirements.txt` — Python dependencies
- `runtime.txt` — Python runtime for supported hosting environments

## Dataset columns

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`

## Prepare the model files

1. Put your existing `heart.csv` in the same folder as `train_model.py`.
2. Install Python 3.11 and the dependencies:
   `pip install -r requirements.txt`
3. Run:
   `python train_model.py`
4. Confirm these files are created:
   `heart_model.pkl`, `scaler.pkl`, `selector.pkl`
5. Run locally:
   `streamlit run app.py`

## Deploy on Render

Create a **Web Service**, not a Static Site. Connect the GitHub repository and set:

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`

No Publish Directory is needed for a Web Service.

Upload the generated `.pkl` files to the repository along with `app.py` and `requirements.txt`. Include `heart.csv` only if you want the dataset stored in the repository; the deployed app does not need it after the `.pkl` files have been generated.

## Important

The selector is applied before the scaler because the scaler is trained on the selected features. Keep the saved artifacts from the same training run together. Pickle/joblib model files should only be loaded from sources you trust.

This project is for educational demonstration and is not a medical diagnostic tool.
