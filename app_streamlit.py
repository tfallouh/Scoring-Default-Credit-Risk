import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# ---------------------------
# Config / constants
# ---------------------------
MODEL_PATH = Path("models/credit_tree.joblib")  # même chemin que dans ton script de train
APP_TITLE = "Credit Risk Scoring — Decision Tree"

# Valeurs proposées pour les selectboxes (adaptées au dataset public classique)
HOME_OWNERSHIP_OPTS = ["RENT", "OWN", "MORTGAGE", "OTHER"]
LOAN_INTENT_OPTS = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
LOAN_GRADE_OPTS = ["A", "B", "C", "D", "E", "F", "G"]

# Colonnes attendues par le pipeline (dans l’ordre d’origine du CSV)
EXPECTED_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

# ---------------------------
# Utils
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Train and save the model first (train_decision_tree.py)."
        )
    bundle = load(MODEL_PATH)
    # bundle was saved as: {"pipeline": pipeline, "feature_order": X.columns.tolist(), "threshold": 0.5}
    pipeline = bundle["pipeline"]
    feature_order = bundle.get("feature_order", EXPECTED_COLUMNS)
    default_threshold = float(bundle.get("threshold", 0.5))
    return pipeline, feature_order, default_threshold

def predict_one(pipeline, feature_order, payload_dict):
    """Build a 1-row DataFrame in the expected column order and predict proba + class."""
    row = {col: payload_dict.get(col, None) for col in feature_order}
    X = pd.DataFrame([row], columns=feature_order)
    proba = float(pipeline.predict_proba(X)[:, 1][0])  # P(default = 1)
    return proba

def predict_batch(pipeline, feature_order, df):
    """Align columns and return probabilities for a batch CSV."""
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in uploaded CSV: {missing}")
    X = df[feature_order]
    probas = pipeline.predict_proba(X)[:, 1]
    return probas

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="centered")
st.title(APP_TITLE)

st.markdown(
    "This app loads a trained Decision Tree pipeline (imputation + one-hot + model) "
    "and returns default probability and a binary decision with a configurable threshold."
)

# Load model
with st.spinner("Loading model..."):
    try:
        pipeline, feature_order, default_threshold = load_artifacts()
    except Exception as e:
        st.error(str(e))
        st.stop()

# Tabs
tab_single, tab_batch, tab_info = st.tabs(["Single prediction", "Batch scoring (CSV)", "Model info"])

# ---------------------------
# Single prediction tab
# ---------------------------
with tab_single:
    st.subheader("Client data")

    # Form
    with st.form("single_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            person_age = st.number_input("Age (years)", min_value=18, max_value=100, value=30, step=1)
            person_income = st.number_input("Annual income", min_value=0, value=50000, step=500)
            person_home_ownership = st.selectbox("Home ownership", options=HOME_OWNERSHIP_OPTS, index=0)
            person_emp_length = st.number_input("Employment length (years)", min_value=0.0, value=5.0, step=0.5)
            loan_intent = st.selectbox("Loan intent", options=LOAN_INTENT_OPTS, index=0)
            loan_grade = st.selectbox("Loan grade", options=LOAN_GRADE_OPTS, index=2)  # default "C"
        with col2:
            loan_amnt = st.number_input("Loan amount", min_value=0, value=10000, step=500)
            loan_int_rate = st.number_input("Interest rate (%)", min_value=0.0, value=4.0, step=0.1)
            loan_percent_income = st.number_input("Loan percent income", min_value=0.0, max_value=5.0, value=0.2, step=0.01)
            cb_person_default_on_file = st.selectbox("Previous default on file (0/1)", options=[0, 1], index=0)
            cb_person_cred_hist_length = st.number_input("Credit history length (years)", min_value=0, value=5, step=1)

        threshold = st.slider("Decision threshold (default class=1 if proba ≥ threshold)", 0.0, 1.0, float(default_threshold), 0.01)

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": float(person_emp_length),
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": float(loan_int_rate),
            "loan_percent_income": float(loan_percent_income),
            "cb_person_default_on_file": int(cb_person_default_on_file),
            "cb_person_cred_hist_length": int(cb_person_cred_hist_length),
        }

        try:
            proba = predict_one(pipeline, feature_order, payload)
            decision = int(proba >= threshold)
            st.success(f"Default probability: {proba:.3f}")
            st.write(f"Decision (1=default, 0=non-default) at threshold {threshold:.2f}: **{decision}**")

            st.caption("Raw features sent to the model:")
            st.json(payload)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------
# Batch tab
# ---------------------------
with tab_batch:
    st.subheader("Upload a CSV for batch scoring")
    st.markdown("The CSV must contain at least these columns (order does not matter):")
    st.code(", ".join(feature_order), language="text")

    file = st.file_uploader("Choose a CSV file", type=["csv"])
    threshold_b = st.slider("Threshold (batch)", 0.0, 1.0, float(default_threshold), 0.01, key="thr_batch")

    if file is not None:
        try:
            df = pd.read_csv(file)
            st.write("Preview:")
            st.dataframe(df.head(10), use_container_width=True)

            probas = predict_batch(pipeline, feature_order, df)
            preds = (probas >= threshold_b).astype(int)

            out = df.copy()
            out["proba_default"] = probas
            out["prediction"] = preds

            st.success("Scoring done.")
            st.dataframe(out.head(20), use_container_width=True)

            # Offer download
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results (CSV)",
                data=csv_bytes,
                file_name="scored_results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Batch scoring failed: {e}")

# ---------------------------
# Info tab
# ---------------------------
with tab_info:
    st.subheader("Model & schema")
    st.write(f"Model file: `{MODEL_PATH}`")
    st.write("Expected feature order (as used during training):")
    st.code(json.dumps(feature_order, indent=2))
    st.write("Notes:")
    st.markdown(
        "- The pipeline includes imputers and One-Hot encoders, so you can pass raw values.\n"
        "- Categorical values not seen during training are safely ignored by the OneHotEncoder (all zeros for that unseen category).\n"
        "- The decision is threshold-based on the predicted probability for class 1 (default)."
    )
