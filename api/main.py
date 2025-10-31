from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
from joblib import load

# Load model artifact
ARTIFACT = load(Path(__file__).resolve().parents[1] / "models" / "credit_tree.joblib")
PIPE = ARTIFACT["pipeline"]
THRESHOLD = float(ARTIFACT.get("threshold", 0.5))

# Expected feature order (saved in the artifact)
FEATURE_ORDER = ARTIFACT["feature_order"]

app = FastAPI(title="Credit Risk API", version="1.0")

class Applicant(BaseModel):
    person_age: conint(ge=18, le=120)
    person_income: conint(ge=0)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: confloat(ge=0)
    loan_intent: Literal["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"]
    loan_grade: Literal["A","B","C","D","E","F","G"]
    loan_amnt: conint(ge=0)
    loan_int_rate: confloat(ge=0)
    loan_percent_income: confloat(ge=0, le=1.0)
    cb_person_default_on_file: conint(ge=0, le=1)     # 0/1 already in df_clean
    cb_person_cred_hist_length: conint(ge=0)

@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD, "features": FEATURE_ORDER}

@app.post("/predict")
def predict(applicant: Applicant):
    try:
        # Build a single-row DataFrame in the exact feature order
        row = pd.DataFrame([[getattr(applicant, f) for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
        proba = float(PIPE.predict_proba(row)[:, 1][0])      # PD = P(default=1 | X)
        decision = int(proba >= THRESHOLD)                   # 1 = default predicted (reject), 0 = accept
        return {
            "prob_default": proba,
            "threshold_used": THRESHOLD,
            "decision": decision,
            "decision_label": "reject" if decision == 1 else "approve"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
