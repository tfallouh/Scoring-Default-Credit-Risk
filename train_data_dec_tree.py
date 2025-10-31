# train_decision_tree.py
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_recall_fscore_support,
)

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "df_clean.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "credit_tree.joblib"


def best_threshold_by_f1(y_true, y_proba, grid=np.linspace(0.05, 0.95, 37)):
    """
    Parcourt une grille de seuils et retourne celui qui maximise le F1,
    ainsi que (precision, recall, f1) au seuil retenu.
    """
    best = None
    for t in grid:
        y_hat = (y_proba >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_hat, average="binary", zero_division=0
        )
        cand = {"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f1)}
        if best is None or cand["f1"] > best["f1"]:
            best = cand
    return best


def main():
    # 1) Load
    df = pd.read_csv(DATA_PATH)

    # 2) Target / Features
    target_col = "loan_status"
    y = df[target_col].astype(int)

    # 2.1) (Optionnel mais recommandé) créer une feature d'interaction
    #      prev_default (0/1) * credit history length
    if "cb_person_default_on_file" in df.columns and "cb_person_cred_hist_length" in df.columns:
        df["interaction_prevdef_hist"] = (
            df["cb_person_default_on_file"].astype(float) * df["cb_person_cred_hist_length"].astype(float)
        )

    X = df.drop(columns=[target_col])

    # 3) Column groups
    cat_cols = ["person_home_ownership", "loan_intent", "loan_grade"]
    # numeric = toutes les autres colonnes (y compris l'interaction si créée)
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 4) Preprocessor (Tree : pas de scaling, juste imputation + one-hot pour cat)
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ]
    )

    # 5) Base model (un peu régularisé + équilibrage)
    base_clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=8,
        min_samples_leaf=25,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", base_clf),
    ])

    # 6) Split global train/test (stratifié)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 6.1) Split train/valid pour choisir le seuil sans toucher au test
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # 7) Hyperparameter tuning avec CV stratifiée (score = ROC-AUC)
    param_grid = {
        "model__max_depth": [6, 8, 10, 12, 14, None],
        "model__min_samples_leaf": [5, 10, 25, 50],
        "model__min_samples_split": [2, 5, 10, 25],
        "model__ccp_alpha": [0.0, 1e-4, 5e-4],
        "model__criterion": ["gini", "entropy"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    gs.fit(X_train, y_train)

    print("\nBest CV AUC:", f"{gs.best_score_:.4f}")
    print("Best params:", gs.best_params_)

    best_pipeline = gs.best_estimator_

    # 8) AUC sur validation puis choix du seuil (F1 max sur valid)
    y_proba_valid = best_pipeline.predict_proba(X_valid)[:, 1]
    valid_auc = roc_auc_score(y_valid, y_proba_valid)
    print("Validation ROC-AUC:", f"{valid_auc:.4f}")

    best_thr = best_threshold_by_f1(y_valid, y_proba_valid)
    print("\nBest threshold by F1 on validation:", best_thr)
    chosen_threshold = best_thr["threshold"]

    # 9) Évaluation finale sur test (AUC + report @ seuil choisi)
    y_proba_test = best_pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba_test)
    print("\n=== Test set evaluation ===")
    print("Test ROC-AUC:", f"{test_auc:.4f}")

    y_pred_test = (y_proba_test >= chosen_threshold).astype(int)
    print(f"\nClassification report @ threshold={chosen_threshold:.2f}")
    print(classification_report(y_test, y_pred_test, digits=4))

    # 10) Top features (après One-Hot)
    try:
        model = best_pipeline.named_steps["model"]
        importances = model.feature_importances_

        ohe = best_pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        numeric_feature_names = np.array(num_cols)
        all_feature_names = np.concatenate([numeric_feature_names, cat_feature_names])

        top_idx = np.argsort(importances)[::-1][:15]
        print("\nTop 15 features by importance:")
        for i in top_idx:
            print(f"{all_feature_names[i]:40s} {importances[i]:.4f}")
    except Exception as e:
        print("\n[Warn] Could not extract expanded feature names.")
        print("Reason:", e)

    # 11) Persist
    dump({
        "pipeline": best_pipeline,
        "feature_order": X.columns.tolist(),     # pour API/Streamlit (ordre d'origine)
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
        "threshold": float(chosen_threshold)
    }, MODEL_PATH)
    print(f"\nSaved pipeline -> {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
