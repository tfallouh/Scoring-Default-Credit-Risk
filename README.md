# Scoring de risque de défaut de crédit

Projet de modélisation du risque de défaut pour des prêts à la consommation, réalisé dans le cadre d’un travail personnel.  
L’objectif est d’estimer la probabilité de défaut d’un emprunteur (PD) à partir de caractéristiques socio-économiques et du prêt, puis de proposer une décision d’octroi basée sur un seuil ajustable.

## Demo en ligne

Application Streamlit (interface de scoring) :  
[Accéder à l’application](https://scoring-default-credit-risks-isaftexaoz2iwn9kpxqhpu.streamlit.app/)

---

## 1. Contenu du dépôt

- `df_clean.csv` : jeu de données prétraité (features + cible `loan_status`).
- `train_data_dec_tree.py` : script d’entraînement du modèle Decision Tree (pipeline sklearn + GridSearchCV + sélection de seuil).
- `models/credit_tree.joblib` : pipeline entraîné (prétraitement + modèle + méta-infos), utilisé par l’API et Streamlit.
- `app_streamlit.py` : interface Streamlit pour scorer un client ou un fichier CSV.
- `api/main.py` (optionnel) : exemple d’API FastAPI permettant d’exposer le modèle en REST.
- `README.md` : documentation du projet.
- `requirements.txt` : dépendances nécessaires pour exécuter le projet.

---

## 2. Modèle et approche

- Variable cible : `loan_status` (0 = non défaut, 1 = défaut).
- Prétraitement :
  - Imputation médiane pour les variables numériques.
  - Imputation par modalité la plus fréquente + One-Hot Encoding pour les variables catégorielles.
- Modèle principal :
  - `DecisionTreeClassifier` avec `class_weight="balanced"`.
  - Recherche d’hyperparamètres par `GridSearchCV` (Stratified K-Fold, scoring ROC-AUC).
  - Choix d’un seuil de décision optimisé sur le F1-score pour la classe défaut.
- Performances (jeu de test tenu à part) :
  - ROC-AUC ≈ 0.92
  - Bon compromis précision / rappel sur la classe défaut, avec un seuil calibré métier.

---

## 3. Lancer l’application en local

### 3.1. Cloner le dépôt

```bash
git clone https://github.com/TON_COMPTE/scoring-default-credit-risk.git
cd scoring-default-credit-risk
