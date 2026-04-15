"""
train.py — Train a HistGradientBoostingClassifier on 429 sequence features
(20 AA composition + 9 physicochemical + 400 dipeptide), tune with
GridSearchCV, evaluate on held-out test set, and save inference artefacts.

Saves:
    models/gb_model.joblib          — trained classifier
    models/scaler.joblib            — fitted StandardScaler
    models/selector.joblib          — fitted SelectKBest (top 80 features)
    models/selected_features.json   — feature name lists for inference
    models/metrics.json             — accuracy, F1, ROC-AUC on test set

Run from the project root:
    python src/train.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

#  Paths 
PROJECT_ROOT  = Path(__file__).parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features.csv"
MODELS_DIR    = PROJECT_ROOT / "models"

K_FEATURES = 80


def main():
    print("=" * 60)
    print("Protein Thermostability — Model Training")
    print(f"Features : 429 total -> top {K_FEATURES} selected")
    print("Model    : HistGradientBoostingClassifier + GridSearchCV")
    print("=" * 60)

    # ── Step 1: Load data 
    if not FEATURES_PATH.exists():
        print(f"ERROR: {FEATURES_PATH} not found.")
        print("Run these first:")
        print("  python data/download_data.py")
        print("  python src/feature_extraction.py")
        sys.exit(1)

    df = pd.read_csv(FEATURES_PATH).dropna()
    X  = df.drop("label", axis=1)
    y  = df["label"]
    print(f"\nLoaded {len(df)} samples  |  {X.shape[1]} features")
    print(f"Class balance: {y.value_counts().to_dict()}")

    #  Step 2: Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    #  Step 3: Standardise 
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    #  Step 4: Feature selection 
    selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
    X_train_sel = selector.fit_transform(X_train_sc, y_train)
    X_test_sel  = selector.transform(X_test_sc)

    selected = X.columns[selector.get_support()].tolist()
    n_dp = sum(1 for f in selected if f.startswith("DP_"))
    n_aa = sum(1 for f in selected if f.startswith("AA_"))
    n_ph = K_FEATURES - n_dp - n_aa
    print(f"\nSelected {K_FEATURES}: "
          f"{n_dp} dipeptide | {n_aa} AA-comp | {n_ph} physicochemical")

    #  Step 5: GridSearchCV 
    print("\nRunning GridSearchCV (5-fold, scoring=f1) ...")
    param_grid = {
        "max_iter":         [200, 400],
        "learning_rate":    [0.05, 0.1],
        "max_depth":        [4, 6, None],
        "min_samples_leaf": [10, 20],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        HistGradientBoostingClassifier(random_state=42),
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train_sel, y_train)
    print(f"\nBest params : {grid.best_params_}")
    print(f"Best CV F1  : {grid.best_score_:.4f}")

    #  Step 6: Cross-validate best model 
    print("\nRunning 5-fold CV on best estimator ...")
    cv_scores = cross_val_score(
        grid.best_estimator_,
        X_train_sel, y_train,
        cv=cv, scoring="f1", n_jobs=-1,
    )
    print(f"CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    #  Step 7: Retrain on full training set 
    print("\nTraining final model on full training set ...")
    model = HistGradientBoostingClassifier(
        random_state=42, **grid.best_params_
    )
    model.fit(X_train_sel, y_train)
    print("  Done.")    

    #  Step 8: Evaluate on held-out test set 
    y_pred = model.predict(X_test_sel)
    y_prob = model.predict_proba(X_test_sel)[:, 1]

    accuracy = round(accuracy_score(y_test, y_pred), 4)
    f1       = round(f1_score(y_test, y_pred), 4)
    roc_auc  = round(roc_auc_score(y_test, y_prob), 4)

    print("\n--- Test Set Results ---")
    print(classification_report(
        y_test, y_pred,
        target_names=["Mesophilic", "Thermophilic"],
    ))
    print(f"Accuracy : {accuracy}")
    print(f"F1       : {f1}")
    print(f"ROC-AUC  : {roc_auc}")

    confs = np.maximum(y_prob, 1 - y_prob) * 100
    print(f"\nConfidence distribution:")
    for t in [95, 90, 80, 70]:
        pct = (confs >= t).mean() * 100
        print(f"  >= {t}%  : {pct:.1f}% of test samples")

    #  Step 9: Save artefacts 
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model,    MODELS_DIR / "gb_model.joblib")
    joblib.dump(scaler,   MODELS_DIR / "scaler.joblib")
    joblib.dump(selector, MODELS_DIR / "selector.joblib")

    with open(MODELS_DIR / "selected_features.json", "w") as f:
        json.dump({
            "all_features":      X.columns.tolist(),
            "selected_features": selected,
        }, f, indent=2)

    metrics = {
        "model":               "HistGradientBoostingClassifier",
        "best_params":         grid.best_params_,
        "n_features_input":    X.shape[1],
        "n_features_selected": int(K_FEATURES),
        "dataset_size":        len(df),
        "train_size":          len(X_train),
        "test_size":           len(X_test),
        "cv_f1_mean":    round(float(cv_scores.mean()), 4),
        "cv_f1_std":     round(float(cv_scores.std()),  4),
        "test_accuracy": accuracy,
        "test_f1":       f1,
        "test_roc_auc":  roc_auc,
        "pct_conf_95":   round(float((confs >= 95).mean() * 100), 1),
        "pct_conf_90":   round(float((confs >= 90).mean() * 100), 1),
        "pct_conf_80":   round(float((confs >= 80).mean() * 100), 1),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — Saved artefacts:")
    for p in sorted(MODELS_DIR.iterdir()):
        print(f"  {p.name:<34} {p.stat().st_size/1024:>8.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
