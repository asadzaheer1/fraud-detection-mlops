"""
Tasks 2, 3, 4: Data Challenges + Model Complexity + Cost-Sensitive Learning
IEEE CIS Fraud Detection
Run: python train_models.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("FRAUD DETECTION - FULL ML PIPELINE")
print("=" * 60)


# ══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════
def load_data(path="data/"):
    print("\n[1] Loading data...")
    trans = pd.read_csv(f"{path}train_transaction.csv")
    ident = pd.read_csv(f"{path}train_identity.csv")
    df = trans.merge(ident, on="TransactionID", how="left")
    print(f"    Shape: {df.shape}  |  Fraud rate: {df['isFraud'].mean():.2%}")
    return df


# ══════════════════════════════════════════════════════════
# 2. TASK 2: DATA CHALLENGES
# ══════════════════════════════════════════════════════════
def handle_data_challenges(df):
    print("\n[2] Handling data challenges...")

    # ── Missing Values (advanced) ──
    # Drop cols with > 50% missing
    drop_cols = df.columns[df.isnull().mean() > 0.50].tolist()
    df.drop(columns=drop_cols, inplace=True)
    print(f"    Dropped {len(drop_cols)} high-missing columns")

    # Numeric: median imputation
    num_cols = df.select_dtypes(include=np.number).columns.difference(["isFraud", "TransactionID"])
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categorical: mode imputation
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        mode_val = df[c].mode()
        df[c] = df[c].fillna(mode_val[0] if not mode_val.empty else "MISSING")

    # ── High-Cardinality Feature Encoding (Target Encoding) ──
    print(f"    Target-encoding {len([c for c in cat_cols if df[c].nunique() > 20])} high-cardinality cols")
    target = df["isFraud"]
    for c in cat_cols:
        if df[c].nunique() > 20:
            # Target encoding with smoothing
            means   = df.groupby(c)["isFraud"].mean()
            global_mean = target.mean()
            counts  = df.groupby(c)["isFraud"].count()
            smoothing = 10
            smooth_means = (counts * means + smoothing * global_mean) / (counts + smoothing)
            df[c] = df[c].map(smooth_means).fillna(global_mean)
        else:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))

    # Drop ID columns
    df.drop(columns=["TransactionID"], errors="ignore", inplace=True)

    # Feature engineering
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"]    = np.log1p(df["TransactionAmt"])
        df["TransactionAmt_scaled"] = (df["TransactionAmt"] - df["TransactionAmt"].mean()) / df["TransactionAmt"].std()

    print(f"    Final shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════
# 3. SPLIT
# ══════════════════════════════════════════════════════════
def split_data(df):
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ══════════════════════════════════════════════════════════
# 4. IMBALANCE STRATEGIES (Task 2 comparison)
# ══════════════════════════════════════════════════════════
def apply_imbalance_strategy(X_train, y_train, strategy="smote"):
    print(f"\n[Imbalance] Strategy: {strategy.upper()}")
    if strategy == "smote":
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X_train, y_train)
    elif strategy == "undersample":
        us = RandomUnderSampler(random_state=42)
        X_res, y_res = us.fit_resample(X_train, y_train)
    elif strategy == "class_weight":
        X_res, y_res = X_train.copy(), y_train.copy()
    else:
        X_res, y_res = X_train.copy(), y_train.copy()
    print(f"    Class distribution → 0:{(y_res==0).sum()}  1:{(y_res==1).sum()}")
    return X_res, y_res


# ══════════════════════════════════════════════════════════
# 5. TASK 3+4: TRAIN ALL MODELS (Standard & Cost-Sensitive)
# ══════════════════════════════════════════════════════════
def train_all_models(X_train, y_train, X_test, y_test):
    results = {}
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # ── XGBoost (Standard) ──
    print("\n[Train] XGBoost - Standard")
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                           use_label_encoder=False, eval_metric="auc",
                           random_state=42, n_jobs=-1)
    m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    results["XGBoost_Standard"] = evaluate_model(m, X_test, y_test)
    joblib.dump(m, "outputs/xgb_standard.pkl")

    # ── XGBoost (Cost-Sensitive) ──
    print("[Train] XGBoost - Cost-Sensitive")
    m_cs = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                              scale_pos_weight=scale_pos,
                              use_label_encoder=False, eval_metric="auc",
                              random_state=42, n_jobs=-1)
    m_cs.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    results["XGBoost_CostSensitive"] = evaluate_model(m_cs, X_test, y_test)
    joblib.dump(m_cs, "outputs/xgb_costsensitive.pkl")

    # ── LightGBM (Standard) ──
    print("[Train] LightGBM - Standard")
    lgb_m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                random_state=42, n_jobs=-1)
    lgb_m.fit(X_train, y_train)
    results["LightGBM_Standard"] = evaluate_model(lgb_m, X_test, y_test)
    joblib.dump(lgb_m, "outputs/lgb_standard.pkl")

    # ── LightGBM (Cost-Sensitive) ──
    print("[Train] LightGBM - Cost-Sensitive")
    lgb_cs = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    lgb_cs.fit(X_train, y_train)
    results["LightGBM_CostSensitive"] = evaluate_model(lgb_cs, X_test, y_test)
    joblib.dump(lgb_cs, "outputs/lgb_costsensitive.pkl")

    # ── Hybrid: RF Feature Selection + Logistic Regression ──
    print("[Train] Hybrid: RF + LR")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances  = rf.feature_importances_
    top_mask     = importances > importances.mean()
    top_features = X_train.columns[top_mask].tolist()
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train[top_features], y_train)
    results["Hybrid_RF_LR"] = evaluate_model(lr, X_test[top_features], y_test)
    joblib.dump((top_features, lr), "outputs/hybrid_model.pkl")

    return results, m_cs  # return best model for SHAP


def evaluate_model(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc   = roc_auc_score(y_test, proba)
    rep   = classification_report(y_test, preds, output_dict=True)
    cm    = confusion_matrix(y_test, preds)

    tn, fp, fn, tp = cm.ravel()
    fraud_loss      = fn * 500   # avg fraud = $500
    false_alarm_cost = fp * 10   # cost of investigating = $10

    return {
        "auc":             round(auc, 4),
        "precision":       round(rep["1"]["precision"], 4),
        "recall":          round(rep["1"]["recall"], 4),
        "f1":              round(rep["1"]["f1-score"], 4),
        "confusion_matrix": cm.tolist(),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "fraud_loss_usd":  int(fraud_loss),
        "false_alarm_cost_usd": int(false_alarm_cost),
        "total_business_cost":  int(fraud_loss + false_alarm_cost),
    }


# ══════════════════════════════════════════════════════════
# 6. PRINT COMPARISON TABLE
# ══════════════════════════════════════════════════════════
def print_results(results):
    print("\n" + "=" * 80)
    print(f"{'Model':<30} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Business Cost':>15}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<30} {m['auc']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} ${m['total_business_cost']:>13,}")
    print("=" * 80)
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to outputs/results.json")


# ══════════════════════════════════════════════════════════
# 7. PLOT CONFUSION MATRICES
# ══════════════════════════════════════════════════════════
def plot_confusion_matrices(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (name, m) in enumerate(results.items()):
        if i >= len(axes):
            break
        cm = np.array(m["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        axes[i].set_title(f"{name}\nAUC={m['auc']:.4f}  Recall={m['recall']:.4f}")
        axes[i].set_ylabel("Actual")
        axes[i].set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Confusion matrices saved → outputs/confusion_matrices.png")


# ══════════════════════════════════════════════════════════
# 8. TASK 9: SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════
def shap_analysis(model, X_test, sample_size=500):
    print("\n[SHAP] Computing feature importance...")
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig("outputs/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ SHAP plots saved → outputs/shap_summary.png, outputs/shap_importance.png")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data()
    df = handle_data_challenges(df)
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n── Imbalance Strategy Comparison ──")
    for strategy in ["smote", "undersample"]:
        X_res, y_res = apply_imbalance_strategy(X_train, y_train, strategy)

    # Use SMOTE for final training
    X_res, y_res = apply_imbalance_strategy(X_train, y_train, "smote")

    results, best_model = train_all_models(X_res, y_res, X_test, y_test)
    print_results(results)
    plot_confusion_matrices(results)
    shap_analysis(best_model, X_test)

    print("\n✅ All tasks complete. Check outputs/ folder.")