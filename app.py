# ==========================================
# Crypto Pump Detection Dashboard
# ==========================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

try:
    from sqlalchemy import create_engine
except ImportError:
    create_engine = None


TARGET_COL = "label"
LEAKAGE_COLUMNS = {
    "future_return_5m",
    "is_pump",
    "is_pump_target",
    "return_5m",
    "momentum_5m",
}
ID_COLUMNS = {"timestamp", "symbol"}


st.set_page_config(
    page_title="Crypto Pump Detection",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
<style>
.main { background-color: #0f1117; }
h1, h2, h3 { color: white; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 Crypto Pump Detection Dashboard")
st.caption("Time-aware machine learning pipeline for short-term crypto pump detection")
st.markdown("---")


@st.cache_data(show_spinner=False)
def load_data():
    """Load from MySQL when configured, otherwise fall back to local CSV."""
    csv_path = os.getenv("DATA15K_CSV", "data15k.csv")
    mysql_url = os.getenv("MYSQL_URL")
    mysql_table = os.getenv("MYSQL_TABLE", "data15k")

    if mysql_url and create_engine is not None:
        engine = create_engine(mysql_url)
        return pd.read_sql(f"SELECT * FROM {mysql_table}", engine), f"MySQL table `{mysql_table}`"

    return pd.read_csv(csv_path), f"CSV `{csv_path}`"


def validate_dataset(df):
    required = {"timestamp", TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Dataset is missing required columns: {sorted(missing)}")
        st.stop()

    if df[TARGET_COL].nunique() != 2:
        st.error(f"Target column `{TARGET_COL}` must contain exactly two classes.")
        st.stop()


def preprocess_data(df):
    df = df.copy()
    validate_dataset(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", TARGET_COL])

    # Global chronological order is required for a true time-based holdout split.
    df = df.sort_values("timestamp").reset_index(drop=True)

    y = df[TARGET_COL].astype(int)
    feature_df = df.drop(columns=[TARGET_COL], errors="ignore")
    feature_df = feature_df.drop(columns=list(LEAKAGE_COLUMNS | ID_COLUMNS), errors="ignore")
    feature_df = feature_df.select_dtypes(include=[np.number])
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    valid_index = feature_df.dropna().index
    removed_rows = len(feature_df) - len(valid_index)

    X = feature_df.loc[valid_index]
    y = y.loc[valid_index]
    model_df = df.loc[valid_index].copy()
    meta = model_df[[col for col in ["timestamp", "symbol"] if col in model_df.columns]]

    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    if train_end == 0 or val_end <= train_end or n <= val_end:
        st.error("Not enough clean rows to create train/validation/test splits.")
        st.stop()

    splits = {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "y_val": y.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_test": y.iloc[val_end:],
        "meta": meta,
        "test_context": model_df.iloc[val_end:].reset_index(drop=True),
        "removed_rows": removed_rows,
    }
    return splits


def find_best_threshold(y_true, y_prob, min_precision=0.45, beta=2.0):
    best_threshold = 0.50
    best_score = -1.0

    for threshold in np.arange(0.01, 0.96, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

        if precision >= min_precision and score > best_score:
            best_score = score
            best_threshold = float(threshold)

    if best_score < 0:
        for threshold in np.arange(0.01, 0.96, 0.01):
            y_pred = (y_prob >= threshold).astype(int)
            score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

    return best_threshold


def evaluate_model(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "F2 Score": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Predicted Pump Rate": y_pred.mean(),
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
    }


with st.spinner("Loading dataset..."):
    df, source_name = load_data()

splits = preprocess_data(df)
X_train = splits["X_train"]
y_train = splits["y_train"]
X_val = splits["X_val"]
y_val = splits["y_val"]
X_test = splits["X_test"]
y_test = splits["y_test"]
test_context = splits["test_context"]

st.success("Dataset loaded and time-aware preprocessing completed.")

st.subheader("🧾 Data Audit")
numeric_df = df.select_dtypes(include=[np.number])
inf_values = int(np.isinf(numeric_df).sum().sum()) if not numeric_df.empty else 0
missing_values = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())
time_series = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

audit_df = pd.DataFrame(
    [
        {"Check": "Rows", "Value": f"{len(df):,}"},
        {"Check": "Columns", "Value": f"{df.shape[1]:,}"},
        {"Check": "Symbols", "Value": f"{df['symbol'].nunique():,}" if "symbol" in df.columns else "N/A"},
        {"Check": "Time range", "Value": f"{time_series.min()} to {time_series.max()}"},
        {"Check": "Missing values", "Value": f"{missing_values:,}"},
        {"Check": "Infinite values", "Value": f"{inf_values:,}"},
        {"Check": "Duplicate rows", "Value": f"{duplicate_rows:,}"},
        {"Check": "Leakage columns removed", "Value": ", ".join(sorted(LEAKAGE_COLUMNS))},
    ]
)
st.dataframe(audit_df, width="stretch", hide_index=True)

st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Source", source_name)
with col2:
    st.metric("Rows", f"{len(df):,}")
with col3:
    st.metric("Features Used", X_train.shape[1])
with col4:
    st.metric("Positive Labels", f"{int(df[TARGET_COL].sum()):,}")

label_counts = df[TARGET_COL].value_counts().sort_index()
st.write("Target distribution")
st.bar_chart(label_counts)

with st.expander("Preview cleaned modeling features"):
    st.dataframe(X_train.head())

st.markdown("---")
st.subheader("📦 Train / Validation / Test Split")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info(f"Train: {len(X_train):,}")
with col2:
    st.info(f"Validation: {len(X_val):,}")
with col3:
    st.info(f"Test: {len(X_test):,}")
with col4:
    st.warning(f"Rows removed: {splits['removed_rows']:,}")

st.caption(
    f"Positive rate after cleaning - train: {y_train.mean():.2%}, "
    f"validation: {y_val.mean():.2%}, test: {y_test.mean():.2%}."
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        min_samples_split=5,
        class_weight="balanced_subsample",
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=0.15,
        scale_pos_weight=max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum())),
        reg_alpha=0.5,
        reg_lambda=2,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    ),
}

st.markdown("---")
st.subheader("⚖️ Recall / Precision Trade-off")
col1, col2 = st.columns(2)
with col1:
    min_precision = st.slider(
        "Minimum validation precision",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.05,
        help="Lower this to catch more pump cases. Higher values reduce false alarms but usually lower recall.",
    )
with col2:
    beta = st.selectbox(
        "Threshold objective",
        options=[1.0, 2.0, 3.0],
        index=1,
        format_func=lambda x: {1.0: "F1 balanced", 2.0: "F2 recall-focused", 3.0: "F3 very recall-focused"}[x],
    )

st.caption("Pump detection is imbalanced, so recall and precision move against each other. The threshold is selected on validation data only.")

st.markdown("---")
st.subheader("🤖 Model Training")

results = []
roc_data = {}
pr_data = {}
confusion_data = {}
importance_data = {}
prediction_data = {}

for name, model in models.items():
    with st.spinner(f"Training {name}..."):
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        best_threshold = find_best_threshold(y_val, val_prob, min_precision=min_precision, beta=beta) 

        test_prob = model.predict_proba(X_test)[:, 1]
        val_metrics = evaluate_model(y_val, val_prob, best_threshold)
        test_metrics = evaluate_model(y_test, test_prob, best_threshold)

        results.append(
            {
                "Model": name,
                "Threshold": round(best_threshold, 2),
                "Val F1": round(val_metrics["F1 Score"], 4),
                "Test Accuracy": round(test_metrics["Accuracy"], 4),
                "Test Precision": round(test_metrics["Precision"], 4),
                "Test Recall": round(test_metrics["Recall"], 4),
                "Test F1": round(test_metrics["F1 Score"], 4),
                "Test F2": round(test_metrics["F2 Score"], 4),
                "Test ROC-AUC": round(test_metrics["ROC-AUC"], 4),
                "Test PR-AUC": round(test_metrics["PR-AUC"], 4),
                "Pred Pump %": f"{test_metrics['Predicted Pump Rate']:.2%}",
            }
        )

        fpr, tpr, _ = roc_curve(y_test, test_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_prob)
        roc_data[name] = (fpr, tpr, test_metrics["ROC-AUC"])
        pr_data[name] = (precision_curve, recall_curve, test_metrics["PR-AUC"])
        confusion_data[name] = test_metrics["Confusion Matrix"]

        y_pred = (test_prob >= best_threshold).astype(int)
        pred_df = test_context.copy()
        pred_df["actual_label"] = y_test.to_numpy()
        pred_df["pump_probability"] = test_prob
        pred_df["predicted_label"] = y_pred
        pred_df["error_type"] = np.select(
            [
                (pred_df["actual_label"] == 1) & (pred_df["predicted_label"] == 1),
                (pred_df["actual_label"] == 0) & (pred_df["predicted_label"] == 1),
                (pred_df["actual_label"] == 1) & (pred_df["predicted_label"] == 0),
            ],
            ["True Positive", "False Positive", "False Negative"],
            default="True Negative",
        )
        prediction_data[name] = pred_df.sort_values("pump_probability", ascending=False).reset_index(drop=True)

        if hasattr(model, "feature_importances_"):
            importance_data[name] = (
                pd.DataFrame(
                    {
                        "Feature": X_train.columns,
                        "Importance": model.feature_importances_,
                    }
                )
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )

st.subheader("📋 Model Comparison on Holdout Test Set")
results_df = pd.DataFrame(results)
st.dataframe(results_df, width="stretch")

st.caption(
    "For imbalanced pump detection, PR-AUC and F2 are usually more informative than accuracy. "
    "Thresholds are selected on validation data, then evaluated once on the holdout test set."
)

best_model_name = results_df.sort_values("Test F2", ascending=False).iloc[0]["Model"]
best_threshold = float(results_df[results_df["Model"] == best_model_name]["Threshold"].iloc[0])
best_pred_df = prediction_data[best_model_name].copy()

st.markdown("---")
st.subheader("⏱️ Real-time Alert Monitor")
st.caption(
    "This section simulates real-time inference by using the latest available candle in the dataset. "
    "In production, the same feature columns would be computed from a live exchange data stream."
)

latest_rows = best_pred_df.sort_values("timestamp").groupby("symbol", as_index=False).tail(1) if "symbol" in best_pred_df.columns else best_pred_df.tail(10)
latest_rows = latest_rows.sort_values("pump_probability", ascending=False).copy()
latest_rows["alert"] = np.where(latest_rows["pump_probability"] >= best_threshold, "PUMP ALERT", "Normal")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Best Model", best_model_name)
with col2:
    st.metric("Alert Threshold", f"{best_threshold:.2f}")
with col3:
    st.metric("Latest Timestamp", str(best_pred_df["timestamp"].max()) if "timestamp" in best_pred_df.columns else "N/A")
with col4:
    st.metric("Active Alerts", int((latest_rows["pump_probability"] >= best_threshold).sum()))

latest_display_cols = [
    col
    for col in [
        "timestamp",
        "symbol",
        "close",
        "pump_probability",
        "alert",
        "future_return_5m",
        "actual_label",
    ]
    if col in latest_rows.columns
]
st.dataframe(latest_rows[latest_display_cols].head(15), width="stretch", hide_index=True)

st.subheader("📡 Alert Backtest Timeline")
alert_timeline = best_pred_df.sort_values("timestamp").copy()
alert_timeline["alert"] = alert_timeline["pump_probability"] >= best_threshold
alert_timeline["date"] = pd.to_datetime(alert_timeline["timestamp"]).dt.date

daily_alerts = (
    alert_timeline.groupby("date")
    .agg(
        alerts=("alert", "sum"),
        true_pumps=("actual_label", "sum"),
        avg_probability=("pump_probability", "mean"),
    )
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(daily_alerts["date"], daily_alerts["alerts"], marker="o", label="Model alerts")
ax.plot(daily_alerts["date"], daily_alerts["true_pumps"], marker="o", label="Actual pumps")
ax.set_title("Daily Alert Count vs Actual Pump Count")
ax.set_xlabel("Date")
ax.set_ylabel("Count")
ax.legend()
ax.grid(True)
plt.xticks(rotation=30, ha="right")
st.pyplot(fig)

alert_cases = alert_timeline[alert_timeline["alert"]].copy()
alert_cols = [
    col
    for col in [
        "timestamp",
        "symbol",
        "close",
        "pump_probability",
        "future_return_5m",
        "actual_label",
        "error_type",
    ]
    if col in alert_cases.columns
]
st.dataframe(alert_cases[alert_cols].sort_values("pump_probability", ascending=False).head(20), width="stretch", hide_index=True)

st.subheader("📈 Metric Comparison")
metric_option = st.selectbox(
    "Select Metric",
    ["Test Accuracy", "Test Precision", "Test Recall", "Test F1", "Test F2", "Test ROC-AUC", "Test PR-AUC"],
)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=results_df, x="Model", y=metric_option, palette="viridis", ax=ax)
ax.set_ylim(0, 1)
ax.set_title(f"{metric_option} Comparison")
st.pyplot(fig)

col_roc, col_pr = st.columns(2)

with col_roc:
    st.subheader("🔥 ROC Curve")
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (fpr, tpr, auc_score) in roc_data.items():
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve on Test Set")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

with col_pr:
    st.subheader("🎯 Precision-Recall Curve")
    fig, ax = plt.subplots(figsize=(7, 5))
    baseline = y_test.mean()
    for name, (precision_curve, recall_curve, ap_score) in pr_data.items():
        ax.plot(recall_curve, precision_curve, linewidth=2, label=f"{name} (AP = {ap_score:.4f})")
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Baseline = {baseline:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve on Test Set")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.subheader("🧩 Model Analysis")
tabs = st.tabs(list(models.keys()))
for tab, model_name in zip(tabs, models.keys()):
    with tab:
        st.subheader(f"{model_name} Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_data[model_name], annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{model_name} - Test Set")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader(f"{model_name} Error Analysis")
        pred_df = prediction_data[model_name]
        error_counts = pred_df["error_type"].value_counts().rename_axis("error_type").reset_index(name="count")
        st.dataframe(error_counts, width="stretch", hide_index=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(
            data=pred_df,
            x="pump_probability",
            hue="actual_label",
            bins=30,
            stat="density",
            common_norm=False,
            ax=ax,
        )
        ax.set_title("Predicted probability distribution by actual label")
        st.pyplot(fig)

        display_cols = [
            col
            for col in [
                "timestamp",
                "symbol",
                "close",
                "future_return_5m",
                "actual_label",
                "predicted_label",
                "pump_probability",
                "error_type",
            ]
            if col in pred_df.columns
        ]
        case_tabs = st.tabs(["Top True Positives", "False Positives", "False Negatives"])
        with case_tabs[0]:
            st.dataframe(
                pred_df[pred_df["error_type"] == "True Positive"][display_cols].head(10),
                width="stretch",
                hide_index=True,
            )
        with case_tabs[1]:
            st.dataframe(
                pred_df[pred_df["error_type"] == "False Positive"][display_cols].head(10),
                width="stretch",
                hide_index=True,
            )
        with case_tabs[2]:
            st.dataframe(
                pred_df[pred_df["error_type"] == "False Negative"].sort_values("pump_probability", ascending=False)[display_cols].head(10),
                width="stretch",
                hide_index=True,
            )

        st.subheader(f"{model_name} Feature Importance")
        importance_df = importance_data[model_name]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", palette="magma", ax=ax)
        ax.set_title(f"Top 10 Features - {model_name}")
        st.pyplot(fig)

st.success("Dashboard completed with validation-based threshold tuning and holdout test evaluation.")
