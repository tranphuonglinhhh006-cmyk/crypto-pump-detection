
# ==========================================
# app.py
# Crypto Pump Detection Dashboard
# ==========================================

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
from urllib.parse import quote_plus
import pandas as pd
import numpy as np

from sqlalchemy import create_engine


from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(

    page_title="Crypto Pump Detection",

    page_icon="📈",

    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""

<style>

.main {
    background-color: #0f1117;
}

h1, h2, h3 {
    color: white;
}

</style>

""", unsafe_allow_html=True)

# ==========================================
# DASHBOARD TITLE
# ==========================================

st.title("📈 Crypto Pump Detection Dashboard")

st.markdown("---")

# ==========================================
# MYSQL CONNECTION
# ==========================================

@st.cache_data
def load_data():

    params = quote_plus(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=LAPTOP-GVLRNBIP;"
        "DATABASE=data15k;"
        "Trusted_Connection=yes;"
    )

    engine = create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}"
    )

    query = """
    SELECT *
    FROM dbo.final_dataset_15k
    """

    df = pd.read_sql(query, engine)

    return df

# ==========================================
# DATA PREPROCESSING FUNCTION
# ==========================================

def preprocess_data(df):

    # ==========================================
    # CONVERT TIMESTAMP
    # ==========================================

    df['timestamp'] = pd.to_datetime(
        df['timestamp']
    )

    # ==========================================
    # SORT TIME
    # ==========================================

    df = df.sort_values('timestamp')

    # ==========================================
    # REMOVE DATA LEAKAGE
    # ==========================================

    drop_columns = [

        "timestamp",

        "symbol",

        "future_return_5m",

        "is_pump",

        "is_pump_target",

        "return_5m",

        "momentum_5m"
    ]

    drop_columns = [

        col for col in drop_columns

        if col in df.columns
    ]

    df = df.drop(columns=drop_columns)

    # ==========================================
    # FEATURES & LABEL
    # ==========================================

    X = df.drop("label", axis=1)

    y = df["label"]

    # ==========================================
    # HANDLE INF VALUES
    # ==========================================

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ==========================================
    # REMOVE MISSING VALUES
    # ==========================================

    X = X.dropna()

    y = y.loc[X.index]

    # ==========================================
    # TIME SERIES SPLIT
    # ==========================================

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]

    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]

    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

# ==========================================
# LOAD DATA
# ==========================================

with st.spinner("Loading data from SQL..."):

    df = load_data()

# ==========================================
# PREPROCESS DATA
# ==========================================

X_train, X_test, y_train, y_test = preprocess_data(df)

st.success("Time-series preprocessing completed successfully.")

# ==========================================
# DATASET OVERVIEW
# ==========================================

st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

with col2:

    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:

    st.metric(
        "Pump Samples",
        int(df['label'].sum())
    )

# Preview dữ liệu

st.dataframe(df.head())

st.markdown("---")

# ==========================================
# TRAIN TEST INFO
# ==========================================

st.subheader("📦 Train/Test Information")

col1, col2 = st.columns(2)

with col1:

    st.info(
        f"Train Size: {len(X_train):,}"
    )

with col2:

    st.info(
        f"Test Size: {len(X_test):,}"
    )

st.markdown("---")

# ==========================================
# MODEL DEFINITIONS
# ==========================================

models = {

    "Random Forest": RandomForestClassifier(

        n_estimators=500,

        min_samples_leaf=3,

        min_samples_split=5,

        class_weight='balanced_subsample',

        max_features='sqrt',

        bootstrap=True,

        random_state=42,

        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(

        n_estimators=400,

        max_depth=5,

        learning_rate=0.03,

        subsample=0.9,

        colsample_bytree=0.9,

        min_child_weight=3,

        gamma=0.15,

        scale_pos_weight=10,

        reg_alpha=0.5,

        reg_lambda=2,

        objective='binary:logistic',

        eval_metric='logloss',

        random_state=42,

        n_jobs=-1
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

        class_weight='balanced',

        random_state=42
    )
}

# ==========================================
# TRAIN MODELS
# ==========================================

st.subheader("🤖 Model Training")

results = []

roc_data = {}

confusion_data = {}

importance_data = {}

# ==========================================
# TRAIN LOOP
# ==========================================

for name, model in models.items():

    with st.spinner(f"Training {name}..."):

        # ==========================================
        # TRAIN MODEL
        # ==========================================

        model.fit(
            X_train,
            y_train
        )

        # ==========================================
        # PREDICT PROBABILITY
        # ==========================================

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        # ==========================================
        # THRESHOLD TUNING
        # ==========================================

        best_threshold = 0.5

        best_f1 = 0

        for t in np.arange(0.1, 0.91, 0.05):

            y_pred_t = (
                y_prob >= t
            ).astype(int)

            precision = precision_score(
                y_test,
                y_pred_t,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred_t,
                zero_division=0
            )

            if precision >= 0.55 and f1 > best_f1:

                best_f1 = f1

                best_threshold = t

        # ==========================================
        # FINAL PREDICTION
        # ==========================================

        y_pred = (
            y_prob >= best_threshold
        ).astype(int)

        # ==========================================
        # EVALUATION METRICS
        # ==========================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        # ==========================================
        # SAVE RESULTS
        # ==========================================

        results.append({

            "Model": name,

            "Accuracy": round(accuracy, 4),

            "Precision": round(precision, 4),

            "Recall": round(recall, 4),

            "F1 Score": round(f1, 4),

            "ROC-AUC": round(roc_auc, 4)
        })

        # ==========================================
        # ROC CURVE DATA
        # ==========================================

        fpr, tpr, _ = roc_curve(
            y_test,
            y_prob
        )

        roc_data[name] = (
            fpr,
            tpr,
            roc_auc
        )

        # ==========================================
        # CONFUSION MATRIX
        # ==========================================

        confusion_data[name] = confusion_matrix(
            y_test,
            y_pred
        )

        # ==========================================
        # FEATURE IMPORTANCE
        # ==========================================

        if hasattr(model, 'feature_importances_'):

            importance_df = pd.DataFrame({

                "Feature": X_train.columns,

                "Importance": model.feature_importances_
            })

            importance_df = importance_df.sort_values(
                by='Importance',
                ascending=False
            )

            importance_data[name] = importance_df

# ==========================================
# MODEL COMPARISON TABLE
# ==========================================

st.subheader("📋 Model Comparison")

results_df = pd.DataFrame(results)

st.dataframe(results_df)

# ==========================================
# METRIC COMPARISON
# ==========================================

st.subheader("📈 Metric Comparison")

metric_option = st.selectbox(

    "Select Metric",

    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)

fig, ax = plt.subplots(figsize=(8,4))

sns.barplot(
    data=results_df,
    x='Model',
    y=metric_option,
    palette='viridis',
    ax=ax
)

ax.set_ylim(0, 1)

ax.set_title(
    f"{metric_option} Comparison"
)

st.pyplot(fig)

# ==========================================
# ROC CURVE
# ==========================================

st.subheader("🔥 ROC Curve Comparison")

fig, ax = plt.subplots(figsize=(9,6))

for name, (fpr, tpr, auc_score) in roc_data.items():

    ax.plot(
        fpr,
        tpr,
        linewidth=3,
        label=f"{name} (AUC = {auc_score:.4f})"
    )

# Random baseline

ax.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

ax.set_xlabel("False Positive Rate")

ax.set_ylabel("True Positive Rate")

ax.set_title("ROC Curve Comparison")

ax.legend()

ax.grid(True)

st.pyplot(fig)

# ==========================================
# CONFUSION MATRIX + FEATURE IMPORTANCE
# ==========================================

st.subheader("🧩 Model Analysis")

tabs = st.tabs([
    "Random Forest",
    "XGBoost",
    "LightGBM"
])

for tab, model_name in zip(
    tabs,
    models.keys()
):

    with tab:

        # ==========================================
        # CONFUSION MATRIX
        # ==========================================

        st.subheader(
            f"{model_name} Confusion Matrix"
        )

        cm = confusion_data[model_name]

        fig, ax = plt.subplots(figsize=(5,4))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax
        )

        ax.set_title(
            f"{model_name} Confusion Matrix"
        )

        ax.set_xlabel("Predicted")

        ax.set_ylabel("Actual")

        st.pyplot(fig)

        # ==========================================
        # FEATURE IMPORTANCE
        # ==========================================

        st.subheader(
            f"{model_name} Feature Importance"
        )

        importance_df = importance_data[model_name]

        fig, ax = plt.subplots(figsize=(10,5))

        sns.barplot(
            data=importance_df.head(10),
            x='Importance',
            y='Feature',
            palette='magma',
            ax=ax
        )

        ax.set_title(
            f"Top 10 Features - {model_name}"
        )

        st.pyplot(fig)

# ==========================================
# FINAL MESSAGE
# ==========================================

st.success("Dashboard completed successfully.")

