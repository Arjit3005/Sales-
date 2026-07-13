"""
Shared data + model utilities for the Superstore Sales Analytics dashboard.
All heavy computations are cached with st.cache_data / st.cache_resource so
that navigating between pages stays fast.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA_PATH = "train.csv"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    return df


# --------------------------------------------------------------------------
# Page 1 helpers
# --------------------------------------------------------------------------
@st.cache_data
def yearly_sales(df):
    return df.groupby("Year", as_index=False)["Sales"].sum()


@st.cache_data
def monthly_sales_series(df):
    ms = df.set_index("Order Date")["Sales"].resample("ME").sum().reset_index()
    ms.columns = ["Order Date", "Sales"]
    return ms


@st.cache_data
def region_category_sales(df, regions, categories):
    filtered = df[df["Region"].isin(regions) & df["Category"].isin(categories)]
    return (
        filtered.groupby(["Region", "Category"], as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )


# --------------------------------------------------------------------------
# Page 2 helpers - Forecasting (XGBoost = best performing model, see Task 3)
# --------------------------------------------------------------------------
def _build_monthly(df, dim, value):
    if dim == "Category":
        sub = df[df["Category"] == value]
    else:
        sub = df[df["Region"] == value]
    monthly = sub.set_index("Order Date")["Sales"].resample("ME").sum().reset_index()
    monthly.columns = ["Order Date", "Sales"]
    return monthly


def _add_features(monthly):
    m = monthly.copy()
    m["Lag1"] = m["Sales"].shift(1)
    m["Lag2"] = m["Sales"].shift(2)
    m["Lag3"] = m["Sales"].shift(3)
    m["RollingMean"] = m["Sales"].rolling(3).mean()
    m["Month"] = m["Order Date"].dt.month
    m["Quarter"] = m["Order Date"].dt.quarter
    return m

FEATURE_COLS = ["Lag1", "Lag2", "Lag3", "RollingMean", "Month", "Quarter"]


@st.cache_data
def forecast_segment(_df_placeholder, dim, value, horizon, df):
    """
    Trains an XGBoost regressor on monthly sales for the chosen Category/Region,
    evaluates it on a held-out tail of known months (MAE/RMSE), then retrains on
    all available data and recursively forecasts `horizon` months ahead.
    """
    monthly = _build_monthly(df, dim, value)
    feat = _add_features(monthly).dropna().reset_index(drop=True)

    if len(feat) < 8:
        return None  # not enough history for this segment

    # --- Hold-out evaluation (last 3 known months) ---
    n_test = min(3, max(1, len(feat) // 5))
    train, test = feat.iloc[:-n_test], feat.iloc[-n_test:]

    eval_model = XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
    eval_model.fit(train[FEATURE_COLS], train["Sales"])
    test_preds = eval_model.predict(test[FEATURE_COLS])

    mae = mean_absolute_error(test["Sales"], test_preds)
    rmse = np.sqrt(mean_squared_error(test["Sales"], test_preds))

    # --- Final model trained on ALL available history ---
    final_model = XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
    final_model.fit(feat[FEATURE_COLS], feat["Sales"])

    # --- Recursive multi-step forecast ---
    history = monthly["Sales"].tolist()
    last_date = monthly["Order Date"].iloc[-1]
    future_rows = []
    for step in range(horizon):
        next_date = last_date + pd.DateOffset(months=step + 1)
        lag1, lag2, lag3 = history[-1], history[-2], history[-3]
        rolling_mean = np.mean(history[-3:])
        x = pd.DataFrame(
            [[lag1, lag2, lag3, rolling_mean, next_date.month, next_date.quarter]],
            columns=FEATURE_COLS,
        )
        pred = float(final_model.predict(x)[0])
        future_rows.append({"Order Date": next_date, "Sales": pred})
        history.append(pred)

    future_df = pd.DataFrame(future_rows)

    return {
        "history": monthly,
        "forecast": future_df,
        "mae": mae,
        "rmse": rmse,
        "n_test": n_test,
    }


# --------------------------------------------------------------------------
# Page 3 helpers - Anomaly detection (Isolation Forest)
# --------------------------------------------------------------------------
@st.cache_data
def detect_anomalies(df, contamination=0.07):
    weekly = df.set_index("Order Date")["Sales"].resample("W").sum().reset_index()
    weekly.columns = ["Order Date", "Sales"]

    weekly["RollingMean"] = weekly["Sales"].rolling(4, min_periods=1).mean()
    weekly["RollingStd"] = weekly["Sales"].rolling(4, min_periods=1).std().fillna(0)

    features = weekly[["Sales", "RollingMean", "RollingStd"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    weekly["Anomaly_Flag"] = iso.fit_predict(scaled)  # -1 = anomaly, 1 = normal
    weekly["Anomaly"] = np.where(weekly["Anomaly_Flag"] == -1, "Anomaly", "Normal")

    return weekly


# --------------------------------------------------------------------------
# Page 4 helpers - Product demand segmentation (KMeans)
# --------------------------------------------------------------------------
@st.cache_data
def cluster_subcategories(df, n_clusters=3):
    total_sales = df.groupby("Sub-Category")["Sales"].sum().rename("Total Sales Volume")
    avg_order = df.groupby("Sub-Category")["Sales"].mean().rename("Average Order Value")

    yearly = df.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack(fill_value=0)
    growth_rate = (yearly.pct_change(axis=1).mean(axis=1) * 100).rename("YoY Growth Rate (%)")
    growth_rate = growth_rate.replace([np.inf, -np.inf], np.nan).fillna(0)

    monthly = (
        df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="ME")])["Sales"]
        .sum()
        .reset_index()
    )
    volatility = monthly.groupby("Sub-Category")["Sales"].std().rename("Sales Volatility")

    summary = pd.concat([total_sales, avg_order, growth_rate, volatility], axis=1).reset_index()
    summary = summary.fillna(0)

    features = summary[
        ["Total Sales Volume", "YoY Growth Rate (%)", "Sales Volatility", "Average Order Value"]
    ]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    summary["Cluster"] = kmeans.fit_predict(scaled)

    # Rank clusters by total sales volume -> High / Medium / Low demand labels
    cluster_rank = (
        summary.groupby("Cluster")["Total Sales Volume"].mean().sort_values(ascending=False)
    )
    ordered_clusters = list(cluster_rank.index)
    label_names = ["High Demand", "Medium Demand", "Low Demand"][:n_clusters]
    cluster_to_label = dict(zip(ordered_clusters, label_names))
    summary["Demand Segment"] = summary["Cluster"].map(cluster_to_label)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    summary["PCA1"] = coords[:, 0]
    summary["PCA2"] = coords[:, 1]

    return summary
