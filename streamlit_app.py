import streamlit as st
from utils import load_data

st.set_page_config(
    page_title="Superstore Sales Analytics",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Superstore Sales Analytics Dashboard")

st.markdown(
    """
Welcome! This app turns the Superstore sales dataset into an interactive
analytics suite. Use the **sidebar** to navigate between pages:

- **1 · Sales Overview** — yearly/monthly sales trends and region × category breakdowns
- **2 · Forecast Explorer** — XGBoost sales forecasts by Category or Region, 1–3 months ahead
- **3 · Anomaly Report** — Isolation Forest anomaly detection on weekly sales
- **4 · Product Demand Segments** — KMeans clustering of sub-categories by demand behavior

Select a page from the sidebar to get started.
"""
)

with st.spinner("Loading data..."):
    df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", f"{df['Order ID'].nunique():,}")
col2.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
col3.metric("Date Range", f"{df['Order Date'].dt.year.min()}–{df['Order Date'].dt.year.max()}")
col4.metric("Sub-Categories", f"{df['Sub-Category'].nunique()}")

st.dataframe(df.head(20), use_container_width=True)
