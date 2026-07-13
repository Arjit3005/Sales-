import plotly.graph_objects as go
import streamlit as st

from utils import forecast_segment, load_data

st.set_page_config(page_title="Forecast Explorer", page_icon="🔮", layout="wide")
st.title("🔮 Forecast Explorer")
st.caption("Forecasts are produced by the best-performing model from Task 3 (XGBoost with lag/rolling features).")

df = load_data()

col1, col2, col3 = st.columns([1, 1, 1.3])

with col1:
    dim = st.selectbox("Select dimension", ["Category", "Region"])

with col2:
    options = sorted(df[dim].unique())
    value = st.selectbox(f"Select {dim}", options)

with col3:
    horizon = st.select_slider(
        "Forecast horizon (months ahead)", options=[1, 2, 3], value=3,
    )

result = forecast_segment(None, dim, value, horizon, df)

if result is None:
    st.warning("Not enough historical data for this selection to build a reliable forecast.")
else:
    history = result["history"]
    forecast = result["forecast"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["Order Date"], y=history["Sales"],
        mode="lines+markers", name="Actual Sales", line=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=forecast["Order Date"], y=forecast["Sales"],
        mode="lines+markers", name="Forecast", line=dict(color="#ff7f0e", dash="dash"),
    ))
    fig.update_layout(
        title=f"{dim}: {value} — Sales Forecast ({horizon} month(s) ahead)",
        xaxis_title="Date", yaxis_title="Sales ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Values")
    st.dataframe(
        forecast.rename(columns={"Sales": "Forecasted Sales"}).style.format({"Forecasted Sales": "${:,.2f}"}),
        use_container_width=True,
    )

    st.subheader("Model Performance (held-out evaluation)")
    m1, m2 = st.columns(2)
    m1.metric("MAE", f"${result['mae']:,.2f}")
    m2.metric("RMSE", f"${result['rmse']:,.2f}")
    st.caption(f"Evaluated on the last {result['n_test']} known month(s) held out from training.")
