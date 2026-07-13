import plotly.graph_objects as go
import streamlit as st

from utils import detect_anomalies, load_data

st.set_page_config(page_title="Anomaly Report", page_icon="🚨", layout="wide")
st.title("🚨 Anomaly Report")
st.caption("Weekly sales anomalies detected using an Isolation Forest model (Task 5).")

df = load_data()

contamination = st.slider(
    "Sensitivity (expected proportion of anomalous weeks)",
    min_value=0.02, max_value=0.15, value=0.07, step=0.01,
)

weekly = detect_anomalies(df, contamination=contamination)
anomalies = weekly[weekly["Anomaly"] == "Anomaly"]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=weekly["Order Date"], y=weekly["Sales"],
    mode="lines", name="Weekly Sales", line=dict(color="#1f77b4", width=2),
))
fig.add_trace(go.Scatter(
    x=anomalies["Order Date"], y=anomalies["Sales"],
    mode="markers", name="Anomaly",
    marker=dict(color="red", size=12, symbol="x"),
))
fig.update_layout(
    title="Weekly Sales with Detected Anomalies (Isolation Forest)",
    xaxis_title="Order Date", yaxis_title="Weekly Sales ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Detected Anomaly Weeks ({len(anomalies)})")
st.dataframe(
    anomalies[["Order Date", "Sales"]]
    .rename(columns={"Order Date": "Week Of", "Sales": "Weekly Sales"})
    .sort_values("Week Of")
    .style.format({"Weekly Sales": "${:,.2f}"}),
    use_container_width=True,
)
