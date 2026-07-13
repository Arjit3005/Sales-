import plotly.express as px
import streamlit as st

from utils import cluster_subcategories, load_data

st.set_page_config(page_title="Product Demand Segments", page_icon="🧩", layout="wide")
st.title("🧩 Product Demand Segments")
st.caption("Sub-categories grouped by demand behavior using KMeans clustering (Task 6).")

df = load_data()
summary = cluster_subcategories(df, n_clusters=3)

fig = px.scatter(
    summary, x="PCA1", y="PCA2", color="Demand Segment", text="Sub-Category",
    size="Total Sales Volume", hover_data=["Average Order Value", "Sales Volatility", "YoY Growth Rate (%)"],
    title="Product Demand Clusters (PCA + K-Means)",
)
fig.update_traces(textposition="top center")
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Sub-Category → Demand Cluster")
display_cols = [
    "Sub-Category", "Demand Segment", "Total Sales Volume",
    "Average Order Value", "YoY Growth Rate (%)", "Sales Volatility",
]
st.dataframe(
    summary[display_cols]
    .sort_values(["Demand Segment", "Total Sales Volume"], ascending=[True, False])
    .style.format({
        "Total Sales Volume": "${:,.0f}",
        "Average Order Value": "${:,.2f}",
        "YoY Growth Rate (%)": "{:.1f}%",
        "Sales Volatility": "${:,.0f}",
    }),
    use_container_width=True,
)
