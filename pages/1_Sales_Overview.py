import plotly.express as px
import streamlit as st

from utils import load_data, monthly_sales_series, region_category_sales, yearly_sales

st.set_page_config(page_title="Sales Overview", page_icon="📊", layout="wide")
st.title("📊 Sales Overview Dashboard")

df = load_data()

# --- Total sales by year ---
st.subheader("Total Sales by Year")
ys = yearly_sales(df)
fig_year = px.bar(
    ys, x="Year", y="Sales", text_auto=".2s",
    labels={"Sales": "Total Sales ($)"}, color="Year",
)
fig_year.update_layout(showlegend=False)
st.plotly_chart(fig_year, use_container_width=True)

# --- Monthly sales trend ---
st.subheader("Monthly Sales Trend")
ms = monthly_sales_series(df)
fig_month = px.line(
    ms, x="Order Date", y="Sales", markers=True,
    labels={"Sales": "Total Sales ($)", "Order Date": "Month"},
)
st.plotly_chart(fig_month, use_container_width=True)

# --- Sales by region and category (interactive filters) ---
st.subheader("Sales by Region and Category")

col1, col2 = st.columns(2)
with col1:
    regions = st.multiselect(
        "Filter Region(s)", options=sorted(df["Region"].unique()),
        default=sorted(df["Region"].unique()),
    )
with col2:
    categories = st.multiselect(
        "Filter Category(ies)", options=sorted(df["Category"].unique()),
        default=sorted(df["Category"].unique()),
    )

if regions and categories:
    rc = region_category_sales(df, regions, categories)
    fig_rc = px.bar(
        rc, x="Region", y="Sales", color="Category", barmode="group",
        labels={"Sales": "Total Sales ($)"},
    )
    st.plotly_chart(fig_rc, use_container_width=True)
    st.dataframe(rc, use_container_width=True)
else:
    st.info("Select at least one Region and one Category to see results.")
