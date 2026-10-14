from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from run_analysis import run_pipeline  # noqa: E402

st.set_page_config(page_title="Real Estate Buyer Intelligence", page_icon="🏢", layout="wide")
st.title("Real Estate Buyer Segmentation & Investment Profiling")
st.caption("Exploratory market intelligence from the supplied client and property datasets.")

@st.cache_data(show_spinner="Cleaning data and fitting clustering models…")
def get_results():
    result = run_pipeline()
    # Return only data objects needed by the app; cache_data serializes these.
    return (
        result["buyers"], result["properties"], result["summary"], result["metrics"],
        result["quality"], result["model_info"]
    )

try:
    buyers, properties, summary, metrics, quality, model_info = get_results()
except Exception as exc:
    st.error(f"Pipeline could not complete: {exc}")
    st.info("Check that data/clients.csv and data/properties.csv match the project schema, then review README.md.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    def options_for(col):
        vals = sorted(buyers[col].dropna().astype(str).unique().tolist())
        return ["All"] + vals
    country = st.selectbox("Country", options_for("country"))
    region = st.selectbox("Region", options_for("region"))
    purpose = st.selectbox("Acquisition purpose", options_for("acquisition_purpose"))
    client_type = st.selectbox("Client type", options_for("client_type"))
    segment_options = ["All"] + sorted(buyers["segment_name"].dropna().unique().tolist())
    segment = st.selectbox("Buyer segment", segment_options)

filtered = buyers.copy()
for col, val in [("country", country), ("region", region),
                 ("acquisition_purpose", purpose), ("client_type", client_type),
                 ("segment_name", segment)]:
    if val != "All":
        filtered = filtered[filtered[col].astype(str).eq(val)]

n_buyers = filtered["client_id"].nunique()
total_sold_value = filtered["total_sold_value"].sum()
sold_units = filtered["sold_property_count"].sum()
investment_share = filtered["acquisition_purpose"].astype("string").str.casefold().eq("investment").mean() if n_buyers else 0
avg_satisfaction = filtered["satisfaction_score"].mean() if n_buyers else np.nan

m1, m2, m3, m4 = st.columns(4)
m1.metric("Buyers in filter", f"{n_buyers:,}")
m2.metric("Sold properties linked", f"{int(sold_units):,}")
m3.metric("Investment-purpose share", f"{investment_share:.1%}" if n_buyers else "—")
m4.metric("Mean satisfaction", f"{avg_satisfaction:.2f}/5" if pd.notna(avg_satisfaction) else "—")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Segmentation Overview", "Investor Behavior", "Geographic Analysis",
    "Segment Insights", "Data & Model Quality"
])

with tab1:
    st.subheader("Buyer segment distribution")
    if filtered.empty:
        st.warning("No buyers match the selected filters.")
    else:
        counts = filtered.groupby(["cluster", "segment_name"], dropna=False).size().reset_index(name="buyers")
        fig = px.bar(counts, x="segment_name", y="buyers", color="segment_name",
                     labels={"segment_name": "Segment", "buyers": "Buyer count"},
                     title="Buyers by model-derived segment")
        st.plotly_chart(fig, use_container_width=True)
        purpose_counts = filtered["acquisition_purpose"].value_counts().rename_axis("purpose").reset_index(name="buyers")
        fig2 = px.pie(purpose_counts, names="purpose", values="buyers", hole=0.45,
                      title="Acquisition purpose mix")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Investment and transaction behavior")
    if filtered.empty:
        st.warning("No buyers match the selected filters.")
    else:
        a, b = st.columns(2)
        purpose_by_segment = filtered.groupby(["segment_name", "acquisition_purpose"]).size().reset_index(name="buyers")
        a.plotly_chart(px.bar(purpose_by_segment, x="segment_name", y="buyers", color="acquisition_purpose",
                              barmode="stack", title="Acquisition purpose by segment"), use_container_width=True)
        loan_by_segment = filtered.groupby(["segment_name", "loan_applied"]).size().reset_index(name="buyers")
        b.plotly_chart(px.bar(loan_by_segment, x="segment_name", y="buyers", color="loan_applied",
                              barmode="stack", title="Loan application by segment"), use_container_width=True)
        spend = filtered.groupby("segment_name", as_index=False).agg(
            mean_total_sold_value=("total_sold_value", "mean"),
            mean_sold_count=("sold_property_count", "mean")
        )
        st.plotly_chart(px.bar(spend, x="segment_name", y="mean_total_sold_value",
                               title="Mean linked sold-property value per buyer",
                               labels={"mean_total_sold_value": "Mean sold value (dataset currency)"}),
                        use_container_width=True)
        st.caption("Sold value is aggregated only from property records marked Sold and linked to a client. It is not net worth or verified investment capital.")

with tab3:
    st.subheader("Geographic buyer analysis")
    if filtered.empty:
        st.warning("No buyers match the selected filters.")
    else:
        geo = filtered.groupby(["country", "region", "segment_name"], dropna=False).size().reset_index(name="buyers")
        top_geo = geo.groupby(["country", "region"], as_index=False)["buyers"].sum().sort_values("buyers", ascending=False).head(25)
        st.plotly_chart(px.bar(top_geo, x="region", y="buyers", color="country",
                               title="Top 25 country-region combinations",
                               labels={"buyers": "Buyer count"}), use_container_width=True)
        matrix = filtered.pivot_table(index="region", columns="segment_name", values="client_id",
                                      aggfunc="nunique", fill_value=0)
        st.dataframe(matrix, use_container_width=True)

with tab4:
    st.subheader("Segment-level descriptive statistics")
    if filtered.empty:
        st.warning("No buyers match the selected filters.")
    else:
        segment_stats = filtered.groupby(["cluster", "segment_name"], as_index=False).agg(
            buyers=("client_id", "nunique"),
            mean_age=("age", "mean"),
            mean_satisfaction=("satisfaction_score", "mean"),
            investment_share=("acquisition_purpose", lambda x: x.astype("string").str.casefold().eq("investment").mean()),
            loan_share=("loan_applied", lambda x: x.astype("string").str.casefold().eq("yes").mean()),
            company_share=("client_type", lambda x: x.astype("string").str.casefold().isin(["company", "corporate"]).mean()),
            mean_sold_properties=("sold_property_count", "mean"),
            mean_total_sold_value=("total_sold_value", "mean")
        )
        st.dataframe(segment_stats.round(3), use_container_width=True)
        st.download_button("Download filtered buyer records", filtered.to_csv(index=False).encode("utf-8"),
                           "filtered_buyer_segments.csv", "text/csv")
        st.download_button("Download segment summary", segment_stats.to_csv(index=False).encode("utf-8"),
                           "filtered_segment_summary.csv", "text/csv")

with tab5:
    st.subheader("Data quality")
    q1, q2 = st.columns(2)
    q1.json(quality)
    q2.json(model_info)
    st.subheader("Cluster-count diagnostics")
    st.dataframe(metrics.round(4), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(metrics, x="k", y="inertia", markers=True, title="Elbow diagnostic (inertia)"),
                    use_container_width=True)
    c2.plotly_chart(px.line(metrics, x="k", y="silhouette_score", markers=True,
                            title="Silhouette score by cluster count"), use_container_width=True)
    st.warning(
        "Interpret clusters as exploratory profiles, not objective buyer classes. "
        "Income, net worth, motivation beyond the recorded purpose, and risk tolerance are not directly observed."
    )

st.divider()
st.caption("Model: K-Means with standardized age/satisfaction and one-hot encoded categorical features; "
           "agglomerative clustering is run on a deterministic sample as a secondary comparison. "
           "Cluster IDs are arbitrary and may change when the data changes.")
