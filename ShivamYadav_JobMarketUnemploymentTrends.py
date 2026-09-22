"""
================================================================================
  ShivamYadav_JobMarketUnemploymentTrends.py
  Author  : Shivam Yadav
  Project : Job Market & Unemployment Trends Dashboard
  Dataset : job_market_unemployment_trends.csv
            (1,000 records · 20 US cities · July 2023 – July 2025)

  Description:
      A fully interactive data analytics dashboard built with Python + Streamlit
      that explores unemployment rates, job postings, in-demand skills, and
      workforce demographics across 20 major US cities.

  How to run:
      streamlit run ShivamYadav_JobMarketUnemploymentTrends.py

  Dependencies (pip install -r requirements.txt):
      streamlit  >= 1.35.0
      pandas     >= 2.0.0
      numpy      >= 1.26.0
      plotly     >= 5.20.0
      scikit-learn >= 1.4.0
================================================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import io
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA PROCESSOR  (all backend analytics & ML logic)
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = "job_market_unemployment_trends.csv"


# ── Load & Clean ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["quarter"]    = df["date"].dt.to_period("Q").astype(str)
    df["skills_list"] = df["in_demand_skills"].apply(_parse_skills)
    return df


def _parse_skills(raw: str) -> list:
    if pd.isna(raw):
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


# ── KPI Summary ───────────────────────────────────────────────────────────────

def get_kpis(df: pd.DataFrame) -> dict:
    return {
        "avg_unemployment"  : round(df["unemployment_rate"].mean(), 2),
        "max_unemployment"  : round(df["unemployment_rate"].max(), 2),
        "min_unemployment"  : round(df["unemployment_rate"].min(), 2),
        "total_job_postings": int(df["job_postings"].sum()),
        "avg_job_postings"  : round(df["job_postings"].mean(), 0),
        "total_records"     : len(df),
        "cities_covered"    : df["location"].nunique(),
        "date_range"        : (df["date"].min().strftime("%b %Y"), df["date"].max().strftime("%b %Y")),
        "avg_age"           : round(df["average_age"].mean(), 1),
        "avg_college_pct"   : round(df["college_degree_percentage"].mean(), 1),
    }


# ── Time Series ───────────────────────────────────────────────────────────────

def monthly_unemployment(df: pd.DataFrame, city: str = "All") -> pd.DataFrame:
    d = df if city == "All" else df[df["location"] == city]
    return (
        d.groupby("year_month")
         .agg(avg_unemployment=("unemployment_rate", "mean"),
              avg_job_postings=("job_postings", "mean"))
         .reset_index()
         .sort_values("year_month")
    )


def quarterly_unemployment(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("quarter")
          .agg(avg_unemployment=("unemployment_rate", "mean"),
               avg_job_postings=("job_postings", "mean"))
          .reset_index()
          .sort_values("quarter")
    )


def yoy_change(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.groupby("year")
          .agg(avg_unemployment=("unemployment_rate", "mean"),
               avg_job_postings=("job_postings", "mean"))
          .reset_index()
    )
    yearly["unemp_change"]   = yearly["avg_unemployment"].diff()
    yearly["postings_change"] = yearly["avg_job_postings"].diff()
    return yearly


# ── City / Geo Analysis ───────────────────────────────────────────────────────

def city_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("location")
          .agg(
              avg_unemployment=("unemployment_rate", "mean"),
              avg_job_postings=("job_postings", "mean"),
              avg_age=("average_age", "mean"),
              avg_college_pct=("college_degree_percentage", "mean"),
              record_count=("id", "count"),
          )
          .reset_index()
          .sort_values("avg_unemployment", ascending=False)
          .round(2)
    )


def city_ranking(df: pd.DataFrame, metric: str = "avg_unemployment", ascending: bool = False) -> pd.DataFrame:
    summary = city_summary(df)
    return summary.sort_values(metric, ascending=ascending).reset_index(drop=True)


# ── Skills Analysis ───────────────────────────────────────────────────────────

def skill_frequency(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    all_skills = [s for sublist in df["skills_list"] for s in sublist]
    counts = Counter(all_skills)
    return (
        pd.DataFrame(counts.items(), columns=["skill", "count"])
          .sort_values("count", ascending=False)
          .head(top_n)
          .reset_index(drop=True)
    )


def skills_by_city(df: pd.DataFrame, city: str) -> pd.DataFrame:
    sub = df[df["location"] == city]
    return skill_frequency(sub, top_n=10)


def skills_unemployment_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """For each skill, compute avg unemployment rate across rows containing that skill."""
    rows = []
    all_skills = list({s for sublist in df["skills_list"] for s in sublist})
    for skill in all_skills:
        mask   = df["skills_list"].apply(lambda lst: skill in lst)
        subset = df[mask]
        rows.append({
            "skill"            : skill,
            "avg_unemployment" : round(subset["unemployment_rate"].mean(), 2),
            "avg_job_postings" : round(subset["job_postings"].mean(), 0),
            "count"            : len(subset),
        })
    return pd.DataFrame(rows).sort_values("avg_unemployment").reset_index(drop=True)


def skills_trend_over_time(df: pd.DataFrame, top_skills: list) -> pd.DataFrame:
    """Return year × skill matrix of occurrence counts."""
    records = []
    for _, row in df.iterrows():
        for skill in row["skills_list"]:
            if skill in top_skills:
                records.append({"year": row["year"], "skill": skill})
    if not records:
        return pd.DataFrame()
    trend = pd.DataFrame(records).groupby(["year", "skill"]).size().reset_index(name="count")
    return trend.pivot(index="year", columns="skill", values="count").fillna(0).reset_index()


# ── Correlations ──────────────────────────────────────────────────────────────

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["unemployment_rate", "job_postings", "average_age", "college_degree_percentage"]
    return df[cols].corr().round(3)


# ── Predictive — simple linear regression per city ───────────────────────────

def predict_unemployment(df: pd.DataFrame, city: str, months_ahead: int = 6) -> pd.DataFrame:
    monthly = monthly_unemployment(df, city)
    if len(monthly) < 4:
        return pd.DataFrame()

    monthly["t"] = np.arange(len(monthly))
    X = monthly[["t"]].values
    y = monthly["avg_unemployment"].values

    model = LinearRegression().fit(X, y)

    last_t    = monthly["t"].max()
    last_date = monthly["year_month"].max()
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=months_ahead, freq="MS")
    future_t     = np.arange(last_t + 1, last_t + 1 + months_ahead).reshape(-1, 1)
    future_pred  = model.predict(future_t)

    hist = monthly[["year_month", "avg_unemployment"]].copy()
    hist["type"] = "Historical"
    hist.rename(columns={"year_month": "date", "avg_unemployment": "unemployment_rate"}, inplace=True)

    pred = pd.DataFrame({
        "date"             : future_dates,
        "unemployment_rate": np.clip(future_pred, 0, 20),
        "type"             : "Forecast",
    })

    return pd.concat([hist, pred], ignore_index=True)


# ── Education vs Unemployment ─────────────────────────────────────────────────

def education_unemployment_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["edu_bucket"] = pd.cut(
        df["college_degree_percentage"],
        bins=[0, 40, 60, 80, 100],
        labels=["<40%", "40–60%", "60–80%", ">80%"],
    )
    return (
        df.groupby("edu_bucket", observed=True)
          .agg(avg_unemployment=("unemployment_rate", "mean"),
               avg_job_postings=("job_postings", "mean"),
               count=("id", "count"))
          .reset_index()
          .round(2)
    )


def age_unemployment_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_bucket"] = pd.cut(
        df["average_age"],
        bins=[20, 30, 35, 40, 45, 55],
        labels=["21–30", "31–35", "36–40", "41–45", "46–55"],
    )
    return (
        df.groupby("age_bucket", observed=True)
          .agg(avg_unemployment=("unemployment_rate", "mean"),
               avg_job_postings=("job_postings", "mean"),
               count=("id", "count"))
          .reset_index()
          .round(2)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — DASHBOARD PAGES
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared helper ─────────────────────────────────────────────────────────────

def _kpi(col, label: str, value: str, sub: str = ""):
    col.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Page: Overview ────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame):
    st.title("🏠 **Job Market Overview**")
    st.markdown("**▸** A bird's-eye view of unemployment rates, job postings, and workforce demographics across US cities.")

    if df.empty:
        st.warning("No data for the selected filters.")
        return

    kpis = get_kpis(df)

    c1, c2, c3, c4 = st.columns(4)
    _kpi(c1, "Avg Unemployment",  f"{kpis['avg_unemployment']}%",  f"Range: {kpis['min_unemployment']}% – {kpis['max_unemployment']}%")
    _kpi(c2, "Total Job Postings", f"{kpis['total_job_postings']:,}", f"Avg per record: {kpis['avg_job_postings']:,.0f}")
    _kpi(c3, "Cities Covered",    str(kpis["cities_covered"]),     f"{kpis['total_records']} records")
    _kpi(c4, "Avg College Grad %", f"{kpis['avg_college_pct']}%",  f"Avg workforce age: {kpis['avg_age']} yrs")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">▸ <strong>Unemployment Rate by City</strong></div>', unsafe_allow_html=True)
        city_data = city_summary(df).sort_values("avg_unemployment", ascending=True)
        fig = px.bar(
            city_data, x="avg_unemployment", y="location",
            orientation="h", color="avg_unemployment",
            color_continuous_scale="RdYlGn_r",
            labels={"avg_unemployment": "Avg Unemployment (%)", "location": "City"},
            text=city_data["avg_unemployment"].apply(lambda v: f"{v:.1f}%"),
        )
        fig.update_traces(textposition="outside", textfont=dict(color="#1e293b", size=12))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=60),
                          coloraxis_showscale=False, plot_bgcolor="#f8fafc",
                          paper_bgcolor="#ffffff", font=dict(color="#1e293b"),
                          xaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                          yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">▸ <strong>Job Postings vs Unemployment</strong></div>', unsafe_allow_html=True)
        fig2 = px.scatter(
            df, x="job_postings", y="unemployment_rate",
            color="location", size="college_degree_percentage",
            hover_data=["location", "date", "in_demand_skills"],
            labels={"job_postings": "Job Postings", "unemployment_rate": "Unemployment Rate (%)", "location": "City"},
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                           plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
                           legend=dict(orientation="h", y=-0.25, x=0, font=dict(color="#1e293b")),
                           font=dict(color="#1e293b"),
                           xaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">▸ <strong>Education Level vs Unemployment</strong></div>', unsafe_allow_html=True)
        edu_data   = education_unemployment_buckets(df)
        EDU_COLORS = ["#22c55e", "#facc15", "#f97316", "#ef4444"]
        fig3 = px.bar(edu_data, x="edu_bucket", y="avg_unemployment",
                      color="edu_bucket", color_discrete_sequence=EDU_COLORS,
                      text=edu_data["avg_unemployment"].apply(lambda v: f"{v:.1f}%"),
                      labels={"edu_bucket": "College Degree %", "avg_unemployment": "Avg Unemployment (%)"})
        fig3.update_traces(textposition="outside", textfont=dict(color="#1e293b", size=13, family="Arial Black"))
        fig3.update_layout(height=360, showlegend=False, plot_bgcolor="#f8fafc",
                           paper_bgcolor="#ffffff", font=dict(color="#1e293b"),
                           xaxis=dict(tickfont=dict(color="#1e293b", size=13), title_font=dict(color="#1e293b")),
                           yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
        edu_min      = edu_data.loc[edu_data["avg_unemployment"].idxmin(), "edu_bucket"]
        edu_max      = edu_data.loc[edu_data["avg_unemployment"].idxmax(), "edu_bucket"]
        edu_low_val  = edu_data["avg_unemployment"].min()
        edu_high_val = edu_data["avg_unemployment"].max()
        st.markdown(
            f"""<div style="background:#f0fdf4;border-left:4px solid #22c55e;padding:10px 14px;
                border-radius:0 8px 8px 0;margin-top:4px;font-size:13px;color:#1e293b;">
                <b>📊 Trend Insight:</b> Areas with the <b>highest college-degree %</b> ({edu_min})
                show the <b>lowest avg unemployment at {edu_low_val:.1f}%</b>, while the
                <b>lowest-education bucket</b> ({edu_max}) peaks at <b>{edu_high_val:.1f}%</b>.
                Higher education attainment correlates with lower unemployment risk.
            </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-header">▸ <strong>Workforce Age vs Unemployment</strong></div>', unsafe_allow_html=True)
        age_data   = age_unemployment_buckets(df)
        AGE_COLORS = ["#6366f1", "#3b82f6", "#06b6d4", "#f59e0b", "#ef4444"]
        fig4 = px.bar(age_data, x="age_bucket", y="avg_unemployment",
                      color="age_bucket", color_discrete_sequence=AGE_COLORS,
                      text=age_data["avg_unemployment"].apply(lambda v: f"{v:.1f}%"),
                      labels={"age_bucket": "Average Age Group", "avg_unemployment": "Avg Unemployment (%)"})
        fig4.update_traces(textposition="outside", textfont=dict(color="#1e293b", size=13, family="Arial Black"))
        fig4.update_layout(height=360, showlegend=False, plot_bgcolor="#f8fafc",
                           paper_bgcolor="#ffffff", font=dict(color="#1e293b"),
                           xaxis=dict(tickfont=dict(color="#1e293b", size=13), title_font=dict(color="#1e293b")),
                           yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)
        age_min      = age_data.loc[age_data["avg_unemployment"].idxmin(), "age_bucket"]
        age_max      = age_data.loc[age_data["avg_unemployment"].idxmax(), "age_bucket"]
        age_low_val  = age_data["avg_unemployment"].min()
        age_high_val = age_data["avg_unemployment"].max()
        st.markdown(
            f"""<div style="background:#eff6ff;border-left:4px solid #3b82f6;padding:10px 14px;
                border-radius:0 8px 8px 0;margin-top:4px;font-size:13px;color:#1e293b;">
                <b>📊 Trend Insight:</b> The <b>{age_min}</b> age group has the
                <b>lowest avg unemployment at {age_low_val:.1f}%</b>, while the
                <b>{age_max}</b> group shows the highest at <b>{age_high_val:.1f}%</b>.
                Mid-career workers tend to face less unemployment volatility.
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">▸ <strong>Correlation Matrix — Key Metrics</strong></div>', unsafe_allow_html=True)
    corr   = correlation_matrix(df)
    z_vals = corr.values.tolist()
    labels = corr.columns.tolist()

    fig5 = go.Figure()
    fig5.add_trace(go.Heatmap(
        z=z_vals, x=labels, y=labels,
        colorscale=[[0.0, "#1d4ed8"], [0.25, "#60a5fa"], [0.5, "#f1f5f9"],
                    [0.75, "#fca5a5"], [1.0, "#b91c1c"]],
        zmid=0, zmin=-1, zmax=1, showscale=True,
        colorbar=dict(tickfont=dict(color="#1e293b", size=12),
                      title=dict(text="r", font=dict(color="#1e293b"))),
    ))
    for i, row_vals in enumerate(z_vals):
        for j, val in enumerate(row_vals):
            text_color = "#ffffff" if abs(val) > 0.5 else "#1e293b"
            fig5.add_annotation(x=labels[j], y=labels[i], text=f"<b>{val:.2f}</b>",
                                showarrow=False, font=dict(color=text_color, size=14, family="Arial Black"),
                                xref="x", yref="y")
    fig5.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10),
                       paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                       font=dict(color="#1e293b", size=13),
                       xaxis=dict(tickfont=dict(color="#1e293b", size=13), side="bottom"),
                       yaxis=dict(tickfont=dict(color="#1e293b", size=13), autorange="reversed"))
    st.plotly_chart(fig5, use_container_width=True)


# ── Page: Unemployment Trends ─────────────────────────────────────────────────

def page_trends(df: pd.DataFrame):
    st.title("📈 **Unemployment Trends**")
    st.markdown("**▸** Explore how unemployment and job postings evolved over time — monthly, quarterly, and year-over-year.")

    if df.empty:
        st.warning("No data for the selected filters.")
        return

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        city_sel = st.selectbox("City for time-series drill-down",
                                options=["All"] + sorted(df["location"].unique().tolist()), key="ts_city")
    with col_ctrl2:
        granularity = st.radio("Granularity", ["Monthly", "Quarterly"], horizontal=True, key="ts_gran")

    st.markdown('<div class="section-header">▸ <strong>Unemployment Rate Over Time</strong></div>', unsafe_allow_html=True)

    if granularity == "Monthly":
        ts = monthly_unemployment(df, city_sel)
        x_col, x_label = "year_month", "Month"
    else:
        ts = quarterly_unemployment(df)
        x_col, x_label = "quarter", "Quarter"

    if ts.empty:
        st.info("Not enough data for the selected city.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts[x_col], y=ts["avg_unemployment"],
                                 name="Unemployment Rate (%)", mode="lines+markers",
                                 line=dict(color="#ef4444", width=2.5), marker=dict(size=6)))
        fig.add_trace(go.Bar(x=ts[x_col], y=ts["avg_job_postings"],
                             name="Avg Job Postings", yaxis="y2",
                             opacity=0.35, marker_color="#3b82f6"))
        fig.update_layout(
            yaxis=dict(title="Unemployment Rate (%)", color="#ef4444",
                       tickfont=dict(color="#1e293b"), title_font=dict(color="#ef4444")),
            yaxis2=dict(title="Avg Job Postings", overlaying="y", side="right", color="#3b82f6",
                        tickfont=dict(color="#1e293b"), title_font=dict(color="#3b82f6")),
            xaxis=dict(title=x_label, tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
            height=420, plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
            font=dict(color="#1e293b"),
            legend=dict(orientation="h", y=1.08, font=dict(color="#1e293b")),
            margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">▸ <strong>Year-over-Year Summary</strong></div>', unsafe_allow_html=True)
    yoy = yoy_change(df)
    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.bar(yoy, x="year", y="avg_unemployment", color="avg_unemployment",
                      color_continuous_scale="RdYlGn_r",
                      text=yoy["avg_unemployment"].apply(lambda v: f"{v:.1f}%"),
                      labels={"year": "Year", "avg_unemployment": "Avg Unemployment (%)"},
                      title="Avg Unemployment by Year")
        fig2.update_traces(textposition="outside", textfont=dict(color="#1e293b", size=12))
        fig2.update_layout(height=360, coloraxis_showscale=False, plot_bgcolor="#f8fafc",
                           paper_bgcolor="#ffffff", font=dict(color="#1e293b"),
                           xaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.bar(yoy, x="year", y="avg_job_postings", color="avg_job_postings",
                      color_continuous_scale="Blues",
                      text=yoy["avg_job_postings"].apply(lambda v: f"{v:,.0f}"),
                      labels={"year": "Year", "avg_job_postings": "Avg Job Postings"},
                      title="Avg Job Postings by Year")
        fig3.update_traces(textposition="outside", textfont=dict(color="#1e293b", size=12))
        fig3.update_layout(height=360, coloraxis_showscale=False, plot_bgcolor="#f8fafc",
                           paper_bgcolor="#ffffff", font=dict(color="#1e293b"),
                           xaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">▸ <strong>Multi-City Unemployment Comparison (Monthly)</strong></div>', unsafe_allow_html=True)
    compare_cities = st.multiselect("Select cities to compare",
                                    options=sorted(df["location"].unique().tolist()),
                                    default=sorted(df["location"].unique().tolist())[:5],
                                    key="compare_cities")
    if compare_cities:
        records = []
        for city in compare_cities:
            m = monthly_unemployment(df, city)
            m["city"] = city
            records.append(m)
        combined = pd.concat(records)
        fig4 = px.line(combined, x="year_month", y="avg_unemployment", color="city",
                       labels={"year_month": "Month", "avg_unemployment": "Avg Unemployment (%)", "city": "City"},
                       color_discrete_sequence=px.colors.qualitative.Plotly)
        fig4.update_layout(height=400, plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
                           font=dict(color="#1e293b"),
                           xaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           legend=dict(orientation="h", y=-0.2, font=dict(color="#1e293b")),
                           margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-header">▸ <strong>Distribution of Unemployment Rates</strong></div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        fig5 = px.histogram(df, x="unemployment_rate", nbins=20,
                            color_discrete_sequence=["#3b82f6"],
                            labels={"unemployment_rate": "Unemployment Rate (%)"},
                            title="Frequency Distribution")
        fig5.update_layout(plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff", height=320,
                           font=dict(color="#1e293b"),
                           xaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig5, use_container_width=True)
    with col4:
        fig6 = px.box(df, x="year", y="unemployment_rate", color="year",
                      labels={"unemployment_rate": "Unemployment Rate (%)", "year": "Year"},
                      title="Spread by Year",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig6.update_layout(plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff", height=320,
                           font=dict(color="#1e293b"),
                           xaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig6, use_container_width=True)


# ── Page: Skills Analysis ─────────────────────────────────────────────────────

def page_skills(df: pd.DataFrame):
    st.title("🛠️ **In-Demand Skills Analysis**")
    st.markdown("**▸** Discover which technical and soft skills dominate the job market and how they correlate with unemployment.")

    if df.empty:
        st.warning("No data for the selected filters.")
        return

    col_ctrl, _ = st.columns([1, 2])
    with col_ctrl:
        top_n = st.slider("Top N skills to display", 5, 20, 12, key="top_n_skills")

    top_skills_df = skill_frequency(df, top_n=top_n)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">▸ <strong>Top In-Demand Skills (Frequency)</strong></div>', unsafe_allow_html=True)
        fig = px.bar(top_skills_df, x="count", y="skill", orientation="h",
                     color="count", color_continuous_scale="Viridis", text="count",
                     labels={"count": "Occurrences", "skill": "Skill"})
        fig.update_traces(textposition="outside", textfont=dict(color="#1e293b", size=12))
        fig.update_layout(height=420, coloraxis_showscale=False, plot_bgcolor="#f8fafc",
                          paper_bgcolor="#ffffff", font=dict(color="#1e293b"),
                          xaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                          yaxis=dict(categoryorder="total ascending", tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">▸ <strong>Skills Share (Pie)</strong></div>', unsafe_allow_html=True)
        fig2 = px.pie(top_skills_df, values="count", names="skill",
                      color_discrete_sequence=px.colors.qualitative.Bold, hole=0.35)
        fig2.update_layout(height=420, paper_bgcolor="#ffffff", font=dict(color="#1e293b"),
                           legend=dict(orientation="h", y=-0.1, x=0, font=dict(color="#1e293b")),
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">▸ <strong>Skills vs. Unemployment Rate</strong></div>', unsafe_allow_html=True)
    corr_df = skills_unemployment_correlation(df)
    fig3 = px.scatter(corr_df, x="avg_job_postings", y="avg_unemployment",
                      size="count", color="avg_unemployment",
                      color_continuous_scale="RdYlGn_r", text="skill",
                      hover_data=["skill", "count"],
                      labels={"avg_job_postings": "Avg Job Postings",
                              "avg_unemployment": "Avg Unemployment Rate (%)",
                              "count": "Occurrences"})
    fig3.update_traces(textposition="top center", textfont=dict(color="#1e293b", size=10))
    fig3.update_layout(height=480, plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
                       coloraxis_showscale=True, font=dict(color="#1e293b"),
                       xaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                       yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">▸ <strong>Skills Demand Trend Over Time</strong></div>', unsafe_allow_html=True)
    top_skill_names = top_skills_df["skill"].head(8).tolist()
    trend = skills_trend_over_time(df, top_skill_names)
    if not trend.empty:
        trend_melted = trend.melt(id_vars="year", var_name="skill", value_name="count")
        fig4 = px.line(trend_melted, x="year", y="count", color="skill", markers=True,
                       labels={"year": "Year", "count": "Occurrences", "skill": "Skill"},
                       color_discrete_sequence=px.colors.qualitative.Plotly)
        fig4.update_layout(height=380, plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
                           font=dict(color="#1e293b"),
                           xaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           legend=dict(orientation="h", y=-0.22, font=dict(color="#1e293b")),
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-header">▸ <strong>Top Skills by City</strong></div>', unsafe_allow_html=True)
    city_for_skills = st.selectbox("Select city", sorted(df["location"].unique().tolist()), key="skill_city")
    city_skills = skills_by_city(df, city_for_skills)
    if city_skills.empty:
        st.info("No skill data for this city in the current filter.")
    else:
        fig5 = px.bar(city_skills, x="skill", y="count",
                      color="count", color_continuous_scale="Tealgrn", text="count",
                      labels={"skill": "Skill", "count": "Occurrences"},
                      title=f"Top Skills — {city_for_skills}")
        fig5.update_traces(textposition="outside", textfont=dict(color="#1e293b", size=12))
        fig5.update_layout(height=360, coloraxis_showscale=False, plot_bgcolor="#f8fafc",
                           paper_bgcolor="#ffffff", font=dict(color="#1e293b"),
                           xaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-header">▸ <strong>Skills Ranked by Unemployment Association</strong></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🟢 Lowest Unemployment Skills**")
        st.dataframe(corr_df.head(8)[["skill", "avg_unemployment", "avg_job_postings", "count"]],
                     use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**🔴 Highest Unemployment Skills**")
        st.dataframe(corr_df.tail(8)[["skill", "avg_unemployment", "avg_job_postings", "count"]].iloc[::-1],
                     use_container_width=True, hide_index=True)


# ── Page: City Intelligence ───────────────────────────────────────────────────

CITY_COORDS = {
    "New York": (40.71, -74.01), "Los Angeles": (34.05, -118.24),
    "Chicago": (41.88, -87.63),  "Houston": (29.76, -95.37),
    "Phoenix": (33.45, -112.07), "Philadelphia": (39.95, -75.17),
    "San Antonio": (29.42, -98.49), "Dallas": (32.78, -96.80),
    "San Jose": (37.34, -121.89), "Austin": (30.27, -97.74),
    "Jacksonville": (30.33, -81.66), "Fort Worth": (32.75, -97.33),
    "Columbus": (39.96, -82.99), "Indianapolis": (39.77, -86.16),
    "Charlotte": (35.23, -80.84), "San Francisco": (37.77, -122.42),
    "Seattle": (47.61, -122.33), "Denver": (39.74, -104.98),
    "Washington": (38.91, -77.04), "San Diego": (32.72, -117.16),
}


def page_city(df: pd.DataFrame):
    st.title("🏙️ **City Intelligence**")
    st.markdown("**▸** Deep-dive into city-level unemployment, job market health, workforce demographics and rankings.")

    if df.empty:
        st.warning("No data for the selected filters.")
        return

    city_data = city_summary(df)
    city_data["lat"] = city_data["location"].map(lambda c: CITY_COORDS.get(c, (0, 0))[0])
    city_data["lon"] = city_data["location"].map(lambda c: CITY_COORDS.get(c, (0, 0))[1])

    st.markdown('<div class="section-header">▸ <strong>US City Map — Unemployment & Job Postings</strong></div>', unsafe_allow_html=True)
    map_metric = st.radio("Color bubble by:", ["avg_unemployment", "avg_job_postings"], horizontal=True,
                          key="map_metric",
                          format_func=lambda v: "Avg Unemployment" if v == "avg_unemployment" else "Avg Job Postings")
    fig_map = px.scatter_geo(
        city_data, lat="lat", lon="lon", size="avg_job_postings",
        color=map_metric,
        color_continuous_scale="RdYlGn_r" if map_metric == "avg_unemployment" else "Blues",
        hover_name="location",
        hover_data={"avg_unemployment": ":.1f", "avg_job_postings": ":,.0f",
                    "avg_college_pct": ":.1f", "lat": False, "lon": False},
        scope="usa", size_max=40,
        labels={"avg_unemployment": "Avg Unemployment (%)", "avg_job_postings": "Avg Job Postings"})
    fig_map.update_layout(height=480,
                          geo=dict(bgcolor="#f0f4f8", landcolor="#e2e8f0", showlakes=True, lakecolor="#bfdbfe"),
                          paper_bgcolor="#ffffff", font=dict(color="#1e293b"),
                          margin=dict(l=0, r=0, t=10, b=0),
                          coloraxis_colorbar=dict(len=0.5, y=0.5, tickfont=dict(color="#1e293b")))
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown('<div class="section-header">▸ <strong>City Rankings</strong></div>', unsafe_allow_html=True)
    rank_col, rank_order_col, _ = st.columns([2, 1, 2])
    with rank_col:
        rank_metric = st.selectbox("Rank by",
                                   ["avg_unemployment", "avg_job_postings", "avg_college_pct", "avg_age"],
                                   format_func=lambda v: {"avg_unemployment": "Avg Unemployment (%)",
                                                          "avg_job_postings": "Avg Job Postings",
                                                          "avg_college_pct": "Avg College Degree %",
                                                          "avg_age": "Avg Workforce Age"}[v],
                                   key="rank_metric")
    with rank_order_col:
        rank_asc = st.toggle("Ascending order", value=False, key="rank_asc")
    ranked = city_ranking(df, metric=rank_metric, ascending=rank_asc).reset_index(drop=True)
    ranked.index = ranked.index + 1
    st.dataframe(
        ranked[["location", "avg_unemployment", "avg_job_postings", "avg_college_pct", "avg_age", "record_count"]]
              .rename(columns={"location": "City", "avg_unemployment": "Avg Unemp %",
                               "avg_job_postings": "Avg Job Postings", "avg_college_pct": "College Grad %",
                               "avg_age": "Avg Age", "record_count": "Records"}),
        use_container_width=True, height=420)

    st.markdown('<div class="section-header">▸ <strong>City Profile Comparison (Radar)</strong></div>', unsafe_allow_html=True)
    radar_cities = st.multiselect("Cities to compare", options=sorted(df["location"].unique().tolist()),
                                  default=sorted(df["location"].unique().tolist())[:5], key="radar_cities")
    if radar_cities:
        sub = city_data[city_data["location"].isin(radar_cities)].copy()
        metrics = ["avg_unemployment", "avg_job_postings", "avg_college_pct", "avg_age"]
        labels  = ["Unemployment", "Job Postings", "College %", "Avg Age"]
        scaler  = MinMaxScaler()
        scaled  = scaler.fit_transform(sub[metrics])
        fig_radar = go.Figure()
        for i, row in enumerate(sub["location"]):
            values = list(scaled[i]) + [scaled[i][0]]
            fig_radar.add_trace(go.Scatterpolar(r=values, theta=labels + [labels[0]],
                                                fill="toself", name=row, opacity=0.6))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#1e293b")),
                       angularaxis=dict(tickfont=dict(color="#1e293b"))),
            showlegend=True, paper_bgcolor="#ffffff", font=dict(color="#1e293b"),
            legend=dict(font=dict(color="#1e293b")), height=460, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<div class="section-header">▸ <strong>Single City Deep Dive</strong></div>', unsafe_allow_html=True)
    deep_city = st.selectbox("Select a city", sorted(df["location"].unique().tolist()), key="deep_city")
    city_df   = df[df["location"] == deep_city]
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Unemployment",  f"{city_df['unemployment_rate'].mean():.1f}%")
    c2.metric("Avg Job Postings",  f"{city_df['job_postings'].mean():,.0f}")
    c3.metric("College Degree %",  f"{city_df['college_degree_percentage'].mean():.1f}%")
    monthly = monthly_unemployment(df, deep_city)
    fig_deep = px.line(monthly, x="year_month", y="avg_unemployment",
                       markers=True, color_discrete_sequence=["#ef4444"],
                       labels={"year_month": "Month", "avg_unemployment": "Avg Unemployment (%)"},
                       title=f"Monthly Unemployment — {deep_city}")
    fig_deep.update_layout(plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff", height=300,
                           font=dict(color="#1e293b"),
                           xaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_deep, use_container_width=True)


# ── Page: Predictive Insights ─────────────────────────────────────────────────

def page_predict(df: pd.DataFrame):
    st.title("🔮 **Predictive Insights**")
    st.markdown(
        "**▸** Linear-regression-based unemployment forecasting plus feature importance insights. "
        "Forecasts are trend extrapolations — treat as directional indicators, not precise predictions."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        forecast_city = st.selectbox("City to forecast",
                                     options=["All"] + sorted(df["location"].unique().tolist()),
                                     key="forecast_city")
    with col2:
        months_ahead = st.slider("Months to forecast", 1, 18, 6, key="months_ahead")

    st.markdown('<div class="section-header">▸ <strong>Unemployment Rate Forecast</strong></div>', unsafe_allow_html=True)
    forecast_df = predict_unemployment(df, forecast_city, months_ahead)

    if forecast_df.empty:
        st.warning("Not enough historical data points to build a forecast for this selection.")
    else:
        hist = forecast_df[forecast_df["type"] == "Historical"]
        pred = forecast_df[forecast_df["type"] == "Forecast"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["unemployment_rate"],
                                 name="Historical", mode="lines+markers",
                                 line=dict(color="#3b82f6", width=2.5), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=pred["date"], y=pred["unemployment_rate"],
                                 name="Forecast", mode="lines+markers",
                                 line=dict(color="#f59e0b", width=2.5, dash="dash"),
                                 marker=dict(size=7, symbol="diamond")))
        upper = pred["unemployment_rate"] * 1.15
        lower = pred["unemployment_rate"] * 0.85
        fig.add_trace(go.Scatter(
            x=pd.concat([pred["date"], pred["date"].iloc[::-1]]),
            y=pd.concat([upper, lower.iloc[::-1]]),
            fill="toself", fillcolor="rgba(245,158,11,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="Confidence band"))
        fig.update_layout(
            xaxis=dict(title="Date", tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
            yaxis=dict(title="Unemployment Rate (%)", tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
            height=420, plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
            font=dict(color="#1e293b"),
            legend=dict(orientation="h", y=1.08, font=dict(color="#1e293b")),
            margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**▸ Forecast values:**")
        st.dataframe(
            pred[["date", "unemployment_rate"]].rename(
                columns={"date": "Month", "unemployment_rate": "Forecast (%)"}
            ).assign(**{"Month": pred["date"].dt.strftime("%b %Y")}),
            use_container_width=False, hide_index=True)

    st.markdown('<div class="section-header">▸ <strong>Feature Importance for Unemployment Rate</strong></div>', unsafe_allow_html=True)
    st.markdown("Absolute Pearson correlation between each numeric feature and unemployment rate "
                "(higher = stronger linear association).")
    corr    = df[["job_postings", "average_age", "college_degree_percentage", "unemployment_rate"]].corr()["unemployment_rate"].drop("unemployment_rate")
    corr_df = corr.abs().reset_index().rename(columns={"index": "Feature", "unemployment_rate": "Correlation"})
    corr_df["Direction"] = corr.values > 0
    corr_df["Direction"] = corr_df["Direction"].map({True: "Positive", False: "Negative"})
    corr_df = corr_df.sort_values("Correlation", ascending=True)
    fig2 = px.bar(corr_df, x="Correlation", y="Feature", orientation="h",
                  color="Direction", color_discrete_map={"Positive": "#ef4444", "Negative": "#22c55e"},
                  text=corr_df["Correlation"].apply(lambda v: f"{v:.3f}"),
                  labels={"Correlation": "|Pearson r|", "Feature": ""})
    fig2.update_traces(textposition="outside", textfont=dict(color="#1e293b", size=12))
    fig2.update_layout(height=280, plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
                       font=dict(color="#1e293b"),
                       xaxis=dict(range=[0, 0.25], tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                       yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                       legend=dict(font=dict(color="#1e293b")),
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">▸ <strong>6-Month Forecast Comparison — All Cities</strong></div>', unsafe_allow_html=True)
    rows = []
    for city in sorted(df["location"].unique()):
        fdf = predict_unemployment(df, city, 6)
        if fdf.empty:
            continue
        last_hist = fdf[fdf["type"] == "Historical"]["unemployment_rate"].iloc[-1]
        last_fore = fdf[fdf["type"] == "Forecast"]["unemployment_rate"].iloc[-1]
        rows.append({"City": city, "Current Unemp %": round(last_hist, 2),
                     "6M Forecast %": round(last_fore, 2),
                     "Change": round(last_fore - last_hist, 2)})
    if rows:
        summary_df = pd.DataFrame(rows).sort_values("Change")
        summary_df["Trend"] = summary_df["Change"].apply(
            lambda v: "🟢 Improving" if v < -0.2 else ("🔴 Worsening" if v > 0.2 else "🟡 Stable"))
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        fig3 = px.bar(summary_df.sort_values("Change"), x="City", y="Change",
                      color="Change", color_continuous_scale="RdYlGn_r",
                      text=summary_df.sort_values("Change")["Change"].apply(lambda v: f"{v:+.2f}%"),
                      labels={"Change": "Projected Change (%)", "City": "City"},
                      title="Projected 6-Month Unemployment Change by City")
        fig3.update_traces(textposition="outside", textfont=dict(color="#1e293b", size=11))
        fig3.update_layout(height=380, plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
                           coloraxis_showscale=False, font=dict(color="#1e293b"),
                           xaxis=dict(tickangle=-35, tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           yaxis=dict(tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")),
                           margin=dict(l=10, r=10, t=40, b=50))
        st.plotly_chart(fig3, use_container_width=True)


# ── Page: Raw Data Explorer ───────────────────────────────────────────────────

def page_rawdata(df: pd.DataFrame):
    st.title("📋 **Raw Data Explorer**")
    st.markdown("**▸** Browse, filter, and export the underlying dataset.")

    if df.empty:
        st.warning("No data for the selected filters.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        search_city = st.multiselect("Filter by City", options=sorted(df["location"].unique()),
                                     default=[], key="raw_city")
    with col2:
        unemp_range = st.slider("Unemployment Rate (%)",
                                float(df["unemployment_rate"].min()), float(df["unemployment_rate"].max()),
                                (float(df["unemployment_rate"].min()), float(df["unemployment_rate"].max())),
                                key="raw_unemp")
    with col3:
        postings_range = st.slider("Job Postings",
                                   int(df["job_postings"].min()), int(df["job_postings"].max()),
                                   (int(df["job_postings"].min()), int(df["job_postings"].max())),
                                   key="raw_postings")

    filtered = df.copy()
    if search_city:
        filtered = filtered[filtered["location"].isin(search_city)]
    filtered = filtered[
        (filtered["unemployment_rate"] >= unemp_range[0]) &
        (filtered["unemployment_rate"] <= unemp_range[1]) &
        (filtered["job_postings"]      >= postings_range[0]) &
        (filtered["job_postings"]      <= postings_range[1])
    ]

    st.markdown(f"**▸ {len(filtered):,} records** matching filters")
    display_cols = ["id", "date", "location", "unemployment_rate", "job_postings",
                    "in_demand_skills", "average_age", "college_degree_percentage"]
    st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True, height=460)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("⬇️ Download CSV",
                           data=filtered[display_cols].to_csv(index=False).encode("utf-8"),
                           file_name="job_market_filtered.csv", mime="text/csv")
    with col_dl2:
        st.download_button("⬇️ Download JSON",
                           data=filtered[display_cols].to_json(orient="records", date_format="iso").encode("utf-8"),
                           file_name="job_market_filtered.json", mime="application/json")

    st.markdown("---")
    st.markdown("### ▸ **Quick Statistics**")
    st.dataframe(
        filtered[["unemployment_rate", "job_postings", "average_age", "college_degree_percentage"]]
        .describe().round(2),
        use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — STREAMLIT APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Job Market & Unemployment Trends",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label { color: #e2e8f0 !important; }

    /* KPI cards */
    .kpi-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        border-left: 4px solid #3b82f6;
        margin-bottom: 8px;
    }
    .kpi-label { color: #94a3b8; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
    .kpi-value { color: #f1f5f9; font-size: 32px; font-weight: 700; margin: 4px 0 0; line-height: 1.1; }
    .kpi-sub   { color: #64748b; font-size: 12px; margin-top: 4px; }

    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 900;
        color: #0f172a;
        letter-spacing: 0.02em;
        border-left: 5px solid #3b82f6;
        border-bottom: 2px solid #e2e8f0;
        padding: 6px 0 8px 14px;
        margin: 32px 0 18px;
        background: linear-gradient(90deg, #eff6ff 0%, transparent 80%);
        border-radius: 0 6px 6px 0;
        text-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }

    /* Hide default Streamlit hamburger watermark */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data
def get_data():
    return load_data()

df_full = get_data()
cities  = sorted(df_full["location"].unique().tolist())

# ── Sidebar navigation & global filters ──────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Job Market Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📈 Unemployment Trends", "🛠️ Skills Analysis",
         "🏙️ City Intelligence", "🔮 Predictive Insights", "📋 Raw Data"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### 🎯 Quick Filters")
    selected_cities = st.multiselect("Filter Cities", options=cities, default=cities, key="global_cities")
    year_range = st.slider(
        "Year Range",
        min_value=int(df_full["year"].min()),
        max_value=int(df_full["year"].max()),
        value=(int(df_full["year"].min()), int(df_full["year"].max())),
        key="global_years",
    )
    st.markdown("---")
    st.caption(f"📁 Dataset: {len(df_full):,} records · {df_full['location'].nunique()} cities")

# ── Apply global filters ──────────────────────────────────────────────────────
filtered_df = df_full[
    (df_full["location"].isin(selected_cities)) &
    (df_full["year"] >= year_range[0]) &
    (df_full["year"] <= year_range[1])
]

# ── Page router ───────────────────────────────────────────────────────────────
if   page == "🏠 Overview":             page_overview(filtered_df)
elif page == "📈 Unemployment Trends":  page_trends(filtered_df)
elif page == "🛠️ Skills Analysis":      page_skills(filtered_df)
elif page == "🏙️ City Intelligence":    page_city(filtered_df)
elif page == "🔮 Predictive Insights":  page_predict(df_full)   # unfiltered — full history for ML
elif page == "📋 Raw Data":             page_rawdata(filtered_df)
