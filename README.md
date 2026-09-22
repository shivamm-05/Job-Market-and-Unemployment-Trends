# 📊 **Job Market & Unemployment Trends Dashboard**

**▸** A fully interactive data analytics dashboard built with **Python + Streamlit** that explores unemployment rates, job postings, in-demand skills, and workforce demographics across 20 major US cities from July 2023 to July 2025.

---

## 🗂️ **Project Structure**

```
Shivam_Yadav_Project/
├── app.py                              # Streamlit entry point + global sidebar/filters
├── data_processor.py                   # Backend: data loading, aggregation, ML logic
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── job_market_unemployment_trends.csv  # Source dataset (1,000 records)
└── pages_app/
    ├── __init__.py
    ├── page_overview.py                # Overview & KPI Dashboard
    ├── page_trends.py                  # Unemployment Trends (time series)
    ├── page_skills.py                  # In-Demand Skills Analysis
    ├── page_city.py                    # City / Location Intelligence
    ├── page_predict.py                 # Predictive Insights (ML forecasting)
    └── page_rawdata.py                 # Raw Data Explorer & Export
```

---

## 🚀 **Getting Started**

### **▸ Prerequisites**

- **•** Python 3.9 or higher
- **•** pip

### **▸ Installation**

1. **•** Clone or download the project folder

2. **•** Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. **•** Run the app

   ```bash
   streamlit run app.py
   ```

4. **•** Open in browser — Streamlit will automatically open `http://localhost:8501`

---

## 📦 **Dependencies**

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.35.0 | Frontend / UI framework |
| `pandas` | ≥ 2.0.0 | Data loading & manipulation |
| `numpy` | ≥ 1.26.0 | Numerical computation |
| `plotly` | ≥ 5.20.0 | Interactive charts & maps |
| `scikit-learn` | ≥ 1.4.0 | Linear regression forecasting |

---

## 📁 **Dataset**

**▸ File:** `job_market_unemployment_trends.csv`

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique record identifier |
| `date` | date | Observation date (YYYY-MM-DD) |
| `location` | string | US city name |
| `unemployment_rate` | float | Unemployment rate (%) |
| `job_postings` | int | Number of active job postings |
| `in_demand_skills` | string | Comma-separated list of top skills |
| `average_age` | int | Average workforce age |
| `college_degree_percentage` | int | % of workforce with a college degree |

**▸ Coverage:** 1,000 records · 20 US cities · July 2023 – July 2025

---

## 🖥️ **Dashboard Pages**

### 🏠 **Overview**
- **▸ 4 KPI cards** — avg/min/max unemployment, total job postings, cities covered, college %
- **▸ City bar chart** — horizontal bar ranked by unemployment rate
- **▸ Scatter plot** — job postings vs unemployment, sized by college degree %
- **▸ Education & Age bucket charts** — how degree attainment and workforce age relate to unemployment
- **▸ Pearson correlation heatmap** — numeric feature relationships

### 📈 **Unemployment Trends**
- **▸ Monthly / Quarterly dual-axis chart** — unemployment rate (line) overlaid with job postings (bar)
- **▸ Year-over-Year bar charts** — avg unemployment and avg postings per year
- **▸ Multi-city comparison** — select any combination of cities for side-by-side line chart
- **▸ Distribution charts** — histogram and box-plot of unemployment across years

### 🛠️ **Skills Analysis**
- **▸ Top-N skills bar & pie chart** — frequency of in-demand skills across all records
- **▸ Skills × Unemployment scatter** — identify which skills correlate with low/high unemployment
- **▸ Demand trend lines** — how the top 8 skills have grown or declined year-over-year
- **▸ Per-city top skills** — drill into any city's most-requested skills
- **▸ Best / worst skill tables** — ranked by unemployment association

### 🏙️ **City Intelligence**
- **▸ US bubble geo-map** — cities plotted on a US map, sized by job postings, coloured by unemployment
- **▸ Sortable rankings table** — rank all 20 cities by any metric
- **▸ Multi-city radar chart** — normalised comparison of unemployment, postings, education, and age
- **▸ Single-city deep-dive** — KPI metrics + monthly unemployment line chart for any city

### 🔮 **Predictive Insights**
- **▸ Unemployment forecast** — linear regression projection up to 18 months ahead with confidence band
- **▸ Feature importance** — absolute Pearson correlation of each feature against unemployment rate
- **▸ All-city 6-month forecast table** — projected change for every city with trend indicator (🟢/🟡/🔴)
- **▸ City comparison bar chart** — visualises projected unemployment change direction

### 📋 **Raw Data Explorer**
- **▸ Filter panel** — filter by city, unemployment range, and job postings range
- **▸ Interactive data table** — browse all 1,000 records
- **▸ Download buttons** — export filtered data as **CSV** or **JSON**
- **▸ Quick statistics** — `describe()` summary of numeric columns

---

## 🌐 **Global Filters (Sidebar)**

**▸** Every page respects two global filters set in the sidebar:

| Filter | Description |
|--------|-------------|
| **Filter Cities** | Multi-select to include/exclude specific cities |
| **Year Range** | Slider to restrict data to a date range |

> **▸ Note:** The **Predictive Insights** page uses the full unfiltered dataset for training forecasts to preserve historical context.

---

## 🧠 **Backend — `data_processor.py`**

**▸** All analytics logic is separated from the UI in `data_processor.py`:

| Function | Description |
|----------|-------------|
| `load_data()` | Reads CSV, parses dates, engineers features |
| `get_kpis()` | Returns key summary statistics |
| `monthly_unemployment()` | Monthly avg unemployment & postings (per city or all) |
| `quarterly_unemployment()` | Quarterly aggregation |
| `yoy_change()` | Year-over-year unemployment and postings changes |
| `city_summary()` | Per-city aggregate metrics |
| `city_ranking()` | Cities ranked by any chosen metric |
| `skill_frequency()` | Top-N skill occurrence counts |
| `skills_by_city()` | Skill counts for a specific city |
| `skills_unemployment_correlation()` | Avg unemployment per skill |
| `skills_trend_over_time()` | Year × skill occurrence matrix |
| `correlation_matrix()` | Pearson correlation of numeric features |
| `predict_unemployment()` | Linear regression forecast with historical + future rows |
| `education_unemployment_buckets()` | Unemployment grouped by education level bands |
| `age_unemployment_buckets()` | Unemployment grouped by age bands |

---

## 📸 **Screenshots**

**▸** Run the app locally with `streamlit run app.py` to explore all interactive charts and filters.

---

## 📝 **Notes**

- **▸** Forecasts use simple **linear regression** on monthly averages and are intended as directional trend indicators, not precise predictions.
- **▸** The confidence band on the forecast chart is an illustrative ±15% envelope.
- **▸** City coordinates for the geo-map are approximate city centres.

---

## 👤 **Author**

- **▸ Shivam Yadav**
- **▸** Project: Job Market & Unemployment Trends Analysis
- **▸** Dataset: `job_market_unemployment_trends.csv`
