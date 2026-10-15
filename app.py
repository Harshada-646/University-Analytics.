"""
UNIVERSITY ANALYTICS USING PYTHON FOR DATA SCIENCE
Department: Computer Engineering | Course: Python for Data Science (GTU)
Single-file Streamlit application covering:
 1. Dataset upload / preview
 2. Data cleaning & preprocessing
 3. Statistics & probability
 4. Data visualization (bar, pie, line, scatter, histogram, heatmap)
 5. Machine learning (Linear Regression, Logistic Regression, Decision Tree,
    Random Forest, XGBoost) for placement probability, backlog risk & package prediction
 6. New student self-registration (appends to the CSV so data persists)
 7. Explainable AI (permutation importance, model-native importance, partial
    dependence, per-student "why this prediction?" explanation, decision-tree rules)

The dashboard stays empty until a CSV is uploaded via the sidebar — nothing
loads automatically on startup.
"""

import base64
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# --------------------------------------------------------------------------------------
# APP CONFIG & THEME
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="University Analytics | Python for Data Science",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_PATH = "university_data.csv"          # lives next to app.py, same as your project folder
REQUIRED_COLS = [
    "Student ID", "Student Name", "Department", "Semester", "Attendance",
    "CGPA", "Backlogs", "Fees Paid", "Scholarship", "Placement", "Package",
]

# --- Generated background: a light, sunlit university-building scene (classical
# facade with columns, pediment and steps, warm sky, soft glow) — a real "place"
# illustration rather than abstract data-graphics, in the chosen 5-tone gradient
# palette (dark navy-teal, slate blue-gray, light gray, warm tan, black). Built as
# SVG and base64-embedded so the app never depends on an external image URL.
_BG_SVG = """<svg xmlns='http://www.w3.org/2000/svg' width='1600' height='1000'>
  <defs>
    <linearGradient id='sky' x1='0%' y1='0%' x2='0%' y2='100%'>
      <stop offset='0%' stop-color='#C9C6BD'/>
      <stop offset='100%' stop-color='#9E9B92'/>
    </linearGradient>
    <radialGradient id='sun' cx='82%' cy='16%' r='38%'>
      <stop offset='0%' stop-color='#D3985F' stop-opacity='0.32'/>
      <stop offset='100%' stop-color='#D3985F' stop-opacity='0'/>
    </radialGradient>
  </defs>

  <rect width='1600' height='1000' fill='url(#sky)'/>
  <rect width='1600' height='1000' fill='url(#sun)'/>

  <!-- distant tree canopies either side, soft and muted -->
  <g fill='#0E0E0E' opacity='0.08'>
    <circle cx='120' cy='650' r='150'/>
    <circle cx='230' cy='690' r='110'/>
    <circle cx='1480' cy='650' r='150'/>
    <circle cx='1370' cy='690' r='110'/>
  </g>

  <!-- ground -->
  <rect x='0' y='930' width='1600' height='70' fill='#5D7A8C' opacity='0.22'/>

  <!-- classical university building: pediment + columns + steps.
       Kept at low opacity throughout — it must never render as a solid dark
       shape, or dark text placed over it (headings, chart legends) would
       become unreadable. This is a faint watermark, not a bold silhouette. -->
  <g opacity='0.16'>
    <polygon points='230,560 800,400 1370,560' fill='#14313F'/>
    <rect x='230' y='555' width='1140' height='30' fill='#14313F'/>
    <rect x='230' y='585' width='1140' height='300' fill='#14313F'/>
    <g fill='#D8D5CC'>
      <rect x='280' y='610' width='46' height='260'/>
      <rect x='420' y='610' width='46' height='260'/>
      <rect x='560' y='610' width='46' height='260'/>
      <rect x='700' y='610' width='46' height='260'/>
      <rect x='840' y='610' width='46' height='260'/>
      <rect x='980' y='610' width='46' height='260'/>
      <rect x='1120' y='610' width='46' height='260'/>
      <rect x='1260' y='610' width='46' height='260'/>
    </g>
    <rect x='170' y='885' width='1260' height='22' fill='#D3985F' opacity='0.85'/>
    <rect x='120' y='907' width='1360' height='22' fill='#D3985F' opacity='0.65'/>
    <rect x='70' y='929' width='1460' height='22' fill='#D3985F' opacity='0.45'/>
  </g>
</svg>"""
_BG_URI = "data:image/svg+xml;base64," + base64.b64encode(_BG_SVG.encode()).decode()

st.markdown(f"""
<style>
    /* Palette: Navy-Teal #14313F · Slate #5D7A8C · Light Gray #C9C6BD · Warm Tan #D3985F · Black #0E0E0E */
    .stApp {{
        background-color: #C9C6BD;
        background-image: linear-gradient(rgba(201,198,189,0.82), rgba(201,198,189,0.82)),
                           url("{_BG_URI}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #0E0E0E;
        border-right: 1px solid rgba(211, 152, 95, 0.5);
    }}
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {{
        color: #C9C6BD !important;
    }}
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        color: #A9A69D !important;
    }}

    h1, h2, h3, h4 {{ color: #14313F !important; font-family: 'Georgia', serif; }}

    /* Scoped to Streamlit's own text blocks — NOT a blanket span/label rule, so it
       never bleeds into other widgets' internal markup (that was the tag-contrast bug). */
    p, li, label, .stMarkdown {{ color: #3A4C54 !important; }}
    .stMarkdown span {{ color: #3A4C54 !important; }}
    [data-testid="stCaptionContainer"] {{ color: #6B7A80 !important; }}
    [data-testid="stWidgetLabel"] p {{ color: #3A4C54 !important; }}

    /* Selected-value chips inside multiselect widgets */
    span[data-baseweb="tag"] {{ background-color: #14313F !important; border-color: #14313F !important; }}
    span[data-baseweb="tag"] span {{ color: #FFFFFF !important; }}
    span[data-baseweb="tag"] svg {{ fill: #FFFFFF !important; }}

    .metric-card {{
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #D3985F;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(14,14,14,0.12);
    }}
    .metric-card h2 {{ color: #14313F !important; margin: 0; font-size: 28px; }}
    .metric-card p {{ color: #6B7A80 !important; margin: 4px 0 0 0; font-size: 13px; }}

    div[data-testid="stDataFrame"] {{ border: 1px solid #D3985F; border-radius: 8px; }}
    .stButton>button {{
        background-color: #D3985F; color: #14313F; font-weight: 700; border: none;
        border-radius: 6px;
    }}
    .stButton>button:hover {{ background-color: #14313F; color: #D3985F; }}
</style>
""", unsafe_allow_html=True)

NAVY, GOLD, BEIGE, CHARCOAL = "#14313F", "#D3985F", "#C9C6BD", "#0E0E0E"
SLATE = "#5D7A8C"
PALETTE = ["#D3985F", "#14313F", "#5D7A8C", "#0E0E0E", "#C9C6BD"]

# Fixed color scale for correlation heatmaps: light gray -> slate -> navy-teal
THEME_COLORSCALE = [[0.0, BEIGE], [0.5, SLATE], [1.0, NAVY]]


def style_fig(fig, height=340):
    """Apply the app's theme to every Plotly chart. paper_bgcolor and the legend
    background are solid-ish (not transparent) on purpose: a fully transparent
    chart background would let the page's illustrated background show through,
    and dark text (titles, legends) could then land on a dark part of that
    illustration and become unreadable. Giving every chart its own light "card"
    background guarantees NAVY text is always readable, regardless of what's
    behind it on the page."""
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0.90)",
        plot_bgcolor="rgba(255,255,255,0.60)",
        font=dict(color=NAVY, family="Georgia, serif", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        legend=dict(bgcolor="rgba(255,255,255,0.90)", font=dict(color=NAVY)),
        hoverlabel=dict(bgcolor=NAVY, font_color="#FFFFFF"),
        coloraxis_colorbar=dict(tickfont=dict(color=NAVY), title_font=dict(color=NAVY)),
    )
    fig.update_xaxes(gridcolor="rgba(20,49,63,0.10)", zerolinecolor="rgba(20,49,63,0.20)",
                      tickfont=dict(color=NAVY), title_font=dict(color=NAVY), color=NAVY)
    fig.update_yaxes(gridcolor="rgba(20,49,63,0.10)", zerolinecolor="rgba(20,49,63,0.20)",
                      tickfont=dict(color=NAVY), title_font=dict(color=NAVY), color=NAVY)
    return fig


# --------------------------------------------------------------------------------------
# DATA HELPERS
# --------------------------------------------------------------------------------------
def parse_package(value):
    """Turn '₹6.6 LPA' into 6.6, and '—' / blanks / garbage into NaN."""
    if pd.isna(value):
        return np.nan
    # Requires at least one digit — the old pattern r"[\d.]+" could match a lone
    # "." with no digits (e.g. a stray "." in messy data), and float(".") crashes.
    match = re.search(r"\d+\.?\d*", str(value))
    if not match:
        return np.nan
    try:
        return float(match.group())
    except ValueError:
        return np.nan


def clean_dataset(df_raw: pd.DataFrame):
    """Generic cleaning pipeline: missing values, duplicates, formats, outliers.
    Written to survive genuinely messy/uncleaned CSVs — blank rows, mixed-type
    numeric columns, empty strings, "N/A" placeholders, fully-blank columns —
    without ever crashing the rest of the app."""
    df = df_raw.copy()
    report = {}

    # Standardise column names, drop fully-blank rows (common trailing-line issue
    # in real-world exports)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)

    numeric_cols = ["Semester", "Attendance", "CGPA", "Backlogs", "Fees Paid"]
    # Known categorical/text columns, fixed by name rather than auto-detected —
    # auto-detection (select_dtypes) misclassifies a numeric column as "text"
    # whenever dirty values (blanks, "N/A") make pandas read it as object dtype,
    # which was the root cause of uncleaned CSVs breaking the app.
    categorical_cols = [c for c in ["Student ID", "Student Name", "Department",
                                     "Scholarship", "Placement"] if c in df.columns]

    for c in categorical_cols:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace(["nan", "None", "NaN", "null", "NULL", ""], np.nan)

    # Canonicalise case for the two fixed-vocabulary columns — "placed", "PLACED "
    # etc. in messy data would otherwise silently fail to match "Placed" anywhere
    # downstream (color maps, ML label matching).
    for c in ["Placement", "Scholarship"]:
        if c in df.columns:
            df[c] = df[c].str.title()

    # Department doesn't follow simple Title Case (e.g. "IT"), so instead of
    # forcing a case rule, consolidate values that only differ by case/whitespace
    # ("Computer Eng." vs "computer eng.") into whichever spelling is most common —
    # otherwise they'd count as two separate departments everywhere downstream.
    if "Department" in df.columns:
        key = df["Department"].str.lower()
        canonical = df.groupby(key)["Department"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
        )
        df["Department"] = key.map(canonical)

    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    report["missing_before"] = int(df[[c for c in numeric_cols if c in df.columns]].isna().sum().sum())
    for c in numeric_cols:
        if c in df.columns and df[c].isna().any():
            median_val = df[c].median()
            df[c] = df[c].fillna(median_val if pd.notna(median_val) else 0)

    for c in categorical_cols:
        if df[c].isna().any():
            mode_val = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    # Outlier capping (IQR method) instead of dropping rows
    outliers_capped = {}
    for c in ["Attendance", "CGPA", "Fees Paid"]:
        if c in df.columns:
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((df[c] < low) | (df[c] > high)).sum())
            outliers_capped[c] = n_out
            df[c] = df[c].clip(lower=low, upper=high)
    report["outliers_capped"] = outliers_capped

    # Realistic domain bounds
    if "Attendance" in df.columns:
        df["Attendance"] = df["Attendance"].clip(0, 100)
    if "CGPA" in df.columns:
        df["CGPA"] = df["CGPA"].clip(0, 10)
    if "Backlogs" in df.columns:
        df["Backlogs"] = df["Backlogs"].clip(lower=0).round().astype(int)
    if "Semester" in df.columns:
        df["Semester"] = df["Semester"].clip(1, 8).round().astype(int)

    if "Package" in df.columns:
        df["Package_LPA"] = df["Package"].apply(parse_package)

    df = df.reset_index(drop=True)
    return df, report


def save_csv(df: pd.DataFrame):
    """Persist the working dataframe back to disk (drops helper columns first)."""
    export_cols = [c for c in REQUIRED_COLS if c in df.columns]
    df[export_cols].to_csv(CSV_PATH, index=False)


def next_student_id(df: pd.DataFrame) -> str:
    nums = df["Student ID"].astype(str).str.extract(r"(\d+)")[0].dropna().astype(int)
    nxt = (nums.max() + 1) if not nums.empty else 1
    return f"STU{nxt:03d}"


# --------------------------------------------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------------------------------------------
# NOTE: we deliberately do NOT auto-load university_data.csv from disk here.
# The dashboard should stay empty until the person uploads a CSV themselves.
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None

# --------------------------------------------------------------------------------------
# SIDEBAR — navigation + data source
# --------------------------------------------------------------------------------------
st.sidebar.markdown("## 🎓 University Analytics")
st.sidebar.caption("Python for Data Science • GTU • Computer Engineering")

uploaded = st.sidebar.file_uploader("Upload / replace CSV", type=["csv"])
if uploaded is not None:
    try:
        new_df = pd.read_csv(uploaded)
        missing = [c for c in REQUIRED_COLS if c not in new_df.columns]
        if missing:
            st.sidebar.error(f"CSV missing columns: {', '.join(missing)}")
        else:
            st.session_state.raw_df = new_df
            save_csv(new_df)
            st.sidebar.success(f"Loaded {len(new_df)} rows and saved as {CSV_PATH}")
    except Exception as e:
        st.sidebar.error(f"Couldn't read that file: {e}")

if st.session_state.raw_df is None:
    st.sidebar.warning("⬆️ Upload a CSV above to unlock the dashboard.")
    st.title("🎓 University Analytics Using Python for Data Science")
    st.info(
        "👋 No dataset loaded yet. Use **Upload / replace CSV** in the sidebar to load your "
        "`university_data.csv` (or any CSV with the same columns) and the dashboard will populate."
    )
    st.caption(f"Required columns: `{', '.join(REQUIRED_COLS)}`")
    st.stop()

try:
    df_clean, clean_report = clean_dataset(st.session_state.raw_df)
except Exception as e:
    st.error(
        f"⚠️ Couldn't process this CSV — {e}\n\n"
        "This usually means a column has an unexpected format. Check that "
        f"`{', '.join(REQUIRED_COLS)}` all exist and try re-uploading."
    )
    st.stop()
st.session_state.df = df_clean

st.sidebar.metric("Total Students", len(df_clean))
st.sidebar.download_button(
    "⬇️ Download current dataset",
    df_clean[[c for c in REQUIRED_COLS if c in df_clean.columns]].to_csv(index=False),
    file_name="university_data.csv",
    mime="text/csv",
)

PAGES = [
    "🏠 Home",
    "📁 Dataset Explorer",
    "🧹 Data Cleaning",
    "📊 Statistics & Probability",
    "📈 Visualizations",
    "🤖 Machine Learning",
    "🔍 Explainable AI",
    "📝 Register New Student",
]
page = st.sidebar.radio("Navigate", PAGES)

df = st.session_state.df  # cleaned working copy used everywhere below

# Fixed color mapping so the same category always gets the same color across
# every chart in the app (Home, Statistics, Visualizations).
DEPT_COLOR_MAP = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(sorted(df["Department"].unique()))}
PLACEMENT_COLOR_MAP = {"Placed": GOLD, "Not Placed": NAVY, "In Process": SLATE}
SCHOLARSHIP_COLOR_MAP = {"Yes": GOLD, "Partial": SLATE, "No": NAVY}

# --------------------------------------------------------------------------------------
# PAGE: HOME
# --------------------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("🎓 University Analytics Dashboard")
    st.caption("Python for Data Science • GTU • Computer Engineering")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Total Students", "Departments", "Avg. CGPA", "Placement Rate"],
        [
            len(df),
            df["Department"].nunique(),
            f"{df['CGPA'].mean():.2f}",
            f"{(df['Placement'] == 'Placed').mean() * 100:.1f}%",
        ],
    ):
        with col:
            st.markdown(f"""<div class="metric-card"><h2>{value}</h2><p>{label}</p></div>""",
                        unsafe_allow_html=True)

    st.markdown("")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("##### Students by Department")
        dept_counts = df["Department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Students"]
        fig = px.bar(dept_counts, x="Department", y="Students", color="Department",
                     color_discrete_map=DEPT_COLOR_MAP, text="Students")
        fig.update_traces(marker_line_color=NAVY, marker_line_width=0.8, textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)
    with v2:
        st.markdown("##### Placement Status")
        vc = df["Placement"].value_counts().reset_index()
        vc.columns = ["Placement", "Count"]
        fig = px.pie(vc, names="Placement", values="Count", color="Placement",
                     color_discrete_map=PLACEMENT_COLOR_MAP, hole=0.35)
        fig.update_traces(marker_line_color=NAVY, marker_line_width=1, textinfo="label+percent", textposition="outside")
        st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)

    st.markdown("##### Latest Records")
    st.dataframe(df.drop(columns=["Package_LPA"], errors="ignore").tail(8), use_container_width=True)

# --------------------------------------------------------------------------------------
# PAGE: DATASET EXPLORER
# --------------------------------------------------------------------------------------
elif page == "📁 Dataset Explorer":
    st.title("📁 Sample University Dataset")
    st.dataframe(df.drop(columns=["Package_LPA"], errors="ignore"), use_container_width=True, height=380)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🎓 Academic Fields**")
        st.caption("Attendance, CGPA, Backlogs, Semester — used for performance evaluation and predictive modelling.")
    with c2:
        st.markdown("**💰 Administrative Fields**")
        st.caption("Fees Paid, Scholarship Status — used for financial tracking, eligibility filtering and reporting.")
    with c3:
        st.markdown("**💼 Placement Fields**")
        st.caption("Placement Status, Package — used for placement analytics and department-wise comparison.")

    st.markdown("### 🔎 Filter")
    dept_filter = st.multiselect("Department", sorted(df["Department"].unique()))
    placement_filter = st.multiselect("Placement Status", sorted(df["Placement"].unique()))
    filtered = df.copy()
    if dept_filter:
        filtered = filtered[filtered["Department"].isin(dept_filter)]
    if placement_filter:
        filtered = filtered[filtered["Placement"].isin(placement_filter)]
    st.dataframe(filtered.drop(columns=["Package_LPA"], errors="ignore"), use_container_width=True)
    st.caption(f"Showing {len(filtered)} of {len(df)} students")

# --------------------------------------------------------------------------------------
# PAGE: DATA CLEANING
# --------------------------------------------------------------------------------------
elif page == "🧹 Data Cleaning":
    st.title("🧹 Data Cleaning & Preprocessing")
    st.write(
        "Raw data may contain missing values, duplicates, inconsistent formats and outliers. "
        "Using Pandas and NumPy, this pipeline cleans the data, handles missing values, "
        "removes duplicates, standardises formats and caps outliers before analysis and modelling."
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card"><h2>{clean_report['duplicates_removed']}</h2>
                <p>Duplicate rows removed</p></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card"><h2>{clean_report['missing_before']}</h2>
                <p>Missing numeric values filled</p></div>""", unsafe_allow_html=True)
    total_outliers = sum(clean_report["outliers_capped"].values())
    c3.markdown(f"""<div class="metric-card"><h2>{total_outliers}</h2>
                <p>Outliers capped (IQR method)</p></div>""", unsafe_allow_html=True)

    st.markdown("#### Outliers capped per column")
    st.table(pd.DataFrame(list(clean_report["outliers_capped"].items()), columns=["Column", "Outliers capped"]))

    st.markdown("#### RAW data (as uploaded)")
    st.dataframe(st.session_state.raw_df.head(15), use_container_width=True)
    st.markdown("#### CLEAN data (after preprocessing)")
    st.dataframe(df.drop(columns=["Package_LPA"], errors="ignore").head(15), use_container_width=True)

# --------------------------------------------------------------------------------------
# PAGE: STATISTICS & PROBABILITY
# --------------------------------------------------------------------------------------
elif page == "📊 Statistics & Probability":
    st.title("📊 Statistics & Probability")

    st.markdown("### Descriptive Statistics")
    numeric_cols = ["Semester", "Attendance", "CGPA", "Backlogs", "Fees Paid"]
    desc = df[numeric_cols].agg(["mean", "median", "std", "min", "max"]).T
    desc["mode"] = [df[c].mode().iloc[0] for c in numeric_cols]
    st.dataframe(desc.round(2), use_container_width=True)

    st.markdown("### Correlation Analysis")
    corr_cols = numeric_cols + (["Package_LPA"] if "Package_LPA" in df.columns else [])
    corr = df[corr_cols].corr()
    fig = px.imshow(corr, color_continuous_scale=THEME_COLORSCALE, aspect="auto",
                    labels=dict(color="Correlation"))
    fig.update_traces(hovertemplate="%{x} × %{y}<br>Correlation: %{z:.2f}<extra></extra>")
    st.plotly_chart(style_fig(fig, height=380), use_container_width=True, theme=None)

    st.markdown("### Probability of Placement")
    thresh_att = st.slider("Attendance threshold (%)", 50, 100, 80)
    thresh_cgpa = st.slider("CGPA threshold", 4.0, 10.0, 5.0)
    p_overall = (df["Placement"] == "Placed").mean()
    subset = df[(df["Attendance"] >= thresh_att) & (df["CGPA"] >= thresh_cgpa)]
    p_conditional = (subset["Placement"] == "Placed").mean() if len(subset) else float("nan")
    c1, c2 = st.columns(2)
    c1.markdown(f"""<div class="metric-card"><h2>{p_overall*100:.1f}%</h2>
                <p>Overall P(Placed)</p></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card"><h2>{'N/A' if pd.isna(p_conditional) else f'{p_conditional*100:.1f}%'}</h2>
                <p>P(Placed | Attendance ≥ {thresh_att}% & CGPA ≥ {thresh_cgpa})<br>n={len(subset)}</p></div>""",
                unsafe_allow_html=True)

    
    st.markdown("### Hypothesis Testing")

    st.caption(
        "Independent-samples t-test — does CGPA genuinely differ "
        "between placed and non-placed students?"
    )

    # Separate CGPA values for both groups
    placed_cgpa = df.loc[
        df["Placement"] == "Placed", "CGPA"
    ].dropna()

    not_placed_cgpa = df.loc[
        df["Placement"] == "Not Placed", "CGPA"
    ].dropna()

    if len(placed_cgpa) > 1 and len(not_placed_cgpa) > 1:

        # Perform Welch's independent-samples t-test
        t_stat, p_val = stats.ttest_ind(
            placed_cgpa,
            not_placed_cgpa,
            equal_var=False
        )

        alpha = 0.05

        # Display hypothesis
        st.write(
            "H0: mean CGPA(Placed) = mean CGPA(Not Placed)"
        )

        st.write(
            f"t-statistic = {t_stat:.3f}, "
            f"p-value = {p_val:.4f} (α = {alpha})"
        )

        # Display the result correctly
        if p_val < alpha:
            st.success(
                "Reject H0 → CGPA difference is statistically significant."
            )
        else:
            st.info(
                "Fail to reject H0 → no significant CGPA difference detected."
            )

    else:
        st.info(
            "Not enough valid CGPA values in both groups to run this test."
        )

# --------------------------------------------------------------------------------------
# PAGE: VISUALIZATIONS
# --------------------------------------------------------------------------------------
elif page == "📈 Visualizations":
    st.title("📈 Data Visualization")
    st.caption("Visualization turns complex data into easy-to-understand insights.")

    row1c1, row1c2 = st.columns(2)
    with row1c1:
        st.markdown("**Bar Chart — Placement count by department**")
        ct = pd.crosstab(df["Department"], df["Placement"]).reset_index().melt(
            id_vars="Department", var_name="Placement", value_name="Students")
        fig = px.bar(ct, x="Department", y="Students", color="Placement", barmode="group",
                     color_discrete_map=PLACEMENT_COLOR_MAP)
        fig.update_traces(marker_line_color=NAVY, marker_line_width=0.6)
        st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)
    with row1c2:
        st.markdown("**Pie Chart — Scholarship distribution**")
        vc = df["Scholarship"].value_counts().reset_index()
        vc.columns = ["Scholarship", "Count"]
        fig = px.pie(vc, names="Scholarship", values="Count", color="Scholarship",
                     color_discrete_map=SCHOLARSHIP_COLOR_MAP, hole=0.35)
        fig.update_traces(marker_line_color=NAVY, marker_line_width=1, textinfo="label+percent", textposition="outside")
        st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)

    row2c1, row2c2 = st.columns(2)
    with row2c1:
        st.markdown("**Line Chart — Average CGPA by semester**")
        trend = df.groupby("Semester")["CGPA"].mean().sort_index().reset_index()
        fig = px.line(trend, x="Semester", y="CGPA", markers=True,
                      color_discrete_sequence=[GOLD])
        fig.update_traces(line_width=3, marker_size=9,
                          marker_line_color=NAVY, marker_line_width=1)
        st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)
    with row2c2:
        st.markdown("**Scatter Plot — Attendance vs CGPA**")
        fig = px.scatter(df, x="Attendance", y="CGPA", color="Placement",
                         color_discrete_map=PLACEMENT_COLOR_MAP,
                         hover_data=["Student Name", "Department"])
        fig.update_traces(marker=dict(size=9, line=dict(width=0.6, color=NAVY)))
        st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)

    row3c1, row3c2 = st.columns(2)
    with row3c1:
        st.markdown("**Histogram — CGPA distribution**")
        fig = px.histogram(df, x="CGPA", nbins=12, color_discrete_sequence=[NAVY])
        fig.update_traces(marker_line_color=GOLD, marker_line_width=1)
        st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)
    with row3c2:
        st.markdown("**Heatmap — Feature correlation**")
        numeric_cols = ["Semester", "Attendance", "CGPA", "Backlogs", "Fees Paid"]
        fig = px.imshow(df[numeric_cols].corr(), color_continuous_scale=THEME_COLORSCALE,
                        aspect="auto", labels=dict(color="Correlation"))
        fig.update_traces(hovertemplate="%{x} × %{y}<br>Correlation: %{z:.2f}<extra></extra>")
        st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)

# --------------------------------------------------------------------------------------
# PAGE: MACHINE LEARNING
# --------------------------------------------------------------------------------------
elif page == "🤖 Machine Learning":
    st.title("🤖 Machine Learning")
    st.write(
        "Models predict student performance, placement probability and backlog risk. "
        "Linear Regression, Logistic Regression, Decision Tree, Random Forest"
        + (" and XGBoost " if XGB_AVAILABLE else " ")
        + "are trained, tested and evaluated using accuracy, precision, recall and F1-score."
    )
    if not XGB_AVAILABLE:
        st.info("`xgboost` isn't installed in this environment, so that model is skipped. "
                "Run `pip install xgboost` and restart the app to include it.")

    # ---- Feature engineering shared by both classifiers ----
    ml_df = df.copy()
    dept_dummies = pd.get_dummies(ml_df["Department"], prefix="Dept")
    scholarship_map = {"No": 0, "Partial": 1, "Yes": 2}
    ml_df["Scholarship_enc"] = ml_df["Scholarship"].map(scholarship_map).fillna(0)
    feature_cols = ["Semester", "Attendance", "CGPA", "Fees Paid", "Scholarship_enc"] + list(dept_dummies.columns)
    X_all = pd.concat([ml_df[["Semester", "Attendance", "CGPA", "Fees Paid", "Scholarship_enc"]], dept_dummies], axis=1)

    def train_classifiers(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
        )
        models = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000)),
            ]),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200, max_depth=4,
                eval_metric="logloss", random_state=42,
            )
        rows, fitted = [], {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds, zero_division=0),
                "Recall": recall_score(y_test, preds, zero_division=0),
                "F1-score": f1_score(y_test, preds, zero_division=0),
            })
            fitted[name] = model
        return pd.DataFrame(rows).set_index("Model").round(3), fitted

    # ---- Placement classification (exclude "In Process" — outcome not known yet) ----
    st.markdown("### 🎯 Placement Prediction")
    known_mask = ml_df["Placement"].isin(["Placed", "Not Placed"])
    y_placement = (ml_df.loc[known_mask, "Placement"] == "Placed").astype(int)
    X_placement = X_all.loc[known_mask]

    if y_placement.nunique() > 1 and len(y_placement) > 10:
        metrics_placement, models_placement = train_classifiers(X_placement, y_placement)
        st.dataframe(metrics_placement, use_container_width=True)
        best_placement_model = models_placement[metrics_placement["Accuracy"].idxmax()]
    else:
        st.info("Need both 'Placed' and 'Not Placed' examples to train this model.")
        best_placement_model = None

    # ---- Backlog risk classification ----
    st.markdown("### ⚠️ Backlog Risk Prediction")
    y_backlog = (ml_df["Backlogs"] > 0).astype(int)
    if y_backlog.nunique() > 1 and len(y_backlog) > 10:
        metrics_backlog, models_backlog = train_classifiers(X_all, y_backlog)
        st.dataframe(metrics_backlog, use_container_width=True)
        best_backlog_model = models_backlog[metrics_backlog["Accuracy"].idxmax()]
    else:
        st.info("Not enough variation in backlog data to train this model.")
        best_backlog_model = None

    # ---- Package regression (Linear Regression, placed students only) ----
    st.markdown("### 💰 Package Prediction (Linear Regression)")
    placed_df = ml_df[(ml_df["Placement"] == "Placed") & ml_df["Package_LPA"].notna()]
    if len(placed_df) > 10:
        X_pkg = pd.concat(
            [placed_df[["Semester", "Attendance", "CGPA", "Fees Paid", "Scholarship_enc"]],
             pd.get_dummies(placed_df["Department"], prefix="Dept")], axis=1
        ).reindex(columns=feature_cols, fill_value=0)
        y_pkg = placed_df["Package_LPA"]
        Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_pkg, y_pkg, test_size=0.2, random_state=42)
        lin_reg = LinearRegression().fit(Xp_train, yp_train)
        pkg_preds = lin_reg.predict(Xp_test)
        c1, c2 = st.columns(2)
        c1.markdown(f"""<div class="metric-card"><h2>{r2_score(yp_test, pkg_preds):.3f}</h2>
                    <p>R² score</p></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card"><h2>{mean_absolute_error(yp_test, pkg_preds):.2f} LPA</h2>
                    <p>Mean Absolute Error</p></div>""", unsafe_allow_html=True)
    else:
        lin_reg = None
        st.info("Need more placed students with package data to train the regression model.")

    st.markdown("---")
    st.markdown("### 🔮 Prediction Dashboard")
    st.caption("Enter a student's profile to estimate placement probability, backlog risk and expected package.")

    with st.form("prediction_form"):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            in_dept = st.selectbox("Department", sorted(df["Department"].unique()))
            in_sem = st.slider("Semester", 1, 8, 6)
        with pc2:
            in_att = st.slider("Attendance (%)", 0, 100, 85)
            in_cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, step=0.1)
        with pc3:
            in_fees = st.number_input("Fees Paid (₹)", 0, 500000, 80000, step=5000)
            in_schol = st.selectbox("Scholarship", ["No", "Partial", "Yes"])
        predict_clicked = st.form_submit_button("Predict")

    if predict_clicked:
        row = {"Semester": in_sem, "Attendance": in_att, "CGPA": in_cgpa,
               "Fees Paid": in_fees, "Scholarship_enc": scholarship_map[in_schol]}
        for c in dept_dummies.columns:
            row[c] = 1 if c == f"Dept_{in_dept}" else 0
        X_input = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)

        r1, r2c, r3 = st.columns(3)
        if best_placement_model is not None:
            prob = best_placement_model.predict_proba(X_input)[0][1] * 100
            r1.markdown(f"""<div class="metric-card"><h2>{prob:.0f}%</h2>
                        <p>Placement Probability</p></div>""", unsafe_allow_html=True)
        if best_backlog_model is not None:
            risk_prob = best_backlog_model.predict_proba(X_input)[0][1]
            risk_label = "Low" if risk_prob < 0.33 else ("Medium" if risk_prob < 0.66 else "High")
            r2c.markdown(f"""<div class="metric-card"><h2>{risk_label}</h2>
                        <p>Backlog Risk ({risk_prob*100:.0f}%)</p></div>""", unsafe_allow_html=True)
        if lin_reg is not None:
            expected_pkg = max(0, lin_reg.predict(X_input)[0])
            r3.markdown(f"""<div class="metric-card"><h2>₹{expected_pkg:.1f} LPA</h2>
                        <p>Expected Package (if placed)</p></div>""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# PAGE: EXPLAINABLE AI  (new — opens the "black box" of the models above)
# --------------------------------------------------------------------------------------
elif page == "🔍 Explainable AI":
    st.title("🔍 Explainable AI (XAI)")
    st.write(
        "Machine-learning models can behave like black boxes: they give an answer but not the reason. "
        "Explainable AI answers **\"why did the model predict this?\"** so that predictions about "
        "students can be understood, questioned and trusted. This page explains the same placement and "
        "backlog-risk models trained on the Machine Learning page, at two levels:"
    )
    st.markdown(
        "- **Global explanation** — which features matter most to the model overall "
        "(permutation importance, model-native importance, partial dependence).\n"
        "- **Local explanation** — why the model gave *one specific student* their prediction "
        "(feature-by-feature contribution against an average student)."
    )

    # ---- Same feature engineering as the Machine Learning page ----
    xai_df = df.copy()
    xai_dummies = pd.get_dummies(xai_df["Department"], prefix="Dept").astype(int)
    xai_schol_map = {"No": 0, "Partial": 1, "Yes": 2}
    xai_df["Scholarship_enc"] = xai_df["Scholarship"].map(xai_schol_map).fillna(0)
    xai_base_cols = ["Semester", "Attendance", "CGPA", "Fees Paid", "Scholarship_enc"]
    xai_feature_cols = xai_base_cols + list(xai_dummies.columns)
    xai_X_all = pd.concat([xai_df[xai_base_cols], xai_dummies], axis=1)

    # Original (human-readable) features -> the model columns that encode them
    XAI_GROUPS = {
        "Semester": ["Semester"],
        "Attendance": ["Attendance"],
        "CGPA": ["CGPA"],
        "Fees Paid": ["Fees Paid"],
        "Scholarship": ["Scholarship_enc"],
        "Department": list(xai_dummies.columns),
    }

    def build_xai_models():
        models = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000)),
            ]),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200, max_depth=4,
                eval_metric="logloss", random_state=42,
            )
        return models

    def group_importance(per_column: pd.Series) -> pd.Series:
        """Sum column-level importances into one number per original feature."""
        out = {g: float(per_column.reindex(cols).fillna(0).sum()) for g, cols in XAI_GROUPS.items()}
        return pd.Series(out).sort_values(ascending=True)

    st.markdown("---")
    xc1, xc2 = st.columns(2)
    with xc1:
        xai_task = st.radio("What should the model predict?",
                            ["Placement Prediction", "Backlog Risk Prediction"], horizontal=True)
    with xc2:
        xai_model_name = st.selectbox("Model to explain", list(build_xai_models().keys()), index=2)

    if xai_task == "Placement Prediction":
        xai_known = xai_df["Placement"].isin(["Placed", "Not Placed"])
        xai_y = (xai_df.loc[xai_known, "Placement"] == "Placed").astype(int)
        xai_X = xai_X_all.loc[xai_known]
        positive_label = "Placement probability"
    else:
        xai_y = (xai_df["Backlogs"] > 0).astype(int)
        xai_X = xai_X_all
        positive_label = "Backlog-risk probability"

    if xai_y.nunique() < 2 or len(xai_y) <= 10:
        st.info("Not enough varied data to train and explain this model yet.")
        st.stop()

    Xtr, Xte, ytr, yte = train_test_split(
        xai_X, xai_y, test_size=0.2, random_state=42, stratify=xai_y
    )
    xai_model = build_xai_models()[xai_model_name]
    xai_model.fit(Xtr, ytr)
    xai_acc = accuracy_score(yte, xai_model.predict(Xte))
    xai_f1 = f1_score(yte, xai_model.predict(Xte), zero_division=0)

    m1, m2, m3 = st.columns(3)
    m1.markdown(f"""<div class="metric-card"><h2>{xai_acc:.3f}</h2><p>Test accuracy</p></div>""",
                unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-card"><h2>{xai_f1:.3f}</h2><p>Test F1-score</p></div>""",
                unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-card"><h2>{len(xai_y)}</h2><p>Students used</p></div>""",
                unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 1. GLOBAL — permutation importance
    # ------------------------------------------------------------------
    st.markdown("### 🌍 Global Explanation 1 — Permutation Importance")
    st.caption(
        "Model-agnostic. Each feature's values are randomly shuffled on the test set; the bigger the drop in "
        "accuracy, the more the model was relying on that feature. Works for every model above."
    )
    perm = permutation_importance(xai_model, Xte, yte, n_repeats=10, random_state=42, scoring="accuracy")
    perm_cols = pd.Series(perm.importances_mean, index=Xte.columns).clip(lower=0)
    perm_grouped = group_importance(perm_cols)
    perm_df = perm_grouped.reset_index()
    perm_df.columns = ["Feature", "Accuracy drop when shuffled"]
    fig = px.bar(perm_df, x="Accuracy drop when shuffled", y="Feature", orientation="h",
                 color_discrete_sequence=[GOLD])
    fig.update_traces(marker_line_color=NAVY, marker_line_width=0.8)
    st.plotly_chart(style_fig(fig, height=330), use_container_width=True, theme=None)
    if perm_grouped.max() > 0:
        top_feat = perm_grouped.idxmax()
        st.success(f"**{top_feat}** is the feature this model depends on most for {xai_task.lower()}.")
    else:
        st.info("Shuffling any single feature barely changes accuracy — the model's signal is spread "
                "thin or the dataset is small.")

    # ------------------------------------------------------------------
    # 2. GLOBAL — model-native importance
    # ------------------------------------------------------------------
    st.markdown("### 🧠 Global Explanation 2 — What the Model Itself Reports")
    if xai_model_name == "Logistic Regression":
        coefs = pd.Series(xai_model.named_steps["clf"].coef_[0], index=Xtr.columns)
        st.caption("Logistic Regression coefficients on standardised features. Positive → pushes the "
                   "prediction toward the positive class; negative → pushes it away.")
        coef_df = pd.DataFrame({"Feature": coefs.index, "Coefficient": coefs.values})
        coef_df = coef_df[~coef_df["Feature"].str.startswith("Dept_")]
        coef_df["Feature"] = coef_df["Feature"].replace({"Scholarship_enc": "Scholarship"})
        coef_df = coef_df.sort_values("Coefficient")
        coef_df["Direction"] = np.where(coef_df["Coefficient"] >= 0, "Increases probability", "Decreases probability")
        fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", color="Direction",
                     color_discrete_map={"Increases probability": GOLD, "Decreases probability": NAVY})
        fig.update_traces(marker_line_color=NAVY, marker_line_width=0.6)
        st.plotly_chart(style_fig(fig, height=300), use_container_width=True, theme=None)
        st.caption("Department dummy columns are omitted here; their combined effect appears in the "
                   "permutation chart above.")
    elif xai_model_name in ("Decision Tree", "Random Forest", "XGBoost"):
        raw_imp = pd.Series(xai_model.feature_importances_, index=Xtr.columns)
        native_grouped = group_importance(raw_imp)
        st.caption("Built-in feature importance of the tree-based model — how much each feature "
                   "contributed to reducing impurity across all splits (Department dummies combined).")
        nat_df = native_grouped.reset_index()
        nat_df.columns = ["Feature", "Importance"]
        fig = px.bar(nat_df, x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=[NAVY])
        fig.update_traces(marker_line_color=GOLD, marker_line_width=0.8)
        st.plotly_chart(style_fig(fig, height=300), use_container_width=True, theme=None)

    if xai_model_name == "Decision Tree":
        with st.expander("🌳 Read the Decision Tree's actual rules (fully transparent model)"):
            st.caption("A decision tree is explainable by design — every prediction follows a chain of "
                       "if/else questions like the ones below.")
            st.code(export_text(xai_model, feature_names=list(Xtr.columns)), language="text")

    # ------------------------------------------------------------------
    # 3. GLOBAL — partial dependence
    # ------------------------------------------------------------------
    st.markdown("### 📉 Global Explanation 3 — Partial Dependence")
    st.caption("Shows how the model's average predicted probability changes as one feature is varied "
               "while all other students' features stay as they are.")
    pdp_label_map = {"Semester": "Semester", "Attendance": "Attendance", "CGPA": "CGPA",
                     "Fees Paid": "Fees Paid", "Scholarship_enc": "Scholarship (0=No, 1=Partial, 2=Yes)"}
    pdp_feature = st.selectbox("Feature", xai_base_cols, format_func=lambda c: pdp_label_map[c])
    uniq_vals = np.sort(Xtr[pdp_feature].unique())
    grid = uniq_vals if len(uniq_vals) <= 10 else np.linspace(Xtr[pdp_feature].min(), Xtr[pdp_feature].max(), 25)
    pdp_avg = []
    for v in grid:
        X_tmp = Xte.copy()
        X_tmp[pdp_feature] = v
        pdp_avg.append(float(xai_model.predict_proba(X_tmp)[:, 1].mean()))
    pdp_df = pd.DataFrame({pdp_label_map[pdp_feature]: grid, "Average predicted probability": pdp_avg})
    fig = px.line(pdp_df, x=pdp_label_map[pdp_feature], y="Average predicted probability", markers=True,
                  color_discrete_sequence=[GOLD])
    fig.update_traces(line_width=3, marker_size=8, marker_line_color=NAVY, marker_line_width=1)
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(style_fig(fig, height=320), use_container_width=True, theme=None)

    # ------------------------------------------------------------------
    # 4. LOCAL — why did the model predict this for ONE student?
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 👤 Local Explanation — Why This Prediction for This Student?")
    st.caption(
        "Each feature's contribution is measured by replacing the student's value with that of a typical "
        "student (median for numbers, most common department) and seeing how the predicted probability "
        "moves. Positive bars pushed the prediction up; negative bars pulled it down."
    )

    with st.form("xai_form"):
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            x_dept = st.selectbox("Department", sorted(df["Department"].unique()), key="xai_dept")
            x_sem = st.slider("Semester", 1, 8, 6, key="xai_sem")
        with lc2:
            x_att = st.slider("Attendance (%)", 0, 100, 85, key="xai_att")
            x_cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, step=0.1, key="xai_cgpa")
        with lc3:
            x_fees = st.number_input("Fees Paid (₹)", 0, 500000, 80000, step=5000, key="xai_fees")
            x_schol = st.selectbox("Scholarship", ["No", "Partial", "Yes"], key="xai_schol")
        explain_clicked = st.form_submit_button("Explain this prediction")

    if explain_clicked:
        row = {"Semester": x_sem, "Attendance": x_att, "CGPA": x_cgpa,
               "Fees Paid": x_fees, "Scholarship_enc": xai_schol_map[x_schol]}
        for c in xai_dummies.columns:
            row[c] = 1 if c == f"Dept_{x_dept}" else 0
        X_one = pd.DataFrame([row]).reindex(columns=xai_feature_cols, fill_value=0)

        # Baseline "typical student" built from the training data
        baseline = {c: float(Xtr[c].median()) for c in xai_base_cols}
        common_dept_col = Xtr[list(xai_dummies.columns)].mean().idxmax()
        for c in xai_dummies.columns:
            baseline[c] = 1 if c == common_dept_col else 0
        X_base = pd.DataFrame([baseline]).reindex(columns=xai_feature_cols, fill_value=0)

        p_student = float(xai_model.predict_proba(X_one)[0][1])
        p_typical = float(xai_model.predict_proba(X_base)[0][1])

        contribs = {}
        for g, cols in XAI_GROUPS.items():
            X_mod = X_one.copy()
            for c in cols:
                X_mod[c] = X_base[c].values[0]
            contribs[g] = p_student - float(xai_model.predict_proba(X_mod)[0][1])
        contrib_df = pd.DataFrame({"Feature": list(contribs.keys()),
                                   "Contribution (percentage points)": [v * 100 for v in contribs.values()]})
        contrib_df = contrib_df.sort_values("Contribution (percentage points)")
        contrib_df["Effect"] = np.where(contrib_df["Contribution (percentage points)"] >= 0,
                                        "Pushes probability up", "Pulls probability down")

        r1, r2 = st.columns(2)
        r1.markdown(f"""<div class="metric-card"><h2>{p_student*100:.0f}%</h2>
                    <p>{positive_label} (this student)</p></div>""", unsafe_allow_html=True)
        r2.markdown(f"""<div class="metric-card"><h2>{p_typical*100:.0f}%</h2>
                    <p>{positive_label} (typical student)</p></div>""", unsafe_allow_html=True)

        fig = px.bar(contrib_df, x="Contribution (percentage points)", y="Feature", orientation="h",
                     color="Effect",
                     color_discrete_map={"Pushes probability up": GOLD, "Pulls probability down": NAVY})
        fig.update_traces(marker_line_color=NAVY, marker_line_width=0.6)
        st.plotly_chart(style_fig(fig, height=330), use_container_width=True, theme=None)

        # Plain-English summary of the top drivers
        ranked = contrib_df.reindex(contrib_df["Contribution (percentage points)"].abs()
                                    .sort_values(ascending=False).index)
        lines = []
        for _, r in ranked.head(3).iterrows():
            if abs(r["Contribution (percentage points)"]) < 0.5:
                continue
            direction = "raised" if r["Contribution (percentage points)"] > 0 else "lowered"
            lines.append(f"**{r['Feature']}** {direction} the probability by "
                         f"{abs(r['Contribution (percentage points)']):.1f} points")
        if lines:
            st.info("**In plain English:** the biggest reasons for this prediction — " + "; ".join(lines) + ".")
        else:
            st.info("**In plain English:** this student looks very close to a typical student, so no single "
                    "feature moved the prediction noticeably.")
        st.caption("These are explanations of how the *model* behaves, not proof of cause and effect in "
                   "real students' lives.")

    # ------------------------------------------------------------------
    # 5. Package regression coefficients
    # ------------------------------------------------------------------
    with st.expander("💰 Explain the Package Prediction (Linear Regression)"):
        xai_placed = xai_df[(xai_df["Placement"] == "Placed") & xai_df["Package_LPA"].notna()]
        if len(xai_placed) > 10:
            X_pk = pd.concat(
                [xai_placed[xai_base_cols],
                 pd.get_dummies(xai_placed["Department"], prefix="Dept").astype(int)], axis=1
            ).reindex(columns=xai_feature_cols, fill_value=0)
            y_pk = xai_placed["Package_LPA"]
            lr_pk = LinearRegression().fit(X_pk, y_pk)
            std_effect = pd.Series(lr_pk.coef_ * X_pk.std().values, index=X_pk.columns)
            eff_df = pd.DataFrame({
                "Feature": [c for c in std_effect.index if not c.startswith("Dept_")],
                "Effect of +1 std-dev (LPA)": [std_effect[c] for c in std_effect.index
                                              if not c.startswith("Dept_")],
            }).replace({"Scholarship_enc": "Scholarship"}).sort_values("Effect of +1 std-dev (LPA)")
            st.caption("A linear model is explainable by its coefficients: each bar shows how much the "
                       "predicted package (in LPA) changes when that feature rises by one standard deviation, "
                       "holding the others fixed.")
            fig = px.bar(eff_df, x="Effect of +1 std-dev (LPA)", y="Feature", orientation="h",
                         color_discrete_sequence=[GOLD])
            fig.update_traces(marker_line_color=NAVY, marker_line_width=0.6)
            st.plotly_chart(style_fig(fig, height=300), use_container_width=True, theme=None)
        else:
            st.info("Need more placed students with package data to explain the regression model.")

# --------------------------------------------------------------------------------------
# PAGE: REGISTER NEW STUDENT
# --------------------------------------------------------------------------------------
elif page == "📝 Register New Student":
    st.title("📝 New Student Registration")
    st.write("New students can register here — the record is appended to `university_data.csv` and "
             "immediately shows up across the whole dashboard.")

    with st.form("registration_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Full Name*")
            dept = st.selectbox("Department*", sorted(df["Department"].unique()))
            sem = st.slider("Current Semester*", 1, 8, 1)
            attendance = st.slider("Attendance (%)*", 0, 100, 75)
        with c2:
            cgpa = st.slider("CGPA*", 0.0, 10.0, 7.0, step=0.1)
            backlogs = st.number_input("Backlogs*", min_value=0, max_value=20, value=0, step=1)
            fees = st.number_input("Fees Paid (₹)*", min_value=0, max_value=500000, value=75000, step=5000)
            schol = st.selectbox("Scholarship*", ["No", "Partial", "Yes"])
        submitted = st.form_submit_button("Register Student")

    if submitted:
        if not name.strip():
            st.error("Please enter the student's name.")
        else:
            new_id = next_student_id(df)
            new_row = {
                "Student ID": new_id, "Student Name": name.strip(), "Department": dept,
                "Semester": sem, "Attendance": attendance, "CGPA": cgpa,
                "Backlogs": backlogs, "Fees Paid": fees, "Scholarship": schol,
                "Placement": "Not Placed", "Package": "—",
            }
            updated_raw = pd.concat(
                [st.session_state.raw_df, pd.DataFrame([new_row])], ignore_index=True
            )
            st.session_state.raw_df = updated_raw
            save_csv(updated_raw)
            st.success(f"✅ Registered successfully as **{new_id}** and saved to {CSV_PATH}.")
            st.balloons()

    st.markdown("### Recently registered")
    st.dataframe(
        st.session_state.raw_df.drop(columns=["Package_LPA"], errors="ignore").tail(5),
        use_container_width=True,
    )
