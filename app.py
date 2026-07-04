"""
Startup Funding Analysis & Prediction App
==========================================
A Streamlit app that recreates the Startup Funding dashboard and adds three
ML-powered prediction tools:
  1. Predict Funding Amount (regression)
  2. Predict Funding Tier: Micro / Small / Mid / Large (classification)
  3. Predict Investment Type / Round (classification)

Run locally:    streamlit run app.py
Run on Colab:   see the instructions the assistant gave you.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from catboost import CatBoostRegressor, CatBoostClassifier
import optuna

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Startup Funding Analysis & Prediction",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# GLOBAL STYLE — fonts, palette, motion. Tokens (used inline in the HTML
# components below too, since iframes can't see this page's CSS variables):
#   bg-deep #0A0F2C · bg-mid #151B4D · card #FFFFFF
#   indigo #5B6EF5 (primary) · emerald #16C79A (growth) · gold #F2B705 (premium)
#   coral #FF6B6B (alert) · text-light #EAF0FF · text-dark #1B1F3B
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #EAF0FF !important; }
p, span, label, .stMarkdown, .stCaption { font-family: 'Inter', sans-serif; }

/* Remove Streamlit's default chrome so the page reads as a standalone site */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stAppViewContainer"] .block-container,
.block-container { padding-top: 1.8rem; }

/* Sidebar collapse toggle button color (highly visible) */
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapsedControl"] button * {
    color: #EAF0FF !important;
}
[data-testid="stSidebarCollapsedControl"] button {
    background-color: rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px;
}
[data-testid="stSidebarCollapsedControl"] button:hover {
    background-color: rgba(255, 255, 255, 0.15) !important;
}

/* Ambient background: soft brand-color glows + a faint rising line-chart motif,
   stacked under the animated gradient — evokes "funding data" without a stock photo. */
.stApp {
    background-image:
        radial-gradient(circle at 12% 12%, rgba(91,110,245,0.16), transparent 38%),
        radial-gradient(circle at 88% 78%, rgba(22,199,154,0.14), transparent 38%),
        url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPSc0NjAnIGhlaWdodD0nMjMwJz4KPHBvbHlsaW5lIHBvaW50cz0nMCwxOTAgMzgsMTY1IDc2LDE3OCAxMTUsMTQwIDE1MywxNTUgMTkxLDExMCAyMzAsMTI4IDI2OCw4MCAzMDYsOTggMzQ1LDU1IDM4Myw3MiA0MjEsMzUgNDYwLDUwJwogIGZpbGw9J25vbmUnIHN0cm9rZT0ncmdiYSgyNTUsMjU1LDI1NSwwLjA1KScgc3Ryb2tlLXdpZHRoPScyJy8+CjxnIGZpbGw9J3JnYmEoMjU1LDI1NSwyNTUsMC4wNiknPgogIDxjaXJjbGUgY3g9JzM4JyBjeT0nMTY1JyByPSczJy8+CiAgPGNpcmNsZSBjeD0nMTE1JyBjeT0nMTQwJyByPSczJy8+CiAgPGNpcmNsZSBjeD0nMTkxJyBjeT0nMTEwJyByPSczJy8+CiAgPGNpcmNsZSBjeD0nMjY4JyBjeT0nODAnIHI9JzMnLz4KICA8Y2lyY2xlIGN4PSczNDUnIGN5PSc1NScgcj0nMycvPgogIDxjaXJjbGUgY3g9JzQyMScgY3k9JzM1JyByPSczJy8+CjwvZz4KPGcgc3Ryb2tlPSdyZ2JhKDI1NSwyNTUsMjU1LDAuMDM1KScgc3Ryb2tlLXdpZHRoPScxJz4KICA8bGluZSB4MT0nMCcgeTE9JzIzMCcgeDI9JzQ2MCcgeTI9JzIzMCcvPgogIDxsaW5lIHgxPScwJyB5MT0nMCcgeDI9JzAnIHkyPScyMzAnLz4KPC9nPgo8L3N2Zz4="),
        linear-gradient(160deg, #0A0F2C 0%, #151B4D 45%, #0A0F2C 100%);
    background-repeat: no-repeat, no-repeat, repeat, no-repeat;
    background-size: 100% 100%, 100% 100%, 460px 230px, 220% 220%;
    background-position: 0 0, 0 0, center, 0% 0%;
    animation: bgShift 24s ease-in-out infinite;
}
@keyframes bgShift {
    0% { background-position: 12% 12%, 88% 78%, center, 0% 0%; }
    50% { background-position: 12% 12%, 88% 78%, center, 100% 100%; }
    100% { background-position: 12% 12%, 88% 78%, center, 0% 0%; }
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0A0F2C;
    border-right: 1px solid rgba(255,255,255,0.08);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #EAF0FF !important;
}
[data-testid="stSidebar"] h1 { font-family: 'Space Grotesk', sans-serif !important; }

/* Sidebar Filter Panel Glassmorphism & Custom Accents */
[data-testid="stSidebar"] div[class*="stSlider"],
[data-testid="stSidebar"] div[data-testid="stMultiSelect"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 12px 14px !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}
[data-testid="stSidebar"] div[class*="stSlider"] label,
[data-testid="stSidebar"] div[data-testid="stMultiSelect"] label {
    color: #F2B705 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 8px !important;
}

/* Sidebar Radio Navigation menu - styled like premium vertical tabs/pills */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 8px;
    display: flex;
    flex-direction: column;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 10px 14px !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #F2B705, #F7D673) !important;
    border-color: transparent !important;
    box-shadow: 0 4px 14px rgba(242,183,5,0.3) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) span {
    color: #0A0F2C !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label span {
    color: #B8C2E8 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #5B6EF5, #16C79A);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 14px rgba(91,110,245,0.35);
}
.stButton button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 22px rgba(22,199,154,0.45);
}
.stButton button:active { transform: translateY(0) scale(0.99); }

/* Card containers (st.container(key="card_...")) */
div[class*="st-key-card_"] {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 1.1rem 1.2rem 0.3rem 1.2rem;
    box-shadow: 0 10px 28px rgba(10,15,44,0.35);
    animation: fadeUp 0.7s ease both;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 1rem;
}
div[class*="st-key-card_"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 36px rgba(10,15,44,0.45);
}
div[class*="st-key-card_dash1"] { animation-delay: 0.05s; }
div[class*="st-key-card_dash2"] { animation-delay: 0.10s; }
div[class*="st-key-card_dash3"] { animation-delay: 0.15s; }
div[class*="st-key-card_dash4"] { animation-delay: 0.20s; }
div[class*="st-key-card_dash5"] { animation-delay: 0.25s; }
div[class*="st-key-card_dash6"] { animation-delay: 0.30s; }
div[class*="st-key-card_dash_timelapse"] { animation-delay: 0.35s; }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Body copy that sits directly on the dark page background (not inside a white
   card) needs a light color — but text inside white cards must stay dark.
   The card-scoped rule below has higher specificity, so it wins there. */
[data-testid="stMarkdownContainer"] p { color: #C9D2F0; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #8B95C9; }
div[class*="st-key-card_"] [data-testid="stMarkdownContainer"] p,
div[class*="st-key-card_"] [data-testid="stCaptionContainer"],
div[class*="st-key-card_"] label,
div[class*="st-key-card_"] span,
div[class*="st-key-card_"] p { color: #1B1F3B !important; }

/* Plotly + dataframe + expander chrome.
   NOTE: no overflow:hidden here — it was clipping Plotly's zoom/fullscreen/reset
   controls, which is why "reset axes" and the fullscreen toggle looked broken. */
[data-testid="stPlotlyChart"] { border-radius: 12px; }
[data-testid="stExpander"] {
    background: #F6F7FC;
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.05);
}
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 14px; animation: fadeUp 0.5s ease both; }

/* Inputs */
[data-baseweb="select"] > div, .stNumberInput input {
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif;
}

@media (prefers-reduced-motion: reduce) {
    .stApp, div[class*="st-key-card_"], [data-testid="stAlert"] { animation: none !important; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_PATH = "startup_funding_final.csv"

# Funding tier cutoffs, chosen on real dollar thresholds rather than quantiles so the
# labels stay interpretable: a "Large" round genuinely means $50M+, not "top 25%".
TIER_BINS = [0, 1_000_000, 10_000_000, 50_000_000, np.inf]
TIER_LABELS = ["Micro", "Small", "Mid", "Large"]
TIER_DESCRIPTIONS = {
    "Micro": "< $1M",
    "Small": "$1M – $10M",
    "Mid": "$10M – $50M",
    "Large": "$50M+",
}


# ----------------------------------------------------------------------------
# DATA LOADING & CLEANING
# ----------------------------------------------------------------------------


def clean_city(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x).strip().replace("\\xc2\\xa0", "").strip()
    low = x.lower()
    mapping = {
        "bangalore": "Bangalore",
        "bengaluru": "Bangalore",
        "gurgaon": "Gurgaon",
        "gurugram": "Gurgaon",
        "new delhi": "New Delhi",
        "delhi": "New Delhi",
    }
    for k, v in mapping.items():
        if k in low:
            return v
    if low in ("unknown", "", "nan"):
        return "Unknown"
    return x.title()


def clean_industry(x):
    if pd.isna(x):
        return "Unknown"
    low = str(x).strip().lower()
    if "ecommerce" in low or "e-commerce" in low or "e commerce" in low:
        return "eCommerce"
    if "fintech" in low or "fin-tech" in low or "fin tech" in low:
        return "FinTech"
    if "edtech" in low or "ed-tech" in low or "education" in low:
        return "EdTech"
    if "consumer internet" in low:
        return "Consumer Internet"
    if "technology" in low or low == "it" or "information technology" in low:
        return "Technology"
    if "health" in low:
        return "Healthcare"
    if "logistic" in low:
        return "Logistics"
    if "food" in low:
        return "Food & Beverage"
    if "transport" in low:
        return "Transportation"
    if "real estate" in low:
        return "Real Estate"
    if "finance" in low:
        return "Finance"
    if low in ("unknown", "", "nan", "others"):
        return "Unknown"
    return x.strip().title()


def clean_investment_type(x):
    if pd.isna(x):
        return "Other"
    low = str(x).strip().lower()
    if "seed" in low or "angel" in low:
        return "Seed/Angel Funding"
    if "private equity" in low:
        return "Private Equity"
    if "series a" in low or "pre-series a" in low or "pre series a" in low:
        return "Series A"
    if "series b" in low:
        return "Series B"
    if "series c" in low:
        return "Series C"
    if any(f"series {l}" in low for l in ["d", "e", "f", "g", "h"]):
        return "Series D+"
    if "debt" in low:
        return "Debt Funding"
    if "venture" in low:
        return "Venture Round"
    if "crowd" in low:
        return "Crowd Funding"
    return "Other"


KEYWORDS = ["saas", "ai", "artificial intelligence", "tech", "mobile", "app", "platform", "online", "marketplace", "b2b", "b2c", "delivery", "health", "edu"]

def get_subvertical_flags(domain_selected):
    flags = {f"SubV_{kw.replace(' ', '_')}": 0 for kw in KEYWORDS}
    domain_selected = str(domain_selected).lower().strip()
    if domain_selected == "saas":
        flags["SubV_saas"] = 1
        flags["SubV_tech"] = 1
    elif domain_selected == "ai":
        flags["SubV_ai"] = 1
        flags["SubV_artificial_intelligence"] = 1
        flags["SubV_tech"] = 1
    elif domain_selected == "platform":
        flags["SubV_platform"] = 1
    elif domain_selected == "marketplace":
        flags["SubV_marketplace"] = 1
        flags["SubV_online"] = 1
    elif domain_selected == "mobile":
        flags["SubV_mobile"] = 1
        flags["SubV_app"] = 1
    elif domain_selected == "b2b":
        flags["SubV_b2b"] = 1
    elif domain_selected == "b2c":
        flags["SubV_b2c"] = 1
    elif domain_selected == "delivery":
        flags["SubV_delivery"] = 1
    elif domain_selected == "healthcare":
        flags["SubV_health"] = 1
    elif domain_selected == "edtech":
        flags["SubV_edu"] = 1
        flags["SubV_tech"] = 1
    return flags

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["City_clean"] = df["City"].apply(clean_city)
    df["Industry_clean"] = df["Industry"].apply(clean_industry)
    df["InvType_clean"] = df["InvestmentType"].apply(clean_investment_type)

    top_ind = df["Industry_clean"].value_counts().head(15).index
    df["Industry_clean"] = df["Industry_clean"].where(
        df["Industry_clean"].isin(top_ind), "Other"
    )
    top_city = df["City_clean"].value_counts().head(15).index
    df["City_clean"] = df["City_clean"].where(df["City_clean"].isin(top_city), "Other")

    df["InvestorCount"] = (
        df["Investors"]
        .fillna("Unknown")
        .apply(lambda s: max(1, len([i for i in str(s).split(",") if i.strip()])))
    )

    df["FundingTier"] = pd.cut(df["FundingAmount"], bins=TIER_BINS, labels=TIER_LABELS)

    # HasTopInvestor Feature Engineering
    all_investors = df["Investors"].fillna("").str.split(",").explode().str.strip()
    all_investors = all_investors[all_investors != ""]
    top_investors = all_investors.value_counts().head(30).index.tolist()
    for label in ["Undisclosed Investors", "Undisclosed Investor", "Undisclosed", "Unknown"]:
        if label in top_investors:
            top_investors.remove(label)

    df["HasTopInvestor"] = df["Investors"].apply(
        lambda x: 1 if any(ti in str(x) for ti in top_investors) else 0
    )

    # SubVertical Keyword Domain Engineering
    df["SubVertical_clean"] = df["SubVertical"].fillna("").str.lower()
    for kw in KEYWORDS:
        col_name = f"SubV_{kw.replace(' ', '_')}"
        df[col_name] = df["SubVertical_clean"].apply(lambda s: 1 if kw in s else 0)

    # ADVANCED FEATURE ENGINEERING:
    # 1. Investor Influence score calculation (count * mean_log_funding)
    df_temp = df.dropna(subset=["Investors", "FundingAmount"]).copy()
    df_temp["LogFunding"] = np.log1p(df_temp["FundingAmount"])
    inv_exploded = df_temp[["Investors", "LogFunding"]].copy()
    inv_exploded["Investor"] = inv_exploded["Investors"].str.split(",")
    inv_exploded = inv_exploded.explode("Investor")
    inv_exploded["Investor"] = inv_exploded["Investor"].str.strip()
    inv_exploded = inv_exploded[inv_exploded["Investor"] != ""]
    
    inv_stats = inv_exploded.groupby("Investor").agg(
        count=("LogFunding", "count"),
        mean_log=("LogFunding", "mean")
    )
    inv_stats["score"] = inv_stats["count"] * inv_stats["mean_log"]
    investor_scores = inv_stats["score"].to_dict()
    
    def get_syndicate_influence(inv_str):
        if pd.isna(inv_str):
            return 0.0
        parts = [p.strip() for p in str(inv_str).split(",") if p.strip()]
        scores = [investor_scores.get(p, 0.0) for p in parts]
        return float(np.max(scores)) if scores else 0.0
        
    df["investor_influence"] = df["Investors"].apply(get_syndicate_influence)

    # 2. Rolling industry and city temporal trends (excluding target leaks via shift(1))
    ind_year_funding = df.groupby(["Year", "Industry_clean"])["FundingAmount"].sum().reset_index()
    ind_year_funding["rolling_ind_funding"] = ind_year_funding.groupby("Industry_clean")["FundingAmount"].shift(1).fillna(0)
    df = pd.merge(df, ind_year_funding[["Year", "Industry_clean", "rolling_ind_funding"]], on=["Year", "Industry_clean"], how="left")
    df["rolling_ind_funding"] = df["rolling_ind_funding"].fillna(0)
    
    city_year_funding = df.groupby(["Year", "City_clean"])["FundingAmount"].sum().reset_index()
    city_year_funding["rolling_city_funding"] = city_year_funding.groupby("City_clean")["FundingAmount"].shift(1).fillna(0)
    df = pd.merge(df, city_year_funding[["Year", "City_clean", "rolling_city_funding"]], on=["Year", "City_clean"], how="left")
    df["rolling_city_funding"] = df["rolling_city_funding"].fillna(0)

    return df


def add_engineered_features(df_input, df_reference):
    df_out = df_input.copy()
    
    top_subset = df_reference[df_reference["HasTopInvestor"] == 1]
    mean_top_influence = float(top_subset["investor_influence"].mean()) if not top_subset.empty else 0.0
    
    df_out["investor_influence"] = df_out["HasTopInvestor"].apply(
        lambda x: mean_top_influence if x == 1 else 0.0
    )
    
    ind_rolling = df_reference.groupby(["Year", "Industry_clean"])["rolling_ind_funding"].first().to_dict()
    ind_avg = df_reference.groupby("Industry_clean")["rolling_ind_funding"].mean().to_dict()
    
    def get_ind_rolling(row):
        year, ind = row["Year"], row["Industry_clean"]
        return ind_rolling.get((year, ind), ind_avg.get(ind, 0.0))
        
    df_out["rolling_ind_funding"] = df_out.apply(get_ind_rolling, axis=1)
    
    city_rolling = df_reference.groupby(["Year", "City_clean"])["rolling_city_funding"].first().to_dict()
    city_avg = df_reference.groupby("City_clean")["rolling_city_funding"].mean().to_dict()
    
    def get_city_rolling(row):
        year, city = row["Year"], row["City_clean"]
        return city_rolling.get((year, city), city_avg.get(city, 0.0))
        
    df_out["rolling_city_funding"] = df_out.apply(get_city_rolling, axis=1)
    
    return df_out


# ----------------------------------------------------------------------------
# MODEL TRAINING (cached as resources so it only runs once per session)
# ----------------------------------------------------------------------------


@st.cache_resource
def train_regression_model(df):
    reg_df = df.dropna(subset=["FundingAmount"]).copy()
    reg_df["LogFunding"] = np.log1p(reg_df["FundingAmount"])

    feat_cols = ["Industry_clean", "City_clean", "InvType_clean", "Year", "Quarter", "InvestorCount", "HasTopInvestor", "investor_influence", "rolling_ind_funding", "rolling_city_funding"] + [f"SubV_{kw.replace(' ', '_')}" for kw in KEYWORDS]
    cat_cols = ["Industry_clean", "City_clean", "InvType_clean"]

    X = reg_df[feat_cols]
    y = reg_df["LogFunding"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optuna hyperparameter optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 300),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "random_seed": 42,
            "verbose": 0
        }
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            pre_fold = ColumnTransformer(
                [("encoder", TargetEncoder(smooth="auto", random_state=42), cat_cols)], remainder="passthrough"
            )
            X_tr_trans = pre_fold.fit_transform(X_tr, y_tr)
            X_val_trans = pre_fold.transform(X_val)
            
            model = CatBoostRegressor(**params)
            model.fit(X_tr_trans, y_tr)
            preds = model.predict(X_val_trans)
            scores.append(mean_absolute_error(y_val, preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=8)
    best_params = study.best_params
    best_params["random_seed"] = 42
    best_params["verbose"] = 0

    pre = ColumnTransformer(
        [("encoder", TargetEncoder(smooth="auto", random_state=42), cat_cols)], remainder="passthrough"
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            ("model", CatBoostRegressor(**best_params)),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, pred),
        "mae_dollar": mean_absolute_error(np.expm1(y_test), np.expm1(pred)),
    }

    resid_std = float(np.std(y_test.values - pred))

    return pipe, metrics, resid_std, feat_cols


@st.cache_resource
def train_tier_classification_model(df):
    tier_df = df.dropna(subset=["FundingAmount"]).copy()

    feat_cols = ["Industry_clean", "City_clean", "InvType_clean", "Year", "Quarter", "InvestorCount", "HasTopInvestor", "investor_influence", "rolling_ind_funding", "rolling_city_funding"] + [f"SubV_{kw.replace(' ', '_')}" for kw in KEYWORDS]
    cat_cols = ["Industry_clean", "City_clean", "InvType_clean"]

    X = tier_df[feat_cols]
    y = tier_df["FundingTier"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Optuna hyperparameter optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 300),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "random_seed": 42,
            "verbose": 0,
            "auto_class_weights": "Balanced"
        }
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            pre_fold = ColumnTransformer(
                [("encoder", TargetEncoder(smooth="auto", random_state=42), cat_cols)], remainder="passthrough"
            )
            X_tr_trans = pre_fold.fit_transform(X_tr, y_tr)
            X_val_trans = pre_fold.transform(X_val)
            
            model = CatBoostClassifier(**params)
            model.fit(X_tr_trans, y_tr)
            preds = model.predict(X_val_trans).flatten()
            scores.append(f1_score(y_val, preds, average="macro"))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=8)
    best_params = study.best_params
    best_params["random_seed"] = 42
    best_params["verbose"] = 0
    best_params["auto_class_weights"] = "Balanced"

    pre = ColumnTransformer(
        [("encoder", TargetEncoder(smooth="auto", random_state=42), cat_cols)], remainder="passthrough"
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            ("model", CatBoostClassifier(**best_params)),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test).flatten()

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
    }

    return pipe, metrics, feat_cols


@st.cache_resource
def train_classification_model(df):
    clf_df = df.copy()
    clf_df["FundingAmount_filled"] = clf_df["FundingAmount"].fillna(clf_df["FundingAmount"].median())
    clf_df["LogFunding"] = np.log1p(clf_df["FundingAmount_filled"])

    vc = clf_df["InvType_clean"].value_counts()
    keep_classes = vc[vc >= 15].index
    clf_df = clf_df[clf_df["InvType_clean"].isin(keep_classes)]

    feat_cols = ["Industry_clean", "City_clean", "Year", "Quarter", "LogFunding", "InvestorCount", "HasTopInvestor", "investor_influence", "rolling_ind_funding", "rolling_city_funding"] + [f"SubV_{kw.replace(' ', '_')}" for kw in KEYWORDS]
    cat_cols = ["Industry_clean", "City_clean"]

    X = clf_df[feat_cols]
    y = clf_df["InvType_clean"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Optuna hyperparameter optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 300),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "random_seed": 42,
            "verbose": 0,
            "auto_class_weights": "Balanced"
        }
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            pre_fold = ColumnTransformer(
                [("encoder", TargetEncoder(smooth="auto", random_state=42), cat_cols)], remainder="passthrough"
            )
            X_tr_trans = pre_fold.fit_transform(X_tr, y_tr)
            X_val_trans = pre_fold.transform(X_val)
            
            model = CatBoostClassifier(**params)
            model.fit(X_tr_trans, y_tr)
            preds = model.predict(X_val_trans).flatten()
            scores.append(f1_score(y_val, preds, average="macro"))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=8)
    best_params = study.best_params
    best_params["random_seed"] = 42
    best_params["verbose"] = 0
    best_params["auto_class_weights"] = "Balanced"

    pre = ColumnTransformer(
        [("encoder", TargetEncoder(smooth="auto", random_state=42), cat_cols)], remainder="passthrough"
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            ("model", CatBoostClassifier(**best_params)),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test).flatten()

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
    }

    return pipe, metrics, feat_cols, sorted(y.unique())


def get_feature_importance(pipe, top_n=12):
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    names = pre.get_feature_names_out()
    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": names, "importance": importances})
    fi["feature"] = fi["feature"].str.replace("cat__", "", regex=False).str.replace(
        "remainder__", "", regex=False
    )
    return fi.sort_values("importance", ascending=False).head(top_n)


def explain_rf_prediction(pipe, x_df, is_regression=True, class_index=None):
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    
    if hasattr(pre, "feature_names_in_"):
        x_df = x_df[pre.feature_names_in_]
        
    x_trans = pre.transform(x_df)
    if hasattr(x_trans, "toarray"):
        x_trans = x_trans.toarray()
        
    feature_names = pre.get_feature_names_out()
    
    from catboost import Pool
    pool = Pool(x_trans)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")
    
    if len(shap_vals.shape) == 3:  # Multiclass
        shap_single = shap_vals[0, class_index or 0]
    else:
        shap_single = shap_vals[0]
        
    contributions = shap_single[:-1]
    bias = shap_single[-1]
    
    mapped_contribs = {}
    for feat_name, contrib in zip(feature_names, contributions):
        category = "Other"
        if "Industry_clean" in feat_name:
            category = "Industry"
        elif "City_clean" in feat_name:
            category = "City"
        elif "InvType_clean" in feat_name:
            category = "Investment Type"
        elif "SubV_" in feat_name:
            category = "Sub-Vertical Domain"
        elif "Year" in feat_name:
            category = "Year"
        elif "Quarter" in feat_name:
            category = "Quarter"
        elif "InvestorCount" in feat_name:
            category = "Investor Count"
        elif "HasTopInvestor" in feat_name:
            category = "Includes Top Investor"
        elif "LogFunding" in feat_name:
            category = "Funding Amount"
        elif "investor_influence" in feat_name:
            category = "Investor Influence"
        elif "rolling_ind" in feat_name:
            category = "Industry Trends"
        elif "rolling_city" in feat_name:
            category = "City Trends"
            
        mapped_contribs[category] = mapped_contribs.get(category, 0.0) + contrib
        
    return bias, mapped_contribs


def make_waterfall_chart(bias, mapped_contribs, final_pred, title, is_regression=True):
    # Sort contributions by absolute value
    sorted_items = sorted(mapped_contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    
    if is_regression:
        x_names = ["Base Value (Average)"]
        y_vals = []
        measures = ["relative"]
        
        current_log_val = bias
        current_dollar_val = np.expm1(current_log_val)
        y_vals.append(current_dollar_val)
        
        for name, contrib_log in sorted_items:
            next_log_val = current_log_val + contrib_log
            next_dollar_val = np.expm1(next_log_val)
            dollar_diff = next_dollar_val - current_dollar_val
            
            x_names.append(name)
            y_vals.append(dollar_diff)
            measures.append("relative")
            
            current_log_val = next_log_val
            current_dollar_val = next_dollar_val
            
        x_names.append("Predicted Funding")
        y_vals.append(np.expm1(final_pred))
        measures.append("total")
        
        text_vals = []
        for v, m in zip(y_vals, measures):
            if m == "total":
                text_vals.append(f"${v:,.0f}")
            else:
                sign = "+" if v >= 0 else "-"
                text_vals.append(f"{sign}${abs(v):,.0f}")
    else:
        # Show probability changes in %
        x_names = ["Base Probability"]
        y_vals = [bias * 100]
        measures = ["relative"]
        
        for name, contrib in sorted_items:
            x_names.append(name)
            y_vals.append(contrib * 100)
            measures.append("relative")
            
        x_names.append("Predicted Probability")
        y_vals.append(final_pred * 100)
        measures.append("total")
        
        text_vals = []
        for v, m in zip(y_vals, measures):
            if m == "total":
                text_vals.append(f"{v:.1f}%")
            else:
                sign = "+" if v >= 0 else "-"
                text_vals.append(f"{sign}{abs(v):.1f}%")
                
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=x_names,
        textposition="outside",
        text=text_vals,
        y=y_vals,
        connector={"line": {"color": "rgba(255, 255, 255, 0.25)"}},
        increasing={"marker": {"color": "#16C79A"}},
        decreasing={"marker": {"color": "#FF6B6B"}},
        totals={"marker": {"color": "#5B6EF5"}}
    ))
    
    fig = style_fig(fig)
    fig.update_layout(
        title=title,
        waterfallgap=0.3,
        showlegend=False,
    )
    return fig


def match_investors(df, industry, city, tier, domain_selected):
    # Filter undisclosed investors
    undisclosed = {"undisclosed investors", "undisclosed investor", "undisclosed", "unknown", "", "others", "nan"}
    
    # We will compute metrics for all VCs
    vc_stats = {}
    
    # Keywords check
    kw_col = None
    if domain_selected != "None":
        kw_col = f"SubV_{domain_selected.lower().replace(' ', '_')}"
        
    for _, row in df.iterrows():
        inv_str = str(row["Investors"])
        if not inv_str or inv_str.lower() in undisclosed:
            continue
        
        # Split by comma
        invs = [i.strip() for i in inv_str.split(",") if i.strip()]
        for inv in invs:
            if inv.lower() in undisclosed:
                continue
                
            if inv not in vc_stats:
                vc_stats[inv] = {
                    "total_deals": 0,
                    "industry_deals": 0,
                    "city_deals": 0,
                    "tier_deals": 0,
                    "subv_deals": 0,
                    "portfolios": []  # list of (startup, amount)
                }
                
            stats = vc_stats[inv]
            stats["total_deals"] += 1
            
            if row["Industry_clean"] == industry:
                stats["industry_deals"] += 1
            if row["City_clean"] == city:
                stats["city_deals"] += 1
            if row["FundingTier"] == tier:
                stats["tier_deals"] += 1
                
            if kw_col and row.get(kw_col) == 1:
                stats["subv_deals"] += 1
                
            amount = row["FundingAmount"] if pd.notna(row["FundingAmount"]) else 0.0
            stats["portfolios"].append((row["StartupName"], amount))
            
    # Calculate scores
    results = []
    for inv, stats in vc_stats.items():
        ind_score = (min(5, stats["industry_deals"]) / 5.0) * 40.0
        city_score = (min(3, stats["city_deals"]) / 3.0) * 30.0
        tier_score = (min(3, stats["tier_deals"]) / 3.0) * 30.0
        
        subv_bonus = 0.0
        if kw_col:
            subv_bonus = (min(2, stats["subv_deals"]) / 2.0) * 10.0
            
        score = min(100.0, ind_score + city_score + tier_score + subv_bonus)
        
        if score < 10.0:  # skip very low matches
            continue
            
        # Get top 3 portfolios by funding amount, unique startup names
        port_dict = {}
        for startup, amt in stats["portfolios"]:
            port_dict[startup] = max(port_dict.get(startup, 0.0), amt)
        sorted_ports = sorted(port_dict.items(), key=lambda x: x[1], reverse=True)
        top_ports = [x[0] for x in sorted_ports[:3]]
        
        results.append({
            "Investor": inv,
            "Match Score (%)": round(score, 1),
            "Total Deals": stats["total_deals"],
            "Industry Deals": stats["industry_deals"],
            "City Deals": stats["city_deals"],
            "Stage/Tier Deals": stats["tier_deals"],
            "Key Portfolios": ", ".join(top_ports) if top_ports else "Unknown"
        })
        
    res_df = pd.DataFrame(results)
    if len(res_df) == 0:
        return pd.DataFrame(columns=["Investor", "Match Score (%)", "Total Deals", "Industry Deals", "City Deals", "Stage/Tier Deals", "Key Portfolios"])
        
    return res_df.sort_values(by="Match Score (%)", ascending=False).head(10)


def get_animated_industry_df(fdf):
    top_ind = fdf.groupby("Industry_clean")["FundingAmount"].sum().nlargest(10).index.tolist()
    if "Other" in top_ind:
        top_ind.remove("Other")
    if "Unknown" in top_ind:
        top_ind.remove("Unknown")
        
    years = sorted(fdf["Year"].unique())
    if not years or not top_ind:
        return pd.DataFrame(columns=["Year", "Industry_clean", "FundingAmount"]), []
        
    import itertools
    combinations = list(itertools.product(years, top_ind))
    base_df = pd.DataFrame(combinations, columns=["Year", "Industry_clean"])
    
    agg = fdf.groupby(["Year", "Industry_clean"])["FundingAmount"].sum().reset_index()
    merged = pd.merge(base_df, agg, on=["Year", "Industry_clean"], how="left").fillna(0)
    
    return merged, top_ind


def render_3d_highcharts(chart_type, df, title, x_col, y_col, height=340, label_suffix="", x_prefix=""):
    data_list = []
    if chart_type == "pie":
        for _, r in df.iterrows():
            data_list.append([str(r[x_col]), float(r[y_col])])
        data_json = str(data_list)
        
        series_config = f"""
        series: [{{
            type: 'pie',
            name: '{y_col}',
            data: {data_json}
        }}]
        """
        chart_options = """
        options3d: {
            enabled: true,
            alpha: 45,
            beta: 0
        }
        """
        plot_options = """
        plotOptions: {
            pie: {
                allowPointSelect: true,
                cursor: 'pointer',
                depth: 35,
                dataLabels: {
                    enabled: true,
                    format: '<b>{{point.name}}</b>: {{point.percentage:.1f}}%',
                    style: {
                        color: '#1B1F3B',
                        textOutline: 'none',
                        fontSize: '11px'
                    }
                }
            }
        }
        """
    else:  # column (3D vertical columns)
        raw_cats = list(df[x_col])
        if x_prefix:
            categories = [f"{x_prefix}{int(float(c))}" for c in raw_cats]
        else:
            categories = [str(c) for c in raw_cats]
            
        values = list(df[y_col].astype(float))
        
        series_config = f"""
        xAxis: {{
            categories: {str(categories)},
            labels: {{
                style: {{ color: '#1B1F3B', fontWeight: 'bold' }}
            }}
        }},
        yAxis: {{
            title: {{ text: '' }},
            labels: {{
                style: {{ color: '#1B1F3B' }}
            }}
        }},
        series: [{{
            name: '{y_col}',
            data: {str(values)},
            color: '#5B6EF5'
        }}]
        """
        chart_options = """
        options3d: {
            enabled: true,
            alpha: 15,
            beta: 15,
            depth: 50,
            viewDistance: 25
        }
        """
        plot_options = """
        plotOptions: {
            column: {
                depth: 25,
                colorByPoint: true
            }
        }
        """
        
    colors_json = str(["#5B6EF5", "#16C79A", "#F2B705", "#FF6B6B", "#8B5CF6", "#22D3EE", "#F97316", "#10B981"])
    
    html = f"""
    <html>
    <head>
    <style>
        html, body {{ margin:0; padding:0; background:transparent; font-family: 'Inter', sans-serif; }}
        #container {{ width: 100%; height: {height}px; }}
    </style>
    <script src="https://code.highcharts.com/highcharts.js"></script>
    <script src="https://code.highcharts.com/highcharts-3d.js"></script>
    </head>
    <body>
        <div id="container"></div>
        <script>
        Highcharts.chart('container', {{
            colors: {colors_json},
            chart: {{
                type: '{chart_type}',
                backgroundColor: 'transparent',
                {chart_options}
            }},
            title: {{
                text: '{title}',
                style: {{
                    color: '#1B1F3B',
                    fontFamily: 'Space Grotesk, sans-serif',
                    fontSize: '15px',
                    fontWeight: 'bold'
                }}
            }},
            credits: {{ enabled: false }},
            legend: {{ enabled: false }},
            tooltip: {{
                pointFormat: '<b>{{point.y:,.0f}}{label_suffix}</b>'
            }},
            {plot_options},
            {series_config}
        }});
        </script>
    </body>
    </html>
    """
    return html






# ----------------------------------------------------------------------------
# VISUAL HELPERS — consistent Plotly theme + the hero/ticker/KPI HTML widgets.
# (components.html renders in a sandboxed iframe, so colors are hardcoded here
# rather than reused from CUSTOM_CSS, which that iframe can't see.)
# ----------------------------------------------------------------------------
PALETTE = ["#5B6EF5", "#16C79A", "#F2B705", "#FF6B6B", "#8B5CF6", "#22D3EE", "#F97316", "#10B981"]


def style_fig(fig, height=380):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color="#1B1F3B", size=12),
        title_font=dict(family="Space Grotesk, sans-serif", size=16, color="#1B1F3B"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
        height=height,
        colorway=PALETTE,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.07)", zerolinecolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.07)", zerolinecolor="rgba(0,0,0,0.1)")
    return fig


def render_hero(total_funding, total_startups, total_deals, top_city, top_industry):
    ticker_text = (
        f"💰 ${total_funding/1e9:,.2f}B total funding tracked &nbsp;•&nbsp; "
        f"🏢 {total_startups:,} startups funded &nbsp;•&nbsp; "
        f"📈 {total_deals:,} deals recorded &nbsp;•&nbsp; "
        f"🌆 {top_city} leads by city &nbsp;•&nbsp; "
        f"🏭 {top_industry} leads by industry &nbsp;•&nbsp; "
    )
    html = f"""
    <html><head><style>
        html, body {{ margin:0; padding:0; background:transparent; }}
        .hero {{
            background: linear-gradient(135deg, #0A0F2C 0%, #1B2470 50%, #0A0F2C 100%);
            border-radius: 20px;
            padding: 1.8rem 2rem 1.1rem 2rem;
            font-family: 'Space Grotesk', sans-serif;
            box-sizing: border-box;
        }}
        .hero-title {{
            font-size: 2.1rem;
            font-weight: 700;
            background: linear-gradient(90deg, #EAF0FF 10%, #16C79A 55%, #F2B705 95%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: heroFade 0.8s ease both;
            margin: 0;
        }}
        .hero-sub {{
            color: #B8C2E8;
            font-family: 'Inter', sans-serif;
            font-size: 0.95rem;
            margin-top: 0.35rem;
            animation: heroFade 0.8s ease 0.15s both;
        }}
        .ticker-wrap {{
            margin-top: 1.1rem;
            overflow: hidden;
            white-space: nowrap;
            border-top: 1px solid rgba(255,255,255,0.15);
            border-bottom: 1px solid rgba(255,255,255,0.15);
            padding: 0.5rem 0;
        }}
        .ticker-move {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            color: #EAF0FF;
            animation: tickerScroll 26s linear infinite;
            padding-left: 100%;
        }}
        @keyframes heroFade {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:translateY(0); }} }}
        @keyframes tickerScroll {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-100%); }} }}
        @media (prefers-reduced-motion: reduce) {{ .ticker-move {{ animation: none; }} }}
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&family=Inter:wght@400&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
    </head><body>
        <div class="hero">
            <div class="hero-title">STARTUP FUNDING INTELLIGENCE</div>
            <div class="hero-sub">Tracking India's startup funding landscape — deals, dollars, and the investors behind them.</div>
            <div class="ticker-wrap">
                <div class="ticker-move">{ticker_text}{ticker_text}</div>
            </div>
        </div>
    </body></html>
    """
    components.html(html, height=175)


def render_kpi_cards(total_funding, total_investors, total_deals, total_startups):
    cards = [
        ("💰", "Total Funding", total_funding / 1e9, "$", "bn", 2),
        ("🧑‍💼", "Total Investors", total_investors, "", "", 0),
        ("🤝", "Total Deals", total_deals, "", "", 0),
        ("🚀", "Total Startups", total_startups, "", "", 0),
    ]
    cards_html = ""
    scripts = ""
    for i, (icon, label, value, prefix, suffix, decimals) in enumerate(cards):
        cards_html += f"""
        <div class="kpi-card" style="animation-delay:{i * 0.08}s">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" id="kpi{i}">{prefix}0{suffix}</div>
        </div>
        """
        scripts += f'animateValue("kpi{i}", {value}, "{prefix}", "{suffix}", {decimals}, 1100);\n'

    html = f"""
    <html><head><style>
        html, body {{ margin:0; padding:0; background:transparent; }}
        .kpi-row {{ display:flex; gap:1rem; font-family:'Space Grotesk', sans-serif; }}
        .kpi-card {{
            flex:1; background:#FFFFFF; border-radius:16px;
            padding:1rem 1.2rem; box-shadow:0 10px 26px rgba(10,15,44,0.28);
            animation: kpiUp 0.6s ease both;
            transition: transform .25s ease, box-shadow .25s ease;
            box-sizing: border-box;
        }}
        .kpi-card:hover {{ transform: translateY(-5px); box-shadow:0 16px 34px rgba(10,15,44,0.4); }}
        .kpi-icon {{ font-size:1.5rem; }}
        .kpi-label {{
            font-size:0.74rem; color:#6B7280; font-family:'Inter',sans-serif;
            text-transform:uppercase; letter-spacing:0.06em; margin-top:0.25rem;
        }}
        .kpi-value {{ font-size:1.7rem; font-weight:700; color:#1B1F3B; margin-top:0.15rem; }}
        @keyframes kpiUp {{ from {{ opacity:0; transform:translateY(14px); }} to {{ opacity:1; transform:translateY(0); }} }}
        @media (prefers-reduced-motion: reduce) {{ .kpi-card {{ animation:none; }} }}
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&family=Inter:wght@400&display=swap" rel="stylesheet">
    </head><body>
        <div class="kpi-row">{cards_html}</div>
        <script>
        function animateValue(id, end, prefix, suffix, decimals, duration) {{
            const el = document.getElementById(id);
            if (!el) return;
            let startTime = null;
            function step(ts) {{
                if (!startTime) startTime = ts;
                const progress = Math.min((ts - startTime) / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                const val = end * eased;
                const shown = decimals > 0 ? val.toFixed(decimals) : Math.round(val).toLocaleString();
                el.textContent = prefix + shown + suffix;
                if (progress < 1) requestAnimationFrame(step);
            }}
            requestAnimationFrame(step);
        }}
        {scripts}
        </script>
    </body></html>
    """
    components.html(html, height=130)


# ----------------------------------------------------------------------------
# LOAD DATA + TRAIN MODELS
# ----------------------------------------------------------------------------
df = load_data(DATA_PATH)
reg_pipe, reg_metrics, resid_std, reg_feat_cols = train_regression_model(df)
tier_pipe, tier_metrics, tier_feat_cols = train_tier_classification_model(df)
clf_pipe, clf_metrics, clf_feat_cols, clf_classes = train_classification_model(df)

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & FILTERS
# ----------------------------------------------------------------------------
st.sidebar.title("🚀 Startup Funding App")
st.sidebar.markdown(
    "Explore the dashboard or use the ML tools to predict funding outcomes "
    "for a hypothetical startup."
)

st.sidebar.markdown("### 🧭 Navigation")
page = st.sidebar.radio(
    "Select a section",
    [
        "📊 Dashboard",
        "💰 Predict Funding Amount",
        "🎯 Predict Funding Tier",
        "🏷️ Predict Investment Type",
        "🤝 Investor Matchmaker & Simulator",
        "📈 Model Performance",
    ],
    label_visibility="collapsed",
)

# Set defaults to avoid NameError if filters aren't rendered
year_range = (int(df["Year"].min()), int(df["Year"].max()))
inv_type_filter = sorted(df["InvType_clean"].unique())
view_3d = False

# Show dashboard filters ONLY when Dashboard is active
if page == "📊 Dashboard":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Dashboard Filters")
    year_range = st.sidebar.slider(
        "Filter by Year Range",
        int(df["Year"].min()),
        int(df["Year"].max()),
        year_range,
        key="dash_year_slider"
    )
    inv_type_filter = st.sidebar.multiselect(
        "Filter by Investment Type",
        sorted(df["InvType_clean"].unique()),
        default=inv_type_filter,
        key="dash_inv_multiselect"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌐 Visual Projection")
    view_3d = st.sidebar.toggle("Enable 3D Chart Mode", value=False, key="toggle_3d_mode")

# ============================== PAGES ===================================
if page == "📊 Dashboard":
    fdf = df[
        (df["Year"] >= year_range[0])
        & (df["Year"] <= year_range[1])
        & (df["InvType_clean"].isin(inv_type_filter))
    ]

    total_funding = fdf["FundingAmount"].sum()
    total_investors = fdf["Investors"].nunique()
    total_deals = len(fdf)
    total_startups = fdf["StartupName"].nunique()
    top_city = (
        fdf.groupby("City_clean")["FundingAmount"].sum().sort_values(ascending=False).index[0]
        if len(fdf) else "—"
    )
    top_industry = (
        fdf.groupby("Industry_clean")["FundingAmount"].sum().sort_values(ascending=False).index[0]
        if len(fdf) else "—"
    )

    render_hero(total_funding, total_startups, total_deals, top_city, top_industry)
    render_kpi_cards(total_funding, total_investors, total_deals, total_startups)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(key="card_dash1"):
            donut_df = fdf["InvType_clean"].value_counts().reset_index()
            donut_df.columns = ["InvestmentType", "Count"]
            if view_3d:
                html = render_3d_highcharts("pie", donut_df, "Count of Startups by Investment Type", "InvestmentType", "Count")
                components.html(html, height=380)
            else:
                fig = px.pie(
                    donut_df, names="InvestmentType", values="Count", hole=0.55,
                    title="Count of Startups by Investment Type",
                )
                st.plotly_chart(style_fig(fig), use_container_width=True)

    with col2:
        with st.container(key="card_dash2"):
            ind_df = (
                fdf.groupby("Industry_clean")["FundingAmount"].sum().sort_values(ascending=False).head(10)
            ).reset_index()
            if view_3d:
                html = render_3d_highcharts("column", ind_df, "Sum of Funding Amount by Industry", "Industry_clean", "FundingAmount", label_suffix=" $")
                components.html(html, height=380)
            else:
                fig = px.bar(
                    ind_df, x="FundingAmount", y="Industry_clean", orientation="h",
                    title="Sum of Funding Amount by Industry",
                    labels={"FundingAmount": "Funding Amount ($)", "Industry_clean": "Industry"},
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(style_fig(fig), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        with st.container(key="card_dash3"):
            city_df = (
                fdf.groupby("City_clean")["FundingAmount"].sum().sort_values(ascending=False).head(10)
            ).reset_index()
            if view_3d:
                html = render_3d_highcharts("column", city_df, "Sum of Funding Amount by City", "City_clean", "FundingAmount", label_suffix=" $")
                components.html(html, height=380)
            else:
                fig = px.bar(
                    city_df, x="FundingAmount", y="City_clean", orientation="h",
                    title="Sum of Funding Amount by City",
                    labels={"FundingAmount": "Funding Amount ($)", "City_clean": "City"},
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(style_fig(fig), use_container_width=True)

    with col4:
        with st.container(key="card_dash4"):
            year_df = fdf.groupby("Year")["FundingAmount"].sum().reset_index()
            if view_3d:
                html = render_3d_highcharts("column", year_df, "Sum of Funding Amount by Year", "Year", "FundingAmount", label_suffix=" $")
                components.html(html, height=380)
            else:
                fig = px.line(
                    year_df, x="Year", y="FundingAmount", markers=True,
                    title="Sum of Funding Amount by Year",
                )
                fig.update_traces(line=dict(width=3))
                st.plotly_chart(style_fig(fig), use_container_width=True)

    col_q, col_inv = st.columns(2)
    with col_q:
        with st.container(key="card_dash_quarter"):
            quarter_df = fdf.groupby("Quarter")["FundingAmount"].sum().reset_index()
            quarter_df = quarter_df.sort_values("FundingAmount", ascending=False)
            
            if view_3d:
                html = render_3d_highcharts("column", quarter_df, "Sum of Funding Amount by Quarter", "Quarter", "FundingAmount", label_suffix=" $", x_prefix="Q")
                components.html(html, height=380)
            else:
                q_x = [f"Q{int(q)}" for q in quarter_df["Quarter"]]
                q_y = list(quarter_df["FundingAmount"])
                q_text = []
                for v in q_y:
                    if v >= 1e9:
                        q_text.append(f"${v/1e9:.2f}B")
                    else:
                        q_text.append(f"${v/1e6:.1f}M")
                        
                total_fund = fdf["FundingAmount"].sum()
                q_x.append("Total")
                q_y.append(total_fund)
                if total_fund >= 1e9:
                    q_text.append(f"${total_fund/1e9:.2f}B")
                else:
                    q_text.append(f"${total_fund/1e6:.1f}M")
                    
                fig_q = go.Figure(go.Waterfall(
                    name="Quarterly Funding",
                    orientation="v",
                    measure=["relative"] * len(quarter_df) + ["total"],
                    x=q_x,
                    textposition="outside",
                    text=q_text,
                    y=q_y,
                    connector={"line": {"color": "rgba(255,255,255,0.2)"}},
                    increasing={"marker": {"color": "#16C79A"}},
                    decreasing={"marker": {"color": "#FF6B6B"}},
                    totals={"marker": {"color": "#5B6EF5"}}
                ))
                fig_q.update_layout(title="Sum of Funding Amount by Quarter")
                st.plotly_chart(style_fig(fig_q), use_container_width=True)

    with col_inv:
        with st.container(key="card_dash_investors"):
            investors_series = fdf["Investors"].fillna("Unknown").str.split(",")
            investors_exploded = investors_series.explode().str.strip()
            top_investors = investors_exploded[investors_exploded != ""].value_counts().head(10).reset_index()
            top_investors.columns = ["Investor", "Count"]
            
            if view_3d:
                html = render_3d_highcharts("column", top_investors, "Count of Startups by Investors", "Investor", "Count")
                components.html(html, height=380)
            else:
                fig_inv = px.bar(
                    top_investors, x="Count", y="Investor", orientation="h",
                    title="Count of Startups by Investors",
                    labels={"Count": "Number of Startups", "Investor": "Investor Name"},
                )
                fig_inv.update_yaxes(autorange="reversed")
                st.plotly_chart(style_fig(fig_inv), use_container_width=True)

    with st.container(key="card_dash5"):
        treemap_df = fdf.groupby("Industry_clean")["FundingAmount"].sum().reset_index()
        treemap_df = treemap_df[treemap_df["FundingAmount"] > 0]
        
        if view_3d:
            html = render_3d_highcharts("column", treemap_df.head(10), "Funding Amount by Industry (3D Column View)", "Industry_clean", "FundingAmount", label_suffix=" $")
            components.html(html, height=380)
        else:
            fig = px.treemap(
                treemap_df, path=["Industry_clean"], values="FundingAmount",
                title="Funding Amount Treemap by Industry",
            )
            st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

    with st.container(key="card_dash6"):
        tier_counts = (
            fdf.dropna(subset=["FundingTier"])["FundingTier"]
            .value_counts()
            .reindex(TIER_LABELS)
            .reset_index()
        )
        tier_counts.columns = ["FundingTier", "Count"]
        
        if view_3d:
            html = render_3d_highcharts("column", tier_counts, "Count of Deals by Funding Tier", "FundingTier", "Count")
            components.html(html, height=380)
        else:
            fig = px.bar(
                tier_counts, x="FundingTier", y="Count", color="FundingTier",
                title="Count of Deals by Funding Tier (Micro < $1M, Small $1-10M, Mid $10-50M, Large $50M+)",
                category_orders={"FundingTier": TIER_LABELS},
                color_discrete_map={"Micro": "#5B6EF5", "Small": "#16C79A", "Mid": "#F2B705", "Large": "#FF6B6B"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with st.container(key="card_dash_timelapse"):
        anim_df, top_ind = get_animated_industry_df(fdf)
        if not anim_df.empty:
            max_val = anim_df["FundingAmount"].max()
            if max_val <= 0:
                max_val = 1.0
            
            fig_anim = px.bar(
                anim_df,
                x="FundingAmount",
                y="Industry_clean",
                color="Industry_clean",
                animation_frame="Year",
                orientation="h",
                title="⏳ Industry Funding Evolution Over Time (Playable Chronological Timelapse)",
                labels={"FundingAmount": "Funding Amount ($)", "Industry_clean": "Industry"},
                category_orders={"Industry_clean": top_ind},
                range_x=[0, max_val * 1.15],
                color_discrete_sequence=PALETTE
            )
            
            fig_anim = style_fig(fig_anim, height=450)
            fig_anim.update_yaxes(autorange="reversed")
            fig_anim.update_layout(showlegend=False)
            
            fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1200
            fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 700
            
            fig_anim.update_layout(
                sliders=[{
                    "currentvalue": {
                        "font": {"family": "Space Grotesk, sans-serif", "size": 18, "color": "#5B6EF5"},
                        "prefix": "Year: ",
                        "visible": True,
                        "xanchor": "right"
                    }
                }]
            )
            
            st.plotly_chart(fig_anim, use_container_width=True)



    with st.expander("View filtered raw data"):
        st.dataframe(fdf.head(200))

# ============================ PREDICT FUNDING ================================
elif page == "💰 Predict Funding Amount":
    st.title("💰 Predict Funding Amount")
    st.markdown(
        "Estimate how much funding a startup with these characteristics is "
        "likely to raise, based on patterns in historical deals."
    )

    with st.container(key="card_reg_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            industries_in = st.multiselect("Industry", sorted(df["Industry_clean"].unique()), default=["eCommerce"], key="reg_ind")
            cities_in = st.multiselect("City", sorted(df["City_clean"].unique()), default=["Bangalore"], key="reg_city")
        with c2:
            invtypes_in = st.multiselect("Investment Type", sorted(df["InvType_clean"].unique()), default=["Seed/Angel Funding"], key="reg_inv")
            years_in = st.multiselect("Year", list(range(2015, 2031)), default=[2020], key="reg_year")
        with c3:
            quarter_in = st.selectbox("Quarter", [1, 2, 3, 4], key="reg_q")
            investors_in = st.number_input("Number of investors", min_value=1, max_value=20, value=1, key="reg_invc")
            has_top_inv_in = st.checkbox("Includes Institutional/Top Investor?", value=False, key="reg_top_inv")
            domain_in = st.selectbox("Sub-Vertical Domain", ["None", "SaaS", "AI", "Platform", "Marketplace", "Mobile", "B2B", "B2C", "Delivery", "Healthcare", "Edtech"], key="reg_dom")

        predict_reg_clicked = st.button("Predict Funding Amount", type="primary")

    if predict_reg_clicked:
        if not industries_in or not cities_in or not invtypes_in or not years_in:
            st.error("Please select at least one option for Industry, City, Investment Type, and Year.")
        else:
            import itertools
            combinations = list(itertools.product(industries_in, cities_in, invtypes_in, years_in))
            rows = []
            domain_flags = get_subvertical_flags(domain_in)
            for ind, city, inv_type, year in combinations:
                row = {
                    "Industry_clean": ind,
                    "City_clean": city,
                    "InvType_clean": inv_type,
                    "Year": year,
                    "Quarter": quarter_in,
                    "InvestorCount": investors_in,
                    "HasTopInvestor": 1 if has_top_inv_in else 0,
                }
                row.update(domain_flags)
                rows.append(row)
            X_new = pd.DataFrame(rows)
            X_new = add_engineered_features(X_new, df)
            log_preds = reg_pipe.predict(X_new)
            pred_amounts = np.expm1(log_preds)
            X_new["Predicted_Amount"] = pred_amounts

            if len(X_new) == 1:
                pred_amount = pred_amounts[0]
                log_pred = log_preds[0]
                low = np.expm1(log_pred - resid_std)
                high = np.expm1(log_pred + resid_std)
                st.success(f"### Estimated Funding: ${pred_amount:,.0f}")
                st.caption(f"Likely range: ${low:,.0f} – ${high:,.0f} (based on model uncertainty)")
                
                # Explainable AI Breakdown
                bias, mapped_contribs = explain_rf_prediction(reg_pipe, X_new, is_regression=True)
                with st.container(key="card_reg_explain"):
                    fig_explain = make_waterfall_chart(
                        bias, mapped_contribs, log_pred,
                        title="Prediction Breakdown — How features impacted this funding estimate",
                        is_regression=True
                    )
                    st.plotly_chart(fig_explain, use_container_width=True)
            else:
                st.success("### Comparative Predictions")
                X_new["Scenario"] = X_new.apply(
                    lambda r: f"{r['Industry_clean']} in {r['City_clean']} ({r['InvType_clean']}, {r['Year']})" + (" + Top Investor" if has_top_inv_in else "") + (f" + {domain_in}" if domain_in != "None" else ""), axis=1
                )
                fig = px.bar(
                    X_new, x="Predicted_Amount", y="Scenario", orientation="h",
                    title="Predicted Funding Amount by Scenario ($)",
                    labels={"Predicted_Amount": "Predicted Funding ($)", "Scenario": "Scenario"},
                    color="Predicted_Amount",
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(coloraxis_showscale=False)
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(style_fig(fig), use_container_width=True)

                st.write("#### Detailed Breakdown (sorted by highest funding)")
                show_df = X_new[["Industry_clean", "City_clean", "InvType_clean", "Year", "HasTopInvestor", "Predicted_Amount"]].sort_values("Predicted_Amount", ascending=False)
                show_df["Predicted_Amount"] = show_df["Predicted_Amount"].apply(lambda v: f"${v:,.0f}")
                st.dataframe(show_df, use_container_width=True)

# ============================ PREDICT FUNDING TIER ================================
elif page == "🎯 Predict Funding Tier":
    st.title("🎯 Predict Funding Tier")
    st.markdown(
        "Classify a startup into a funding **tier** rather than guessing an exact dollar "
        "figure — Micro (<$1M), Small ($1-10M), Mid ($10-50M), or Large ($50M+). This uses "
        "the same inputs as the dollar prediction above, but bucketing the highly skewed "
        "funding amounts into 4 classes is usually an easier, more reliable problem for "
        "these features than predicting an exact number."
    )

    with st.container(key="card_tier_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            industry_in3 = st.multiselect("Industry", sorted(df["Industry_clean"].unique()), default=["eCommerce"], key="tier_ind")
            city_in3 = st.multiselect("City", sorted(df["City_clean"].unique()), default=["Bangalore"], key="tier_city")
        with c2:
            invtype_in3 = st.multiselect("Investment Type", sorted(df["InvType_clean"].unique()), default=["Seed/Angel Funding"], key="tier_inv")
            year_in3 = st.multiselect("Year", list(range(2015, 2031)), default=[2020], key="tier_year")
        with c3:
            quarter_in3 = st.selectbox("Quarter", [1, 2, 3, 4], key="tier_q")
            investors_in3 = st.number_input("Number of investors", min_value=1, max_value=20, value=1, key="tier_invc")
            has_top_inv_in3 = st.checkbox("Includes Institutional/Top Investor?", value=False, key="tier_top_inv")
            domain_in3 = st.selectbox("Sub-Vertical Domain", ["None", "SaaS", "AI", "Platform", "Marketplace", "Mobile", "B2B", "B2C", "Delivery", "Healthcare", "Edtech"], key="tier_dom")

        predict_tier_clicked = st.button("Predict Funding Tier", type="primary")

    if predict_tier_clicked:
        if not industry_in3 or not city_in3 or not invtype_in3 or not year_in3:
            st.error("Please select at least one option for Industry, City, Investment Type, and Year.")
        else:
            import itertools
            combinations = list(itertools.product(industry_in3, city_in3, invtype_in3, year_in3))
            rows = []
            domain_flags = get_subvertical_flags(domain_in3)
            for ind, city, inv_type, year in combinations:
                row = {
                    "Industry_clean": ind,
                    "City_clean": city,
                    "InvType_clean": inv_type,
                    "Year": year,
                    "Quarter": quarter_in3,
                    "InvestorCount": investors_in3,
                    "HasTopInvestor": 1 if has_top_inv_in3 else 0,
                }
                row.update(domain_flags)
                rows.append(row)
            X_new3 = pd.DataFrame(rows)
            X_new3 = add_engineered_features(X_new3, df)
            pred_tiers = tier_pipe.predict(X_new3)
            proba3_matrix = tier_pipe.predict_proba(X_new3)
            X_new3["Predicted_Tier"] = pred_tiers

            if len(X_new3) == 1:
                pred_tier = pred_tiers[0]
                proba3 = proba3_matrix[0]
                proba_df3 = pd.DataFrame({"FundingTier": tier_pipe.classes_, "Probability": proba3})
                proba_df3["FundingTier"] = pd.Categorical(proba_df3["FundingTier"], categories=TIER_LABELS, ordered=True)
                proba_df3 = proba_df3.sort_values("FundingTier")
                st.success(f"### Predicted Tier: {pred_tier} ({TIER_DESCRIPTIONS[pred_tier]})")
                
                # Explainable AI Breakdown
                pred_class_idx = list(tier_pipe.classes_).index(pred_tier)
                bias, mapped_contribs = explain_rf_prediction(
                    tier_pipe, X_new3, is_regression=False, class_index=pred_class_idx
                )
                
                c_res1, c_res2 = st.columns(2)
                with c_res1:
                    with st.container(key="card_tier_result"):
                        fig = px.bar(
                            proba_df3, x="FundingTier", y="Probability", color="FundingTier",
                            title="Predicted Probability by Tier",
                            category_orders={"FundingTier": TIER_LABELS},
                            color_discrete_map={"Micro": "#5B6EF5", "Small": "#16C79A", "Mid": "#F2B705", "Large": "#FF6B6B"},
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(style_fig(fig), use_container_width=True)
                with c_res2:
                    with st.container(key="card_tier_explain"):
                        fig_explain = make_waterfall_chart(
                            bias, mapped_contribs, proba3[pred_class_idx],
                            title=f"Prediction Breakdown — Impact on '{pred_tier}' Probability",
                            is_regression=False
                        )
                        st.plotly_chart(fig_explain, use_container_width=True)
            else:
                st.success("### Comparative Predictions")
                X_new3["Scenario"] = X_new3.apply(
                    lambda r: f"{r['Industry_clean']} in {r['City_clean']} ({r['InvType_clean']}, {r['Year']})" + (" + Top Investor" if has_top_inv_in3 else "") + (f" + {domain_in3}" if domain_in3 != "None" else ""), axis=1
                )
                prob_rows = []
                for idx, r in X_new3.iterrows():
                    for tier_idx, tier_name in enumerate(tier_pipe.classes_):
                        prob_rows.append({
                            "Scenario": r["Scenario"],
                            "FundingTier": tier_name,
                            "Probability": proba3_matrix[idx][tier_idx]
                        })
                prob_df = pd.DataFrame(prob_rows)
                prob_df["FundingTier"] = pd.Categorical(prob_df["FundingTier"], categories=TIER_LABELS, ordered=True)
                prob_df = prob_df.sort_values(["Scenario", "FundingTier"])

                fig = px.bar(
                    prob_df, x="Scenario", y="Probability", color="FundingTier", barmode="group",
                    title="Predicted Probabilities by Tier per Scenario",
                    category_orders={"FundingTier": TIER_LABELS},
                    color_discrete_map={"Micro": "#5B6EF5", "Small": "#16C79A", "Mid": "#F2B705", "Large": "#FF6B6B"}
                )
                fig.update_xaxes(tickangle=30)
                st.plotly_chart(style_fig(fig, height=450), use_container_width=True)

                st.write("#### Detailed Breakdown")
                show_df = X_new3[["Industry_clean", "City_clean", "InvType_clean", "Year", "HasTopInvestor", "Predicted_Tier"]]
                st.dataframe(show_df, use_container_width=True)

# ============================ PREDICT INV TYPE ================================
elif page == "🏷️ Predict Investment Type":
    st.title("🏷️ Predict Investment Type / Round")
    st.markdown(
        "Estimate the most likely funding round type for a startup with these "
        "characteristics."
    )

    with st.container(key="card_clf_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            industry_in2 = st.multiselect("Industry", sorted(df["Industry_clean"].unique()), default=["eCommerce"], key="clf_ind")
            city_in2 = st.multiselect("City", sorted(df["City_clean"].unique()), default=["Bangalore"], key="clf_city")
        with c2:
            year_in2 = st.multiselect("Year", list(range(2015, 2031)), default=[2020], key="clf_year")
            quarter_in2 = st.selectbox("Quarter", [1, 2, 3, 4], key="clf_q")
        with c3:
            funding_in2 = st.number_input(
                "Funding amount ($)", min_value=10000, max_value=4_000_000_000, value=1_000_000, step=10000, key="clf_amt"
            )
            investors_in2 = st.number_input("Number of investors", min_value=1, max_value=20, value=1, key="clf_invc")
            has_top_inv_in2 = st.checkbox("Includes Institutional/Top Investor?", value=False, key="clf_top_inv")
            domain_in2 = st.selectbox("Sub-Vertical Domain", ["None", "SaaS", "AI", "Platform", "Marketplace", "Mobile", "B2B", "B2C", "Delivery", "Healthcare", "Edtech"], key="clf_dom")

        predict_clf_clicked = st.button("Predict Investment Type", type="primary")

    if predict_clf_clicked:
        if not industry_in2 or not city_in2 or not year_in2:
            st.error("Please select at least one option for Industry, City, and Year.")
        else:
            import itertools
            combinations = list(itertools.product(industry_in2, city_in2, year_in2))
            rows = []
            domain_flags = get_subvertical_flags(domain_in2)
            for ind, city, year in combinations:
                row = {
                    "Industry_clean": ind,
                    "City_clean": city,
                    "Year": year,
                    "Quarter": quarter_in2,
                    "LogFunding": np.log1p(funding_in2),
                    "InvestorCount": investors_in2,
                    "HasTopInvestor": 1 if has_top_inv_in2 else 0,
                }
                row.update(domain_flags)
                rows.append(row)
            X_new2 = pd.DataFrame(rows)
            X_new2 = add_engineered_features(X_new2, df)
            pred_classes = clf_pipe.predict(X_new2)
            proba2_matrix = clf_pipe.predict_proba(X_new2)
            X_new2["Predicted_Round"] = pred_classes

            if len(X_new2) == 1:
                pred_class = pred_classes[0]
                proba = proba2_matrix[0]
                proba_df = pd.DataFrame({"InvestmentType": clf_pipe.classes_, "Probability": proba}).sort_values(
                    "Probability", ascending=False
                )
                st.success(f"### Most Likely Round: {pred_class}")
                
                # Explainable AI Breakdown
                pred_class_idx = list(clf_pipe.classes_).index(pred_class)
                bias, mapped_contribs = explain_rf_prediction(
                    clf_pipe, X_new2, is_regression=False, class_index=pred_class_idx
                )
                
                c_res1, c_res2 = st.columns(2)
                with c_res1:
                    with st.container(key="card_clf_result"):
                        fig = px.bar(
                            proba_df, x="Probability", y="InvestmentType", orientation="h",
                            title="Predicted Probability by Round Type",
                        )
                        fig.update_yaxes(autorange="reversed")
                        st.plotly_chart(style_fig(fig), use_container_width=True)
                with c_res2:
                    with st.container(key="card_clf_explain"):
                        fig_explain = make_waterfall_chart(
                            bias, mapped_contribs, proba[pred_class_idx],
                            title=f"Prediction Breakdown — Impact on '{pred_class}' Probability",
                            is_regression=False
                        )
                        st.plotly_chart(fig_explain, use_container_width=True)
            else:
                st.success("### Comparative Predictions")
                X_new2["Scenario"] = X_new2.apply(
                    lambda r: f"{r['Industry_clean']} in {r['City_clean']} ({r['Year']})" + (" + Top Investor" if has_top_inv_in2 else "") + (f" + {domain_in2}" if domain_in2 != "None" else ""), axis=1
                )
                prob_rows = []
                for idx, r in X_new2.iterrows():
                    for class_idx, class_name in enumerate(clf_pipe.classes_):
                        prob_rows.append({
                            "Scenario": r["Scenario"],
                            "InvestmentType": class_name,
                            "Probability": proba2_matrix[idx][class_idx]
                        })
                prob_df = pd.DataFrame(prob_rows)
                prob_df = prob_df.sort_values(["Scenario", "Probability"], ascending=[True, False])

                fig = px.bar(
                    prob_df, x="Scenario", y="Probability", color="InvestmentType", barmode="group",
                    title="Predicted Probabilities by Round Type per Scenario"
                )
                fig.update_xaxes(tickangle=30)
                st.plotly_chart(style_fig(fig, height=450), use_container_width=True)

                st.write("#### Detailed Breakdown")
                show_df = X_new2[["Industry_clean", "City_clean", "Year", "HasTopInvestor", "Predicted_Round"]]
                st.dataframe(show_df, use_container_width=True)

# ============================ MATCHMAKER & SIMULATOR ============================
elif page == "🤝 Investor Matchmaker & Simulator":
    st.title("🤝 Investor Matchmaker & Funding Simulator")
    st.markdown(
        "Find the most compatible venture capital firms for your startup based on "
        "historical deals, and simulate the dilution impact of your upcoming funding round."
    )
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🔍 Investor Matchmaker")
        with st.container(key="card_matchmaker_inputs"):
            c1, c2 = st.columns(2)
            with c1:
                match_industry = st.selectbox("Industry", sorted(df["Industry_clean"].unique()), index=0, key="match_ind")
                match_city = st.selectbox("City / Hub", sorted(df["City_clean"].unique()), index=0, key="match_cit")
            with c2:
                match_tier = st.selectbox("Expected Stage (Funding Tier)", TIER_LABELS, index=1, key="match_tie")
                match_domain = st.selectbox("Sub-Vertical Domain", ["None", "SaaS", "AI", "Platform", "Marketplace", "Mobile", "B2B", "B2C", "Delivery", "Healthcare", "Edtech"], key="match_dom")
            
            run_matchmaker = st.button("Find Matching Investors", type="primary", key="btn_matchmaker")
            
        if run_matchmaker:
            st.write("")
            matching_df = match_investors(df, match_industry, match_city, match_tier, match_domain)
            if len(matching_df) == 0:
                st.warning("No matching investors found in our database for this combination. Try broadening your criteria.")
            else:
                st.success("### Top 10 Investor Matches")
                for idx, r in matching_df.iterrows():
                    score = r["Match Score (%)"]
                    color = "#16C79A" if score >= 70 else "#F2B705" if score >= 40 else "#FF6B6B"
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 12px 16px; border-radius: 12px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; color: #EAF0FF; font-family: 'Space Grotesk', sans-serif; font-size: 1rem;">{r['Investor']}</span>
                            <span style="background: {color}; color: #0A0F2C; font-weight: 700; padding: 2px 8px; border-radius: 20px; font-size: 0.8rem;">{score}% Match</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #B8C2E8; margin-bottom: 8px;">
                            💼 <b>Total Deals:</b> {r['Total Deals']} &nbsp;•&nbsp; 
                            🏭 <b>{match_industry} Deals:</b> {r['Industry Deals']} &nbsp;•&nbsp; 
                            🌆 <b>{match_city} Deals:</b> {r['City Deals']}
                        </div>
                        <div style="font-size: 0.8rem; color: #8B95C9;">
                            🚀 <b>Top Portfolios:</b> <span style="color: #F2B705;">{r['Key Portfolios']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
    with col_right:
        st.subheader("📊 Cap Table Dilution Simulator")
        with st.container(key="card_simulator"):
            target_raise = st.number_input("Target Raise Amount ($)", min_value=10000, max_value=500000000, value=1000000, step=50000, key="sim_raise")
            pre_money_val = st.number_input("Pre-Money Valuation ($)", min_value=10000, max_value=2000000000, value=4000000, step=100000, key="sim_preval")
            
            st.markdown("##### Current Equity Distribution (%)")
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                founder1_pct = st.slider("Founder 1 Equity", 0, 100, 60, key="sim_f1")
            with c_f2:
                founder2_pct = st.slider("Founder 2 Equity", 0, 100 - founder1_pct, 100 - founder1_pct, key="sim_f2")
                
            other_pct = 100 - founder1_pct - founder2_pct
            if other_pct > 0:
                st.caption(f"Other Shareholders: {other_pct}%")
                
            # Perform simulator calculations
            post_money_val = pre_money_val + target_raise
            dilution_rate = target_raise / post_money_val
            dilution_pct = dilution_rate * 100
            
            # Post-round equity calculations
            new_investor_pct = dilution_pct
            new_f1_pct = founder1_pct * (1 - dilution_rate)
            new_f2_pct = founder2_pct * (1 - dilution_rate)
            new_other_pct = other_pct * (1 - dilution_rate)
            
            st.markdown("---")
            st.markdown("#### Dilution Outcome")
            
            c_m1, c_m2 = st.columns(2)
            c_m1.metric("Post-Money Valuation", f"${post_money_val:,.0f}")
            c_m2.metric("New Investor Share", f"{dilution_pct:.1f}%")
            
            # Advisor Badges
            if dilution_pct < 10.0:
                st.info("💡 **Low Dilution**: Excellent term conservation for founders, but ensure the high valuation is justified to avoid down-rounds in the future.")
            elif 15.0 <= dilution_pct <= 25.0:
                st.success("✅ **Healthy Venture Standard**: Dilution is in the standard range (15% - 25%) for a standard institutional round.")
            elif dilution_pct > 30.0:
                st.warning("⚠️ **High Dilution**: Founders are diluting heavily in this round. Consider raising less capital or negotiating a higher valuation.")
                
            # Plotly Cap Table Pie Chart
            pie_data = {
                "Shareholder": ["Founder 1", "Founder 2", "New Investor"],
                "Equity": [new_f1_pct, new_f2_pct, new_investor_pct]
            }
            if new_other_pct > 0:
                pie_data["Shareholder"].append("Other Shareholders")
                pie_data["Equity"].append(new_other_pct)
                
            df_pie = pd.DataFrame(pie_data)
            fig_pie = px.pie(
                df_pie, names="Shareholder", values="Equity",
                title="Post-Round Cap Table",
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(style_fig(fig_pie, height=340), use_container_width=True)

# ============================ MODEL PERFORMANCE ================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    st.subheader("Funding Amount Regression")
    with st.container(key="card_perf1"):
        c1, c2 = st.columns(2)
        c1.metric("R² (test set)", f"{reg_metrics['r2']:.3f}")
        c2.metric("Mean Abs. Error", f"${reg_metrics['mae_dollar']:,.0f}")
        fi_reg = get_feature_importance(reg_pipe)
        fig = px.bar(fi_reg, x="importance", y="feature", orientation="h", title="Top Features")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Funding Tier Classification")
    with st.container(key="card_perf2"):
        c1, c2 = st.columns(2)
        c1.metric("Accuracy (test set)", f"{tier_metrics['accuracy']:.3f}")
        c2.metric("F1 Macro", f"{tier_metrics['f1_macro']:.3f}")
        fi_tier = get_feature_importance(tier_pipe)
        fig = px.bar(fi_tier, x="importance", y="feature", orientation="h", title="Top Features")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.caption(
            "Same input features as the regression model above, but bucketed into 4 tiers "
            "instead of predicting an exact dollar figure — typically more reliable given how "
            "skewed funding amounts are."
        )

    st.subheader("Investment Type Classification")
    with st.container(key="card_perf3"):
        c1, c2 = st.columns(2)
        c1.metric("Accuracy (test set)", f"{clf_metrics['accuracy']:.3f}")
        c2.metric("F1 Macro", f"{clf_metrics['f1_macro']:.3f}")
        fi_clf = get_feature_importance(clf_pipe)
        fig = px.bar(fi_clf, x="importance", y="feature", orientation="h", title="Top Features")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.info(
        "Models are trained fresh each time the app restarts (cached for the session). "
        "The dataset has many missing/'Undisclosed' funding amounts and very long-tail "
        "categories, so predictions are directional estimates, not precise forecasts."
    )

