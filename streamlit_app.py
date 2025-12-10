import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import time
from pathlib import Path

# ---------------------------------------------------
# PAGE CONFIG + STYLING
# ---------------------------------------------------
st.set_page_config(
    page_title="Banff Parking – ML Dashboard",
    layout="wide"
)

st.markdown(
    """
    <style>
        .main {
            background-color: #0f172a;
            color: #e5e7eb;
        }
        .app-header {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }
        .app-subtitle {
            color: #9ca3af;
            font-size: 0.9rem;
            margin-bottom: 1.2rem;
        }
        .card {
            padding: 1rem 1.25rem;
            border-radius: 0.85rem;
            background: radial-gradient(circle at top left, #1d283a, #020617);
            border: 1px solid rgba(148, 163, 184, 0.3);
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.7);
            margin-bottom: 0.75rem;
        }
        .metric-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            color: #9ca3af;
            letter-spacing: 0.08em;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 600;
            margin-top: 0.3rem;
        }
        .metric-badge {
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            margin-top: 0.4rem;
        }
        .pill-tabs > div[role="tablist"] {
            gap: 0.5rem;
        }
        .pill-tabs button[role="tab"] {
            border-radius: 999px !important;
            padding-top: 0.4rem !important;
            padding-bottom: 0.4rem !important;
        }
        .stDataFrame {
            border-radius: 0.75rem;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# DATA & MODEL LOADING
# ---------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    # Try multiple possible CSV names
    possible_files = [
        "banff_parking_engineered_HOURLY.csv",
        "banff_parking_engineered_HOURLY (1).csv",
        "banff_parking_engineered_HOURLY(1).csv",
    ]
    csv_path = None
    for fname in possible_files:
        if Path(fname).exists():
            csv_path = fname
            break

    if csv_path is None:
        st.error(
            "CSV file not found. Please make sure one of these files is in the repo "
            "root: 'banff_parking_engineered_HOURLY.csv' or 'banff_parking_engineered_HOURLY (1).csv'."
        )
        st.stop()

    df = pd.read_csv(csv_path)

    # Timestamp
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Ensure engineered features exist
    if "Hour" in df.columns and "hour" not in df.columns:
        df["hour"] = df["Hour"]
    if "DayOfWeek" in df.columns and "dow" not in df.columns:
        df["dow"] = df["DayOfWeek"]
    if "Timestamp" in df.columns and "day_of_year" not in df.columns:
        df["day_of_year"] = df["Timestamp"].dt.dayofyear

    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    if "dow" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    if "DayOfWeek" in df.columns and "is_weekend" not in df.columns:
        df["is_weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    return df


@st.cache_resource(show_spinner=True)
def load_models_and_features():
    reg = joblib.load("banff_best_xgb_reg.pkl")

    # Try LGBM first, then XGB classifier
    try:
        cls = joblib.load("banff_best_lgbm_cls.pkl")
    except FileNotFoundError:
        cls = joblib.load("banff_best_xgb_cls.pkl")

    scaler = joblib.load("banff_scaler.pkl")
    features = joblib.load("banff_features.pkl")
    if not isinstance(features, list):
        features = list(features)
    return reg, cls, scaler, features


def make_predictions(row, reg, cls, scaler, features):
    missing = [f for f in features if f not in row.index]
    if missing:
        raise ValueError(f"Missing features in data row: {missing}")

    X = row[features].to_frame().T
    X_scaled = scaler.transform(X)

    demand_pred = float(reg.predict(X_scaled)[0])
    cls_proba = cls.predict_proba(X_scaled)[0]
    full_prob = float(cls_proba[1])
    label = int(cls.predict(X_scaled)[0])

    return demand_pred, full_prob, label


def congestion_level(percent_occupancy):
    if percent_occupancy < 0.4:
        return "Low", "✅ Easy parking"
    elif percent_occupancy < 0.75:
        return "Medium", "⚠️ Getting busy"
    else:
        return "High", "🚨 Very crowded"


data = load_data()
reg_model, cls_model, scaler, feature_list = load_models_and_features()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<div class="app-header">Banff Parking AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Smart parking demand & congestion forecasts for Banff lots.</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# TABS (NO SIDEBAR)
# ---------------------------------------------------
with st.container():
    st.markdown('<div class="pill-tabs">', unsafe_allow_html=True)
    tab_dashboard, tab_predict, tab_insights = st.tabs(
        ["🏠 Dashboard", "🔮 Predict", "📊 Insights"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 🏠 DASHBOARD TAB
# ---------------------------------------------------
with tab_dashboard:
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Records</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(data):,}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        units = data["Unit"].nunique()
        st.markdown('<div class="metric-title">Parking lots</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{units}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        min_date = data["Timestamp"].min().strftime("%Y-%m-%d")
        max_date = data["Timestamp"].max().strftime("%Y-%m-%d")
        st.markdown('<div class="metric-title">Data range</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{min_date}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="metric-badge">to {max_date}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Targets</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">Occupancy</div>', unsafe_allow_html=True)
        st.markdown('<span class="metric-badge">Is_Full (0/1)</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Layout: chart + sample data
    left, right = st.columns([1.6, 1])

    with left:
        st.markdown("#### ⏰ Daily pattern by lot")

        unit_list = sorted(data["Unit"].unique())
        selected_unit = st.selectbox("Lot", unit_list, key="dash_unit")

        unit_df = data[data["Unit"] == selected_unit].copy()
        available_dates = sorted(unit_df["Timestamp"].dt.date.unique())

        if available_dates:
            selected_date = st.date_input(
                "Date",
                value=available_dates[0],
                min_value=min(available_dates),
                max_value=max(available_dates),
                key="dash_date"
            )
            day_df = unit_df[unit_df["Timestamp"].dt.date == selected_date]

            if not day_df.empty:
                fig, ax = plt.subplots()
                ax.plot(day_df["Hour"], day_df["Percent_Occupancy"] * 100, marker="o")
                ax.set_xlabel("Hour")
                ax.set_ylabel("Occupancy (%)")
                ax.set_title(f"{selected_unit} on {selected_date}")
                ax.set_xticks(range(0, 24, 2))
                ax.grid(True, linestyle="--", linewidth=0.4)
                st.pyplot(fig)
            else:
                st.info("No data for this date.")
        else:
            st.info("No dates available for this lot.")

    with right:
        st.markdown("#### 🔎 Sample data")
        st.dataframe(
            data[["Timestamp", "Unit", "Occupancy", "Capacity", "Percent_Occupancy"]]
            .head(15)
            .reset_index(drop=True)
        )

# ---------------------------------------------------
# 🔮 PREDICT TAB
# ---------------------------------------------------
with tab_predict:
    st.markdown("#### Scenario selection")

    # Controls row
    c1, c2, c3 = st.columns(3)
    with c1:
        pred_unit = st.selectbox(
            "Lot",
            sorted(data["Unit"].unique()),
            key="pred_unit"
        )

    unit_df = data[data["Unit"] == pred_unit].copy()
    available_dates = sorted(unit_df["Timestamp"].dt.date.unique())

    with c2:
        if available_dates:
            pred_date = st.date_input(
                "Date",
                value=available_dates[0],
                min_value=min(available_dates),
                max_value=max(available_dates),
                key="pred_date"
            )
        else:
            pred_date = None

    with c3:
        pred_time = st.time_input(
            "Time",
            value=time(12, 0),
            step=3600,  # hourly
            key="pred_time"
        )

    if not available_dates or pred_date is None:
        st.warning("No data available for this lot.")
    else:
        # Filter for date + hour
        selected_hour = pred_time.hour
        filtered = unit_df[unit_df["Timestamp"].dt.date == pred_date]
        row_match = filtered[filtered["Hour"] == selected_hour]

        if row_match.empty:
            st.info("No exact hour found for this date. Try a different time.")
        else:
            row = row_match.iloc[0]

            st.markdown("#### Selected context")
            st.markdown(
                f"- **Lot:** {pred_unit}  \n"
                f"- **Date:** {pred_date}  \n"
                f"- **Time:** {selected_hour:02d}:00  \n"
                f"- **Weather:** {row['Max Temp (°C)']}°C / {row['Min Temp (°C)']}°C • "
                f"Precip: {row['Total Precip (mm)']} mm • Wind gust: {row['Spd of Max Gust (km/h)']} km/h"
            )

            run = st.button("🚗 Predict parking demand")

            if run:
                try:
                    y_pred, full_prob, label = make_predictions(
                        row, reg_model, cls_model, scaler, feature_list
                    )
                    capacity = row["Capacity"]
                    if capacity > 0:
                        pred_percent = float(np.clip(y_pred / capacity, 0, 1))
                    else:
                        pred_percent = 0.0

                    level, msg = congestion_level(pred_percent)

                    top_a, top_b, top_c = st.columns(3)

                    with top_a:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown(
                            '<div class="metric-title">Predicted occupancy</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="metric-value">{y_pred:.0f} / {int(capacity)}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<span class="metric-badge">{pred_percent * 100:.1f}% full</span>',
                            unsafe_allow_html=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    with top_b:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown(
                            '<div class="metric-title">Full / near-full risk</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="metric-value">{full_prob * 100:.1f}%</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<span class="metric-badge">Level: {level}</span>',
                            unsafe_allow_html=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    with top_c:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown(
                            '<div class="metric-title">Quick note</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="metric-value">{msg}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("#### Historical value (same hour)")
                    st.markdown(
                        f"- Actual: **{row['Occupancy']} / {int(row['Capacity'])}** "
                        f"({row['Percent_Occupancy']*100:.1f}%)  \n"
                        f"- Actual Is_Full: **{row['Is_Full']}**"
                    )

                except Exception as e:
                    st.error(f"Prediction error: {e}")

# ---------------------------------------------------
# 📊 INSIGHTS TAB
# ---------------------------------------------------
with tab_insights:
    st.markdown("#### Feature importance")

    left, right = st.columns(2)

    # Regression model
    with left:
        st.markdown("**Regression model (Occupancy)**")
        try:
            if hasattr(reg_model, "feature_importances_"):
                fi_reg = pd.DataFrame({
                    "Feature": feature_list,
                    "Importance": reg_model.feature_importances_
                }).sort_values("Importance", ascending=False)[:15]

                fig1, ax1 = plt.subplots(figsize=(5, 5))
                ax1.barh(fi_reg["Feature"], fi_reg["Importance"])
                ax1.set_xlabel("Importance")
                ax1.set_ylabel("")
                ax1.set_title("Top features")
                ax1.invert_yaxis()
                st.pyplot(fig1)
            else:
                st.info("Model does not expose feature_importances_.")
        except Exception as e:
            st.error(f"Could not plot regression features: {e}")

    # Classification model
    with right:
        st.markdown("**Classification model (Is_Full)**")
        try:
            if hasattr(cls_model, "feature_importances_"):
                fi_cls = pd.DataFrame({
                    "Feature": feature_list,
                    "Importance": cls_model.feature_importances_
                }).sort_values("Importance", ascending=False)[:15]

                fig2, ax2 = plt.subplots(figsize=(5, 5))
                ax2.barh(fi_cls["Feature"], fi_cls["Importance"])
                ax2.set_xlabel("Importance")
                ax2.set_ylabel("")
                ax2.set_title("Top features")
                ax2.invert_yaxis()
                st.pyplot(fig2)
            else:
                st.info("Model does not expose feature_importances_.")
        except Exception as e:
            st.error(f"Could not plot classification features: {e}")

    st.markdown("---")
    st.markdown(
        "Key drivers usually include: time of day, weekday/weekend, recent occupancy history, "
        "and weather (temperature, precipitation, wind)."
    )
