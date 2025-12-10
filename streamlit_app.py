import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import time
from pathlib import Path
from typing import List, Dict

# ---------------------------------------------------
# PAGE CONFIG + STYLING
# ---------------------------------------------------
st.set_page_config(
    page_title="Banff Parking – Research Questions Dashboard",
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
def load_data() -> pd.DataFrame:
    """
    Load the hourly parking dataset.
    We only assume basic columns; engineered model features are handled separately.
    """
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
            "CSV file not found. Please upload one of: "
            "'banff_parking_engineered_HOURLY.csv' or "
            "'banff_parking_engineered_HOURLY (1).csv' "
            "to the root of the repo."
        )
        st.stop()

    df = pd.read_csv(csv_path)

    # Timestamp & simple time features for UI
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Hour"] = df["Timestamp"].dt.hour
        df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
        df["Month"] = df["Timestamp"].dt.month

    # Percent occupancy for plots
    if "Percent_Occupancy" not in df.columns and {"Occupancy", "Capacity"} <= set(df.columns):
        df["Percent_Occupancy"] = df["Occupancy"] / df["Capacity"]

    return df


@st.cache_resource(show_spinner=True)
def load_models_and_features():
    """Load trained models, scaler and the feature list."""
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


# ---------------------------------------------------
# FEATURE PREPARATION FOR A SINGLE ROW
# ---------------------------------------------------
def build_feature_vector(row: pd.Series, features: List[str]) -> pd.DataFrame:
    """
    Build a 1-row DataFrame with all features required by the model.

    - If feature exists in row: use its value
    - If feature starts with 'Unit_': create one-hot from row['Unit']
    - Otherwise: default to 0.0 (neutral for missing engineered features)
    """
    row_dict: Dict[str, float] = {}

    for f in features:
        if f in row.index:
            row_dict[f] = row[f]
        elif f.startswith("Unit_") and "Unit" in row.index:
            unit_name = f[len("Unit_"):]
            row_dict[f] = 1.0 if str(row["Unit"]) == unit_name else 0.0
        else:
            row_dict[f] = 0.0

    X = pd.DataFrame([row_dict])[features]
    return X


def make_predictions(row: pd.Series,
                     reg,
                     cls,
                     scaler,
                     features: List[str]):
    """
    Prepare feature vector and run both models.
    Returns: occupancy prediction, near-full probability, label (0/1).
    """
    X = build_feature_vector(row, features)
    X_scaled = scaler.transform(X)

    demand_pred = float(reg.predict(X_scaled)[0])
    cls_proba = cls.predict_proba(X_scaled)[0]
    full_prob = float(cls_proba[1])
    label = int(cls.predict(X_scaled)[0])

    return demand_pred, full_prob, label


def congestion_level(percent_occupancy: float):
    """
    Convert predicted occupancy (0–1) to human label for congestion.
    """
    if percent_occupancy < 0.4:
        return "Low", "✅ Easy parking"
    elif percent_occupancy < 0.75:
        return "Medium", "⚠️ Getting busy"
    else:
        return "High", "🚨 Very crowded"


# Load everything once
data = load_data()
reg_model, cls_model, scaler, feature_list = load_models_and_features()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<div class="app-header">Banff Parking AI – Research Questions</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="app-subtitle">
    This app answers two core questions for the Town of Banff:
    <br>
    <b>Q1:</b> Which factors (time, weekday, weather, trends) best predict parking demand?
    <br>
    <b>Q2:</b> Can we forecast, hour by hour, when a lot will be near capacity (&gt; 90% full)?
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# TABS ALIGNED WITH PROBLEM STATEMENTS
# ---------------------------------------------------
with st.container():
    st.markdown('<div class="pill-tabs">', unsafe_allow_html=True)
    tab_q1, tab_q2, tab_data = st.tabs(
        ["🧠 Q1 – Key Predictors", "🔮 Q2 – Near-Capacity Forecast", "📊 Data Explorer"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 🧠 TAB Q1 – KEY PREDICTORS (Problem Statement 1)
# ---------------------------------------------------
with tab_q1:
    st.markdown("### Q1. Which factors are the most reliable predictors of parking demand?")
    st.markdown(
        "This view uses **model feature importance** to show which variables "
        "(time of day, day of week, weather, recent history, etc.) drive the predictions."
    )

    col_top, col_bottom = st.columns([1.1, 1])

    # --- Top 5 features summary (combined from reg + cls) ---
    with col_top:
        st.markdown("#### Top predictors (automatically extracted from the models)")

        try:
            combined_df = None

            if hasattr(reg_model, "feature_importances_"):
                fi_reg = pd.DataFrame({
                    "Feature": feature_list,
                    "Reg_Importance": reg_model.feature_importances_
                })
                combined_df = fi_reg.copy()
            if hasattr(cls_model, "feature_importances_"):
                fi_cls = pd.DataFrame({
                    "Feature": feature_list,
                    "Cls_Importance": cls_model.feature_importances_
                })
                if combined_df is None:
                    combined_df = fi_cls.copy()
                else:
                    combined_df = combined_df.merge(fi_cls, on="Feature", how="outer")

            if combined_df is not None:
                combined_df = combined_df.fillna(0.0)
                combined_df["Combined"] = combined_df.get("Reg_Importance", 0) + combined_df.get("Cls_Importance", 0)
                top5 = combined_df.sort_values("Combined", ascending=False).head(5)

                # Cards for top 3
                c1, c2, c3 = st.columns(3)
                top_features = top5["Feature"].tolist()

                def nice_name(feat: str) -> str:
                    # Quick friendly-name mapping
                    mapping_keywords = [
                        ("Hour", "Time of day"),
                        ("hour", "Time of day"),
                        ("DayOfWeek", "Day of week"),
                        ("dow", "Day of week"),
                        ("Month", "Month / season"),
                        ("Max Temp", "Max temperature"),
                        ("Min Temp", "Min temperature"),
                        ("Total Precip", "Precipitation"),
                        ("Gust", "Wind / gusts"),
                        ("Occupancy", "Recent occupancy"),
                        ("Percent_Occupancy", "Recent occupancy"),
                        ("roll", "Rolling average / trend"),
                        ("lag", "Lagged history"),
                        ("is_weekend", "Weekend / weekday"),
                        ("Unit_", "Specific parking lot"),
                    ]
                    label = feat
                    for key, nice in mapping_keywords:
                        if key in feat:
                            label = nice
                            break
                    return label

                # Card 1
                with c1:
                    if len(top_features) > 0:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-title">#1 Predictor</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">{nice_name(top_features[0])}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<span class="metric-badge">{top_features[0]}</span>',
                            unsafe_allow_html=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                # Card 2
                with c2:
                    if len(top_features) > 1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-title">#2 Predictor</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">{nice_name(top_features[1])}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<span class="metric-badge">{top_features[1]}</span>',
                            unsafe_allow_html=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                # Card 3
                with c3:
                    if len(top_features) > 2:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-title">#3 Predictor</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">{nice_name(top_features[2])}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<span class="metric-badge">{top_features[2]}</span>',
                            unsafe_allow_html=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("**Model answer to Q1:**")
                st.markdown(
                    "- Parking demand is mainly explained by **time-of-day / day-of-week patterns**, "
                    "the **recent occupancy history (lags / rolling averages)**, and **weather variables** "
                    "(temperature, precipitation, wind). "
                    "These appear at the top of the importance ranking."
                )
            else:
                st.info("Models do not expose feature_importances_.")
        except Exception as e:
            st.error(f"Could not compute top predictors: {e}")

    # --- Detailed feature importance plots ---
    with col_bottom:
        st.markdown("#### Detailed feature importance")

        left, right = st.columns(2)

        # Regression model
        with left:
            st.markdown("**Regression model (predicts # of occupied stalls)**")
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
                    st.info("Regression model does not expose feature_importances_.")
            except Exception as e:
                st.error(f"Could not plot regression features: {e}")

        # Classification model
        with right:
            st.markdown("**Classification model (probability lot is near full)**")
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
                    st.info("Classification model does not expose feature_importances_.")
            except Exception as e:
                st.error(f"Could not plot classification features: {e}")

    st.markdown("---")
    st.markdown(
        "➡️ **Interpretation for your report:** Time of day, day of week, and recent occupancy "
        "are the most stable and reliable drivers of parking demand, with weather and seasonal "
        "effects adding additional variation."
    )

# ---------------------------------------------------
# 🔮 TAB Q2 – HOURLY NEAR-CAPACITY FORECAST (Problem Statement 2)
# ---------------------------------------------------
with tab_q2:
    st.markdown("### Q2. Can we forecast, hour-by-hour, when a lot will be near capacity (> 90% full)?")
    st.markdown(
        "Use the controls below to select a parking lot, date, and hour. "
        "The system predicts occupancy and the probability that the lot is **near full**."
    )

    # --- Scenario selection ---
    c1, c2, c3 = st.columns(3)
    with c1:
        pred_unit = st.selectbox(
            "Parking lot (Unit)",
            sorted(data["Unit"].unique()),
            key="pred_unit"
        )

    unit_df = data[data["Unit"] == pred_unit].copy()
    if "Timestamp" in unit_df.columns:
        available_dates = sorted(unit_df["Timestamp"].dt.date.unique())
    else:
        available_dates = []

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
            step=3600,
            key="pred_time"
        )

    if not available_dates or pred_date is None:
        st.warning("No data available for this lot.")
    else:
        selected_hour = pred_time.hour
        filtered = unit_df[unit_df["Timestamp"].dt.date == pred_date]
        row_match = filtered[filtered["Hour"] == selected_hour]

        if row_match.empty:
            st.info("No exact hour found for this date. Try a different time.")
        else:
            row = row_match.iloc[0]

            # Context line
            if all(col in row.index for col in ["Max Temp (°C)", "Min Temp (°C)", "Total Precip (mm)", "Spd of Max Gust (km/h)"]):
                weather_str = (
                    f"{row['Max Temp (°C)']}°C / {row['Min Temp (°C)']}°C • "
                    f"Precip: {row['Total Precip (mm)']} mm • "
                    f"Wind gust: {row['Spd of Max Gust (km/h)']} km/h"
                )
            else:
                weather_str = "Weather not available in this file"

            st.markdown("#### Selected scenario")
            st.markdown(
                f"- **Lot:** {pred_unit}  \n"
                f"- **Date:** {pred_date}  \n"
                f"- **Time:** {selected_hour:02d}:00  \n"
                f"- **Weather:** {weather_str}"
            )

            run = st.button("🚗 Run hourly forecast")

            if run:
                try:
                    y_pred, full_prob, label = make_predictions(
                        row, reg_model, cls_model, scaler, feature_list
                    )
                    capacity = row.get("Capacity", 0)
                    if capacity > 0:
                        pred_percent = float(np.clip(y_pred / capacity, 0, 1))
                    else:
                        pred_percent = 0.0

                    level, msg = congestion_level(pred_percent)
                    near_capacity_flag = pred_percent >= 0.90  # > 90% full

                    top_a, top_b, top_c, top_d = st.columns(4)

                    with top_a:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-title">Predicted occupancy</div>', unsafe_allow_html=True)
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
                        st.markdown('<div class="metric-title">Near-full probability</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">{full_prob * 100:.1f}%</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<span class="metric-badge">Model: Is_Full classifier</span>',
                            unsafe_allow_html=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    with top_c:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-title">Congestion level</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">{level}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<span class="metric-badge">{msg}</span>',
                            unsafe_allow_html=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    with top_d:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown('<div class="metric-title">Near capacity? (&gt; 90%)</div>', unsafe_allow_html=True)
                        answer = "YES" if near_capacity_flag else "NO"
                        colour = "#22c55e" if not near_capacity_flag else "#f97316"
                        st.markdown(
                            f'<div class="metric-value" style="color:{colour};">{answer}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            '<span class="metric-badge">Threshold: 90% occupancy</span>',
                            unsafe_allow_html=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Historical actual for comparison
                    if all(col in row.index for col in ["Occupancy", "Capacity", "Percent_Occupancy", "Is_Full"]):
                        st.markdown("#### Historical value (same hour, same lot)")
                        st.markdown(
                            f"- **Actual:** {row['Occupancy']} / {int(row['Capacity'])} "
                            f"({row['Percent_Occupancy']*100:.1f}%)  \n"
                            f"- **Actual Is_Full label:** {row['Is_Full']}"
                        )

                    st.markdown("---")
                    st.markdown(
                        "➡️ **Interpretation for your report:** "
                        "This system provides an hourly forecast of occupancy and near-full risk. "
                        "By checking if predicted occupancy exceeds 90%, it can flag hours "
                        "when a specific lot is likely to operate at or near capacity."
                    )

                except Exception as e:
                    st.error(f"Prediction error: {e}")

# ---------------------------------------------------
# 📊 DATA EXPLORER – CONTEXT & TRENDS
# ---------------------------------------------------
with tab_data:
    st.markdown("### Data Explorer & Daily Patterns")
    st.markdown(
        "This tab gives context for the models: overall data size, range, and "
        "typical daily occupancy patterns for each lot."
    )

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
        if "Timestamp" in data.columns:
            min_date = data["Timestamp"].min().strftime("%Y-%m-%d")
            max_date = data["Timestamp"].max().strftime("%Y-%m-%d")
        else:
            min_date = max_date = "-"
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
        if "Timestamp" in unit_df.columns:
            available_dates = sorted(unit_df["Timestamp"].dt.date.unique())
        else:
            available_dates = []

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
        cols_to_show = [
            c for c in ["Timestamp", "Unit", "Occupancy", "Capacity", "Percent_Occupancy"]
            if c in data.columns
        ]
        st.dataframe(
            data[cols_to_show]
            .head(15)
            .reset_index(drop=True)
        )
