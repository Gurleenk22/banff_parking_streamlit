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
    page_title="Banff Smart Parking – ML & XAI Dashboard",
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
            font-size: 2.3rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }
        .app-subtitle {
            color: #9ca3af;
            font-size: 0.9rem;
            margin-bottom: 1.2rem;
        }
        .card {
            padding: 1rem 1.25rem;
            border-radius: 0.9rem;
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
    """Load the hourly parking dataset for UI + plots."""
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
# FEATURE BUILDING + PREDICTIONS
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
    """Convert predicted occupancy (0–1) to label for congestion."""
    if percent_occupancy < 0.4:
        return "Low", "✅ Easy parking"
    elif percent_occupancy < 0.75:
        return "Medium", "⚠️ Getting busy"
    else:
        return "High", "🚨 Very crowded"


# Cached SHAP computation for regression model
@st.cache_resource(show_spinner=True)
def compute_shap_global(data: pd.DataFrame,
                        feature_list: List[str],
                        reg_model,
                        scaler,
                        sample_size: int = 200):
    """
    Compute global SHAP values on a random sample for the regression model.
    Done lazily so the app stays responsive.
    """
    try:
        import shap
    except ModuleNotFoundError:
        return None, None

    if len(data) == 0:
        return None, None

    sample_size = min(sample_size, len(data))
    idx = np.random.choice(len(data), size=sample_size, replace=False)
    sample = data.iloc[idx].copy()

    # Build feature matrix from sample
    rows = []
    for _, r in sample.iterrows():
        rows.append(build_feature_vector(r, feature_list).iloc[0])
    X = pd.DataFrame(rows)[feature_list]

    X_scaled = scaler.transform(X)
    explainer = shap.TreeExplainer(reg_model)
    shap_values = explainer.shap_values(X_scaled)

    return shap_values, X


# Load data & models
data = load_data()
reg_model, cls_model, scaler, feature_list = load_models_and_features()

# ---------------------------------------------------
# HEADER (always visible)
# ---------------------------------------------------
st.markdown('<div class="app-header">Banff Smart Parking</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Machine-learning & XAI dashboard for hourly parking demand forecasting.</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# TABS – PURE APP STYLE
# ---------------------------------------------------
with st.container():
    st.markdown('<div class="pill-tabs">', unsafe_allow_html=True)
    tab_overview, tab_forecast, tab_xai, tab_data = st.tabs(
        ["🏠 Overview", "🚗 Forecast", "🧠 XAI Insights", "📊 Data"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 🏠 OVERVIEW TAB
# ---------------------------------------------------
with tab_overview:
    # Top cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Records</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(data):,}</div>', unsafe_allow_html=True)
        st.markdown('<span class="metric-badge">Hourly observations</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        units = data["Unit"].nunique()
        st.markdown('<div class="metric-title">Parking lots</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{units}</div>', unsafe_allow_html=True)
        st.markdown('<span class="metric-badge">Different units in Banff</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
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

    with c4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Models</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">XGBoost & LGBM</div>', unsafe_allow_html=True)
        st.markdown('<span class="metric-badge">Reg: occupancy • Cls: near-full</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Quick explanation row (short)
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown("#### How this app helps")
        st.markdown(
            "- Forecasts **hourly occupancy** per parking lot.\n"
            "- Estimates **risk of near capacity (> 90%)**.\n"
            "- Shows **which factors drive demand** using ML + XAI.\n"
            "- Built on: time-of-day, weekday/weekend, weather, and past occupancy trends."
        )

    with col_right:
        st.markdown("#### Typical daily pattern (example)")
        unit_example = data["Unit"].value_counts().index[0]
        df_ex = data[data["Unit"] == unit_example].copy()
        if "Timestamp" in df_ex.columns:
            date_ex = df_ex["Timestamp"].dt.date.mode()[0]
            day_df = df_ex[df_ex["Timestamp"].dt.date == date_ex]
            if not day_df.empty:
                fig, ax = plt.subplots()
                ax.plot(day_df["Hour"], day_df["Percent_Occupancy"] * 100, marker="o")
                ax.set_xlabel("Hour")
                ax.set_ylabel("Occupancy (%)")
                ax.set_title(f"{unit_example} on {date_ex}")
                ax.set_xticks(range(0, 24, 2))
                ax.grid(True, linestyle="--", linewidth=0.4)
                st.pyplot(fig)
            else:
                st.info("No data for sample pattern.")
        else:
            st.info("Timestamp column missing in data.")

# ---------------------------------------------------
# 🚗 FORECAST TAB – MAIN PREDICTION UI
# ---------------------------------------------------
with tab_forecast:
    st.markdown("### Hourly parking demand forecast")
    st.markdown(
        "Select a parking lot, date, and time. The app predicts occupancy and near-full risk."
    )

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
            "Time (hourly)",
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

            run = st.button("🚗 Run forecast for this hour")

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
                            '<span class="metric-badge">Model: classifier</span>',
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

                    # Optional: compare with actual historic value
                    if all(col in row.index for col in ["Occupancy", "Capacity", "Percent_Occupancy", "Is_Full"]):
                        st.markdown("#### Historical value (same hour, same lot)")
                        st.markdown(
                            f"- **Actual:** {row['Occupancy']} / {int(row['Capacity'])} "
                            f"({row['Percent_Occupancy']*100:.1f}%)  \n"
                            f"- **Actual Is_Full label:** {row['Is_Full']}"
                        )

                except Exception as e:
                    st.error(f"Prediction error: {e}")

            # Daily curve (hour-by-hour forecast for selected day)
            st.markdown("### Hour-by-hour forecast for this day")
            day_df = filtered.copy()
            if not day_df.empty:
                preds = []
                for _, r in day_df.iterrows():
                    try:
                        yp, _, _ = make_predictions(
                            r, reg_model, cls_model, scaler, feature_list
                        )
                        cap = r.get("Capacity", 0)
                        if cap > 0:
                            preds.append(min(max(yp / cap, 0), 1))
                        else:
                            preds.append(0.0)
                    except Exception:
                        preds.append(np.nan)

                fig2, ax2 = plt.subplots()
                ax2.plot(day_df["Hour"], np.array(preds) * 100, marker="o")
                ax2.set_xlabel("Hour")
                ax2.set_ylabel("Predicted occupancy (%)")
                ax2.set_title(f"Forecast for {pred_unit} on {pred_date}")
                ax2.set_xticks(range(0, 24, 2))
                ax2.grid(True, linestyle="--", linewidth=0.4)
                st.pyplot(fig2)
            else:
                st.info("No rows for this day to plot.")

# ---------------------------------------------------
# 🧠 XAI INSIGHTS TAB – FEATURE IMPORTANCE + SHAP
# ---------------------------------------------------
with tab_xai:
    st.markdown("### Explainable AI: what drives the predictions?")
    st.markdown(
        "This view shows **global drivers** (feature importance, SHAP summary) "
        "and **local explanation** for a single prediction."
    )

    # GLOBAL – SHAP summary for regression model
    shap_values, X_shap = compute_shap_global(data, feature_list, reg_model, scaler)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Global importance (model built-in)")
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
                ax1.set_title("Top features – regression model")
                ax1.invert_yaxis()
                st.pyplot(fig1)
            else:
                st.info("Regression model does not expose feature_importances_.")
        except Exception as e:
            st.error(f"Could not plot feature importance: {e}")

    with col_right:
        st.markdown("#### Global SHAP summary (regression)")
        if shap_values is None or X_shap is None:
            st.info("SHAP library not available or SHAP could not be computed.")
        else:
            try:
                import shap
                fig, ax = plt.subplots(figsize=(5, 5))
                shap.summary_plot(shap_values, X_shap, show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not render SHAP summary plot: {e}")

    st.markdown("---")
    st.markdown("### Local explanation for one prediction")

    # Select any row from data for explanation
    idx = st.slider(
        "Choose an hourly record to explain",
        min_value=0,
        max_value=len(data) - 1,
        value=0,
        step=1,
    )
    row_local = data.iloc[idx]

    st.markdown(
        f"- **Unit:** {row_local['Unit']}  \n"
        f"- **Timestamp:** {row_local['Timestamp']}  \n"
        f"- **Observed occupancy:** "
        f"{row_local.get('Occupancy', 'N/A')} / {int(row_local.get('Capacity', 0))}"
    )

    if st.button("Explain this record", key="explain_button"):
        # Make prediction for this row
        try:
            y_pred_l, full_prob_l, _ = make_predictions(
                row_local, reg_model, cls_model, scaler, feature_list
            )
        except Exception as e:
            st.error(f"Could not predict for local explanation: {e}")
            y_pred_l, full_prob_l = None, None

        if shap_values is None or X_shap is None:
            st.info("SHAP global sample not available; local explanation disabled.")
        else:
            try:
                import shap
                # Rebuild features for this single row
                X_local = build_feature_vector(row_local, feature_list)
                X_local_scaled = scaler.transform(X_local)

                explainer_local = shap.TreeExplainer(reg_model)
                shap_local = explainer_local.shap_values(X_local_scaled)

                st.markdown("#### SHAP contribution (top 10 features)")
                fig_loc, ax_loc = plt.subplots(figsize=(5, 5))
                shap.plots.bar(
                    shap.Explanation(values=shap_local[0],
                                     base_values=explainer_local.expected_value,
                                     data=X_local.iloc[0],
                                     feature_names=feature_list),
                    max_display=10,
                    show=False
                )
                st.pyplot(fig_loc)
            except Exception as e:
                st.error(f"Could not compute local SHAP explanation: {e}")

# ---------------------------------------------------
# 📊 DATA TAB – CONTEXT & EXPLORER
# ---------------------------------------------------
with tab_data:
    st.markdown("### Data explorer & daily patterns")

    # Metrics again for context
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

    # Pattern + table
    left, right = st.columns([1.6, 1])

    with left:
        st.markdown("#### Daily pattern by lot")

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
        st.markdown("#### Sample data")
        cols_to_show = [
            c for c in ["Timestamp", "Unit", "Occupancy", "Capacity", "Percent_Occupancy"]
            if c in data.columns
        ]
        st.dataframe(
            data[cols_to_show]
            .head(20)
            .reset_index(drop=True)
        )
