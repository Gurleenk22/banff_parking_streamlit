import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import time
from pathlib import Path
from typing import List, Dict

# ---------------------------------------------------
# PAGE CONFIG + PASTEL GLASS STYLING
# ---------------------------------------------------
st.set_page_config(
    page_title="Path Finders – Banff Smart Parking",
    layout="wide"
)

st.markdown(
    """
    <style>
        /* Pastel gradient background */
        .main {
            background: linear-gradient(135deg, #e0f4ff, #fce4ff, #fff5e6);
        }

        /* Remove default padding a bit */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }

        .app-header {
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            text-align: center;
            color: #334155;
            margin-bottom: 0.3rem;
        }

        .app-subtitle {
            text-align: center;
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 2.0rem;
        }

        /* Glassmorphism card */
        .glass-card {
            padding: 1.4rem 1.7rem;
            border-radius: 1.4rem;
            background: rgba(255, 255, 255, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.7);
            box-shadow: 0 18px 50px rgba(148, 163, 184, 0.5);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }

        .metric-title {
            font-size: 0.78rem;
            text-transform: uppercase;
            color: #64748b;
            letter-spacing: 0.12em;
        }

        .metric-value {
            font-size: 1.6rem;
            font-weight: 650;
            margin-top: 0.3rem;
            color: #0f172a;
        }

        .metric-badge {
            display: inline-block;
            padding: 0.18rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            background: rgba(56, 189, 248, 0.17);
            color: #0284c7;
            margin-top: 0.4rem;
        }

        /* Buttons – pill style */
        .stButton > button {
            border-radius: 999px;
            padding: 0.5rem 1.6rem;
            border: none;
            font-weight: 600;
            font-size: 0.95rem;
            background: linear-gradient(135deg, #a5b4fc, #f9a8d4);
            color: #0f172a;
            box-shadow: 0 10px 30px rgba(148, 163, 184, 0.7);
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 15px 35px rgba(148, 163, 184, 0.9);
        }

        /* Small page chips on top of inner pages */
        .page-chip {
            display: inline-block;
            padding: 0.25rem 0.8rem;
            border-radius: 999px;
            font-size: 0.75rem;
            background: rgba(148, 163, 184, 0.24);
            color: #475569;
            margin-bottom: 0.6rem;
        }

        .page-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.3rem;
        }

        .page-subtitle {
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 1.2rem;
        }

        .stDataFrame {
            border-radius: 1rem;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.5);
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
        return "Low", "Easy parking"
    elif percent_occupancy < 0.75:
        return "Medium", "Getting busy"
    else:
        return "High", "Very crowded"


@st.cache_resource(show_spinner=True)
def compute_shap_global(data: pd.DataFrame,
                        feature_list: List[str],
                        reg_model,
                        scaler,
                        sample_size: int = 200):
    """Compute SHAP values on a random sample for global explanation."""
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


# ---------------------------------------------------
# LOAD DATA & MODELS
# ---------------------------------------------------
data = load_data()
reg_model, cls_model, scaler, feature_list = load_models_and_features()

# ---------------------------------------------------
# SIMPLE PAGE ROUTING (HOME + INNER PAGES)
# ---------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "home"


def goto(page_name: str):
    st.session_state["page"] = page_name


# ---------------------------------------------------
# HOME / LANDING PAGE – PATH FINDERS ONLY
# ---------------------------------------------------
if st.session_state["page"] == "home":
    # Big centered title + subtitle
    st.markdown('<div class="app-header">PATH FINDERS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        'Banff Smart Parking – ML & Explainable AI Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    # Centered glass card with navigation buttons
    col_center = st.columns([0.15, 0.7, 0.15])[1]
    with col_center:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        # Quick stats row
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="metric-title">Records</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{len(data):,}</div>',
                unsafe_allow_html=True
            )
        with c2:
            units = data["Unit"].nunique()
            st.markdown('<div class="metric-title">Parking lots</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{units}</div>',
                unsafe_allow_html=True
            )
        with c3:
            if "Timestamp" in data.columns:
                min_date = data["Timestamp"].min().strftime("%Y-%m-%d")
                max_date = data["Timestamp"].max().strftime("%Y-%m-%d")
            else:
                min_date = max_date = "-"
            st.markdown('<div class="metric-title">Data range</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value" style="font-size:1rem;">{min_date} → {max_date}</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        st.markdown(
            "<p style='text-align:center; color:#6b7280; font-size:0.9rem;'>"
            "Choose what you want to explore:"
            "</p>",
            unsafe_allow_html=True
        )

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("🚗 Forecast Dashboard"):
                goto("forecast")
        with b2:
            if st.button("🧠 XAI Insights"):
                goto("xai")
        with b3:
            if st.button("📊 Data Explorer"):
                goto("data")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# INNER PAGE: FORECAST DASHBOARD
# ---------------------------------------------------
elif st.session_state["page"] == "forecast":
    st.markdown('<span class="page-chip">Forecast</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Hourly Parking Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Select lot, date, and time to predict occupancy and near-full risk.</div>',
        unsafe_allow_html=True
    )
    st.button("← Back to Path Finders", on_click=lambda: goto("home"))

    # Glass container
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        pred_unit = st.selectbox(
            "Parking lot",
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
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        selected_hour = pred_time.hour
        filtered = unit_df[unit_df["Timestamp"].dt.date == pred_date]
        row_match = filtered[filtered["Hour"] == selected_hour]

        if row_match.empty:
            st.info("No exact hour found for this date. Try a different time.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            row = row_match.iloc[0]

            if all(col in row.index for col in ["Max Temp (°C)", "Min Temp (°C)", "Total Precip (mm)", "Spd of Max Gust (km/h)"]):
                weather_str = (
                    f"{row['Max Temp (°C)']}°C / {row['Min Temp (°C)']}°C • "
                    f"Precip: {row['Total Precip (mm)']} mm • "
                    f"Wind gust: {row['Spd of Max Gust (km/h)']} km/h"
                )
            else:
                weather_str = "Weather not available"

            st.markdown(
                f"**Scenario:** {pred_unit} • {pred_date} • {selected_hour:02d}:00 • {weather_str}"
            )

            run = st.button("Run forecast", key="run_forecast")

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
                    near_capacity_flag = pred_percent >= 0.90

                    m1, m2, m3, m4 = st.columns(4)

                    with m1:
                        st.markdown('<div class="metric-title">Predicted occupancy</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">{y_pred:.0f} / {int(capacity)}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<span class="metric-badge">{pred_percent*100:.1f}% full</span>',
                            unsafe_allow_html=True
                        )

                    with m2:
                        st.markdown('<div class="metric-title">Near-full probability</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">{full_prob*100:.1f}%</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            '<span class="metric-badge">Model: classifier</span>',
                            unsafe_allow_html=True
                        )

                    with m3:
                        st.markdown('<div class="metric-title">Congestion level</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="metric-value">{level}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<span class="metric-badge">{msg}</span>',
                            unsafe_allow_html=True
                        )

                    with m4:
                        st.markdown('<div class="metric-title">Near capacity? (&gt; 90%)</div>', unsafe_allow_html=True)
                        ans = "YES" if near_capacity_flag else "NO"
                        col = "#16a34a" if not near_capacity_flag else "#fb923c"
                        st.markdown(
                            f'<div class="metric-value" style="color:{col};">{ans}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            '<span class="metric-badge">Threshold at 90%</span>',
                            unsafe_allow_html=True
                        )

                except Exception as e:
                    st.error(f"Prediction error: {e}")

            st.markdown("---")

            # Hour-by-hour forecast curve for the same day
            st.markdown("**Hourly forecast for selected day**")
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
                ax2.set_title(f"Forecast – {pred_unit} on {pred_date}")
                ax2.set_xticks(range(0, 24, 2))
                ax2.grid(True, linestyle="--", linewidth=0.4)
                st.pyplot(fig2)
            else:
                st.info("No rows for this day to plot.")

            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# INNER PAGE: XAI INSIGHTS
# ---------------------------------------------------
elif st.session_state["page"] == "xai":
    st.markdown('<span class="page-chip">XAI</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Explainable AI Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">See which features drive the model and explain single predictions.</div>',
        unsafe_allow_html=True
    )
    st.button("← Back to Path Finders", on_click=lambda: goto("home"))

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    shap_values, X_shap = compute_shap_global(data, feature_list, reg_model, scaler)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Global feature importance (regression model)**")
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
                ax1.invert_yaxis()
                st.pyplot(fig1)
            else:
                st.info("Regression model does not expose feature_importances_.")
        except Exception as e:
            st.error(f"Could not plot feature importance: {e}")

    with col_right:
        st.markdown("**Global SHAP summary (regression)**")
        if shap_values is None or X_shap is None:
            st.info("SHAP not available or could not be computed.")
        else:
            try:
                import shap
                fig, ax = plt.subplots(figsize=(5, 5))
                shap.summary_plot(shap_values, X_shap, show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not render SHAP summary plot: {e}")

    st.markdown("---")
    st.markdown("**Local explanation – one hourly record**")

    idx = st.slider(
        "Pick a record index",
        min_value=0,
        max_value=len(data) - 1,
        value=0,
        step=1,
    )
    row_local = data.iloc[idx]

    st.markdown(
        f"- Unit: **{row_local['Unit']}**  \n"
        f"- Time: **{row_local['Timestamp']}**"
    )

    if st.button("Explain this prediction", key="explain_button"):
        if shap_values is None or X_shap is None:
            st.info("SHAP global sample not available; local explanation disabled.")
        else:
            try:
                import shap
                X_local = build_feature_vector(row_local, feature_list)
                X_local_scaled = scaler.transform(X_local)

                explainer_local = shap.TreeExplainer(reg_model)
                shap_local = explainer_local.shap_values(X_local_scaled)

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

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# INNER PAGE: DATA EXPLORER
# ---------------------------------------------------
elif st.session_state["page"] == "data":
    st.markdown('<span class="page-chip">Data</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">View raw hourly records and daily patterns for each lot.</div>',
        unsafe_allow_html=True
    )
    st.button("← Back to Path Finders", on_click=lambda: goto("home"))

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-title">Records</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{len(data):,}</div>',
            unsafe_allow_html=True
        )
    with c2:
        units = data["Unit"].nunique()
        st.markdown('<div class="metric-title">Parking lots</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{units}</div>',
            unsafe_allow_html=True
        )
    with c3:
        if "Timestamp" in data.columns:
            min_date = data["Timestamp"].min().strftime("%Y-%m-%d")
            max_date = data["Timestamp"].max().strftime("%Y-%m-%d")
        else:
            min_date = max_date = "-"
        st.markdown('<div class="metric-title">Data range</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value" style="font-size:1rem;">{min_date} → {max_date}</div>',
            unsafe_allow_html=True
        )
    with c4:
        st.markdown('<div class="metric-title">Targets</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-value" style="font-size:1.1rem;">Occupancy / Is_Full</div>',
            unsafe_allow_html=True
        )

    left, right = st.columns([1.6, 1])

    with left:
        st.markdown("**Daily pattern by lot**")
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
        st.markdown("**Sample records**")
        cols_to_show = [
            c for c in ["Timestamp", "Unit", "Occupancy", "Capacity", "Percent_Occupancy"]
            if c in data.columns
        ]
        st.dataframe(
            data[cols_to_show]
            .head(25)
            .reset_index(drop=True)
        )

    st.markdown('</div>', unsafe_allow_html=True)
