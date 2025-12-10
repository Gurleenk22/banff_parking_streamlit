import os
from datetime import date, time, datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import streamlit as st
import streamlit.components.v1 as components

from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

# ---------------------------------------------------
# BASIC PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Banff Parking – ML & XAI Dashboard",
    layout="wide",
)

# 🔗 Put your real Power BI public embed URL here (if you have one)
POWERBI_EMBED_URL = ""  # e.g. "https://app.powerbi.com/view?r=..."


# ---------------------------------------------------
# HELPER – SAFE OPENAI CLIENT
# ---------------------------------------------------
def get_openai_client():
    """
    Try to create an OpenAI client.
    If no API key is configured, return None so the app does NOT crash.
    """
    try:
        client = OpenAI()  # expects OPENAI_API_KEY env / Streamlit secrets
        # simple test access to trigger error early if misconfigured
        _ = os.getenv("OPENAI_API_KEY", None)
        if not _:
            return None
        return client
    except Exception:
        return None


# ---------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# ---------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load trained models, scaler, feature list, and XAI test data.
    Uses the exact filenames that are in your repository.
    """
    # ---- Models and artefacts ----
    reg = joblib.load("banff_best_xgb_reg.pkl")      # tuned XGBoost regressor
    cls = joblib.load("banff_best_lgbm_cls.pkl")     # tuned LightGBM classifier
    scaler = joblib.load("banff_scaler.pkl")         # StandardScaler
    features = joblib.load("banff_features.pkl")     # list of feature names

    # ---- XAI / evaluation data ----
    X_test_scaled = np.load("X_test_scaled.npy")
    y_reg_test = np.load("y_reg_test.npy")

    return reg, cls, scaler, features, X_test_scaled, y_reg_test


# try to load; if something is missing we show a clean error later
try:
    best_reg, best_cls, SCALER, FEATURES, X_TEST_SCALED, Y_REG_TEST = load_models_and_data()
    MODELS_OK = True
except Exception as e:
    MODELS_OK = False
    LOAD_ERROR = e


# ---------------------------------------------------
# RAG: LOAD KNOWLEDGE + BUILD VECTORIZER
# ---------------------------------------------------
@st.cache_resource
def load_rag_knowledge():
    """
    Loads banff_knowledge.txt and builds TF-IDF vectors.
    Each non-empty line is treated as a small document.
    """
    knowledge_path = "banff_knowledge.txt"

    if not os.path.exists(knowledge_path):
        docs = [
            "This is Gurleen's Banff parking assistant. The banff_knowledge.txt "
            "file is missing, so answers are based only on general parking logic."
        ]
    else:
        with open(knowledge_path, "r", encoding="utf-8") as f:
            docs = [line.strip() for line in f.readlines() if line.strip()]

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_embeddings = vectorizer.fit_transform(docs)

    return docs, vectorizer, doc_embeddings


def retrieve_context(query, docs, vectorizer, doc_embeddings, k=5):
    """Returns top-k most relevant lines from the knowledge base."""
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_embeddings).flatten()
    top_idx = sims.argsort()[::-1][:k]
    selected = [docs[i] for i in top_idx if sims[i] > 0.0]

    if not selected:
        return "No strong matches in the knowledge base. Answer based on general parking logic."

    return "\n".join(selected)


def generate_chat_answer(user_question, chat_history):
    """
    Uses RAG + OpenAI (if available) to answer user questions.
    If OpenAI is not configured, falls back to returning context only.
    """
    docs, vectorizer, doc_embeddings = load_rag_knowledge()
    context = retrieve_context(user_question, docs, vectorizer, doc_embeddings, k=5)

    client = get_openai_client()

    # If no client (no API key), fall back gracefully
    if client is None:
        return (
            "I don't have access to the OpenAI API in this deployment, "
            "so I can't generate a full natural-language answer.\n\n"
            "Here is the most relevant information from the project notes:\n\n"
            f"{context}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly project assistant helping Gurleen explain a Banff "
                "parking analytics project. Speak clearly and simply, as if you are "
                "presenting to classmates and instructors who are not data scientists. "
                "Use the provided 'Context' from the project notes as your main source "
                "of truth. If the context does not clearly contain the answer, say that "
                "openly and give a short, reasonable guess based on typical parking "
                "behaviour."
            ),
        },
        {
            "role": "system",
            "content": f"Context from project notes:\n{context}",
        },
    ]

    # keep last few turns of history
    for h in chat_history[-4:]:
        messages.append(
            {
                "role": h["role"],
                "content": h["content"],
            }
        )

    messages.append({"role": "user", "content": user_question})

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Friendly fallback when quota is exhausted or API not reachable
        return (
            "I couldn’t contact the language-model service right now "
            "(this usually means the OpenAI API quota or free credits are used up "
            "for this key).\n\n"
            "Here is the most relevant information I can give based only on "
            "the project notes:\n\n"
            f"{context}"
        )


# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.title("Banff Parking Dashboard")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "📘 Project Guide",
        "🎯 Make Prediction",
        "📊 Lot Status Overview",
        "🔍 XAI – Explainable AI",
        "💬 Chat Assistant (RAG)",
    ],
)

if not MODELS_OK and page != "💬 Chat Assistant (RAG)":
    st.error(
        "⚠️ Model files could not be loaded.\n\n"
        f"Details: `{LOAD_ERROR}`\n\n"
        "Check that all `.pkl` and `.npy` files are present in the repo and "
        "match the filenames in `load_models_and_data()`."
    )


# ---------------------------------------------------
# SMALL UTILITY: BUILD BASE FEATURE TEMPLATE
# ---------------------------------------------------
def build_base_feature_template():
    """Return a dict of all features initialised to 0."""
    return {f: 0 for f in FEATURES}


def set_time_weather_features(base, month, day_of_week, hour, max_temp, total_precip, wind_gust):
    """Fill common time + weather features into the base feature dict."""
    if "Month" in base:
        base["Month"] = month
    if "DayOfWeek" in base:
        base["DayOfWeek"] = day_of_week
    if "Hour" in base:
        base["Hour"] = hour
    if "IsWeekend" in base:
        base["IsWeekend"] = 1 if day_of_week in [5, 6] else 0
    if "Max Temp (°C)" in base:
        base["Max Temp (°C)"] = max_temp
    if "Total Precip (mm)" in base:
        base["Total Precip (mm)"] = total_precip
    if "Spd of Max Gust (km/h)" in base:
        base["Spd of Max Gust (km/h)"] = wind_gust
    return base


# ---------------------------------------------------
# PAGE 1 – DASHBOARD (NEW)
# ---------------------------------------------------
if page == "🏠 Dashboard":
    st.title("🏠 Banff Parking – Interactive Dashboard")

    st.markdown(
        """
        This dashboard gives a **quick snapshot** of parking pressure in Banff
        using your trained machine learning models.
        Use the calendar and time selector to explore a *typical hour* in the
        tourist season, then see which lots are most at risk of being full.
        """
    )

    if MODELS_OK:
        # --- Calendar & time controls ---
        c1, c2 = st.columns(2)
        with c1:
            selected_date = st.date_input(
                "Choose date (tourist season May–Sept 2025)",
                value=date(2025, 7, 15),
            )
        with c2:
            selected_time = st.time_input(
                "Choose hour of day",
                value=time(14, 0),
            )

        # Derive features from date/time
        month = selected_date.month
        day_of_week = selected_date.weekday()  # 0 = Monday
        hour = selected_time.hour

        # Simple default weather for dashboard (can be refined)
        max_temp = st.slider("Assumed Max Temperature (°C)", -10.0, 35.0, 22.0)
        total_precip = st.slider("Assumed Total Precipitation (mm)", 0.0, 30.0, 0.5)
        wind_gust = st.slider("Assumed Speed of Max Gust (km/h)", 0.0, 80.0, 15.0)

        # --- Model performance KPIs ---
        y_pred_dashboard = best_reg.predict(X_TEST_SCALED)
        r2 = r2_score(Y_REG_TEST, y_pred_dashboard)
        mae = mean_absolute_error(Y_REG_TEST, y_pred_dashboard)
        rmse = np.sqrt(mean_squared_error(Y_REG_TEST, y_pred_dashboard))

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Regression R² on test set", f"{r2:.3f}")
        with k2:
            st.metric("MAE (vehicles)", f"{mae:.2f}")
        with k3:
            st.metric("RMSE (vehicles)", f"{rmse:.2f}")

        st.markdown("---")

        # --- Compute lot risk for the chosen date/time (similar to Lot Status) ---
        lot_features = [f for f in FEATURES if f.startswith("Unit_")]
        lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]
        if lot_features:
            pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
            lot_features, lot_display_names = zip(*pairs)
            lot_features, lot_display_names = list(lot_features), list(lot_display_names)

            base = build_base_feature_template()
            base = set_time_weather_features(
                base, month, day_of_week, hour, max_temp, total_precip, wind_gust
            )

            rows = []
            for lf, lname in zip(lot_features, lot_display_names):
                lot_input = base.copy()
                if lf in lot_input:
                    lot_input[lf] = 1

                x_vec = np.array([lot_input[f] for f in FEATURES]).reshape(1, -1)
                x_scaled = SCALER.transform(x_vec)

                occ_pred = best_reg.predict(x_scaled)[0]
                full_prob = best_cls.predict_proba(x_scaled)[0, 1]

                rows.append(
                    {
                        "Lot": lname,
                        "Predicted occupancy": occ_pred,
                        "Probability full": full_prob,
                    }
                )

            df_all = pd.DataFrame(rows).sort_values("Probability full", ascending=False)

            st.subheader("Top 3 lots at risk for the selected hour")

            top3 = df_all.head(3)
            c1, c2, c3 = st.columns(3)
            cols = [c1, c2, c3]
            for col, (_, row) in zip(cols, top3.iterrows()):
                status = "High" if row["Probability full"] > 0.7 else (
                    "Medium" if row["Probability full"] > 0.4 else "Low"
                )
                with col:
                    st.metric(
                        row["Lot"],
                        f"{row['Probability full']:.1%}",
                        help=f"Predicted occupancy ≈ {row['Predicted occupancy']:.1f} vehicles\nRisk level: {status}",
                    )

            st.markdown("### All lots – risk table")
            st.dataframe(
                df_all.assign(
                    Status=np.where(
                        df_all["Probability full"] > 0.7,
                        "🟥 High risk full",
                        np.where(
                            df_all["Probability full"] > 0.4,
                            "🟧 Busy",
                            "🟩 Comfortable",
                        ),
                    )
                ).style.format(
                    {
                        "Predicted occupancy": "{:.2f}",
                        "Probability full": "{:.1%}",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.warning(
                "No features starting with `Unit_` were found in the FEATURES list, "
                "so the dashboard cannot compute lot-level risk."
            )

    st.markdown("---")

    # --- Optional Power BI embed ---
    st.subheader("Power BI – Parking Overview (optional)")
    if POWERBI_EMBED_URL:
        components.iframe(POWERBI_EMBED_URL, height=480, scrolling=True)
    else:
        st.info(
            "Add your Power BI public embed URL to `POWERBI_EMBED_URL` at the top of "
            "`streamlit_app.py` to display your interactive Power BI dashboard here."
        )


# ---------------------------------------------------
# PAGE 2 – PROJECT GUIDE (EXPLAINS OTHER PAGES)
# ---------------------------------------------------
if page == "📘 Project Guide":
    st.title("📘 Project Guide – How This App Works")

    st.markdown(
        """
        This app turns your **Banff parking ML project** into an interactive tool
        for city staff and non-technical stakeholders.

        Below is a simple explanation of what each page does.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            ### 🏠 Dashboard  
            - Choose a **date & time** with the calendar  
            - Assumed **weather conditions** can be adjusted  
            - Shows overall **model accuracy (R², MAE, RMSE)**  
            - Highlights the **top 3 lots most likely to be full**  
            - Displays a **risk table** for all lots  

            Use this when a manager asks:  
            > *“Which lots should we worry about for Saturday at 2 PM?”*
            """
        )

        st.markdown(
            """
            ### 🎯 Make Prediction  
            - Focuses on **one parking lot at a time**  
            - You can pick:
              - Lot  
              - Month, day of week, hour  
              - Temperature, precipitation, wind  
            - The model returns:
              - **Predicted occupancy** (regression)  
              - **Probability of being near full** (classification)  

            This page is ideal for **what-if scenarios**, such as trying different
            weather or time-of-day combinations.
            """
        )

    with c2:
        st.markdown(
            """
            ### 📊 Lot Status Overview  
            - Uses a **single hour & weather scenario**  
            - Compares **all lots at once**  
            - Gives a simple status:
              - 🟥 High risk full  
              - 🟧 Busy  
              - 🟩 Comfortable  

            This supports **operational decisions**, such as where to direct
            vehicles or when to close a lot.
            """
        )

        st.markdown(
            """
            ### 🔍 XAI – Explainable AI  
            Here you show **why** the models behave the way they do:

            - **SHAP summary plot** – which features push predictions up or down  
            - **SHAP bar plot** – overall feature importance  
            - **Partial Dependence Plots** – how Hour, Month, Temperature affect demand  
            - **Residual plot** – checks if predictions are unbiased  

            This page is especially useful for your **presentation** and **report**,
            because it connects the ML model to real-world behaviour.
            """
        )

    st.markdown("---")

    st.markdown(
        """
        ### 💬 Chat Assistant (RAG)

        - Uses a small **knowledge file (`banff_knowledge.txt`)**  
        - Retrieves the most relevant lines for a question  
        - If an OpenAI key is available, it generates a **friendly answer**  
        - If not, it still shows the **relevant context** from your notes  

        You can update `banff_knowledge.txt` with:
        - EDA findings  
        - Feature engineering decisions  
        - Model comparison results  
        - Policy recommendations  

        This turns your project into an **interactive explanation tool**.
        """
    )


# ---------------------------------------------------
# PAGE 3 – MAKE PREDICTION
# ---------------------------------------------------
if page == "🎯 Make Prediction" and MODELS_OK:
    st.title("🎯 Interactive Parking Demand Prediction")

    st.markdown(
        """
        Use this page to explore *what-if* scenarios for a single Banff parking lot.

        1. Select a **parking lot**  
        2. Choose a **scenario** (or adjust the sliders)  
        3. See:
           - Predicted **occupancy** for the selected hour  
           - **Probability** the lot is near full  
        """
    )

    # Find lot indicator features (one-hot encoded units)
    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

    # Sort lot list alphabetically
    if lot_features:
        lot_pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
        lot_features, lot_display_names = zip(*lot_pairs)
        lot_features = list(lot_features)
        lot_display_names = list(lot_display_names)

    if not lot_features:
        st.warning(
            "No parking-lot indicator features (starting with 'Unit_') were "
            "found in FEATURES. Lot selection is disabled; generic features only."
        )

    # Scenario presets
    scenario_options = {
        "Custom (use sliders below)": None,
        "Sunny Weekend Midday": {
            "month": 7,
            "dow": 5,
            "hour": 13,
            "max_temp": 24.0,
            "precip": 0.0,
            "gust": 10.0,
        },
        "Rainy Weekday Afternoon": {
            "month": 6,
            "dow": 2,
            "hour": 16,
            "max_temp": 15.0,
            "precip": 5.0,
            "gust": 20.0,
        },
        "Cold Morning (Shoulder Season)": {
            "month": 5,
            "dow": 1,
            "hour": 9,
            "max_temp": 5.0,
            "precip": 0.0,
            "gust": 15.0,
        },
        "Warm Evening (Busy Day)": {
            "month": 8,
            "dow": 6,
            "hour": 19,
            "max_temp": 22.0,
            "precip": 0.0,
            "gust": 8.0,
        },
    }

    st.subheader("Step 1 – Choose Lot & Scenario")

    col_lot, col_scenario = st.columns([1.2, 1])

    with col_lot:
        if lot_features:
            selected_lot_label = st.selectbox(
                "Select parking lot",
                lot_display_names,
                index=0,
            )
            selected_lot_feature = lot_features[lot_display_names.index(selected_lot_label)]
        else:
            selected_lot_label = None
            selected_lot_feature = None

    with col_scenario:
        selected_scenario = st.selectbox(
            "Scenario",
            list(scenario_options.keys()),
            index=1,
        )

    # Default slider values – overridden by scenario, if chosen
    default_vals = {
        "month": 7,
        "dow": 5,
        "hour": 13,
        "max_temp": 22.0,
        "precip": 0.5,
        "gust": 12.0,
    }

    if scenario_options[selected_scenario] is not None:
        default_vals.update(scenario_options[selected_scenario])

    st.subheader("Step 2 – Adjust Conditions (if needed)")

    col1, col2 = st.columns(2)

    with col1:
        month = st.slider(
            "Month (1 = Jan, 12 = Dec)", 1, 12, int(default_vals["month"])
        )
        day_of_week = st.slider(
            "Day of Week (0 = Monday, 6 = Sunday)", 0, 6, int(default_vals["dow"])
        )
        hour = st.slider("Hour of Day (0–23)", 0, 23, int(default_vals["hour"]))

    with col2:
        max_temp = st.slider(
            "Max Temperature (°C)", -20.0, 40.0, float(default_vals["max_temp"])
        )
        total_precip = st.slider(
            "Total Precipitation (mm)", 0.0, 30.0, float(default_vals["precip"])
        )
        wind_gust = st.slider(
            "Speed of Max Gust (km/h)", 0.0, 100.0, float(default_vals["gust"])
        )

    st.caption(
        "Lag features (previous-hour occupancy, rolling averages) are set automatically "
        "by the model and are not entered manually here."
    )

    # Build feature dict starting from all zeros
    base_input = build_base_feature_template()
    base_input = set_time_weather_features(
        base_input, month, day_of_week, hour, max_temp, total_precip, wind_gust
    )

    # Lot indicator – one-hot
    if selected_lot_feature is not None and selected_lot_feature in base_input:
        base_input[selected_lot_feature] = 1

    # Vector in the exact training feature order
    x_vec = np.array([base_input[f] for f in FEATURES]).reshape(1, -1)
    x_scaled = SCALER.transform(x_vec)

    if st.button("🔮 Predict for this scenario"):
        occ_pred = best_reg.predict(x_scaled)[0]
        full_prob = best_cls.predict_proba(x_scaled)[0, 1]

        st.subheader("Step 3 – Results for Selected Hour")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(
                "Predicted occupancy (model units)",
                f"{occ_pred:.2f}",
            )
        with col_res2:
            st.metric(
                "Probability lot is near full",
                f"{full_prob:.1%}",
            )

        if full_prob > 0.7:
            st.warning(
                "⚠️ High risk this lot will be full. Consider redirecting drivers "
                "to other parking areas or adjusting signage."
            )
        elif full_prob > 0.4:
            st.info(
                "Moderate risk of heavy usage. Monitoring and dynamic guidance "
                "could be useful."
            )
        else:
            st.success(
                "Low risk of the lot being at full capacity for this hour."
            )


# ---------------------------------------------------
# PAGE 4 – LOT STATUS OVERVIEW (ALL LOTS)
# ---------------------------------------------------
if page == "📊 Lot Status Overview" and MODELS_OK:
    st.title("📊 Lot Status Overview – Which Lots Are Likely Full?")

    st.markdown(
        """
        This page shows, for a selected hour and conditions, the predicted:

        - **Occupancy** for each parking lot  
        - **Probability that the lot is near full**  
        - Simple status: 🟥 High risk, 🟧 Busy, 🟩 Comfortable
        """
    )

    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

    # sort lots alphabetically
    if lot_features:
        lot_pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
        lot_features, lot_display_names = zip(*lot_pairs)
        lot_features = list(lot_features)
        lot_display_names = list(lot_display_names)

    if not lot_features:
        st.error(
            "No parking-lot indicator features (starting with 'Unit_') were "
            "found in FEATURES. This view needs those to work."
        )
    else:
        st.subheader("Step 1 – Choose time & weather")

        col1, col2 = st.columns(2)

        with col1:
            month = st.slider("Month (1 = Jan, 12 = Dec)", 1, 12, 7)
            day_of_week = st.slider(
                "Day of Week (0 = Monday, 6 = Sunday)", 0, 6, 5
            )
            hour = st.slider("Hour of Day", 0, 23, 14)

        with col2:
            max_temp = st.slider("Max Temperature (°C)", -20.0, 40.0, 22.0)
            total_precip = st.slider("Total Precipitation (mm)", 0.0, 30.0, 0.5)
            wind_gust = st.slider("Speed of Max Gust (km/h)", 0.0, 100.0, 12.0)

        st.caption(
            "Lag features (previous-hour occupancy, rolling averages) are set to 0 "
            "for this overview. In a real system they would come from live feeds."
        )

        if st.button("Compute lot status"):
            rows = []

            base_input = build_base_feature_template()
            base_input = set_time_weather_features(
                base_input, month, day_of_week, hour, max_temp, total_precip, wind_gust
            )

            for lot_feat, lot_name in zip(lot_features, lot_display_names):
                lot_input = base_input.copy()
                if lot_feat in lot_input:
                    lot_input[lot_feat] = 1

                x_vec = np.array([lot_input[f] for f in FEATURES]).reshape(1, -1)
                x_scaled = SCALER.transform(x_vec)

                occ_pred = best_reg.predict(x_scaled)[0]
                full_prob = best_cls.predict_proba(x_scaled)[0, 1]

                if full_prob > 0.7:
                    status = "🟥 High risk full"
                elif full_prob > 0.4:
                    status = "🟧 Busy"
                else:
                    status = "🟩 Comfortable"

                rows.append(
                    {
                        "Lot": lot_name,
                        "Predicted occupancy": occ_pred,
                        "Probability full": full_prob,
                        "Status": status,
                    }
                )

            df = pd.DataFrame(rows)
            df = df.sort_values("Lot")

            def lot_status_row_style(row):
                if "High risk" in row["Status"]:
                    return ["background-color: #ffe5e5"] * len(row)
                elif "Busy" in row["Status"]:
                    return ["background-color: #fff4e0"] * len(row)
                else:
                    return ["background-color: #e9f7ef"] * len(row)

            styled_df = (
                df.style.format(
                    {
                        "Predicted occupancy": "{:.2f}",
                        "Probability full": "{:.1%}",
                    }
                ).apply(lot_status_row_style, axis=1)
            )

            st.subheader("Step 2 – Lot status for selected hour")
            st.dataframe(
                styled_df,
                use_container_width=True,
            )

            st.caption(
                "Lots are shown in numeric order (e.g., BANFF02, BANFF03, …). "
                "Row colour shows risk level: red = high risk, orange = busy, "
                "green = comfortable."
            )


# ---------------------------------------------------
# PAGE 5 – XAI (EXPLAINABLE AI)
# ---------------------------------------------------
if page == "🔍 XAI – Explainable AI" and MODELS_OK:
    st.title("🔍 Explainable AI – Understanding the Models")

    st.markdown(
        """
        This page explains **why** the models make their predictions,
        using Explainable AI tools:

        - **SHAP summary plot**: which features contribute most to predictions  
        - **SHAP bar plot**: overall feature importance  
        - **Partial Dependence Plots (PDPs)**: effect of one feature at a time  
        - **Residual plot**: how close predictions are to the true values  
        """
    )

    # ---------- SHAP EXPLANATIONS FOR REGRESSION ----------
    st.subheader("SHAP Summary – Regression Model (Occupancy)")

    try:
        explainer_reg = shap.TreeExplainer(best_reg)
        shap_values_reg = explainer_reg.shap_values(X_TEST_SCALED)

        fig1, _ = plt.subplots()
        shap.summary_plot(
            shap_values_reg,
            X_TEST_SCALED,
            feature_names=FEATURES,
            show=False,
        )
        st.pyplot(fig1)
        st.caption(
            "Each point represents a sample. Colour shows feature value, and position "
            "shows how much that feature pushed the prediction up or down."
        )

        st.subheader("SHAP Feature Importance – Regression")
        fig2, _ = plt.subplots()
        shap.summary_plot(
            shap_values_reg,
            X_TEST_SCALED,
            feature_names=FEATURES,
            plot_type="bar",
            show=False,
        )
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Could not generate SHAP plots: {e}")

    # ---------- PARTIAL DEPENDENCE PLOTS ----------
    st.subheader("Partial Dependence – Key Features")

    pd_feature_names = []
    for name in ["Max Temp (°C)", "Month", "Hour"]:
        if name in FEATURES:
            pd_feature_names.append(name)

    if len(pd_feature_names) > 0:
        feature_indices = [FEATURES.index(f) for f in pd_feature_names]
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        PartialDependenceDisplay.from_estimator(
            best_reg,
            X_TEST_SCALED,
            feature_indices,
            feature_names=FEATURES,
            ax=ax3,
        )
        st.pyplot(fig3)
        st.caption(
            "Partial dependence shows the average effect of each feature on predicted "
            "occupancy while holding other features constant."
        )
    else:
        st.info(
            "Could not find the configured PDP features ('Max Temp (°C)', 'Month', 'Hour') "
            "in the FEATURES list. You may need to adjust the feature names."
        )

    # ---------- RESIDUAL ANALYSIS ----------
    st.subheader("Residual Plot – Regression Model")

    try:
        y_pred = best_reg.predict(X_TEST_SCALED)
        residuals = Y_REG_TEST - y_pred

        fig4, ax4 = plt.subplots()
        ax4.scatter(y_pred, residuals, alpha=0.3)
        ax4.axhline(0, linestyle="--")
        ax4.set_xlabel("Predicted Occupancy")
        ax4.set_ylabel("Residual (Actual - Predicted)")
        st.pyplot(fig4)
        st.caption(
            "Residuals scattered symmetrically around zero suggest that the model "
            "captures the main patterns without strong systematic bias."
        )
    except Exception as e:
        st.error(f"Could not compute residuals: {e}")


# ---------------------------------------------------
# PAGE 6 – CHAT ASSISTANT (RAG)
# ---------------------------------------------------
if page == "💬 Chat Assistant (RAG)":
    st.title("💬 Banff Parking Chat Assistant (RAG)")

    st.markdown(
        """
        Ask questions about parking patterns, busy times, or model behaviour.

        This chatbot uses **RAG (Retrieval-Augmented Generation)**:
        1. It first retrieves relevant lines from your `banff_knowledge.txt` file  
        2. Then it uses an OpenAI model (if available) to answer, grounded in that context  
        """
    )

    # Initialize chat history
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []

    # Show previous messages
    for msg in st.session_state.rag_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask something about Banff parking...")

    if user_input:
        st.session_state.rag_chat_history.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking with project context..."):
                answer = generate_chat_answer(
                    user_input,
                    st.session_state.rag_chat_history,
                )
                st.markdown(answer)

        st.session_state.rag_chat_history.append(
            {"role": "assistant", "content": answer}
        )

    st.caption(
        "Tip: edit `banff_knowledge.txt` in your repo to control what the chatbot knows "
        "about your EDA, feature engineering, and model findings."
    )
