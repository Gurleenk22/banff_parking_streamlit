import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.inspection import PartialDependenceDisplay
import datetime as dt

# Tree-model libs are only needed because they were used when pickling
import xgboost as xgb     # noqa: F401
import lightgbm as lgb    # noqa: F401

# ==== RAG / Chatbot imports ====
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
import streamlit.components.v1 as components

client = OpenAI()  # uses OPENAI_API_KEY from env / Streamlit secrets

# ---------------------------------------------------
# BASIC PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Banff Parking – ML & XAI Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# ---------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """Load trained models, scaler, feature list, and test data."""
    reg = joblib.load("banff_best_xgb_reg.pkl")      # Tuned XGBoost regressor
    cls = joblib.load("banff_best_xgb_cls.pkl")      # Tuned classifier (LightGBM)
    scaler = joblib.load("banff_scaler.pkl")         # Scaler used in training
    features = joblib.load("banff_features.pkl")     # List of feature names

    # Test data for XAI and residual analysis
    X_test_scaled = np.load("X_test_scaled.npy")
    y_reg_test = np.load("y_reg_test.npy")

    return reg, cls, scaler, features, X_test_scaled, y_reg_test


best_xgb_reg, best_cls_model, scaler, FEATURES, X_test_scaled, y_reg_test = load_models_and_data()

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
            "This is Gurleen's Banff parking assistant. The banff_knowledge.txt file is "
            "missing, so answers are based only on general parking logic."
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
    Calls OpenAI with retrieved context + short chat history.
    If the API fails (e.g., insufficient_quota), fall back to
    a simple answer built only from the retrieved context.
    """
    docs, vectorizer, doc_embeddings = load_rag_knowledge()
    context = retrieve_context(user_question, docs, vectorizer, doc_embeddings, k=5)

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
# SMALL HELPERS
# ---------------------------------------------------
def compute_time_features(selected_date: dt.date, hour: int):
    """Return dictionary with all time / cyclic features."""
    dow = selected_date.weekday()                  # 0 = Monday
    day_of_year = selected_date.timetuple().tm_yday
    is_weekend = 1 if dow in [5, 6] else 0

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    return {
        "Month": selected_date.month,
        "DayOfWeek": dow,
        "Hour": hour,
        "IsWeekend": is_weekend,
        "day_of_year": day_of_year,
        "hour": hour,          # lower-case version from feature engineering
        "dow": dow,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "is_weekend": is_weekend,
    }

# Lot feature names (one-hot encoded Units)
LOT_FEATURES = [f for f in FEATURES if f.startswith("Unit_")]
LOT_DISPLAY_NAMES = [lf.replace("Unit_", "").replace("_", " ") for lf in LOT_FEATURES]

if LOT_FEATURES:
    lot_pairs = sorted(zip(LOT_FEATURES, LOT_DISPLAY_NAMES), key=lambda x: x[1])
    LOT_FEATURES, LOT_DISPLAY_NAMES = zip(*lot_pairs)
    LOT_FEATURES = list(LOT_FEATURES)
    LOT_DISPLAY_NAMES = list(LOT_DISPLAY_NAMES)

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.title("Banff Parking Dashboard")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "🧭 App Tour & Logic",
        "🎯 Make Prediction",
        "📊 Lot Status Overview",
        "🔍 XAI – Explainable AI",
        "💬 Chat Assistant (RAG)",
    ]
)

# ===================================================
# PAGE 1 – DASHBOARD (quick overview)
# ===================================================
if page == "🏠 Dashboard":
    st.title("🏠 Banff Parking – Project Dashboard")

    col_top1, col_top2, col_top3 = st.columns(3)

    # Quick summary cards
    with col_top1:
        st.metric("Models in App", "2",
                  help="XGBoost Regressor (Occupancy) + LightGBM Classifier (Is_Full)")
    with col_top2:
        st.metric("Engineered Features", f"{len(FEATURES)}+",
                  help="Includes time, weather, lags, rolling stats, and lot indicators")
    with col_top3:
        st.metric("Test Points", f"{len(y_reg_test)}",
                  help="Size of the held-out test set used for evaluation")

    st.markdown("---")

    # Date + hour selection as a calendar-style widget
    st.subheader("📅 Pick a Day & Hour (Tourist Season 2025)")

    col_date, col_hour = st.columns([1.4, 1])

    with col_date:
        selected_date = st.date_input(
            "Select a date",
            value=dt.date(2025, 7, 15),
            min_value=dt.date(2025, 5, 1),
            max_value=dt.date(2025, 9, 30),
            help="Model was trained on May–September 2025."
        )

    with col_hour:
        selected_hour = st.slider(
            "Select an hour of day (0–23)",
            0, 23, 14
        )

    st.caption(
        "This calendar-style input is used on the prediction pages to convert the date "
        "into Month, DayOfWeek, and cyclic time features."
    )

    # Optional Power BI (or any external) dashboard embed
    st.subheader("📈 Power BI Overview (optional)")

    st.markdown(
        "If you published a Power BI dashboard, paste its **public embed URL** "
        "in the code below. For now this is just a placeholder."
    )
    # 👉 Replace 'YOUR_POWERBI_EMBED_URL' with your actual published Power BI link.
    POWERBI_URL = "https://app.powerbi.com/links/YOUR_POWERBI_EMBED_URL"
    if "YOUR_POWERBI_EMBED_URL" not in POWERBI_URL:
        components.iframe(POWERBI_URL, height=520, scrolling=True)
    else:
        st.info(
            "Power BI not embedded yet. In the code, replace "
            "`YOUR_POWERBI_EMBED_URL` with your own link after publishing the report."
        )

    st.markdown("---")
    st.subheader("What this dashboard answers at a glance")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("✅ **When** parking lots get busy")
        st.write("- Hourly patterns across the season\n- Differences between weekdays/weekends")
    with cols[1]:
        st.markdown("✅ **Where** pressure is highest")
        st.write("- Compare lots at the same hour\n- Flag lots with high risk of being full")
    with cols[2]:
        st.markdown("✅ **Why** the model predicts that")
        st.write("- XAI page shows SHAP plots\n- Partial dependence and residual analysis")

# ===================================================
# PAGE 2 – APP TOUR & LOGIC
# ===================================================
elif page == "🧭 App Tour & Logic":
    st.title("🧭 How This App Works – Step-by-Step Tour")

    st.markdown(
        """
        This page is your **explainer for instructors and clients**.  
        It walks through the logic behind each part of the app and how it connects to the data pipeline.
        """
    )

    with st.expander("1️⃣ Data & Feature Engineering", expanded=True):
        st.markdown(
            """
            - Source: **Hourly Banff parking & weather data (May–September 2025)**  
            - Target for regression: **`Occupancy`** (vehicles in the lot)  
            - Target for classification: **`Is_Full`** (1 if >90% of capacity, else 0)  
            - Key engineered features:
              - **Time features** – Hour, DayOfWeek, Month, weekend flag  
              - **Cyclic encodings** – `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`  
              - **Lags** – occupancy 1, 2, 3, 24, 168 hours ago  
              - **Rolling stats** – rolling mean/std over 3, 7, 30 time windows  
              - **Lot identity** – one-hot encoded `Unit_*` columns
            """
        )

    with st.expander("2️⃣ Modelling approach", expanded=True):
        st.markdown(
            """
            - **Regression model (Occupancy):** tuned **XGBoost Regressor**  
            - **Classification model (Is_Full):** tuned **LightGBM Classifier**  
            - Split: chronological 70% train / 20% validation / 10% test  
            - Metrics:
              - Regression – R², MAE, RMSE  
              - Classification – accuracy, F1, ROC-AUC, precision–recall
            """
        )

    with st.expander("3️⃣ What each app page does", expanded=True):
        st.markdown(
            """
            - **🏠 Dashboard** – high-level summary, calendar selector, optional Power BI view.  
            - **🎯 Make Prediction** – *what-if* analysis for a **single lot**:
              - Choose date + hour + weather + lot  
              - Get predicted occupancy and probability of being full.  
            - **📊 Lot Status Overview** – compare **all lots at once** for one hour:
              - Shows occupancy, full-lot probability, and colour-coded status.  
            - **🔍 XAI – Explainable AI**:
              - SHAP global explanations (which features drive predictions).  
              - Partial dependence plots for Month, Hour, and Temperature.  
              - Residual plots to show how well regression fits the data.  
            - **💬 Chat Assistant (RAG)**:
              - Uses your `banff_knowledge.txt` notes + OpenAI to answer questions.
            """
        )

    with st.expander("4️⃣ How this supports Banff’s decisions", expanded=True):
        st.markdown(
            """
            - **Operations** – identify peak hours and high-risk lots to adjust signage, staffing,
              and dynamic messaging.  
            - **Planning** – use seasonal patterns to redesign parking allocation or pricing.  
            - **Communication with stakeholders** – XAI plots make the model transparent and easier
              to trust for non-technical audiences.
            """
        )

# ===================================================
# PAGE 3 – MAKE PREDICTION
# ===================================================
elif page == "🎯 Make Prediction":
    st.title("🎯 Interactive Parking Demand Prediction – Single Lot")

    st.markdown(
        """
        Use this page to explore *what-if* scenarios for a single Banff parking lot.

        1. Select a **date & hour** (via the calendar)  
        2. Choose a **lot** and **weather conditions**  
        3. See:
           - Predicted **occupancy**  
           - **Probability** that the lot is near full (>90% capacity)  
        """
    )

    if not LOT_FEATURES:
        st.warning(
            "No parking-lot indicator features (starting with 'Unit_') were found. "
            "Lot selection will be disabled."
        )

    # ---- Step 1: Date & lot selection ----
    st.subheader("Step 1 – Choose Date, Hour & Lot")

    col_date, col_hour, col_lot = st.columns([1.2, 0.8, 1.2])

    with col_date:
        sel_date = st.date_input(
            "Date (tourist season 2025)",
            value=dt.date(2025, 7, 15),
            min_value=dt.date(2025, 5, 1),
            max_value=dt.date(2025, 9, 30),
        )

    with col_hour:
        sel_hour = st.slider("Hour of Day (0–23)", 0, 23, 13)

    with col_lot:
        if LOT_FEATURES:
            selected_lot_label = st.selectbox(
                "Parking lot",
                LOT_DISPLAY_NAMES,
                index=0
            )
            selected_lot_feature = LOT_FEATURES[LOT_DISPLAY_NAMES.index(selected_lot_label)]
        else:
            selected_lot_label = None
            selected_lot_feature = None

    # ---- Step 2: Weather scenario ----
    st.subheader("Step 2 – Weather & Scenario")

    scenario_options = {
        "Custom (use sliders below)": None,
        "Sunny Weekend Midday": {"max_temp": 24.0, "precip": 0.0, "gust": 10.0},
        "Rainy Afternoon": {"max_temp": 15.0, "precip": 5.0, "gust": 20.0},
        "Cold Morning": {"max_temp": 5.0, "precip": 0.0, "gust": 15.0},
        "Warm Evening": {"max_temp": 22.0, "precip": 0.0, "gust": 8.0},
    }

    col_scenario, col_spac = st.columns([1, 1])
    with col_scenario:
        selected_scenario = st.selectbox("Scenario", list(scenario_options.keys()), index=1)

    # Default slider values
    default_vals = {"max_temp": 22.0, "precip": 0.5, "gust": 12.0}
    if scenario_options[selected_scenario] is not None:
        default_vals.update(scenario_options[selected_scenario])

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        max_temp = st.slider(
            "Max Temperature (°C)",
            -20.0, 40.0, float(default_vals["max_temp"])
        )
    with col_w2:
        total_precip = st.slider(
            "Total Precipitation (mm)",
            0.0, 30.0, float(default_vals["precip"])
        )
    with col_w3:
        wind_gust = st.slider(
            "Speed of Max Gust (km/h)",
            0.0, 100.0, float(default_vals["gust"])
        )

    # ---- Build feature vector ----
    time_feats = compute_time_features(sel_date, sel_hour)

    # Start from all zeros
    base_input = {f: 0 for f in FEATURES}

    # Set time features if present
    for k, v in time_feats.items():
        if k in base_input:
            base_input[k] = v

    # Weather features (only set if they exist)
    if "Max Temp (°C)" in base_input:
        base_input["Max Temp (°C)"] = max_temp
    if "Min Temp (°C)" in base_input:
        # simple heuristic: min temp = max_temp - 10
        base_input["Min Temp (°C)"] = max_temp - 10
    if "Total Precip (mm)" in base_input:
        base_input["Total Precip (mm)"] = total_precip
    if "Spd of Max Gust (km/h)" in base_input:
        base_input["Spd of Max Gust (km/h)"] = wind_gust

    # Lot indicator – one-hot
    if selected_lot_feature is not None and selected_lot_feature in base_input:
        base_input[selected_lot_feature] = 1

    # Vector in training feature order
    x_vec = np.array([base_input[f] for f in FEATURES]).reshape(1, -1)
    x_scaled = scaler.transform(x_vec)

    if st.button("🔮 Predict for this scenario"):
        occ_pred = best_xgb_reg.predict(x_scaled)[0]
        full_prob = best_cls_model.predict_proba(x_scaled)[0, 1]

        st.subheader("Step 3 – Prediction for Selected Lot")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Predicted occupancy (vehicles)",
                      f"{occ_pred:.2f}")
        with col_res2:
            st.metric("Probability lot is near full",
                      f"{full_prob:.1%}")

        if full_prob > 0.7:
            st.warning(
                "⚠️ **High risk** this lot will be full. Consider redirecting drivers "
                "or updating signage in advance."
            )
        elif full_prob > 0.4:
            st.info(
                "Moderate risk of heavy usage. Monitoring and dynamic guidance could help."
            )
        else:
            st.success(
                "Low risk that this lot will be at full capacity at the selected time."
            )

        st.caption(
            "Note: Lag and rolling features (previous hours / days) are filled with "
            "neutral values here, so predictions should be interpreted as **what-if "
            "scenarios**, not exact forecasts."
        )

# ===================================================
# PAGE 4 – LOT STATUS OVERVIEW
# ===================================================
elif page == "📊 Lot Status Overview":
    st.title("📊 Lot Status Overview – Compare All Lots")

    st.markdown(
        """
        This page shows, for a selected **day & hour**:

        - Predicted **occupancy** for each parking lot  
        - **Probability** that the lot is near full  
        - Colour-coded status: 🟥 High risk, 🟧 Busy, 🟩 Comfortable  
        """
    )

    if not LOT_FEATURES:
        st.error(
            "No parking-lot indicator features (starting with 'Unit_') were found "
            "in FEATURES. This view needs those to work."
        )
    else:
        st.subheader("Step 1 – Choose Date, Hour & Weather")

        col1, col2, col3 = st.columns([1.2, 0.8, 1.2])

        with col1:
            date_ls = st.date_input(
                "Date (tourist season 2025)",
                value=dt.date(2025, 7, 15),
                min_value=dt.date(2025, 5, 1),
                max_value=dt.date(2025, 9, 30),
            )
        with col2:
            hour_ls = st.slider("Hour of Day", 0, 23, 14)
        with col3:
            max_temp_ls = st.slider("Max Temperature (°C)", -20.0, 40.0, 22.0)

        col4, col5 = st.columns(2)
        with col4:
            total_precip_ls = st.slider("Total Precipitation (mm)", 0.0, 30.0, 0.5)
        with col5:
            wind_gust_ls = st.slider("Speed of Max Gust (km/h)", 0.0, 100.0, 12.0)

        time_feats_ls = compute_time_features(date_ls, hour_ls)

        st.caption(
            "Lag and rolling features are set to neutral values. In a production system "
            "these would come from live occupancy feeds."
        )

        if st.button("Compute lot status"):
            rows = []

            # Base template for all lots
            base_input = {f: 0 for f in FEATURES}

            for k, v in time_feats_ls.items():
                if k in base_input:
                    base_input[k] = v

            if "Max Temp (°C)" in base_input:
                base_input["Max Temp (°C)"] = max_temp_ls
            if "Min Temp (°C)" in base_input:
                base_input["Min Temp (°C)"] = max_temp_ls - 10
            if "Total Precip (mm)" in base_input:
                base_input["Total Precip (mm)"] = total_precip_ls
            if "Spd of Max Gust (km/h)" in base_input:
                base_input["Spd of Max Gust (km/h)"] = wind_gust_ls

            # Loop over each lot, one-hot encode, and predict
            for lot_feat, lot_name in zip(LOT_FEATURES, LOT_DISPLAY_NAMES):
                lot_input = base_input.copy()
                if lot_feat in lot_input:
                    lot_input[lot_feat] = 1

                x_vec = np.array([lot_input[f] for f in FEATURES]).reshape(1, -1)
                x_scaled = scaler.transform(x_vec)

                occ_pred = best_xgb_reg.predict(x_scaled)[0]
                full_prob = best_cls_model.predict_proba(x_scaled)[0, 1]

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

            df = pd.DataFrame(rows).sort_values("Lot")

            def lot_status_row_style(row):
                if "High risk" in row["Status"]:
                    return ["background-color: #ffe5e5"] * len(row)  # light red
                elif "Busy" in row["Status"]:
                    return ["background-color: #fff4e0"] * len(row)  # light orange
                else:
                    return ["background-color: #e9f7ef"] * len(row)  # light green

            styled_df = (
                df.style
                .format(
                    {"Predicted occupancy": "{:.2f}",
                     "Probability full": "{:.1%}"}
                )
                .apply(lot_status_row_style, axis=1)
            )

            st.subheader("Step 2 – Lot status for selected hour")
            st.dataframe(
                styled_df,
                use_container_width=True,
            )

            st.caption(
                "Row colour shows risk level: red = high risk, orange = busy, "
                "green = comfortable."
            )

# ===================================================
# PAGE 5 – XAI (EXPLAINABLE AI)
# ===================================================
elif page == "🔍 XAI – Explainable AI":
    st.title("🔍 Explainable AI – Understanding the Models")

    st.markdown(
        """
        This page explains **why** the models make their predictions,
        using Explainable AI tools:

        - **SHAP summary plot**: which features contribute most  
        - **SHAP bar plot**: global feature importance  
        - **Partial Dependence Plots (PDPs)** for Month, Hour, Temperature  
        - **Residual plot**: how close predictions are to the true values  
        """
    )

    # ---------- SHAP EXPLANATIONS FOR REGRESSION ----------
    st.subheader("SHAP Summary – XGBoost Regressor (Occupancy)")

    try:
        X_shap = X_test_scaled.copy()
        if X_shap.shape[0] > 2000:
            # subsample for speed
            idx = np.random.choice(X_shap.shape[0], 2000, replace=False)
            X_shap = X_shap[idx, :]

        explainer_reg = shap.TreeExplainer(best_xgb_reg)
        shap_values_reg = explainer_reg.shap_values(X_shap)

        fig1, ax1 = plt.subplots()
        shap.summary_plot(
            shap_values_reg,
            X_shap,
            feature_names=FEATURES,
            show=False
        )
        st.pyplot(fig1)
        st.caption(
            "Each point is an hour for one lot. Colour shows the feature value; "
            "position shows how much that feature pushed the prediction up or down."
        )

        st.subheader("SHAP Feature Importance – Regression")
        fig2, ax2 = plt.subplots()
        shap.summary_plot(
            shap_values_reg,
            X_shap,
            feature_names=FEATURES,
            plot_type="bar",
            show=False
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
            best_xgb_reg,
            X_test_scaled,
            feature_indices,
            feature_names=FEATURES,
            ax=ax3
        )
        st.pyplot(fig3)
        st.caption(
            "Partial dependence shows the average effect of each feature on predicted "
            "occupancy while holding all other features constant."
        )
    else:
        st.info(
            "Could not find the configured PDP features ('Max Temp (°C)', 'Month', 'Hour') "
            "in the FEATURES list."
        )

    # ---------- RESIDUAL ANALYSIS ----------
    st.subheader("Residual Plot – Regression Model")

    try:
        y_pred = best_xgb_reg.predict(X_test_scaled)
        residuals = y_reg_test - y_pred

        fig4, ax4 = plt.subplots()
        ax4.scatter(y_pred, residuals, alpha=0.3)
        ax4.axhline(0, color="red", linestyle="--")
        ax4.set_xlabel("Predicted Occupancy")
        ax4.set_ylabel("Residual (Actual - Predicted)")
        st.pyplot(fig4)
        st.caption(
            "If residuals are scattered roughly around zero, the model is capturing "
            "the main patterns without strong bias."
        )
    except Exception as e:
        st.error(f"Could not compute residuals: {e}")

# ===================================================
# PAGE 6 – CHAT ASSISTANT (RAG)
# ===================================================
elif page == "💬 Chat Assistant (RAG)":
    st.title("💬 Banff Parking Chat Assistant (RAG)")

    st.markdown(
        """
        Ask questions about parking patterns, busy times, or model behaviour.

        This chatbot uses **RAG (Retrieval-Augmented Generation)**:
        1. It first retrieves relevant lines from your `banff_knowledge.txt` file  
        2. Then it uses an OpenAI model to answer, grounded in that context  
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
        # Add user message to history
        st.session_state.rag_chat_history.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant response
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
