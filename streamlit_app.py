import os
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import streamlit as st
import streamlit.components.v1 as components

from sklearn.inspection import PartialDependenceDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Optional: OpenAI for RAG chat ----
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

# ---- Optional: LightGBM (needed for classifier .pkl) ----
try:
    import lightgbm as lgb  # noqa: F401
except ImportError:
    lgb = None

# ---- Power BI public embed URL (replace with your own) ----
POWERBI_EMBED_URL = ""  # e.g. "https://app.powerbi.com/view?r=YOUR_REPORT_ID"

# ---- OpenAI client (only if key + package available) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OpenAI is not None and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# ---------------------------------------------------
# BASIC PAGE CONFIG + GLOBAL STYLING
# ---------------------------------------------------
st.set_page_config(
    page_title="Banff Parking – ML & XAI Dashboard",
    layout="wide",
)

# Soft background + tighter layout + nicer sidebar
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fb;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.4rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models_and_data():
    """
    Load tuned models, scaler, feature list and test data.
    Files must be present in the same repo:
      - banff_best_xgb_reg.pkl
      - banff_best_lgbm_cls.pkl
      - banff_scaler.pkl
      - banff_features.pkl
      - X_test_scaled.npy
      - y_reg_test.npy
    """
    # This will fail if lightgbm is not installed because the classifier was
    # trained with LightGBM and joblib needs that package to unpickle it.
    reg = joblib.load("banff_best_xgb_reg.pkl")
    cls = joblib.load("banff_best_lgbm_cls.pkl")
    scaler = joblib.load("banff_scaler.pkl")
    features = joblib.load("banff_features.pkl")

    X_test_scaled = np.load("X_test_scaled.npy")
    y_reg_test = np.load("y_reg_test.npy")

    return reg, cls, scaler, features, X_test_scaled, y_reg_test


# Try to load everything once and keep a flag
try:
    (
        best_xgb_reg,
        best_lgbm_cls,
        scaler,
        FEATURES,
        X_test_scaled,
        y_reg_test,
    ) = load_models_and_data()
    MODELS_OK = True
    MODEL_ERROR = ""
except Exception as e:  # noqa: BLE001
    best_xgb_reg = None
    best_lgbm_cls = None
    scaler = None
    FEATURES = []
    X_test_scaled = None
    y_reg_test = None
    MODELS_OK = False
    MODEL_ERROR = str(e)

# ---------------------------------------------------
# RAG: LOAD KNOWLEDGE + BUILD VECTORIZER
# ---------------------------------------------------
@st.cache_resource(show_spinner=False)
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
    If the API fails or there is no API key, fall back to a simple
    answer based only on the retrieved context.
    """
    docs, vectorizer, doc_embeddings = load_rag_knowledge()
    context = retrieve_context(user_question, docs, vectorizer, doc_embeddings, k=5)

    # If we have no client or no key, just return a context-based answer
    if client is None:
        return (
            "The OpenAI API key is not configured for this app, "
            "so I can only share information directly from the project notes:\n\n"
            f"{context}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly project assistant helping Gurleen explain a Banff "
                "parking analytics project. Speak clearly and simply for classmates and "
                "instructors who are not data scientists. Use the provided 'Context' "
                "from the project notes as your main source of truth."
            ),
        },
        {"role": "system", "content": f"Context from project notes:\n{context}"},
    ]

    # keep last few turns of history
    for h in chat_history[-4:]:
        messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": user_question})

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages={ "messages": messages } if False else messages,  # small trick: allow old/new clients
            temperature=0.3,
        )
        # depending on client version; handle generically
        choice = response.choices[0]
        content = getattr(choice, "message", choice).content
        return content.strip()
    except Exception:
        return (
            "I couldn’t contact the language-model service right now.\n\n"
            "Here is the most relevant information from the project notes:\n\n"
            f"{context}"
        )


# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.title("Banff Parking Dashboard")
st.sidebar.markdown(
    """
- 🏠 **Dashboard** – quick overview  
- 📘 **Project Guide** – what each page does  
- 🎯 **Make Prediction** – what-if for 1 lot  
- 📊 **Lot Status** – compare all lots  
- 🔍 **XAI** – model insights  
- 💬 **Chat** – RAG assistant  
"""
)

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

# ---------------------------------------------------
# Helper: compute lot status for all lots
# ---------------------------------------------------
def compute_lot_status(month, day_of_week, hour, max_temp, total_precip, wind_gust):
    """
    Returns a DataFrame with predicted occupancy, probability full and status
    for each parking lot, using the tuned models.
    """
    if not MODELS_OK:
        return pd.DataFrame()

    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

    if not lot_features:
        return pd.DataFrame()

    # sort by name
    lot_pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
    lot_features, lot_display_names = zip(*lot_pairs)
    lot_features = list(lot_features)
    lot_display_names = list(lot_display_names)

    is_weekend = 1 if day_of_week in [5, 6] else 0

    base_input = {f: 0 for f in FEATURES}
    if "Month" in base_input:
        base_input["Month"] = month
    if "DayOfWeek" in base_input:
        base_input["DayOfWeek"] = day_of_week
    if "Hour" in base_input:
        base_input["Hour"] = hour
    if "IsWeekend" in base_input:
        base_input["IsWeekend"] = is_weekend
    if "Max Temp (°C)" in base_input:
        base_input["Max Temp (°C)"] = max_temp
    if "Total Precip (mm)" in base_input:
        base_input["Total Precip (mm)"] = total_precip
    if "Spd of Max Gust (km/h)" in base_input:
        base_input["Spd of Max Gust (km/h)"] = wind_gust

    rows = []
    for lot_feat, lot_name in zip(lot_features, lot_display_names):
        row_input = base_input.copy()
        if lot_feat in row_input:
            row_input[lot_feat] = 1

        x_vec = np.array([row_input[f] for f in FEATURES]).reshape(1, -1)
        x_scaled = scaler.transform(x_vec)

        occ_pred = best_xgb_reg.predict(x_scaled)[0]
        full_prob = best_lgbm_cls.predict_proba(x_scaled)[0, 1]

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
    return df


# ---------------------------------------------------
# GLOBAL MODEL-ERROR BANNER (if any)
# ---------------------------------------------------
if not MODELS_OK:
    st.error(
        f"Model files could not be loaded.\n\n**Details:** `{MODEL_ERROR}`\n\n"
        "If you are running this on Streamlit Cloud, make sure:\n"
        "- All `.pkl` and `.npy` files are in the repo\n"
        "- The `lightgbm` package is added to `requirements.txt` "
        "because the classifier model was trained with LightGBM."
    )

# ---------------------------------------------------
# PAGE 1 – DASHBOARD
# ---------------------------------------------------
if page == "🏠 Dashboard":
    st.title("🚗 Banff Parking – Interactive Dashboard")

    # Hero cards
    hcol1, hcol2, hcol3 = st.columns([2, 1.2, 1.2])
    with hcol1:
        st.markdown(
            """
            <div style="background-color:#ffffff;border-radius:12px;padding:1rem 1.2rem;
                        box-shadow:0 2px 6px rgba(15,23,42,0.08);">
              <p style="font-size:0.95rem;line-height:1.5;margin-bottom:0;">
                This dashboard turns your Banff parking model into an
                <strong>operations tool</strong>. You can explore a typical
                tourist-season day and quickly see which lots are most at risk
                of being full.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hcol2:
        st.markdown(
            """
            <div style="background-color:#e0f2fe;border-radius:12px;padding:0.8rem 1rem;
                        box-shadow:0 1px 4px rgba(15,23,42,0.05);">
              <p style="font-size:0.8rem;margin:0;">
              <strong>Best for:</strong><br>
              Planning which lots to monitor and where to place signs.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hcol3:
        st.markdown(
            """
            <div style="background-color:#eef2ff;border-radius:12px;padding:0.8rem 1rem;
                        box-shadow:0 1px 4px rgba(15,23,42,0.05);">
              <p style="font-size:0.8rem;margin:0;">
              <strong>Models:</strong><br>
              Tuned XGBoost (occupancy) + LightGBM (near-full).
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Power BI section
    st.subheader("📈 Power BI – Parking Overview (optional)")

    if POWERBI_EMBED_URL:
        components.iframe(POWERBI_EMBED_URL, height=450)
    else:
        st.info(
            "Add your Power BI public embed URL to `POWERBI_EMBED_URL` near the top of "
            "`streamlit_app.py` to display your interactive Power BI dashboard here."
        )

    st.markdown("---")
    st.subheader("🗓️ Quick Scenario – Which lots are risky for a typical hour?")

    c1, c2, c3 = st.columns(3)
    with c1:
        chosen_date = st.date_input(
            "Pick a date in the tourist season",
            value=date(2025, 7, 15),
        )
    with c2:
        hour = st.slider("Hour of day", 0, 23, 14)
    with c3:
        max_temp = st.slider("Max Temperature (°C)", -5.0, 35.0, 22.0)

    # Simple default weather assumptions
    c4, c5 = st.columns(2)
    with c4:
        total_precip = st.slider("Total Precipitation (mm)", 0.0, 20.0, 0.5)
    with c5:
        wind_gust = st.slider("Speed of Max Gust (km/h)", 0.0, 80.0, 12.0)

    month = chosen_date.month
    day_of_week = chosen_date.weekday()

    if MODELS_OK:
        df_status = compute_lot_status(
            month,
            day_of_week,
            hour,
            max_temp,
            total_precip,
            wind_gust,
        )
        if not df_status.empty:
            # Top 3 highest probability of being full
            top3 = df_status.sort_values("Probability full", ascending=False).head(3)

            tcol1, tcol2 = st.columns([1.2, 1])
            with tcol1:
                st.markdown("#### 🔝 Top 3 highest-risk lots")
                st.dataframe(
                    top3.style.format(
                        {
                            "Predicted occupancy": "{:.2f}",
                            "Probability full": "{:.1%}",
                        }
                    ),
                    use_container_width=True,
                )
            with tcol2:
                st.markdown("#### ℹ️ How to read this")
                st.markdown(
                    """
                    - **Predicted occupancy** is in model units (vehicles).  
                    - **Probability full** is the chance the lot is above 90% capacity.  
                    - **Status** combines this into a simple signal:
                      🟥 high risk, 🟧 busy, 🟩 comfortable.
                    """
                )
        else:
            st.warning("No lot indicator features were found in the model feature list.")
    else:
        st.info("Models are not loaded, so the quick scenario view is disabled.")

# ---------------------------------------------------
# PAGE 2 – PROJECT GUIDE
# ---------------------------------------------------
elif page == "📘 Project Guide":
    st.title("📘 Project Guide – How this app works")

    st.markdown(
        """
        ### 1. Problem & Data

        - **Goal:** Help Banff anticipate where and when parking lots will be under pressure
          during the May–September tourist season.
        - **Data used:**
          - Hourly parking management data (occupancy per lot)
          - Hourly weather features (temperature, precipitation, wind)
        - **Engineered features:**
          - Time features: `Hour`, `DayOfWeek`, `Month`, `IsWeekend`
          - History features: lagged occupancies (1h, 2h, 3h, 24h, 168h)
          - Rolling averages and standard deviations
          - Lot capacity and one-hot encoded `Unit` names
        """
    )

    st.markdown(
        """
        ### 2. Models

        - **Regression model (XGBoost):**
          - Target: `Occupancy` (vehicles)
          - Output: predicted number of parked vehicles each hour for each lot.
        - **Classification model (LightGBM):**
          - Target: `Is_Full` (1 if ≥ 90% of capacity, else 0)
          - Output: probability that a lot will be near-full.

        Both models are tuned with time-series cross-validation and then evaluated on a
        held-out test period.
        """
    )

    st.markdown(
        """
        ### 3. What each page does

        - **🏠 Dashboard**  
          Quick snapshot. Choose a date & hour, see a top-3 list of lots by risk.
          There is also space for your **Power BI** overview.
        - **🎯 Make Prediction**  
          Pick one lot, set time & weather manually or with presets (e.g., sunny weekend).
          You get:
          - Predicted occupancy  
          - Probability the lot is near full  
          and simple guidance messages.
        - **📊 Lot Status Overview**  
          Same inputs for time & weather, but shows **all lots at once** in a table so that
          dispatchers can see which ones are high risk, busy, or comfortable.
        - **🔍 XAI – Explainable AI**  
          SHAP plots, partial dependence plots, and residual analysis to explain:
          - Which features drive predictions  
          - How temperature, hour, and month affect occupancy  
          - Whether the model is systematically biased.
        - **💬 Chat Assistant (RAG)**  
          A chatbot that reads `banff_knowledge.txt` (your project notes) and answers
          questions in simple language. Change that file to control what the chatbot knows.
        """
    )

    st.markdown(
        """
        ### 4. Example questions to explore

        You can ask the **Chat Assistant** things like:

        - “Which features were most important for predicting parking occupancy?”  
        - “How does weather affect parking usage in Banff?”  
        - “Which lots tend to be high risk on sunny Saturdays?”  
        - “How should the town use this dashboard in daily operations?”
        """
    )

# ---------------------------------------------------
# PAGE 3 – MAKE PREDICTION (ONE LOT)
# ---------------------------------------------------
elif page == "🎯 Make Prediction":
    st.title("🎯 Interactive Parking Demand Prediction (single lot)")

    if not MODELS_OK:
        st.warning("Models are not loaded, so predictions are disabled on this page.")
    else:
        st.markdown(
            """
            Use this page to explore *what-if* scenarios for a single Banff parking lot:

            1. Select a **parking lot**  
            2. Choose a **scenario** or adjust the sliders  
            3. See predicted **occupancy** and **near-full probability**
            """
        )

        lot_features = [f for f in FEATURES if f.startswith("Unit_")]
        lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

        if lot_features:
            lot_pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
            lot_features, lot_display_names = zip(*lot_pairs)
            lot_features = list(lot_features)
            lot_display_names = list(lot_display_names)

        if not lot_features:
            st.error(
                "No parking-lot indicator features (starting with 'Unit_') were "
                "found in FEATURES. Lot selection is disabled."
            )
        else:
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
                selected_lot_label = st.selectbox(
                    "Select parking lot",
                    lot_display_names,
                    index=0,
                )
                selected_lot_feature = lot_features[lot_display_names.index(selected_lot_label)]

            with col_scenario:
                selected_scenario = st.selectbox(
                    "Scenario",
                    list(scenario_options.keys()),
                    index=1,
                )

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
                month = st.slider("Month (1 = Jan, 12 = Dec)", 1, 12, int(default_vals["month"]))
                day_of_week = st.slider(
                    "Day of Week (0 = Monday, 6 = Sunday)", 0, 6, int(default_vals["dow"])
                )
                hour = st.slider("Hour of Day (0–23)", 0, 23, int(default_vals["hour"]))

            with col2:
                max_temp = st.slider(
                    "Max Temperature (°C)",
                    -20.0,
                    40.0,
                    float(default_vals["max_temp"]),
                )
                total_precip = st.slider(
                    "Total Precipitation (mm)",
                    0.0,
                    30.0,
                    float(default_vals["precip"]),
                )
                wind_gust = st.slider(
                    "Speed of Max Gust (km/h)",
                    0.0,
                    100.0,
                    float(default_vals["gust"]),
                )

            is_weekend = 1 if day_of_week in [5, 6] else 0

            st.caption(
                "Lag features (previous-hour occupancy, rolling averages) are set automatically "
                "by the model and are not entered manually here."
            )

            base_input = {f: 0 for f in FEATURES}

            if "Month" in base_input:
                base_input["Month"] = month
            if "DayOfWeek" in base_input:
                base_input["DayOfWeek"] = day_of_week
            if "Hour" in base_input:
                base_input["Hour"] = hour
            if "IsWeekend" in base_input:
                base_input["IsWeekend"] = is_weekend
            if "Max Temp (°C)" in base_input:
                base_input["Max Temp (°C)"] = max_temp
            if "Total Precip (mm)" in base_input:
                base_input["Total Precip (mm)"] = total_precip
            if "Spd of Max Gust (km/h)" in base_input:
                base_input["Spd of Max Gust (km/h)"] = wind_gust

            if selected_lot_feature in base_input:
                base_input[selected_lot_feature] = 1

            x_vec = np.array([base_input[f] for f in FEATURES]).reshape(1, -1)
            x_scaled = scaler.transform(x_vec)

            if st.button("🔮 Predict for this scenario"):
                occ_pred = best_xgb_reg.predict(x_scaled)[0]
                full_prob = best_lgbm_cls.predict_proba(x_scaled)[0, 1]

                st.subheader("Step 3 – Results for Selected Hour")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Predicted occupancy (vehicles)", f"{occ_pred:.2f}")
                with col_res2:
                    st.metric("Probability lot is near full", f"{full_prob:.1%}")

                if full_prob > 0.7:
                    st.warning(
                        "⚠️ High risk this lot will be full. Consider re-routing drivers "
                        "or highlighting alternative lots on signs."
                    )
                elif full_prob > 0.4:
                    st.info(
                        "Moderate risk of heavy usage. Monitoring and dynamic guidance "
                        "could be useful."
                    )
                else:
                    st.success("Low risk of this lot being near full for this hour.")

# ---------------------------------------------------
# PAGE 4 – LOT STATUS OVERVIEW (ALL LOTS)
# ---------------------------------------------------
elif page == "📊 Lot Status Overview":
    st.title("📊 Lot Status Overview – Which Lots Are Likely Full?")

    if not MODELS_OK:
        st.warning("Models are not loaded, so lot status overview is disabled.")
    else:
        st.markdown(
            """
            This page shows, for a selected hour and conditions, the predicted:

            - **Occupancy** for each parking lot  
            - **Probability that the lot is near full**  
            - Simple status: 🟥 High risk, 🟧 Busy, 🟩 Comfortable
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            month = st.slider("Month (1 = Jan, 12 = Dec)", 1, 12, 7)
            day_of_week = st.slider("Day of Week (0 = Monday, 6 = Sunday)", 0, 6, 5)
            hour = st.slider("Hour of Day", 0, 23, 14)
        with col2:
            max_temp = st.slider("Max Temperature (°C)", -20.0, 40.0, 22.0)
            total_precip = st.slider("Total Precipitation (mm)", 0.0, 30.0, 0.5)
            wind_gust = st.slider("Speed of Max Gust (km/h)", 0.0, 100.0, 12.0)

        st.caption(
            "Lag features (previous-hour occupancy, rolling averages) are set to 0 for this overview. "
            "In a live system they would come from live sensor feeds."
        )

        if st.button("Compute lot status"):
            df = compute_lot_status(
                month,
                day_of_week,
                hour,
                max_temp,
                total_precip,
                wind_gust,
            )
            if df.empty:
                st.warning("Could not compute lot status – check that feature names match.")
            else:
                def lot_status_row_style(row):
                    if "High risk" in row["Status"]:
                        return ["background-color: #ffe5e5"] * len(row)
                    if "Busy" in row["Status"]:
                        return ["background-color: #fff4e0"] * len(row)
                    return ["background-color: #e9f7ef"] * len(row)

                styled_df = (
                    df.style.format(
                        {
                            "Predicted occupancy": "{:.2f}",
                            "Probability full": "{:.1%}",
                        }
                    ).apply(lot_status_row_style, axis=1)
                )

                st.subheader("Lot status for selected hour")
                st.dataframe(styled_df, use_container_width=True)

                st.caption(
                    "Row colour shows risk level: red = high risk, orange = busy, green = comfortable. "
                    "You can sort columns directly in this table during your demo."
                )

# ---------------------------------------------------
# PAGE 5 – XAI (EXPLAINABLE AI)
# ---------------------------------------------------
elif page == "🔍 XAI – Explainable AI":
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

    if not MODELS_OK:
        st.warning("Models are not loaded, so XAI plots cannot be generated.")
    else:
        # SHAP for regression
        st.subheader("SHAP Summary – Regression Model (Occupancy)")
        try:
            explainer_reg = shap.TreeExplainer(best_xgb_reg)
            shap_values_reg = explainer_reg.shap_values(X_test_scaled)

            fig1, _ = plt.subplots()
            shap.summary_plot(
                shap_values_reg,
                X_test_scaled,
                feature_names=FEATURES,
                show=False,
            )
            st.pyplot(fig1)
            st.caption(
                "Each point is one sample. Colour shows feature value; position shows how much "
                "that feature pushed the prediction up or down for that sample."
            )

            st.subheader("SHAP Feature Importance – Regression")
            fig2, _ = plt.subplots()
            shap.summary_plot(
                shap_values_reg,
                X_test_scaled,
                feature_names=FEATURES,
                plot_type="bar",
                show=False,
            )
            st.pyplot(fig2)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not generate SHAP plots: {e}")

        # Partial dependence plots
        st.subheader("Partial Dependence – Key Features")
        pd_feature_names = [f for f in ["Max Temp (°C)", "Month", "Hour"] if f in FEATURES]

        if pd_feature_names:
            feature_indices = [FEATURES.index(f) for f in pd_feature_names]
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            PartialDependenceDisplay.from_estimator(
                best_xgb_reg,
                X_test_scaled,
                feature_indices,
                feature_names=FEATURES,
                ax=ax3,
            )
            st.pyplot(fig3)
            st.caption(
                "Partial dependence shows the average effect of each feature on predicted occupancy "
                "while holding all other features constant."
            )
        else:
            st.info(
                "Could not find the configured PDP features ('Max Temp (°C)', 'Month', 'Hour') "
                "in the FEATURES list. You may need to adjust the feature names."
            )

        # Residual analysis
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
                "Residuals scattered symmetrically around zero suggest that the model "
                "captures the main patterns without strong systematic bias."
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not compute residuals: {e}")

# ---------------------------------------------------
# PAGE 6 – CHAT ASSISTANT (RAG)
# ---------------------------------------------------
elif page == "💬 Chat Assistant (RAG)":
    st.title("💬 Banff Parking Chat Assistant (RAG)")

    st.markdown(
        """
        Ask questions about parking patterns, busy times, or model behaviour.

        This chatbot uses **RAG (Retrieval-Augmented Generation)**:

        1. It first retrieves relevant lines from `banff_knowledge.txt`  
        2. Then it uses an OpenAI model (if configured) to answer, grounded in that context  
        """
    )

    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []

    for msg in st.session_state.rag_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about Banff parking or this project...")

    if user_input:
        st.session_state.rag_chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking with project context..."):
                answer = generate_chat_answer(
                    user_input,
                    st.session_state.rag_chat_history,
                )
                st.markdown(answer)

        st.session_state.rag_chat_history.append({"role": "assistant", "content": answer})

    st.caption(
        "Tip: edit `banff_knowledge.txt` in your repo to control what the chatbot knows "
        "about your EDA, feature engineering, and model findings."
    )
