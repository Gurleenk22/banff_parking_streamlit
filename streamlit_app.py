import os
from datetime import datetime, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import streamlit as st
import streamlit.components.v1 as components
from sklearn.inspection import PartialDependenceDisplay

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


# ---------------------------------------------------
# SAFE OPENAI CLIENT CREATION
# ---------------------------------------------------
def get_openai_client():
    """
    Returns an OpenAI client if an API key is configured,
    otherwise returns None so the app does not crash.
    """
    # 1) Try environment variable
    api_key = os.getenv("OPENAI_API_KEY", None)

    # 2) Try Streamlit secrets
    if api_key is None:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None

    if api_key is None:
        return None

    return OpenAI(api_key=api_key)


# ---------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# ---------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load trained models, scaler, feature list, and test data.
    Make sure these files exist in your repo.
    """
    reg = joblib.load("banff_best_xgb_reg.pkl")      # tuned XGBoost regressor
    cls = joblib.load("banff_best_xgb_cls.pkl")      # tuned classifier (XGB or LGBM)
    scaler = joblib.load("banff_scaler.pkl")         # StandardScaler used in training
    features = joblib.load("banff_features.pkl")     # List of feature names

    # Test data for XAI and residual analysis
    X_test_scaled = np.load("X_test_scaled.npy")
    y_reg_test = np.load("y_reg_test.npy")

    return reg, cls, scaler, features, X_test_scaled, y_reg_test


best_xgb_reg, best_xgb_cls, scaler, FEATURES, X_test_scaled, y_reg_test = load_models_and_data()


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
    If the API is not configured or fails, fall back to
    a simple answer built only from the retrieved context.
    """
    docs, vectorizer, doc_embeddings = load_rag_knowledge()
    context = retrieve_context(user_question, docs, vectorizer, doc_embeddings, k=5)

    client = get_openai_client()
    if client is None:
        return (
            "⚠️ The chat assistant is currently running **without an OpenAI API key**, "
            "so I can’t generate new answers.\n\n"
            "However, here is the most relevant information from the project notes:\n\n"
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
        messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": user_question})

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
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
st.sidebar.markdown("## Banff Parking Dashboard")

st.sidebar.markdown(
    """
- 📊 **Dashboard** – quick overview  
- 🎯 **Make Prediction** – what-if for 1 lot  
- 🧭 **Lot Status** – compare all lots  
- 🔍 **XAI** – model insights  
- 💬 **Chat** – RAG assistant  
"""
)

page = st.sidebar.radio(
    "Go to",
    [
        "📊 Dashboard",
        "🎯 Make Prediction",
        "🧭 Lot Status Overview",
        "🔍 XAI – Explainable AI",
        "💬 Chat Assistant (RAG)",
    ],
)


# ---------------------------------------------------
# PAGE 1 – DASHBOARD (NEW)
# ---------------------------------------------------
if page == "📊 Dashboard":
    st.title("📊 Banff Parking – Project Dashboard")

    # --- Top summary blocks / hero section ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Season Modelled", "May–September 2025")
        st.caption("Tourist peak season used for training.")
    with col2:
        st.metric("Lots Modelled", "Multiple units")
        st.caption("On-street & off-street lots in Banff.")
    with col3:
        st.metric("Best R² (Occupancy)", "≈ 0.93")
        st.caption("XGBoost regressor performance on test set.")

    st.markdown("---")

    # --- Date + time selector (calendar style) ---
    st.subheader("📅 Choose a Day & Time to Explore")

    col_date, col_time = st.columns([1.2, 1])
    with col_date:
        selected_date = st.date_input(
            "Select date (within May–Sep 2025)",
            value=datetime(2025, 7, 15),
            min_value=datetime(2025, 5, 1),
            max_value=datetime(2025, 9, 30),
        )
    with col_time:
        selected_time = st.slider(
            "Select hour of day",
            min_value=0,
            max_value=23,
            value=14,
        )

    dt = datetime.combine(selected_date, time(selected_time))
    dow = dt.weekday()
    is_weekend = "Weekend" if dow in [5, 6] else "Weekday"

    st.info(
        f"Selected: **{dt.strftime('%Y-%m-%d %H:00')}** "
        f"({is_weekend}, Month = {dt.month}, DayOfWeek = {dow}).\n\n"
        "You can use the same date & hour on the **Make Prediction** and "
        "**Lot Status** pages to keep the story consistent during your demo."
    )

    st.markdown("---")

    # --- Short explanation of each page with cards ---
    st.subheader("🧭 How the App is Organized")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            ### 🎯 Make Prediction – Single Lot *What-If*

            - Pick **one lot** and set **time & weather**  
            - Model returns:
              - predicted **hourly occupancy**
              - **probability the lot is near full** (>90%)  
            - Use this when you want to tell a story like:  
              *“What happens if we have rain on a Saturday afternoon?”*
            """
        )
    with c2:
        st.markdown(
            """
            ### 🧭 Lot Status Overview – Compare All Lots

            - Use **one set** of conditions (date, hour, weather)  
            - See **all lots** in a table:
              - predicted occupancy  
              - probability of being full  
              - status: 🟥 high risk, 🟧 busy, 🟩 comfortable  
            - Great for operational decisions:  
              *“Which lots should show ‘FULL’ on signage at 2 PM?”*
            """
        )

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(
            """
            ### 🔍 XAI – Explainable AI

            - **SHAP plots** show which features matter most  
            - **Partial Dependence** shows how temperature, hour, and month
              affect predicted occupancy  
            - **Residual plot** checks whether predictions are unbiased  
            - This page is for instructors / clients who ask:  
              *“Why should we trust this model?”*
            """
        )
    with c4:
        st.markdown(
            """
            ### 💬 Chat Assistant (RAG)

            - Uses your `banff_knowledge.txt` as a mini knowledge base  
            - You can ask about:
              - peak hours / busy days  
              - interpretation of features  
              - model behaviour  
            - Acts like a **friendly helper** during your presentation.
            """
        )

    st.markdown("---")

    # --- Optional Power BI embed (replace URL with your own) ---
    st.subheader("📈 Power BI Overview (Optional)")

    POWERBI_EMBED_URL = "https://app.powerbi.com/view?r=YOUR_EMBED_URL_HERE"

    st.caption(
        "If you have published a Power BI dashboard, paste its public embed URL "
        "into `POWERBI_EMBED_URL` in the code."
    )

    if "YOUR_EMBED_URL_HERE" not in POWERBI_EMBED_URL:
        components.iframe(POWERBI_EMBED_URL, height=500)
    else:
        st.info(
            "Power BI URL not configured yet. Edit `POWERBI_EMBED_URL` in `app.py` "
            "to show your live dashboard here."
        )


# ---------------------------------------------------
# PAGE 2 – MAKE PREDICTION
# ---------------------------------------------------
if page == "🎯 Make Prediction":
    st.title("🎯 Interactive Parking Demand Prediction")

    st.markdown(
        """
        Use this page to explore **what-if scenarios for a single Banff parking lot**.

        1. Select a **parking lot**  
        2. Choose a **scenario** (or adjust the sliders)  
        3. See:
           - Predicted **occupancy** for the selected hour  
           - **Probability** the lot is near full (>90%)  
        """
    )

    # Find lot indicator features (one-hot encoded units)
    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

    # Sort lot list alphabetically so numbers appear in order (BANFF02, BANFF03, …)
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
        "Sunny Weekend Midday": {"month": 7, "dow": 5, "hour": 13,
                                 "max_temp": 24.0, "precip": 0.0, "gust": 10.0},
        "Rainy Weekday Afternoon": {"month": 6, "dow": 2, "hour": 16,
                                    "max_temp": 15.0, "precip": 5.0, "gust": 20.0},
        "Cold Morning (Shoulder Season)": {"month": 5, "dow": 1, "hour": 9,
                                           "max_temp": 5.0, "precip": 0.0, "gust": 15.0},
        "Warm Evening (Busy Day)": {"month": 8, "dow": 6, "hour": 19,
                                    "max_temp": 22.0, "precip": 0.0, "gust": 8.0},
    }

    st.subheader("Step 1 – Choose Lot & Scenario")

    col_lot, col_scenario = st.columns([1.2, 1])

    with col_lot:
        if lot_features:
            selected_lot_label = st.selectbox(
                "Select parking lot",
                lot_display_names,
                index=0
            )
            selected_lot_feature = lot_features[lot_display_names.index(selected_lot_label)]
        else:
            selected_lot_label = None
            selected_lot_feature = None

    with col_scenario:
        selected_scenario = st.selectbox(
            "Scenario",
            list(scenario_options.keys()),
            index=1
        )

    # Default slider values – will be overwritten by scenario if chosen
    default_vals = {"month": 7, "dow": 5, "hour": 13,
                    "max_temp": 22.0, "precip": 0.5, "gust": 12.0}

    if scenario_options[selected_scenario] is not None:
        default_vals.update(scenario_options[selected_scenario])

    st.subheader("Step 2 – Adjust Conditions (if needed)")

    col1, col2 = st.columns(2)

    with col1:
        month = st.slider("Month (1 = Jan, 12 = Dec)",
                          1, 12, int(default_vals["month"]))
        day_of_week = st.slider("Day of Week (0 = Monday, 6 = Sunday)",
                                0, 6, int(default_vals["dow"]))
        hour = st.slider("Hour of Day (0–23)",
                         0, 23, int(default_vals["hour"]))

    with col2:
        max_temp = st.slider("Max Temperature (°C)",
                             -20.0, 40.0, float(default_vals["max_temp"]))
        total_precip = st.slider("Total Precipitation (mm)",
                                 0.0, 30.0, float(default_vals["precip"]))
        wind_gust = st.slider("Speed of Max Gust (km/h)",
                              0.0, 100.0, float(default_vals["gust"]))

    is_weekend = 1 if day_of_week in [5, 6] else 0

    st.caption(
        "Lag features (previous-hour occupancy, rolling averages) are set automatically "
        "by the model and are not entered manually here."
    )

    # Build feature dict starting from all zeros
    base_input = {f: 0 for f in FEATURES}

    # Time & weather
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

    # Lot indicator – one-hot
    if selected_lot_feature is not None and selected_lot_feature in base_input:
        base_input[selected_lot_feature] = 1

    # Vector in the exact training feature order
    x_vec = np.array([base_input[f] for f in FEATURES]).reshape(1, -1)
    x_scaled = scaler.transform(x_vec)

    if st.button("🔮 Predict for this scenario"):
        # Current-hour predictions
        occ_pred = best_xgb_reg.predict(x_scaled)[0]
        full_prob = best_xgb_cls.predict_proba(x_scaled)[0, 1]

        st.subheader("Step 3 – Results for Selected Hour")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Predicted occupancy (model units)",
                      f"{occ_pred:.2f}")
        with col_res2:
            st.metric("Probability lot is near full",
                      f"{full_prob:.1%}")

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
# PAGE 3 – LOT STATUS OVERVIEW (ALL LOTS AT ONCE)
# ---------------------------------------------------
if page == "🧭 Lot Status Overview":
    st.title("🧭 Lot Status Overview – Which Lots Are Likely Full?")

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

    # sort lots alphabetically so numbers are in sequence
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
            day_of_week = st.slider("Day of Week (0 = Monday, 6 = Sunday)", 0, 6, 5)
            hour = st.slider("Hour of Day", 0, 23, 14)

        with col2:
            max_temp = st.slider("Max Temperature (°C)", -20.0, 40.0, 22.0)
            total_precip = st.slider("Total Precipitation (mm)", 0.0, 30.0, 0.5)
            wind_gust = st.slider("Speed of Max Gust (km/h)", 0.0, 100.0, 12.0)

        is_weekend = 1 if day_of_week in [5, 6] else 0

        st.caption(
            "Lag features (previous-hour occupancy, rolling averages) are set to 0 "
            "for this overview. In a real system they would come from live feeds."
        )

        if st.button("Compute lot status"):
            rows = []

            # Base feature template
            base_input = {f: 0 for f in FEATURES}

            # Common time & weather fields
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

            # Loop over each lot, one-hot encode, and predict
            for lot_feat, lot_name in zip(lot_features, lot_display_names):
                lot_input = base_input.copy()
                if lot_feat in lot_input:
                    lot_input[lot_feat] = 1

                x_vec = np.array([lot_input[f] for f in FEATURES]).reshape(1, -1)
                x_scaled = scaler.transform(x_vec)

                occ_pred = best_xgb_reg.predict(x_scaled)[0]
                full_prob = best_xgb_cls.predict_proba(x_scaled)[0, 1]

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
            # sort by lot name so numbers are in sequence
            df = df.sort_values("Lot")

            # ---------- nice colour styling ----------
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
                "Lots are shown in numeric order (BANFF02, BANFF03, …). "
                "Row colour shows risk level: red = high risk, orange = busy, "
                "green = comfortable."
            )


# ---------------------------------------------------
# PAGE 4 – XAI (EXPLAINABLE AI)
# ---------------------------------------------------
if page == "🔍 XAI – Explainable AI":
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
        explainer_reg = shap.TreeExplainer(best_xgb_reg)
        shap_values_reg = explainer_reg.shap_values(X_test_scaled)

        # Summary dot plot
        fig1, ax1 = plt.subplots()
        shap.summary_plot(
            shap_values_reg,
            X_test_scaled,
            feature_names=FEATURES,
            show=False
        )
        st.pyplot(fig1)
        st.caption(
            "Each point represents a sample. Colour shows feature value, and position "
            "shows how much that feature pushed the prediction up or down."
        )

        # Summary bar plot
        st.subheader("SHAP Feature Importance – Regression")
        fig2, ax2 = plt.subplots()
        shap.summary_plot(
            shap_values_reg,
            X_test_scaled,
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
    except Exception as e:
        st.error(f"Could not compute residuals: {e}")


# ---------------------------------------------------
# PAGE 5 – CHAT ASSISTANT (RAG)
# ---------------------------------------------------
if page == "💬 Chat Assistant (RAG)":
    st.title("💬 Banff Parking Chat Assistant (RAG)")

    if get_openai_client() is None:
        st.warning(
            "No OpenAI API key is configured. The chatbot will answer only with "
            "static context from `banff_knowledge.txt`."
        )

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
