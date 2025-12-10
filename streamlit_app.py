import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.inspection import PartialDependenceDisplay
from datetime import date
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

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
    reg = joblib.load("banff_best_xgb_reg.pkl")      # XGBoost regressor
    cls = joblib.load("banff_best_xgb_cls.pkl")      # XGBoost classifier
    scaler = joblib.load("banff_scaler.pkl")         # Scaler used in training
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
                "parking analytics project. Speak clearly and simply. Use the provided "
                "'Context' as your main source of truth. If the context does not clearly "
                "contain the answer, say that openly and give a small reasonable guess."
            ),
        },
        {
            "role": "system",
            "content": f"Context from project notes:\n{context}",
        },
    ]

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
st.sidebar.markdown(
    """
    - 📊 **Dashboard** – quick overview  
    - 🎯 **Make Prediction** – what-if for 1 lot  
    - 🗺️ **Lot Status** – compare all lots  
    - 🔍 **XAI** – model insights  
    - 💬 **Chat** – RAG assistant  
    """
)

page = st.sidebar.radio(
    "Go to",
    [
        "📊 Dashboard",
        "🎯 Make Prediction",
        "🗺️ Lot Status Overview",
        "🔍 XAI – Explainable AI",
        "💬 Chat Assistant (RAG)",
    ]
)

# Small helper: get sorted lot feature names
def get_lot_features():
    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]
    if lot_features:
        lot_pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
        lot_features, lot_display_names = zip(*lot_pairs)
        return list(lot_features), list(lot_display_names)
    return [], []


# ===================================================
# PAGE 1 – MAIN DASHBOARD (NEW)
# ===================================================
if page == "📊 Dashboard":
    st.title("📊 Banff Parking – Main Dashboard")

    lot_features, lot_display_names = get_lot_features()
    if not lot_features:
        st.error("No parking-lot indicator features (starting with 'Unit_') were found.")
    else:
        # --- Controls row ---------------------------------------------------
        with st.container():
            c1, c2, c3, c4 = st.columns([1.2, 0.8, 1, 1])

            with c1:
                sel_date = st.date_input(
                    "Date",
                    value=date(2025, 7, 15),
                    help="Pick a calendar date – we use it to get month and weekday."
                )
                day_of_week = sel_date.weekday()  # 0=Mon, 6=Sun
                month = sel_date.month
                st.caption(sel_date.strftime("Selected: %A, %d %b %Y"))

            with c2:
                hour = st.slider("Hour (0–23)", 0, 23, 14)

            with c3:
                max_temp = st.slider("Max Temp (°C)", -20.0, 40.0, 22.0)

            with c4:
                total_precip = st.slider("Total Precip (mm)", 0.0, 30.0, 0.5)
                wind_gust = st.slider("Max Gust (km/h)", 0.0, 100.0, 12.0)

        is_weekend = 1 if day_of_week in [5, 6] else 0

        # --- Compute predictions for all lots (similar to Lot Status page) --
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
            lot_input = base_input.copy()
            if lot_feat in lot_input:
                lot_input[lot_feat] = 1

            x_vec = np.array([lot_input[f] for f in FEATURES]).reshape(1, -1)
            x_scaled = scaler.transform(x_vec)

            occ_pred = best_xgb_reg.predict(x_scaled)[0]
            full_prob = best_xgb_cls.predict_proba(x_scaled)[0, 1]

            if full_prob > 0.7:
                status = "High risk"
            elif full_prob > 0.4:
                status = "Busy"
            else:
                status = "Comfortable"

            rows.append(
                {
                    "Lot": lot_name,
                    "Predicted occupancy": occ_pred,
                    "Probability full": full_prob,
                    "Status": status,
                }
            )

        df = pd.DataFrame(rows).sort_values("Lot")

        # --- KPI cards ------------------------------------------------------
        high_risk = (df["Status"] == "High risk").sum()
        busy = (df["Status"] == "Busy").sum()
        comfy = (df["Status"] == "Comfortable").sum()
        busiest_row = df.sort_values("Probability full", ascending=False).iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("High-risk lots", high_risk)
        k2.metric("Busy lots", busy)
        k3.metric("Comfortable lots", comfy)
        k4.metric(
            "Busiest lot right now",
            busiest_row["Lot"],
            f"{busiest_row['Probability full']:.0%}",
        )

        st.markdown("---")

        # --- Tabs for different views --------------------------------------
        tab1, tab2 = st.tabs(["Top risk lots", "All lots table"])

        with tab1:
            top_df = df.sort_values("Probability full", ascending=False).head(10)
            st.subheader("Top 10 lots by probability of being full")

            fig, ax = plt.subplots()
            ax.barh(top_df["Lot"], top_df["Probability full"])
            ax.invert_yaxis()
            ax.set_xlabel("Probability full")
            ax.set_xlim(0, 1)
            st.pyplot(fig)

        with tab2:
            st.subheader("All lots – status snapshot")
            st.dataframe(
                df.style.format(
                    {"Predicted occupancy": "{:.2f}",
                     "Probability full": "{:.1%}"}
                ),
                use_container_width=True,
            )

# ===================================================
# PAGE 2 – MAKE PREDICTION (WITH CALENDAR)
# ===================================================
if page == "🎯 Make Prediction":
    st.title("🎯 What-If Prediction for a Single Lot")

    st.caption("Choose a lot, pick a date & hour, set weather, and see risk for that lot.")

    lot_features, lot_display_names = get_lot_features()

    if not lot_features:
        st.warning(
            "No parking-lot indicator features (starting with 'Unit_') were "
            "found in FEATURES. Lot selection is disabled."
        )
    else:
        scenario_options = {
            "Custom (use controls)": None,
            "Sunny Weekend Midday": {"month": 7, "dow": 5, "hour": 13,
                                     "max_temp": 24.0, "precip": 0.0, "gust": 10.0},
            "Rainy Weekday Afternoon": {"month": 6, "dow": 2, "hour": 16,
                                        "max_temp": 15.0, "precip": 5.0, "gust": 20.0},
            "Cold Morning (Shoulder Season)": {"month": 5, "dow": 1, "hour": 9,
                                               "max_temp": 5.0, "precip": 0.0, "gust": 15.0},
            "Warm Evening (Busy Day)": {"month": 8, "dow": 6, "hour": 19,
                                        "max_temp": 22.0, "precip": 0.0, "gust": 8.0},
        }

        c_lot, c_scen = st.columns([1.2, 1])
        with c_lot:
            selected_lot_label = st.selectbox("Parking lot", lot_display_names, index=0)
            selected_lot_feature = lot_features[lot_display_names.index(selected_lot_label)]

        with c_scen:
            selected_scenario = st.selectbox("Scenario", list(scenario_options.keys()), index=1)

        default_vals = {"month": 7, "dow": 5, "hour": 13,
                        "max_temp": 22.0, "precip": 0.5, "gust": 12.0}
        if scenario_options[selected_scenario] is not None:
            default_vals.update(scenario_options[selected_scenario])

        col1, col2 = st.columns(2)

        with col1:
            sel_date = st.date_input(
                "Date",
                value=date(2025, int(default_vals["month"]), 15)
            )
            hour = st.slider("Hour (0–23)", 0, 23, int(default_vals["hour"]))
            month = sel_date.month
            day_of_week = sel_date.weekday()
            is_weekend = 1 if day_of_week in [5, 6] else 0
            st.caption(sel_date.strftime("Selected: %A, %d %b"))

        with col2:
            max_temp = st.slider("Max Temp (°C)", -20.0, 40.0, float(default_vals["max_temp"]))
            total_precip = st.slider("Total Precip (mm)", 0.0, 30.0, float(default_vals["precip"]))
            wind_gust = st.slider("Max Gust (km/h)", 0.0, 100.0, float(default_vals["gust"]))

        st.caption(
            "Lag features (previous-hour occupancy, rolling averages) are handled by the model "
            "and are not entered manually."
        )

        # Build feature dict
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

        if st.button("🔮 Predict"):
            occ_pred = best_xgb_reg.predict(x_scaled)[0]
            full_prob = best_xgb_cls.predict_proba(x_scaled)[0, 1]

            r1, r2 = st.columns(2)
            with r1:
                st.metric("Predicted occupancy (model units)", f"{occ_pred:.2f}")
            with r2:
                st.metric("Probability lot is near full", f"{full_prob:.1%}")

            if full_prob > 0.7:
                st.warning("⚠️ High risk this lot will be full – consider redirecting drivers.")
            elif full_prob > 0.4:
                st.info("Moderate risk – worth monitoring.")
            else:
                st.success("Low risk of being at full capacity for this hour.")

# ===================================================
# PAGE 3 – LOT STATUS OVERVIEW (ALL LOTS)
# ===================================================
if page == "🗺️ Lot Status Overview":
    st.title("🗺️ Lot Status Overview – All Lots at Once")

    lot_features, lot_display_names = get_lot_features()
    if not lot_features:
        st.error(
            "No parking-lot indicator features (starting with 'Unit_') were "
            "found in FEATURES. This view needs those to work."
        )
    else:
        c1, c2, c3 = st.columns([1.2, 1, 1])

        with c1:
            sel_date = st.date_input("Date", value=date(2025, 7, 15))
            month = sel_date.month
            day_of_week = sel_date.weekday()
            is_weekend = 1 if day_of_week in [5, 6] else 0
            st.caption(sel_date.strftime("Selected: %A, %d %b"))

        with c2:
            hour = st.slider("Hour (0–23)", 0, 23, 14)
            max_temp = st.slider("Max Temp (°C)", -20.0, 40.0, 22.0)

        with c3:
            total_precip = st.slider("Total Precip (mm)", 0.0, 30.0, 0.5)
            wind_gust = st.slider("Max Gust (km/h)", 0.0, 100.0, 12.0)

        st.caption(
            "Lag features (previous-hour occupancy, rolling averages) are set to 0 "
            "for this quick overview."
        )

        if st.button("Compute lot status"):
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

            df = pd.DataFrame(rows).sort_values("Lot")

            def lot_status_row_style(row):
                if "High risk" in row["Status"]:
                    return ["background-color: #ffe5e5"] * len(row)
                elif "Busy" in row["Status"]:
                    return ["background-color: #fff4e0"] * len(row)
                else:
                    return ["background-color: #e9f7ef"] * len(row)

            styled_df = (
                df.style
                .format(
                    {"Predicted occupancy": "{:.2f}",
                     "Probability full": "{:.1%}"}
                )
                .apply(lot_status_row_style, axis=1)
            )

            st.subheader("Lot status for selected hour")
            st.dataframe(styled_df, use_container_width=True)

# ===================================================
# PAGE 4 – XAI (EXPLAINABLE AI)
# ===================================================
if page == "🔍 XAI – Explainable AI":
    st.title("🔍 Explainable AI – Model Insights")

    st.markdown(
        "Global explanations for the occupancy regression model "
        "using SHAP and Partial Dependence."
    )

    st.subheader("SHAP Summary – Regression Model (Occupancy)")
    try:
        explainer_reg = shap.TreeExplainer(best_xgb_reg)
        shap_values_reg = explainer_reg.shap_values(X_test_scaled)

        fig1, _ = plt.subplots()
        shap.summary_plot(
            shap_values_reg,
            X_test_scaled,
            feature_names=FEATURES,
            show=False
        )
        st.pyplot(fig1)

        st.subheader("SHAP Feature Importance – Regression")
        fig2, _ = plt.subplots()
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

    st.subheader("Partial Dependence – Key Features")
    pd_feature_names = [n for n in ["Max Temp (°C)", "Month", "Hour"] if n in FEATURES]

    if pd_feature_names:
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
    else:
        st.info("Configured PDP features not found in FEATURES list.")

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
    except Exception as e:
        st.error(f"Could not compute residuals: {e}")

# ===================================================
# PAGE 5 – CHAT ASSISTANT (RAG)
# ===================================================
if page == "💬 Chat Assistant (RAG)":
    st.title("💬 Banff Parking Chat Assistant (RAG)")

    st.caption(
        "Ask questions about parking patterns, busy times, or how the model works. "
        "The assistant uses your `banff_knowledge.txt` file as context."
    )

    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []

    for msg in st.session_state.rag_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
        "Edit `banff_knowledge.txt` in your repo to control what the chatbot knows "
        "about your EDA, feature engineering, and model findings."
    )
