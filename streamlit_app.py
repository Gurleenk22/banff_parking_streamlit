import os
from datetime import date

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# BASIC PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Banff Parking – ML & XAI Dashboard",
    layout="wide",
)

# ---------------------------------------------------
# LOAD MODELS + DATA (CACHED)
# ---------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Load trained models, scaler, feature list, and test data.

    Regression model is mandatory. Classification model is optional.
    """
    try:
        reg = joblib.load("banff_best_xgb_reg.pkl")          # XGBoost regressor
        scaler = joblib.load("banff_scaler.pkl")             # Scaler used in training
        features = joblib.load("banff_features.pkl")         # List of feature names
        X_test_scaled = np.load("X_test_scaled.npy")         # For XAI / residuals
        y_reg_test = np.load("y_reg_test.npy")               # True y for residuals
    except Exception as e:
        return None, None, None, [], None, None, f"Core model files missing: {e}"

    cls = None
    cls_error = None
    try:
        cls = joblib.load("banff_best_lgbm_cls.pkl")         # LightGBM classifier
    except Exception as e:
        cls_error = f"Classification model not loaded: {e}"

    return reg, cls, scaler, features, X_test_scaled, y_reg_test, cls_error


best_xgb_reg, best_cls_model, scaler, FEATURES, X_test_scaled, y_reg_test, CLS_ERROR = (
    load_models_and_data()
)

if best_xgb_reg is None or scaler is None or not FEATURES:
    st.error("❌ Could not load core model files. Check filenames in your repo.")
else:
    if CLS_ERROR:
        st.warning(
            CLS_ERROR
            + " – classification-based probabilities will be unavailable."
        )

# ---------------------------------------------------
# LOAD CAPACITY FROM RAW CSV (FOR REALISTIC INPUTS)
# ---------------------------------------------------
@st.cache_resource
def load_capacity_map():
    """
    Returns a dict {Unit name -> median Capacity} from the engineered hourly CSV.
    Used so that the app passes realistic Capacity values into the models.
    """
    path = "banff_parking_engineered_HOURLY (1).csv"
    try:
        df = pd.read_csv(path)
        if "Unit" in df.columns and "Capacity" in df.columns:
            cap_map = df.groupby("Unit")["Capacity"].median().to_dict()
            return cap_map
    except Exception:
        pass
    return {}

CAPACITY_MAP = load_capacity_map()

# ---------------------------------------------------
# OPENAI CLIENT (SAFE – WON'T CRASH IF KEY MISSING)
# ---------------------------------------------------
@st.cache_resource
def get_openai_client():
    """
    Try to create an OpenAI client.

    Looks for OPENAI_API_KEY in st.secrets first, then in environment variables.
    Returns None if no key is found so the app can fall back gracefully.
    """
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)

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
            "This is Gurleen's Banff parking assistant. The banff_knowledge.txt file "
            "is missing, so answers are based only on general parking logic."
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
        return (
            "No strong matches in the knowledge base. "
            "Answer based on general parking logic."
        )

    return "\n".join(selected)


def generate_chat_answer(user_question, chat_history):
    """
    Calls OpenAI with retrieved context + short chat history.
    If the API fails or no key is set, fall back to a simple
    answer built only from the retrieved context.
    """
    docs, vectorizer, doc_embeddings = load_rag_knowledge()
    context = retrieve_context(user_question, docs, vectorizer, doc_embeddings, k=5)

    client = get_openai_client()
    fallback = (
        "I couldn’t contact the language-model service right now "
        "(this usually means the OpenAI API key is missing or the quota is used up).\n\n"
        "Here is the most relevant information I can give based only on "
        "the project notes:\n\n"
        f"{context}"
    )

    if client is None:
        return fallback

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
        return fallback

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.title("Banff Parking Dashboard")
st.sidebar.markdown(
    """
Use this app to:

- Explore hourly parking demand  
- Check which lots may be full  
- Understand the model using XAI  
- Chat with a **parking assistant** using RAG  
"""
)

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "App Guide – What This Dashboard Does",
        "Make Prediction",
        "Lot Status Overview",
        "XAI – Explainable AI",
        "💬 Chat Assistant (RAG)",
    ],
)

# ---------------------------------------------------
# PAGE 1 – OVERVIEW
# ---------------------------------------------------
if page == "Overview":
    st.title("🚗 Banff Parking Demand – Machine Learning Overview")

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown(
            """
### Project Question

**How can Banff use real data to anticipate parking pressure and avoid full lots during the May–September tourist season?**

This project combines:

- **Parking management data** – when and where people park  
- **Weather data** – temperature, rain, and wind  
- **Engineered features** – hour, weekday/weekend, lagged occupancy, rolling averages  

A Gradient-boosted tree model (**XGBoost**) predicts:

- Hourly **occupancy level** for each lot  
- **Probability that a lot is near full** (> 90% capacity)  
"""
        )

    with col_right:
        st.markdown("### Quick Facts (from engineered data)")
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st.metric("Tourist season", "May–September 2025")
        with kpi2:
            st.metric("Lots modelled", "Multiple Banff units")
        kpi3, kpi4 = st.columns(2)
        with kpi3:
            st.metric("Target 1", "Hourly occupancy")
        with kpi4:
            st.metric("Target 2", "Full / Not-full")

        st.markdown(
            """
✅ Models trained on **historical hourly data**  
✅ Includes **time, weather, and history** features  
✅ Deployed as this **Streamlit decision-support app**
"""
        )

    st.markdown("---")

    st.info(
        "Tip: move between pages using the left sidebar. "
        "Start with **App Guide** if you want a walkthrough."
    )

# ---------------------------------------------------
# PAGE 2 – APP GUIDE
# ---------------------------------------------------
if page == "App Guide – What This Dashboard Does":
    st.title("📘 App Guide – What Each Page Shows")

    st.markdown(
        """
This page is a simple tour of the dashboard and how each page supports
your Banff parking problem.
"""
    )

    st.subheader("1️⃣ Make Prediction – “What if” for one lot")
    st.markdown(
        """
- Pick a **lot** and a **date/time**  
- Adjust basic weather conditions  
- See predicted occupancy and chance the lot is near full  
"""
    )

    st.subheader("2️⃣ Lot Status Overview – Compare all lots")
    st.markdown(
        """
- Choose one **date/time**  
- See **every lot** with predicted occupancy and risk level  
- Colours highlight high-risk vs comfortable lots  
"""
    )

    st.subheader("3️⃣ XAI – Explainable AI")
    st.markdown(
        """
- SHAP plots show which features (hour, month, weather, history) matter most  
- PDPs show how occupancy changes across time or temperature  
- Residual plot shows how close predictions are to true values  
"""
    )

    st.subheader("4️⃣ Chat Assistant (RAG)")
    st.markdown(
        """
- Ask questions in plain English about the project  
- Answers are grounded in `banff_knowledge.txt` plus the OpenAI model  
"""
    )

# ---------------------------------------------------
# SMALL HELPER: FILL CAPACITY
# ---------------------------------------------------
def fill_capacity_for_lot(lot_label: str) -> float:
    """Return a sensible capacity for a given lot label."""
    if CAPACITY_MAP and lot_label in CAPACITY_MAP:
        return float(CAPACITY_MAP[lot_label])

    # Fallback: median over all lots or default 50
    if CAPACITY_MAP:
        return float(np.median(list(CAPACITY_MAP.values())))
    return 50.0

# ---------------------------------------------------
# SMALL HELPER: BUILD CLASSIFICATION INPUT
# ---------------------------------------------------
def build_cls_input_from_base(base_input: dict, occ_pred: float) -> dict:
    """
    Take base_input (time, weather, lot, capacity, lags=0)
    and fill occupancy-related lag / rolling features using the
    regression prediction so the classifier sees a realistic
    'already busy' state.
    """
    cls_input = base_input.copy()
    for feat in FEATURES:
        if "Occupancy" in feat and feat in cls_input:
            # Use same occupancy value as proxy for all occupancy-based features
            cls_input[feat] = occ_pred
    return cls_input

# ---------------------------------------------------
# PAGE 3 – MAKE PREDICTION
# ---------------------------------------------------
if page == "Make Prediction":
    st.title("🎯 Interactive Parking Demand Prediction")

    if best_xgb_reg is None or scaler is None or not FEATURES:
        st.warning("Models are not available – cannot run predictions.")
    else:
        st.markdown(
            """
Use this page to explore *what-if* scenarios for a **single Banff parking lot**.
"""
        )

        # Find lot indicator features (one-hot encoded units)
        lot_features = [f for f in FEATURES if f.startswith("Unit_")]
        lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

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

        st.subheader("Step 1 – Choose lot & day")

        col_lot, col_date = st.columns([1.2, 1])

        with col_lot:
            if lot_features:
                selected_lot_label = st.selectbox(
                    "Select parking lot",
                    lot_display_names,
                    index=0,
                )
                selected_lot_feature = lot_features[lot_display_names.index(selected_lot_label)]
            else:
                selected_lot_label = "Generic lot"
                selected_lot_feature = None

        with col_date:
            chosen_date = st.date_input(
                "Day (tourist season 2025)",
                value=date(2025, 7, 20),
                min_value=date(2025, 5, 1),
                max_value=date(2025, 9, 1),
            )

        month = chosen_date.month
        day_of_week = chosen_date.weekday()
        is_weekend = 1 if day_of_week in [5, 6] else 0

        st.subheader("Step 2 – Time & weather")

        col1, col2 = st.columns(2)
        with col1:
            hour = st.slider("Hour of Day (0–23)", 0, 23, 13)
        with col2:
            max_temp = st.slider("Max Temperature (°C)", -20.0, 40.0, 22.0)
            total_precip = st.slider("Total Precipitation (mm)", 0.0, 30.0, 0.5)
            wind_gust = st.slider("Speed of Max Gust (km/h)", 0.0, 100.0, 12.0)

        st.caption(
            "Lag features (previous-hour occupancy, rolling averages) are estimated "
            "automatically from the regression prediction so the classifier sees a"
            " realistic busy/quiet state."
        )

        # Build base feature dict starting from all zeros
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

        # Capacity – get realistic value from CSV
        if "Capacity" in base_input and selected_lot_label is not None:
            base_input["Capacity"] = fill_capacity_for_lot(selected_lot_label)

        # Lot indicator – one-hot
        if selected_lot_feature is not None and selected_lot_feature in base_input:
            base_input[selected_lot_feature] = 1

        # ---------- Regression prediction ----------
        x_vec_reg = np.array([base_input[f] for f in FEATURES]).reshape(1, -1)
        x_scaled_reg = scaler.transform(x_vec_reg)
        occ_pred = best_xgb_reg.predict(x_scaled_reg)[0]

        # ---------- Classification prediction (using occupancy-based lags) ----------
        full_prob = None
        if best_cls_model is not None:
            try:
                cls_input = build_cls_input_from_base(base_input, occ_pred)
                x_vec_cls = np.array([cls_input[f] for f in FEATURES]).reshape(1, -1)
                x_scaled_cls = scaler.transform(x_vec_cls)
                full_prob = best_cls_model.predict_proba(x_scaled_cls)[0, 1]
            except Exception:
                full_prob = None

        if st.button("🔮 Predict for this scenario"):
            st.subheader("Step 3 – Results")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric(
                    "Predicted occupancy (model units)",
                    f"{occ_pred:.2f}",
                )

            if full_prob is not None:
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
                    st.success("Low risk of the lot being at full capacity for this hour.")
            else:
                st.info(
                    "Classification model is not available, so only occupancy is shown."
                )

# ---------------------------------------------------
# PAGE 4 – LOT STATUS OVERVIEW (ALL LOTS)
# ---------------------------------------------------
if page == "Lot Status Overview":
    st.title("📊 Lot Status Overview – Which Lots Are Likely Full?")

    if best_xgb_reg is None or scaler is None or not FEATURES:
        st.warning("Models are not available – cannot run predictions.")
    else:
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
                "found in FEATURES. This view needs those to work."
            )
        else:
            st.subheader("Step 1 – Choose date, time & weather")

            col1, col2 = st.columns(2)

            with col1:
                chosen_date = st.date_input(
                    "Day",
                    value=date(2025, 7, 20),
                    min_value=date(2025, 5, 1),
                    max_value=date(2025, 9, 1),
                    key="lot_status_date",
                )
                hour = st.slider("Hour of Day", 0, 23, 14)

            with col2:
                max_temp = st.slider("Max Temperature (°C)", -20.0, 40.0, 22.0)
                total_precip = st.slider("Total Precipitation (mm)", 0.0, 30.0, 0.5)
                wind_gust = st.slider("Speed of Max Gust (km/h)", 0.0, 100.0, 12.0)

            month = chosen_date.month
            day_of_week = chosen_date.weekday()
            is_weekend = 1 if day_of_week in [5, 6] else 0

            st.caption(
                "Lag features are estimated from the regression prediction for each lot "
                "so the classifier can highlight realistic busy / full situations."
            )

            if st.button("Compute lot status"):
                rows = []

                for lot_feat, lot_name in zip(lot_features, lot_display_names):
                    # Base input per lot
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
                    if "Capacity" in base_input:
                        base_input["Capacity"] = fill_capacity_for_lot(lot_name)

                    if lot_feat in base_input:
                        base_input[lot_feat] = 1

                    # Regression prediction
                    x_vec_reg = np.array([base_input[f] for f in FEATURES]).reshape(1, -1)
                    x_scaled_reg = scaler.transform(x_vec_reg)
                    occ_pred = best_xgb_reg.predict(x_scaled_reg)[0]

                    # Classification prediction
                    full_prob = None
                    if best_cls_model is not None:
                        try:
                            cls_input = build_cls_input_from_base(base_input, occ_pred)
                            x_vec_cls = np.array(
                                [cls_input[f] for f in FEATURES]
                            ).reshape(1, -1)
                            x_scaled_cls = scaler.transform(x_vec_cls)
                            full_prob = best_cls_model.predict_proba(x_scaled_cls)[0, 1]
                        except Exception:
                            full_prob = None

                    if full_prob is not None:
                        if full_prob > 0.7:
                            status = "🟥 High risk full"
                        elif full_prob > 0.4:
                            status = "🟧 Busy"
                        else:
                            status = "🟩 Comfortable"
                    else:
                        status = "N/A"

                    rows.append(
                        {
                            "Lot": lot_name,
                            "Predicted occupancy": occ_pred,
                            "Probability full": (
                                full_prob if full_prob is not None else np.nan
                            ),
                            "Status": status,
                        }
                    )

                df = pd.DataFrame(rows).sort_values("Lot")

                def lot_status_row_style(row):
                    if "High risk" in row["Status"]:
                        return ["background-color: #ffe5e5"] * len(row)
                    elif "Busy" in row["Status"]:
                        return ["background-color: #fff4e0"] * len(row)
                    elif "Comfortable" in row["Status"]:
                        return ["background-color: #e9f7ef"] * len(row)
                    else:
                        return [""] * len(row)

                styled_df = (
                    df.style.format(
                        {
                            "Predicted occupancy": "{:.2f}",
                            "Probability full": "{:.1%}",
                        }
                    ).apply(lot_status_row_style, axis=1)
                )

                st.subheader("Step 2 – Lot status for selected hour")
                st.dataframe(styled_df, use_container_width=True)

# ---------------------------------------------------
# PAGE 5 – XAI (EXPLAINABLE AI)
# ---------------------------------------------------
if page == "XAI – Explainable AI":
    st.title("🔍 Explainable AI – Understanding the Models")

    if (
        best_xgb_reg is None
        or scaler is None
        or not FEATURES
        or X_test_scaled is None
        or y_reg_test is None
    ):
        st.warning("XAI cannot run because model or test data is missing.")
    else:
        st.markdown(
            """
This page explains **why** the models make their predictions.
"""
        )

        # SHAP – Regression
        st.subheader("SHAP Summary – Regression Model (Occupancy)")
        try:
            explainer_reg = shap.TreeExplainer(best_xgb_reg)
            shap_values_reg = explainer_reg.shap_values(X_test_scaled)

            fig1, ax1 = plt.subplots()
            shap.summary_plot(
                shap_values_reg,
                X_test_scaled,
                feature_names=FEATURES,
                show=False,
            )
            st.pyplot(fig1)

            st.subheader("SHAP Feature Importance – Regression")
            fig2, ax2 = plt.subplots()
            shap.summary_plot(
                shap_values_reg,
                X_test_scaled,
                feature_names=FEATURES,
                plot_type="bar",
                show=False,
            )
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Could not generate SHAP plots: {e}")

        # PDPs
        st.subheader("Partial Dependence – Key Features")
        pd_feature_names = [
            name for name in ["Max Temp (°C)", "Month", "Hour"] if name in FEATURES
        ]
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
        else:
            st.info(
                "Configured PDP features ('Max Temp (°C)', 'Month', 'Hour') "
                "were not found in FEATURES."
            )

        # Residuals
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
2. Then it uses an OpenAI model to answer, grounded in that context  
"""
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
                    user_input, st.session_state.rag_chat_history
                )
                st.markdown(answer)

        st.session_state.rag_chat_history.append(
            {"role": "assistant", "content": answer}
        )

    st.caption(
        "Tip: edit `banff_knowledge.txt` in your repo to control what the chatbot knows "
        "about your EDA, feature engineering, and model findings."
    )
