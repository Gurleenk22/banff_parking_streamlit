import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.inspection import PartialDependenceDisplay

# ==== RAG / Chatbot imports ====
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from datetime import date

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
    """
    Load trained models, scaler, feature list, and test data.
    Tries LightGBM classifier first; falls back to XGBoost if needed.
    """
    MODELS_OK = True
    reg = cls = scaler = features = X_test_scaled = y_reg_test = None
    error_msg = ""

    try:
        reg = joblib.load("banff_best_xgb_reg.pkl")
    except Exception as e:
        MODELS_OK = False
        error_msg += f"Error loading banff_best_xgb_reg.pkl: {e}\n"

    # Try LightGBM classifier first
    if MODELS_OK:
        try:
            cls = joblib.load("banff_best_lgbm_cls.pkl")
        except Exception:
            try:
                cls = joblib.load("banff_best_xgb_cls.pkl")
            except Exception as e:
                MODELS_OK = False
                error_msg += f"Error loading classifier model: {e}\n"

    if MODELS_OK:
        try:
            scaler = joblib.load("banff_scaler.pkl")
            features = joblib.load("banff_features.pkl")
            X_test_scaled = np.load("X_test_scaled.npy")
            y_reg_test = np.load("y_reg_test.npy")
        except Exception as e:
            MODELS_OK = False
            error_msg += f"Error loading scaler/features/test arrays: {e}\n"

    return MODELS_OK, error_msg, reg, cls, scaler, features, X_test_scaled, y_reg_test


MODELS_OK, MODEL_ERR, best_reg, best_cls, scaler, FEATURES, X_test_scaled, y_reg_test = load_models_and_data()

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
    If the API fails, fall back to a simple context-based answer.
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
            "I couldn’t contact the language-model service right now. "
            "Here is the most relevant information I can give based only on "
            "the project notes:\n\n"
            f"{context}"
        )

# ---------------------------------------------------
# HELPER: LOT STATUS WITH BETTER RISK THRESHOLDS
# ---------------------------------------------------
def compute_lot_status(month, dow, hour, max_temp, precip, gust):
    """
    Return occupancy & near-full probability for every lot.
    Uses better thresholds for imbalanced data:
    - High risk: p >= 0.08
    - Medium:   0.02 <= p < 0.08
    - Low:      p < 0.02
    Also sets lag/rolling occupancy features to a small typical value
    instead of 0 to avoid unrealistically tiny probabilities.
    """
    if not MODELS_OK:
        return pd.DataFrame()

    lot_feats = [f for f in FEATURES if f.startswith("Unit_")]
    lot_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_feats]
    if not lot_feats:
        return pd.DataFrame()

    # sort so BANFF02, BANFF03, … are in order
    pairs = sorted(zip(lot_feats, lot_names), key=lambda x: x[1])
    lot_feats, lot_names = zip(*pairs)
    lot_feats, lot_names = list(lot_feats), list(lot_names)

    is_weekend = 1 if dow in [5, 6] else 0
    base = {f: 0 for f in FEATURES}

    # time & weather
    if "Month" in base:
        base["Month"] = month
    if "DayOfWeek" in base:
        base["DayOfWeek"] = dow
    if "Hour" in base:
        base["Hour"] = hour
    if "IsWeekend" in base:
        base["IsWeekend"] = is_weekend
    if "Max Temp (°C)" in base:
        base["Max Temp (°C)"] = max_temp
    if "Total Precip (mm)" in base:
        base["Total Precip (mm)"] = precip
    if "Spd of Max Gust (km/h)" in base:
        base["Spd of Max Gust (km/h)"] = gust

    # Set lag / rolling occupancy features to a small typical value
    for f in FEATURES:
        if "Occupancy" in f and ("lag" in f or "roll" in f):
            base[f] = 3.0  # approx. typical number of vehicles

    rows = []
    for lf, name in zip(lot_feats, lot_names):
        lot_row = base.copy()
        if lf in lot_row:
            lot_row[lf] = 1

        x = np.array([lot_row[f] for f in FEATURES]).reshape(1, -1)
        x_scaled = scaler.transform(x)

        occ = best_reg.predict(x_scaled)[0]
        prob = best_cls.predict_proba(x_scaled)[0, 1]

        # new thresholds based on imbalanced data
        if prob >= 0.08:
            status = "🟥 High risk"
        elif prob >= 0.02:
            status = "🟧 Medium"
        else:
            status = "🟩 Low"

        rows.append(
            {
                "Lot": name,
                "Predicted occupancy": occ,
                "Probability full (>90%)": prob,
                "Status": status,
            }
        )

    return pd.DataFrame(rows).sort_values("Lot")


# ---------------------------------------------------
# SIMPLE TOP NAV BAR WITH BOXES
# ---------------------------------------------------
PAGES = [
    "Dashboard",
    "Explore Factors",
    "Forecast Single Lot",
    "Lot Status Overview",
    "XAI – Explainable AI",
    "Chat Assistant",
]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard"


def page_selector():
    st.markdown("### Banff Parking – ML Decision Support")

    cols = st.columns(len(PAGES))
    for idx, page_name in enumerate(PAGES):
        with cols[idx]:
            if st.button(page_name, use_container_width=True):
                st.session_state["current_page"] = page_name


page_selector()
page = st.session_state["current_page"]

# Show a warning at the top if models failed to load
if not MODELS_OK:
    st.error("Models or data failed to load. Check file names and paths.")
    st.code(MODEL_ERR, language="text")

# ---------------------------------------------------
# PAGE: DASHBOARD (POWER BI + SHORT OVERVIEW)
# ---------------------------------------------------
if page == "Dashboard":
    st.subheader("📊 High-Level Dashboard")

    # Problem statement cards
    c1, c2 = st.columns(2)
    with c1:
        st.info(
            "**Problem 1**  \n"
            "Which factors like **time of day, day of week, weather, and 2025 trends** "
            "are the most reliable predictors of parking demand?"
        )
    with c2:
        st.info(
            "**Problem 2**  \n"
            "Can we forecast, **hour by hour**, when a specific lot is most likely "
            "to be near capacity (> 90% full)?"
        )

    # Top-level quick filters (just for user feel)
    st.markdown("#### Quick Scenario Filters")
    cold1, cold2, cold3 = st.columns([1, 1, 2])

    with cold1:
        d = st.date_input("Date", value=date(2025, 7, 15))
    with cold2:
        hour = st.slider("Hour", 0, 23, 14)
    with cold3:
        max_temp = st.slider("Max Temp (°C)", -10.0, 35.0, 22.0)

    st.caption("Use the other pages (tabs above) to see the detailed forecasts and lot risk.")

    # Power BI dashboard embed (replace URL with your Power BI embed link if available)
    st.markdown("#### Embedded Power BI Dashboard")

    POWER_BI_URL = (
        "https://norquest-my.sharepoint.com/:u:/g/personal/"
        "mcranton312_norquest_ca/IQCGmJFaQHgmSZMFXLzhHNCDAZjUw8zWjltT7xmGbJh-Buw"
        "?e=1bLfFd"
    )

    st.markdown(
        f"""
        <iframe src="{POWER_BI_URL}"
                width="100%" height="600"
                style="border:none;">
        </iframe>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "If the dashboard does not display, use the official Power BI 'Embed' URL "
        "and replace the link in the code."
    )

# ---------------------------------------------------
# PAGE: EXPLORE FACTORS (FEATURE IMPORTANCE)
# ---------------------------------------------------
elif page == "Explore Factors":
    st.subheader("📌 Which Factors Matter Most?")

    if not MODELS_OK:
        st.stop()

    lot_features = [f for f in FEATURES if f.startswith("Unit_")]

    col_top = st.columns(3)
    with col_top[0]:
        st.metric("Total features", len(FEATURES))
    with col_top[1]:
        st.metric("Lot indicators", len(lot_features))
    with col_top[2]:
        st.metric("Test points", len(X_test_scaled))

    st.markdown("---")

    st.markdown("**1. Top features for predicting occupancy (regression)**")
    try:
        import seaborn as sns

        importances_reg = best_reg.feature_importances_
        df_imp_reg = pd.DataFrame(
            {"Feature": FEATURES, "Importance": importances_reg}
        ).sort_values("Importance", ascending=False)

        N = 15
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            x="Importance",
            y="Feature",
            data=df_imp_reg.head(N),
        )
        ax.set_title(f"Top {N} features – Hourly occupancy")
        st.pyplot(fig)

        st.caption(
            "Lagged occupancy features and time-of-day are usually the strongest "
            "predictors of how many vehicles are in the lot."
        )
    except Exception as e:
        st.error(f"Could not plot regression feature importance: {e}")

    st.markdown("**2. Top features for predicting near-full risk (classification)**")
    try:
        import seaborn as sns

        importances_cls = best_cls.feature_importances_
        df_imp_cls = pd.DataFrame(
            {"Feature": FEATURES, "Importance": importances_cls}
        ).sort_values("Importance", ascending=False)

        N = 15
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.barplot(
            x="Importance",
            y="Feature",
            data=df_imp_cls.head(N),
        )
        ax2.set_title(f"Top {N} features – Near-full risk (Is_Full)")
        st.pyplot(fig2)

        st.caption(
            "Recent occupancy (last hour and rolling averages), capacity, and specific "
            "lots play a big role in predicting when a lot is likely to be > 90% full."
        )
    except Exception as e:
        st.error(f"Could not plot classification feature importance: {e}")

# ---------------------------------------------------
# PAGE: FORECAST SINGLE LOT
# ---------------------------------------------------
elif page == "Forecast Single Lot":
    st.subheader("🎯 Forecast a Single Parking Lot")

    if not MODELS_OK:
        st.stop()

    # Date + sliders (calendar)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        chosen_date = st.date_input("Date", value=date(2025, 7, 15))
    with col_t2:
        hour = st.slider("Hour (0–23)", 0, 23, 14)

    month = chosen_date.month
    dow = chosen_date.weekday()  # 0=Mon, 6=Sun
    is_weekend = 1 if dow in [5, 6] else 0

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        max_temp = st.slider("Max Temp (°C)", -20.0, 40.0, 22.0)
    with col_w2:
        total_precip = st.slider("Total Precip (mm)", 0.0, 30.0, 0.5)
    with col_w3:
        wind_gust = st.slider("Max Gust (km/h)", 0.0, 100.0, 12.0)

    st.caption("Date picker gives **month** and **day of week** to the model; year is not used.")

    # Lot selection (sorted, nice names)
    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]
    if lot_features:
        lot_pairs = sorted(zip(lot_features, lot_display_names), key=lambda x: x[1])
        lot_features, lot_display_names = zip(*lot_pairs)
        lot_features, lot_display_names = list(lot_features), list(lot_display_names)

    if not lot_features:
        st.warning(
            "No features starting with 'Unit_' found in FEATURES. "
            "Lot selection is disabled."
        )
        selected_lot_feature = None
        selected_lot_label = "Selected lot"
    else:
        selected_lot_label = st.selectbox(
            "Parking lot", lot_display_names, index=0
        )
        selected_lot_feature = lot_features[lot_display_names.index(selected_lot_label)]

    # Build feature dict from zeros
    base_input = {f: 0 for f in FEATURES}
    if "Month" in base_input:
        base_input["Month"] = month
    if "DayOfWeek" in base_input:
        base_input["DayOfWeek"] = dow
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

    # Lag / rolling occupancy features set to small typical value
    for f in FEATURES:
        if "Occupancy" in f and ("lag" in f or "roll" in f):
            base_input[f] = 3.0

    # Lot one-hot
    if selected_lot_feature is not None and selected_lot_feature in base_input:
        base_input[selected_lot_feature] = 1

    x_vec = np.array([base_input[f] for f in FEATURES]).reshape(1, -1)
    x_scaled = scaler.transform(x_vec)

    if st.button("🔮 Predict this hour for this lot"):
        occ_pred = best_reg.predict(x_scaled)[0]
        prob = best_cls.predict_proba(x_scaled)[0, 1]

        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.metric(
                "Predicted occupancy (vehicles)",
                f"{occ_pred:.1f}",
            )
        with c_res2:
            st.metric(
                "Probability near full (>90%)",
                f"{prob:.1%}",
            )

        if prob >= 0.08:
            st.warning(
                "High risk – this lot is likely to be near capacity (>90%) at this hour."
            )
        elif prob >= 0.02:
            st.info(
                "Medium risk – this hour is busier than usual; the lot may feel crowded."
            )
        else:
            st.success(
                "Low risk – this lot is unlikely to be near capacity at this hour."
            )

# ---------------------------------------------------
# PAGE: LOT STATUS OVERVIEW
# ---------------------------------------------------
elif page == "Lot Status Overview":
    st.subheader("📍 Compare All Lots for One Hour")

    if not MODELS_OK:
        st.stop()

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        chosen_date = st.date_input("Date", value=date(2025, 7, 15), key="overview_date")
    with col_t2:
        hour = st.slider("Hour (0–23)", 0, 23, 14, key="overview_hour")

    month = chosen_date.month
    dow = chosen_date.weekday()
    is_weekend = 1 if dow in [5, 6] else 0

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        max_temp = st.slider("Max Temp (°C)", -20.0, 40.0, 22.0, key="overview_temp")
    with col_w2:
        total_precip = st.slider("Total Precip (mm)", 0.0, 30.0, 0.5, key="overview_precip")
    with col_w3:
        wind_gust = st.slider("Max Gust (km/h)", 0.0, 100.0, 12.0, key="overview_gust")

    if st.button("Compute lot status", type="primary"):
        df_status = compute_lot_status(
            month=month,
            dow=dow,
            hour=hour,
            max_temp=max_temp,
            precip=total_precip,
            gust=wind_gust,
        )

        if df_status.empty:
            st.warning("No lot features found. Check your FEATURES list.")
        else:
            def lot_status_row_style(row):
                if "High" in row["Status"]:
                    return ["background-color: #ffe5e5"] * len(row)  # light red
                elif "Medium" in row["Status"]:
                    return ["background-color: #fff4e0"] * len(row)  # light orange
                else:
                    return ["background-color: #e9f7ef"] * len(row)  # light green

            styled_df = (
                df_status.style
                .format(
                    {
                        "Predicted occupancy": "{:.2f}",
                        "Probability full (>90%)": "{:.1%}",
                    }
                )
                .apply(lot_status_row_style, axis=1)
            )

            st.dataframe(styled_df, use_container_width=True)
            st.caption(
                "Rows are sorted by lot name. Colours show risk: red = high, "
                "orange = medium, green = low."
            )

# ---------------------------------------------------
# PAGE: XAI – SHAP & PDP
# ---------------------------------------------------
elif page == "XAI – Explainable AI":
    st.subheader("🔍 Explainable AI – Why the Model Predicts This")

    if not MODELS_OK:
        st.stop()

    st.markdown(
        "Below you can see **SHAP** and **Partial Dependence** plots that show "
        "how time-of-day, weather, and history affect the predicted occupancy."
    )

    # ---------- SHAP EXPLANATIONS FOR REGRESSION ----------
    st.markdown("**1. SHAP summary – regression model (occupancy)**")
    try:
        explainer_reg = shap.TreeExplainer(best_reg)
        shap_values_reg = explainer_reg.shap_values(X_test_scaled)

        fig1, ax1 = plt.subplots()
        shap.summary_plot(
            shap_values_reg,
            X_test_scaled,
            feature_names=FEATURES,
            show=False
        )
        st.pyplot(fig1)
        st.caption(
            "Each point is one hour. Colour shows the feature value; left/right shows "
            "whether that feature pushed occupancy down or up."
        )

        st.markdown("**2. SHAP bar plot – overall feature importance**")
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
    st.markdown("**3. Partial dependence – key features**")

    pd_feature_names = []
    for name in ["Max Temp (°C)", "Month", "Hour"]:
        if name in FEATURES:
            pd_feature_names.append(name)

    if len(pd_feature_names) > 0:
        feature_indices = [FEATURES.index(f) for f in pd_feature_names]
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        PartialDependenceDisplay.from_estimator(
            best_reg,
            X_test_scaled,
            feature_indices,
            feature_names=FEATURES,
            ax=ax3
        )
        st.pyplot(fig3)
        st.caption(
            "These curves show how **Hour**, **Month**, or **Max Temp** change the "
            "predicted occupancy on average."
        )
    else:
        st.info(
            "Could not find the configured PDP features ('Max Temp (°C)', 'Month', 'Hour') "
            "in the FEATURES list."
        )

    # ---------- RESIDUAL ANALYSIS ----------
    st.markdown("**4. Residual plot – model fit**")

    try:
        y_pred = best_reg.predict(X_test_scaled)
        residuals = y_reg_test - y_pred

        fig4, ax4 = plt.subplots()
        ax4.scatter(y_pred, residuals, alpha=0.3)
        ax4.axhline(0, color="red", linestyle="--")
        ax4.set_xlabel("Predicted occupancy")
        ax4.set_ylabel("Residual (Actual - Predicted)")
        st.pyplot(fig4)
        st.caption(
            "Points scattered roughly around zero mean the model captures the main patterns "
            "without strong systematic error."
        )
    except Exception as e:
        st.error(f"Could not compute residuals: {e}")

# ---------------------------------------------------
# PAGE: CHAT ASSISTANT (RAG)
# ---------------------------------------------------
elif page == "Chat Assistant":
    st.subheader("💬 Banff Parking Chat Assistant (RAG)")

    st.markdown(
        "Ask questions about parking patterns, busy times, or why certain features matter. "
        "The assistant uses your `banff_knowledge.txt` plus a language model."
    )

    # Initialize chat history
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []

    # Show previous messages
    for msg in st.session_state.rag_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask something about Banff parking…")

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
        "Edit `banff_knowledge.txt` in your repo to control what the chatbot knows "
        "about your EDA, feature engineering, and model findings."
    )
