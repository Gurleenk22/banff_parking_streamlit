import os
import warnings
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics.pairwise import cosine_similarity
import joblib

warnings.filterwarnings("ignore")

# =========================================================
# -------- BASIC PAGE CONFIG + CUSTOM STYLING -------------
# =========================================================
st.set_page_config(page_title="Banff Parking – ML & XAI Dashboard", layout="wide")

# Custom CSS: font, colours, cards, nav buttons
st.markdown(
    """
<style>
body {
    font-family: "Poppins", sans-serif;
}
h1, h2, h3, h4 {
    font-family: "Poppins", sans-serif;
    font-weight: 700;
}

/* Top nav */
.nav-button > button {
    width: 100%;
    border-radius: 999px !important;
    padding: 10px 0 !important;
    font-weight: 600 !important;
    border: none !important;
    color: white !important;
    background: linear-gradient(135deg, #4F46E5, #EC4899) !important;
}
.nav-button-active > button {
    width: 100%;
    border-radius: 999px !important;
    padding: 10px 0 !important;
    font-weight: 700 !important;
    border: none !important;
    color: #111827 !important;
    background: #FBBF24 !important;
}

/* Generic buttons */
.stButton > button {
    border-radius: 10px !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
}

/* Metric card */
.metric-card {
    background-color: #EEF2FF;
    border-radius: 14px;
    padding: 14px;
}

/* Small pill badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    background-color: #E5E7EB;
    color: #111827;
}

/* Table row colours will still come from pandas Styler */
</style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# ----------------- POWER BI DASHBOARD URL ----------------
# =========================================================
POWERBI_EMBED_URL = (
    "https://norquest-my.sharepoint.com/:u:/g/personal/"
    "mcranton312_norquest_ca/IQCGmJFaQHgmSZMFXLzhHNCDAZjUw8zWjltT7xmGbJh-Buw?download=1"
)
# (If Power BI complains about embedding, you may need the 'Publish to web' URL instead.)

# =========================================================
# ---------------- LOAD MODELS & DATA (CACHED) ------------
# =========================================================
@st.cache_resource
def load_models_and_data():
    """Load XGBoost regression model + scaler + features + test data.

    Classification model is optional; if it fails (e.g., lightgbm missing),
    we just return None and use rule-based risk.
    """
    try:
        reg = joblib.load("banff_best_xgb_reg.pkl")
        scaler = joblib.load("banff_scaler.pkl")
        features = joblib.load("banff_features.pkl")
        X_test_scaled = np.load("X_test_scaled.npy")
        y_reg_test = np.load("y_reg_test.npy")
    except Exception as e:
        raise RuntimeError(f"Core model files could not be loaded: {e}")

    cls = None
    cls_error = None
    # OPTIONAL classification model – safe failure
    try:
        cls = joblib.load("banff_best_lgbm_cls.pkl")
    except Exception as e:
        cls_error = f"Classification model not loaded (using rule-based risk instead): {e}"

    return reg, cls, scaler, features, X_test_scaled, y_reg_test, cls_error


try:
    best_xgb_reg, best_cls_opt, scaler, FEATURES, X_test_scaled, y_reg_test, CLS_ERROR = (
        load_models_and_data()
    )
    MODEL_ERROR = None
except Exception as e:
    best_xgb_reg = None
    best_cls_opt = None
    scaler = None
    FEATURES = []
    X_test_scaled = None
    y_reg_test = None
    CLS_ERROR = None
    MODEL_ERROR = str(e)

# =========================================================
# ----------------------- OPENAI CLIENT -------------------
# =========================================================
@st.cache_resource
def get_openai_client():
    """Create OpenAI client using Streamlit secrets or environment variable."""
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# =========================================================
# -------------------- RAG KNOWLEDGE BASE -----------------
# =========================================================
@st.cache_resource
def load_rag_knowledge():
    """Load banff_knowledge.txt and build TF-IDF embeddings."""
    path = "banff_knowledge.txt"
    if not os.path.exists(path):
        docs = [
            "This is the Banff parking assistant. The knowledge file "
            "is missing, so answers are based only on general parking "
            "logic and common-sense patterns in tourist season 2025."
        ]
    else:
        with open(path, "r", encoding="utf-8") as f:
            docs = [ln.strip() for ln in f.readlines() if ln.strip()]

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_embeddings = vectorizer.fit_transform(docs)
    return docs, vectorizer, doc_embeddings


def retrieve_context(query, docs, vectorizer, doc_embeddings, k=5):
    """Return top-k most relevant lines from knowledge base."""
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_embeddings).flatten()
    top_idx = sims.argsort()[::-1][:k]
    selected = [docs[i] for i in top_idx if sims[i] > 0.0]
    if not selected:
        return (
            "No strong matches in project notes. Answer based on typical "
            "parking patterns for Banff tourist season."
        )
    return "\n".join(selected)


def generate_chat_answer(user_question, chat_history):
    """Use RAG + OpenAI (if available). Fallback: context-only answer."""
    docs, vectorizer, doc_embeddings = load_rag_knowledge()
    context = retrieve_context(user_question, docs, vectorizer, doc_embeddings, k=5)

    fallback = (
        "I couldn't reach the language model service, so here is the "
        "best answer I can give based on the project notes:\n\n"
        f"{context}"
    )

    client = get_openai_client()
    if client is None:
        return (
            fallback
            + "\n\n(OPENAI_API_KEY is not set, so only RAG context is used.)"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly assistant helping explain a Banff parking "
                "forecasting project. The key questions are:\n"
                "1) Which factors (time of day, day of week, weather, 2025 trends) "
                "predict parking demand?\n"
                "2) For a given lot and hour, how likely is it to be near capacity "
                "(> 90% full)?\n\n"
                "Always ground your answer in the provided context. "
                "Keep explanations short and clear."
            ),
        },
        {
            "role": "system",
            "content": f"Context from project notes:\n{context}",
        },
    ]

    # Brief history (last 4 messages)
    for m in chat_history[-4:]:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_question})

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Do not crash the app if OpenAI fails
        return fallback


# =========================================================
# ------------- RISK LABEL CALCULATION --------------------
# =========================================================
def compute_risk_label(occ_pred, capacity, cls_prob=None):
    """Combine occupancy ratio and (optional) classifier prob into risk bands."""
    if capacity is None or capacity <= 0:
        capacity = 80.0

    ratio = float(occ_pred) / float(capacity)
    ratio = max(0.0, min(ratio, 2.0))

    # Rule-based probability: 0 below 70%, then grows to 1 at 95%
    if ratio <= 0.7:
        rule_prob = 0.0
    elif ratio >= 0.95:
        rule_prob = 1.0
    else:
        rule_prob = (ratio - 0.7) / (0.95 - 0.7)

    if cls_prob is not None:
        prob = 0.5 * rule_prob + 0.5 * cls_prob
    else:
        prob = rule_prob

    if prob >= 0.85:
        label = "🔴 Very high – almost full"
        band = "very_high"
    elif prob >= 0.6:
        label = "🟥 High – strong chance of > 90% capacity"
        band = "high"
    elif prob >= 0.35:
        label = "🟧 Medium – could get busy"
        band = "medium"
    else:
        label = "🟩 Low – comfortable capacity"
        band = "low"

    return prob, ratio, label, band


# =========================================================
# ------------------------ NAVIGATION ---------------------
# =========================================================
PAGES = ["Dashboard", "Make Prediction", "Lot Status", "XAI", "Chat"]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard"


def render_top_nav():
    st.markdown("## 🚗 Banff Parking – Machine Learning & XAI Dashboard")

    cols = st.columns(len(PAGES))
    for i, page_name in enumerate(PAGES):
        key = f"nav-{page_name}"
        active = st.session_state["current_page"] == page_name
        with cols[i]:
            klass = "nav-button-active" if active else "nav-button"
            st.markdown(
                f'<div class="{klass}">', unsafe_allow_html=True
            )
            if st.button(page_name, key=key):
                st.session_state["current_page"] = page_name
            st.markdown("</div>", unsafe_allow_html=True)


render_top_nav()

# Global warning if models failed
if MODEL_ERROR:
    st.error(
        f"Core model files could not be loaded, so predictions/XAI pages will not work.\n\n"
        f"Details: {MODEL_ERROR}"
    )

if CLS_ERROR and not MODEL_ERROR:
    st.warning(CLS_ERROR)


# =========================================================
# --------------------- PAGE: DASHBOARD -------------------
# =========================================================
def show_dashboard():
    st.markdown(
        """
**Problem focus**

- 🔍 *Which factors (time, weekday/weekend, weather, 2025 seasonal trends) predict parking demand?*  
- 🕒 *For each lot and hour, how likely is it to be near capacity (> 90% full)?*
        """
    )

    # Top-level calendar + time controls describing a "typical scenario"
    st.markdown("### 🎛️ Choose a typical scenario")

    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])

    with c1:
        sel_date = st.date_input(
            "Day (tourist season 2025)",
            value=date(2025, 6, 15),
            min_value=date(2025, 5, 1),
            max_value=date(2025, 9, 1),
        )

    with c2:
        sel_hour = st.slider("Hour of day", 0, 23, 14)

    with c3:
        sel_temp = st.slider("Max temp (°C)", -5.0, 35.0, 22.0)

    with c4:
        sel_precip = st.slider("Total precip (mm)", 0.0, 20.0, 0.5)

    st.caption(
        "These controls describe a typical hour in 2025 used in your scenario pages."
    )

    st.markdown("---")

    st.markdown("### 📊 Power BI – Parking Overview (optional)")
    components.iframe(POWERBI_EMBED_URL, height=520, scrolling=True)

    st.markdown("---")
    st.markdown(
        """
<span class="badge">Hint</span> Use the tabs above to:
- **Make Prediction** – what-if for one lot  
- **Lot Status** – compare all lots for one hour  
- **XAI** – see which features matter most  
- **Chat** – ask natural-language questions using RAG + OpenAI  
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# --------------- COMMON INPUT BUILDING -------------------
# =========================================================
def build_feature_vector(
    month,
    day_of_week,
    hour,
    max_temp,
    total_precip,
    wind_gust,
    capacity,
    selected_lot_feature=None,
):
    """Build a 1-row feature vector in the correct order for the model."""
    base = {f: 0.0 for f in FEATURES}

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
    if "Capacity" in base:
        base["Capacity"] = capacity

    if selected_lot_feature is not None and selected_lot_feature in base:
        base[selected_lot_feature] = 1.0

    x_vec = np.array([base[f] for f in FEATURES]).reshape(1, -1)
    x_scaled = scaler.transform(x_vec)
    return x_scaled


# =========================================================
# ---------------- PAGE: MAKE PREDICTION ------------------
# =========================================================
def show_make_prediction():
    if MODEL_ERROR or best_xgb_reg is None:
        st.warning("Models are not available – cannot run predictions.")
        return

    st.markdown("### 🎯 What-if for one parking lot")

    # Lot list from one-hot features
    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]

    if lot_features:
        pairs = sorted(zip(lot_features, lot_display), key=lambda x: x[1])
        lot_features, lot_display = zip(*pairs)
        lot_features = list(lot_features)
        lot_display = list(lot_display)

    c1, c2, c3 = st.columns(3)

    with c1:
        selected_lot_label = (
            st.selectbox("Parking lot", lot_display)
            if lot_features
            else "Generic lot"
        )
        selected_lot_feature = (
            lot_features[lot_display.index(selected_lot_label)]
            if lot_features
            else None
        )

    with c2:
        chosen_date = st.date_input(
            "Day (2025 tourist season)",
            value=date(2025, 7, 20),
            min_value=date(2025, 5, 1),
            max_value=date(2025, 9, 1),
        )
        hour = st.slider("Hour", 0, 23, 13)

    with c3:
        capacity = st.slider("Lot capacity (stalls)", 10, 300, 80, step=5)

    c4, c5, c6 = st.columns(3)
    with c4:
        max_temp = st.slider("Max temp (°C)", -5.0, 35.0, 24.0)
    with c5:
        total_precip = st.slider("Precipitation (mm)", 0.0, 20.0, 0.0)
    with c6:
        wind_gust = st.slider("Max gust (km/h)", 0.0, 80.0, 15.0)

    month = chosen_date.month
    day_of_week = chosen_date.weekday()

    st.caption("Weekend and time-of-day are inferred from your chosen date & hour.")

    if st.button("🔮 Predict occupancy & risk"):
        x_scaled = build_feature_vector(
            month,
            day_of_week,
            hour,
            max_temp,
            total_precip,
            wind_gust,
            capacity,
            selected_lot_feature,
        )
        occ_pred = best_xgb_reg.predict(x_scaled)[0]

        # Optional classifier probability if available
        cls_prob = None
        if best_cls_opt is not None:
            try:
                cls_prob = best_cls_opt.predict_proba(x_scaled)[0, 1]
            except Exception:
                cls_prob = None

        prob_full, ratio, label, band = compute_risk_label(
            occ_pred, capacity, cls_prob
        )

        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Predicted occupancy (vehicles)",
                f"{occ_pred:.1f}",
                help="Regression model output for this lot & hour.",
            )
            st.metric(
                "Assumed capacity (stalls)",
                f"{capacity}",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c_right:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Occupancy ratio",
                f"{ratio*100:.1f}%",
                help="Predicted vehicles / capacity.",
            )
            st.metric(
                "Near-full risk (model + rule)",
                f"{prob_full*100:.1f}%",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.success(
            f"**Risk band:** {label}  "
            f"(lot: **{selected_lot_label}**, hour: **{hour}:00**, "
            f"date: **{chosen_date}**)"
        )


# =========================================================
# ---------------- PAGE: LOT STATUS OVERVIEW --------------
# =========================================================
def show_lot_status():
    if MODEL_ERROR or best_xgb_reg is None:
        st.warning("Models are not available – cannot run predictions.")
        return

    st.markdown("### 📊 Compare all lots for one hour")

    lot_features = [f for f in FEATURES if f.startswith("Unit_")]
    lot_display = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_features]
    if not lot_features:
        st.error("No unit one-hot features found (Unit_*).")
        return

    pairs = sorted(zip(lot_features, lot_display), key=lambda x: x[1])
    lot_features, lot_display = zip(*pairs)
    lot_features = list(lot_features)
    lot_display = list(lot_display)

    c1, c2, c3 = st.columns(3)

    with c1:
        chosen_date = st.date_input(
            "Day",
            value=date(2025, 7, 20),
            min_value=date(2025, 5, 1),
            max_value=date(2025, 9, 1),
            key="lot_status_date",
        )
        hour = st.slider("Hour", 0, 23, 14, key="lot_status_hour")

    with c2:
        max_temp = st.slider(
            "Max temp (°C)", -5.0, 35.0, 22.0, key="lot_status_temp"
        )
        total_precip = st.slider(
            "Precipitation (mm)", 0.0, 20.0, 0.5, key="lot_status_precip"
        )

    with c3:
        wind_gust = st.slider(
            "Max gust (km/h)", 0.0, 80.0, 12.0, key="lot_status_gust"
        )
        capacity_assumed = st.slider(
            "Assumed capacity per lot (stalls)", 10, 300, 80, step=5
        )

    month = chosen_date.month
    day_of_week = chosen_date.weekday()

    if st.button("Compute status for all lots"):
        rows = []
        for lf, name in zip(lot_features, lot_display):
            x_scaled = build_feature_vector(
                month,
                day_of_week,
                hour,
                max_temp,
                total_precip,
                wind_gust,
                capacity_assumed,
                lf,
            )
            occ_pred = best_xgb_reg.predict(x_scaled)[0]

            cls_prob = None
            if best_cls_opt is not None:
                try:
                    cls_prob = best_cls_opt.predict_proba(x_scaled)[0, 1]
                except Exception:
                    cls_prob = None

            prob_full, ratio, label, band = compute_risk_label(
                occ_pred, capacity_assumed, cls_prob
            )

            rows.append(
                {
                    "Lot": name,
                    "Predicted occupancy": occ_pred,
                    "Occupancy %": ratio * 100.0,
                    "Near-full risk %": prob_full * 100.0,
                    "Risk band": label,
                    "_band_code": band,
                }
            )

        df = pd.DataFrame(rows).sort_values(
            ["_band_code", "Near-full risk %"], ascending=[False, False]
        )

        def row_style(row):
            band = row["_band_code"]
            if band == "very_high":
                colour = "#FEE2E2"  # red
            elif band == "high":
                colour = "#FEF3C7"  # amber
            elif band == "medium":
                colour = "#E0F2FE"  # blue
            else:
                colour = "#DCFCE7"  # green
            return [f"background-color: {colour}"] * len(row)

        styled = (
            df.drop(columns=["_band_code"])
            .style.format(
                {
                    "Predicted occupancy": "{:.1f}",
                    "Occupancy %": "{:.1f}%",
                    "Near-full risk %": "{:.1f}%",
                }
            )
            .apply(row_style, axis=1)
        )

        st.dataframe(styled, use_container_width=True)


# =========================================================
# ---------------------- PAGE: XAI ------------------------
# =========================================================
def show_xai():
    if MODEL_ERROR or best_xgb_reg is None or X_test_scaled is None:
        st.warning("XAI cannot run because model or test data is missing.")
        return

    st.markdown("### 🔍 Explainable AI – which features matter?")

    # SHAP summary & bar plots
    st.subheader("SHAP summary – regression model (occupancy)")
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

        st.caption(
            "Each dot is one hour-lot observation. Position shows impact on prediction; colour shows feature value."
        )

        st.subheader("Top features by average impact")
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
        st.error(f"Could not compute SHAP plots: {e}")

    # Partial dependence for a few key features
    st.subheader("Partial dependence – time & weather")
    try:
        pd_features = [f for f in ["Hour", "Month", "Max Temp (°C)"] if f in FEATURES]
        if pd_features:
            idx = [FEATURES.index(f) for f in pd_features]
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            PartialDependenceDisplay.from_estimator(
                best_xgb_reg,
                X_test_scaled,
                idx,
                feature_names=FEATURES,
                ax=ax3,
            )
            st.pyplot(fig3)
            st.caption(
                "These curves show the average effect of each feature on occupancy while holding others fixed."
            )
        else:
            st.info(
                "Configured PDP features not found in FEATURES. "
                "Check your feature names."
            )
    except Exception as e:
        st.error(f"Could not compute partial dependence plots: {e}")

    # Residual plot
    st.subheader("Residual plot – regression fit")
    try:
        y_pred = best_xgb_reg.predict(X_test_scaled)
        residuals = y_reg_test - y_pred

        fig4, ax4 = plt.subplots()
        ax4.scatter(y_pred, residuals, alpha=0.3)
        ax4.axhline(0, color="red", linestyle="--")
        ax4.set_xlabel("Predicted occupancy")
        ax4.set_ylabel("Residual (actual – predicted)")
        st.pyplot(fig4)
        st.caption(
            "Residuals roughly centered around zero suggest the model captures the main patterns."
        )
    except Exception as e:
        st.error(f"Could not plot residuals: {e}")


# =========================================================
# ---------------------- PAGE: CHAT -----------------------
# =========================================================
def show_chat():
    st.markdown("### 💬 Banff Parking Chat Assistant (RAG + OpenAI)")

    st.caption(
        "Ask about busy hours, risk of being full, or which factors drive demand. "
        "Answers are grounded in your `banff_knowledge.txt` plus OpenAI."
    )

    if "rag_chat_history" not in st.session_state:
        st.session_state["rag_chat_history"] = []

    # Show history
    for msg in st.session_state["rag_chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about Banff parking or the model...")
    if user_input:
        st.session_state["rag_chat_history"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking with project context..."):
                answer = generate_chat_answer(
                    user_input, st.session_state["rag_chat_history"]
                )
                st.markdown(answer)

        st.session_state["rag_chat_history"].append(
            {"role": "assistant", "content": answer}
        )


# =========================================================
# ------------------------ ROUTER -------------------------
# =========================================================
page = st.session_state["current_page"]

if page == "Dashboard":
    show_dashboard()
elif page == "Make Prediction":
    show_make_prediction()
elif page == "Lot Status":
    show_lot_status()
elif page == "XAI":
    show_xai()
elif page == "Chat":
    show_chat()
