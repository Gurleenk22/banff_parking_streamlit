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

# ---- Optional libs ----
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import lightgbm as lgb  # noqa: F401
except Exception:
    lgb = None

# ---- CONFIG ----
st.set_page_config(page_title="Banff Parking – Dashboard", layout="wide")

POWERBI_EMBED_URL = ""  # put your Power BI public URL here if you want

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

# Simple background
st.markdown(
    """
    <style>
    .main { background-color:#f5f7fb; }
    .block-container { padding-top:1.2rem;padding-bottom:1.5rem; }
    section[data-testid="stSidebar"] { background:#ffffff;border-right:1px solid #e5e7eb; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# LOAD MODELS + DATA
# ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models_and_data():
    reg = joblib.load("banff_best_xgb_reg.pkl")
    cls = joblib.load("banff_best_lgbm_cls.pkl")
    scaler = joblib.load("banff_scaler.pkl")
    features = joblib.load("banff_features.pkl")
    X_test_scaled = np.load("X_test_scaled.npy")
    y_reg_test = np.load("y_reg_test.npy")
    return reg, cls, scaler, features, X_test_scaled, y_reg_test


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
except Exception as e:
    best_xgb_reg = None
    best_lgbm_cls = None
    scaler = None
    FEATURES = []
    X_test_scaled = None
    y_reg_test = None
    MODELS_OK = False
    MODEL_ERROR = str(e)

# ---------------------------------------------------
# RAG: KNOWLEDGE + CHAT
# ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_rag_knowledge():
    path = "banff_knowledge.txt"
    if not os.path.exists(path):
        docs = [
            "Banff parking assistant. banff_knowledge.txt is missing, "
            "so answers are generic."
        ]
    else:
        with open(path, "r", encoding="utf-8") as f:
            docs = [line.strip() for line in f if line.strip()]
    vec = TfidfVectorizer(stop_words="english")
    emb = vec.fit_transform(docs)
    return docs, vec, emb


def retrieve_context(query, docs, vec, emb, k=5):
    qv = vec.transform([query])
    sims = cosine_similarity(qv, emb).flatten()
    idx = sims.argsort()[::-1][:k]
    lines = [docs[i] for i in idx if sims[i] > 0.0]
    if not lines:
        return "No strong matches in project notes."
    return "\n".join(lines)


def generate_chat_answer(user_question, history):
    docs, vec, emb = load_rag_knowledge()
    ctx = retrieve_context(user_question, docs, vec, emb, k=5)

    # no API key: just return context
    if client is None:
        return "OpenAI key not set. Context from notes:\n\n" + ctx

    messages = [
        {
            "role": "system",
            "content": "Explain this Banff parking ML project simply.",
        },
        {
            "role": "system",
            "content": f"Project notes:\n{ctx}",
        },
    ]
    for h in history[-4:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_question})

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "API error. Here is info from notes:\n\n" + ctx


# ---------------------------------------------------
# Helper: lot status for all lots
# ---------------------------------------------------
def compute_lot_status(month, dow, hour, max_temp, precip, gust):
    if not MODELS_OK:
        return pd.DataFrame()

    lot_feats = [f for f in FEATURES if f.startswith("Unit_")]
    lot_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_feats]
    if not lot_feats:
        return pd.DataFrame()

    pairs = sorted(zip(lot_feats, lot_names), key=lambda x: x[1])
    lot_feats, lot_names = zip(*pairs)
    lot_feats, lot_names = list(lot_feats), list(lot_names)

    is_weekend = 1 if dow in [5, 6] else 0
    base = {f: 0 for f in FEATURES}
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

    rows = []
    for lf, name in zip(lot_feats, lot_names):
        row = base.copy()
        if lf in row:
            row[lf] = 1
        x = np.array([row[f] for f in FEATURES]).reshape(1, -1)
        x_scaled = scaler.transform(x)
        occ = best_xgb_reg.predict(x_scaled)[0]
        prob = best_lgbm_cls.predict_proba(x_scaled)[0, 1]
        if prob > 0.7:
            status = "🟥 High risk"
        elif prob > 0.4:
            status = "🟧 Busy"
        else:
            status = "🟩 OK"
        rows.append(
            {
                "Lot": name,
                "Predicted occupancy": occ,
                "Probability full": prob,
                "Status": status,
            }
        )
    return pd.DataFrame(rows).sort_values("Lot")


# ---------------------------------------------------
# SIDEBAR NAV
# ---------------------------------------------------
st.sidebar.title("Banff Parking")
page = st.sidebar.radio(
    "Pages",
    [
        "🏠 Dashboard",
        "🎯 Make Prediction",
        "📊 Lot Status Overview",
        "🔍 XAI",
        "💬 Chat",
    ],
)

# Global error banner if models not loaded
if not MODELS_OK:
    st.error(
        "Models not loaded.\n\n"
        f"`{MODEL_ERROR}`\n\n"
        "Check that all .pkl/.npy files are in the repo and `lightgbm` "
        "is in requirements.txt."
    )

# ---------------------------------------------------
# PAGE: DASHBOARD
# ---------------------------------------------------
if page == "🏠 Dashboard":
    st.title("🏠 Banff Parking – Dashboard")
    st.write("Pick a date & hour, see which lots are most at risk.")

    c1, c2, c3 = st.columns(3)
    with c1:
        chosen_date = st.date_input(
            "📅 Choose date (tourist season)",
            value=date(2025, 7, 15),
        )
    with c2:
        hour = st.slider("🕒 Hour", 0, 23, 14)
    with c3:
        max_temp = st.slider("🌡 Max Temp (°C)", -10.0, 35.0, 22.0)

    c4, c5 = st.columns(2)
    with c4:
        precip = st.slider("☔ Total Precip (mm)", 0.0, 30.0, 0.5)
    with c5:
        gust = st.slider("💨 Max Gust (km/h)", 0.0, 80.0, 12.0)

    month = chosen_date.month
    dow = chosen_date.weekday()

    if MODELS_OK:
        df = compute_lot_status(month, dow, hour, max_temp, precip, gust)
        if df.empty:
            st.warning("No lot features starting with 'Unit_'.")
        else:
            top3 = df.sort_values("Probability full", ascending=False).head(3)
            colA, colB = st.columns([1.3, 1])
            with colA:
                st.subheader("Top 3 high-risk lots")
                st.dataframe(
                    top3.style.format(
                        {
                            "Predicted occupancy": "{:.1f}",
                            "Probability full": "{:.1%}",
                        }
                    ),
                    use_container_width=True,
                )
            with colB:
                high = (df["Status"] == "🟥 High risk").sum()
                busy = (df["Status"] == "🟧 Busy").sum()
                ok = (df["Status"] == "🟩 OK").sum()
                st.metric("🟥 High-risk lots", high)
                st.metric("🟧 Busy lots", busy)
                st.metric("🟩 OK lots", ok)

    st.markdown("---")
    st.subheader("Power BI overview (optional)")
    if POWERBI_EMBED_URL:
        components.iframe(POWERBI_EMBED_URL, height=420)
    else:
        st.info("Set POWERBI_EMBED_URL in code to show your Power BI dashboard here.")

# ---------------------------------------------------
# PAGE: MAKE PREDICTION
# ---------------------------------------------------
elif page == "🎯 Make Prediction":
    st.title("🎯 Make Prediction – Single Lot")

    if not MODELS_OK:
        st.info("Models not loaded.")
    else:
        lot_feats = [f for f in FEATURES if f.startswith("Unit_")]
        lot_names = [lf.replace("Unit_", "").replace("_", " ") for lf in lot_feats]
        if lot_feats:
            pairs = sorted(zip(lot_feats, lot_names), key=lambda x: x[1])
            lot_feats, lot_names = zip(*pairs)
            lot_feats, lot_names = list(lot_feats), list(lot_names)

        if not lot_feats:
            st.error("No lot features starting with 'Unit_'.")
        else:
            col0, col1, col2 = st.columns(3)
            with col0:
                chosen_date = st.date_input(
                    "📅 Date",
                    value=date(2025, 7, 15),
                    key="pred_date",
                )
            with col1:
                hour = st.slider("🕒 Hour", 0, 23, 14, key="pred_hour")
            with col2:
                lot_label = st.selectbox("🚗 Lot", lot_names)
                lot_feat = lot_feats[lot_names.index(lot_label)]

            dow = chosen_date.weekday()
            month = chosen_date.month

            scenario_opt = {
                "Custom": None,
                "Sunny weekend": {"temp": 24.0, "precip": 0.0, "gust": 10.0},
                "Rainy weekday": {"temp": 15.0, "precip": 5.0, "gust": 25.0},
            }
            scen = st.selectbox("Scenario", list(scenario_opt.keys()), index=1)

            defaults = {"temp": 22.0, "precip": 0.5, "gust": 12.0}
            if scenario_opt[scen] is not None:
                defaults.update(scenario_opt[scen])

            c3, c4, c5 = st.columns(3)
            with c3:
                max_temp = st.slider("🌡 Max Temp (°C)", -20.0, 40.0, float(defaults["temp"]))
            with c4:
                precip = st.slider("☔ Total Precip (mm)", 0.0, 30.0, float(defaults["precip"]))
            with c5:
                gust = st.slider("💨 Max Gust (km/h)", 0.0, 100.0, float(defaults["gust"]))

            is_weekend = 1 if dow in [5, 6] else 0
            base = {f: 0 for f in FEATURES}
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
            if lot_feat in base:
                base[lot_feat] = 1

            x = np.array([base[f] for f in FEATURES]).reshape(1, -1)
            x_scaled = scaler.transform(x)

            if st.button("Predict"):
                occ = best_xgb_reg.predict(x_scaled)[0]
                prob = best_lgbm_cls.predict_proba(x_scaled)[0, 1]

                cA, cB = st.columns(2)
                with cA:
                    st.metric("Predicted occupancy (vehicles)", f"{occ:.1f}")
                with cB:
                    st.metric("Probability near full", f"{prob:.1%}")

                if prob > 0.7:
                    st.warning("High risk this lot is full at this time.")
                elif prob > 0.4:
                    st.info("Moderate risk – keep an eye on this lot.")
                else:
                    st.success("Low risk – lot should be comfortable.")

# ---------------------------------------------------
# PAGE: LOT STATUS OVERVIEW
# ---------------------------------------------------
elif page == "📊 Lot Status Overview":
    st.title("📊 Lot Status – All Lots")

    if not MODELS_OK:
        st.info("Models not loaded.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            chosen_date = st.date_input(
                "📅 Date",
                value=date(2025, 7, 15),
                key="status_date",
            )
        with c2:
            hour = st.slider("🕒 Hour", 0, 23, 14, key="status_hour")
        with c3:
            max_temp = st.slider("🌡 Max Temp (°C)", -20.0, 40.0, 22.0)

        c4, c5 = st.columns(2)
        with c4:
            precip = st.slider("☔ Total Precip (mm)", 0.0, 30.0, 0.5)
        with c5:
            gust = st.slider("💨 Max Gust (km/h)", 0.0, 100.0, 12.0)

        month = chosen_date.month
        dow = chosen_date.weekday()

        if st.button("Show lot status"):
            df = compute_lot_status(month, dow, hour, max_temp, precip, gust)
            if df.empty:
                st.warning("Could not compute lot status.")
            else:
                def row_style(row):
                    if "High risk" in row["Status"]:
                        return ["background-color:#ffe5e5"] * len(row)
                    if "Busy" in row["Status"]:
                        return ["background-color:#fff4e0"] * len(row)
                    return ["background-color:#e9f7ef"] * len(row)

                styled = (
                    df.style.format(
                        {
                            "Predicted occupancy": "{:.1f}",
                            "Probability full": "{:.1%}",
                        }
                    ).apply(row_style, axis=1)
                )
                st.dataframe(styled, use_container_width=True)

# ---------------------------------------------------
# PAGE: XAI
# ---------------------------------------------------
elif page == "🔍 XAI":
    st.title("🔍 XAI – Model Insight")
    st.write("SHAP, partial dependence, and residuals for the regression model.")

    if not MODELS_OK:
        st.info("Models not loaded.")
    else:
        # SHAP summary + bar
        try:
            explainer = shap.TreeExplainer(best_xgb_reg)
            shap_vals = explainer.shap_values(X_test_scaled)

            st.subheader("SHAP summary")
            fig1, _ = plt.subplots()
            shap.summary_plot(shap_vals, X_test_scaled, feature_names=FEATURES, show=False)
            st.pyplot(fig1)

            st.subheader("SHAP feature importance")
            fig2, _ = plt.subplots()
            shap.summary_plot(
                shap_vals,
                X_test_scaled,
                feature_names=FEATURES,
                plot_type="bar",
                show=False,
            )
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"SHAP error: {e}")

        # PDP
        pd_feats = [f for f in ["Max Temp (°C)", "Month", "Hour"] if f in FEATURES]
        if pd_feats:
            st.subheader("Partial dependence")
            idxs = [FEATURES.index(f) for f in pd_feats]
            fig3, ax3 = plt.subplots(figsize=(9, 4))
            PartialDependenceDisplay.from_estimator(
                best_xgb_reg,
                X_test_scaled,
                idxs,
                feature_names=FEATURES,
                ax=ax3,
            )
            st.pyplot(fig3)

        # Residual plot
        try:
            st.subheader("Residuals")
            y_pred = best_xgb_reg.predict(X_test_scaled)
            residuals = y_reg_test - y_pred
            fig4, ax4 = plt.subplots()
            ax4.scatter(y_pred, residuals, alpha=0.3)
            ax4.axhline(0, color="red", linestyle="--")
            ax4.set_xlabel("Predicted")
            ax4.set_ylabel("Actual - Predicted")
            st.pyplot(fig4)
        except Exception as e:
            st.error(f"Residual error: {e}")

# ---------------------------------------------------
# PAGE: CHAT
# ---------------------------------------------------
elif page == "💬 Chat":
    st.title("💬 Banff Parking Chat")

    st.write("Ask questions about this project. The bot uses your `banff_knowledge.txt`.")

    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []

    for msg in st.session_state.rag_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user = st.chat_input("Type your question...")
    if user:
        st.session_state.rag_chat_history.append({"role": "user", "content": user})
        with st.chat_message("user"):
            st.markdown(user)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans = generate_chat_answer(user, st.session_state.rag_chat_history)
                st.markdown(ans)
        st.session_state.rag_chat_history.append({"role": "assistant", "content": ans})
