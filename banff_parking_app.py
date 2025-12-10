import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay

# Basic Streamlit page setup
st.set_page_config(
    page_title="Banff Parking – ML & XAI",
    layout="wide"
)

# ---------- LOAD MODELS & DATA ----------

@st.cache_resource
def load_models_and_data():
    reg = joblib.load("banff_best_xgb_reg.pkl")
    cls = joblib.load("banff_best_xgb_cls.pkl")
    scaler = joblib.load("banff_scaler.pkl")
    features = joblib.load("banff_features.pkl")

    X_test_scaled = np.load("X_test_scaled.npy")
    y_reg_test = np.load("y_reg_test.npy")

    return reg, cls, scaler, features, X_test_scaled, y_reg_test

best_xgb_reg, best_xgb_cls, scaler, FEATURES, X_test_scaled, y_reg_test = load_models_and_data()

# ---------- SIDEBAR NAVIGATION ----------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Make Prediction", "XAI – Explainable AI"]
)

# ---------- PAGE 1: OVERVIEW ----------

if page == "Overview":
    st.title("Banff Parking Demand – Machine Learning Project")

    st.write("""
    This application is based on a machine learning project that predicts 
    hourly parking demand in Banff using weather, temporal, and historical occupancy data.
    
    **Main objectives:**
    - Analyze how time, weather, and location affect parking occupancy.
    - Predict hourly occupancy and the probability that a lot will be full (>90%).
    - Provide explainable AI (XAI) so that results are interpretable for stakeholders.
    """)

    st.write("""
    Use the menu on the left to:
    - **Make Prediction**: Input month, hour, weather and get model predictions.
    - **XAI – Explainable AI**: See why the model makes those predictions.
    """)

# ---------- PAGE 2: MAKE PREDICTION ----------

if page == "Make Prediction":
    st.title("Predict Parking Occupancy and Full-Lot Probability")

    st.write("Provide basic information to estimate parking demand for a given hour.")

    col1, col2 = st.columns(2)

    with col1:
        month = st.slider("Month (1 = Jan, 12 = Dec)", 1, 12, 7)
        day_of_week = st.slider("Day of Week (0 = Mon, 6 = Sun)", 0, 6, 5)
        hour = st.slider("Hour of Day (0–23)", 0, 23, 14)

    with col2:
        max_temp = st.slider("Max Temperature (°C)", -10.0, 35.0, 22.0)
        precip = st.slider("Total Precipitation (mm)", 0.0, 20.0, 0.5)
        wind = st.slider("Wind Gust (km/h)", 0.0, 80.0, 15.0)

    is_weekend = 1 if day_of_week in [5, 6] else 0

    # Build feature dictionary (adjust keys to match your FEATURES names)
    input_dict = {
        "Month": month,
        "DayOfWeek": day_of_week,
        "Hour": hour,
        "IsWeekend": is_weekend,
        "Max Temp (°C)": max_temp,
        "Total Precip (mm)": precip,
        "Spd of Max Gust (km/h)": wind,
        # any extra features your model expects will default to 0 below
    }

    # Align to the exact FEATURE order used in training
    x = np.array([input_dict.get(f, 0) for f in FEATURES]).reshape(1, -1)
    x_scaled = scaler.transform(x)

    if st.button("Predict"):
        occ_pred = best_xgb_reg.predict(x_scaled)[0]
        full_prob = best_xgb_cls.predict_proba(x_scaled)[0, 1]

        st.subheader("Model Predictions")
        st.write(f"**Predicted Occupancy (model units):** {occ_pred:.2f}")
        st.write(f"**Probability Lot is Full (>90%):** {full_prob:.1%}")

        if full_prob > 0.7:
            st.warning("High risk of this lot being full. Consider redirecting to alternative parking.")
        elif full_prob > 0.4:
            st.info("Moderate risk. Monitor this lot closely.")
        else:
            st.success("Low risk of full capacity at this time.")

# ---------- PAGE 3: XAI – EXPLAINABLE AI ----------

if page == "XAI – Explainable AI":
    st.title("Explainable AI (XAI) – Why Does the Model Predict This?")

    st.write("""
    This page explains **which features** most strongly influence the model's predictions,
    using SHAP values, Partial Dependence Plots, and Residual analysis.
    """)

    # ---- SHAP for Regression Model ----
    st.subheader("SHAP Summary – Regression Model (Occupancy)")

    explainer_reg = shap.TreeExplainer(best_xgb_reg)
    shap_vals_reg = explainer_reg.shap_values(X_test_scaled)

    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_vals_reg, X_test_scaled,
                      feature_names=FEATURES,
                      show=False)
    st.pyplot(fig1)
    st.caption("Each point shows how a feature value increases or decreases the predicted occupancy.")

    st.subheader("SHAP Feature Importance – Regression")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_vals_reg, X_test_scaled,
                      feature_names=FEATURES,
                      plot_type='bar',
                      show=False)
    st.pyplot(fig2)

    # ---- Partial Dependence Plots ----
    st.subheader("Partial Dependence – Key Features")

    # choose 3 important features by name; change if needed
    pd_features = []
    for feat_name in ["Max Temp (°C)", "Month", "Hour"]:
        if feat_name in FEATURES:
            pd_features.append(FEATURES.index(feat_name))

    if len(pd_features) > 0:
        fig3, ax3 = plt.subplots(figsize=(9, 4))
        PartialDependenceDisplay.from_estimator(
            best_xgb_reg,
            X_test_scaled,
            pd_features,
            feature_names=FEATURES,
            ax=ax3
        )
        st.pyplot(fig3)
    else:
        st.info("Configured PDP feature names were not found in FEATURES list.")

    # ---- Residual Plot ----
    st.subheader("Residual Plot – Regression Model")

    y_pred = best_xgb_reg.predict(X_test_scaled)
    residuals = y_reg_test - y_pred

    fig4, ax4 = plt.subplots()
    ax4.scatter(y_pred, residuals, alpha=0.3)
    ax4.axhline(0, color="red", linestyle="--")
    ax4.set_xlabel("Predicted Occupancy")
    ax4.set_ylabel("Residuals (Actual - Predicted)")
    st.pyplot(fig4)

    st.caption("Residuals scattered around zero indicate a reasonably well-fitting model.")
