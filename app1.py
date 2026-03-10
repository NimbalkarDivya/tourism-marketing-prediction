import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Wellness Tourism Marketing Intelligence", layout="centered")
st.title("🌿 Wellness Tourism Marketing Intelligence System")

# ------------------------------------------------
# Load Model
# ------------------------------------------------
@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------
st.sidebar.header("📋 Customer Information")

income = st.sidebar.number_input("Monthly Income", 10000, 500000, 50000)
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
trips = st.sidebar.slider("Number of Trips per Year", 0, 10, 2)
passport = st.sidebar.selectbox("Passport Available", [0, 1])
pitch_score = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
followups = st.sidebar.slider("Number of Followups", 0, 10, 3)
contact_type = st.sidebar.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

# ------------------------------------------------
# Create DataFrame for Prediction
# ------------------------------------------------
input_df = pd.DataFrame({
    "MonthlyIncome": [income],
    "CityTier": [city_tier],
    "NumberOfTrips": [trips],
    "Passport": [passport],
    "PitchSatisfactionScore": [pitch_score],
    "NumberOfFollowups": [followups],
    "TypeofContact": [contact_type]
})

# ------------------------------------------------
# Prediction
# ------------------------------------------------
if st.button("🔍 Predict Customer"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")
    st.metric("Purchase Probability", f"{round(probability*100,2)} %")

    # ------------------------------------------------
    # Package Recommendation Logic
    # ------------------------------------------------
    if income < 40000:
        package = "Basic"
    elif income < 80000:
        package = "Standard"
    elif income < 150000:
        package = "Deluxe"
    elif income < 250000:
        package = "Super Deluxe"
    else:
        package = "King"

    st.subheader("🏷 Recommended Package")
    st.info(f"Suggested Package: **{package}**")

    # ------------------------------------------------
    # Loyal Customer Identification
    # ------------------------------------------------
    st.subheader("⭐ Loyalty Evaluation")

    if probability > 0.7 and trips >= 4 and pitch_score >= 4:
        st.success("🔥 Potential Loyal Customer")
        st.write("High lifetime value. Prioritize for premium campaigns.")
    elif probability > 0.5:
        st.warning("⚡ Moderate Potential Customer")
        st.write("Offer discounts or personalized offers.")
    else:
        st.error("❌ Low Potential Customer")
        st.write("Avoid high marketing expenditure.")

    # ------------------------------------------------
    # Business Recommendation
    # ------------------------------------------------
    st.markdown("---")
    st.subheader("💼 Marketing Strategy Recommendation")

    if probability > 0.75:
        st.write("Immediate follow-up by senior sales team recommended.")
    elif probability > 0.4:
        st.write("Send promotional email / discount offer.")
    else:
        st.write("Do not allocate marketing budget to this customer.")