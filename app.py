import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Trips & Travel Dashboard",
    layout="wide"
)

# ---------------- CUSTOM BLACK & BLUE THEME ---------------- #

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    h1, h2, h3, h4 {
        color: #1f77ff;
    }

    .stMetric {
        background-color: #1c1f26;
        padding: 10px;
        border-radius: 10px;
    }

    .stButton>button {
        background-color: #1f77ff;
        color: white;
        border-radius: 8px;
    }

    </style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")
sns.set_style("darkgrid")

st.title("🌍 Trips & Travel - Customer Conversion Dashboard")

# ---------------- FILE UPLOAD ---------------- #

uploaded_file = st.file_uploader("Upload Travel Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload the dataset to continue.")
    st.stop()

# ---------------- DATA CLEANING ---------------- #

# Fill missing values properly
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["Age"] = df["Age"].fillna(df["Age"].median())
df["DurationOfPitch"] = df["DurationOfPitch"].fillna(df["DurationOfPitch"].median())
df["NumberOfTrips"] = df["NumberOfTrips"].fillna(df["NumberOfTrips"].median())
df["NumberOfFollowups"] = df["NumberOfFollowups"].fillna(df["NumberOfFollowups"].median())
df["NumberOfChildrenVisiting"] = df["NumberOfChildrenVisiting"].fillna(df["NumberOfChildrenVisiting"].median())
df["PreferredPropertyStar"] = df["PreferredPropertyStar"].fillna(df["PreferredPropertyStar"].median())
df["TypeofContact"] = df["TypeofContact"].fillna(df["TypeofContact"].mode()[0])

# ---------------- SIDEBAR FILTERS ---------------- #

st.sidebar.header("🔎 Customer Filters")

city = st.sidebar.selectbox("City Tier", ["All"] + sorted(df["CityTier"].astype(str).unique().tolist()))
gender = st.sidebar.selectbox("Gender", ["All"] + df["Gender"].astype(str).unique().tolist())
passport = st.sidebar.selectbox("Passport", ["All", 0, 1])
product = st.sidebar.selectbox("Product Pitched", ["All"] + df["ProductPitched"].astype(str).unique().tolist())

filtered_df = df.copy()

if city != "All":
    filtered_df = filtered_df[filtered_df["CityTier"].astype(str) == city]

if gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"].astype(str) == gender]

if passport != "All":
    filtered_df = filtered_df[filtered_df["Passport"] == passport]

if product != "All":
    filtered_df = filtered_df[filtered_df["ProductPitched"].astype(str) == product]

# ---------------- KPI SECTION ---------------- #

st.subheader("📊 Key Business Metrics")

if filtered_df.empty:
    st.warning("No data available for selected filters.")
else:
    total_customers = filtered_df.shape[0]
    conversion_rate = round(filtered_df["ProdTaken"].mean() * 100, 2)
    avg_income = int(filtered_df["MonthlyIncome"].mean())
    avg_age = round(filtered_df["Age"].mean(), 1)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", total_customers)
    col2.metric("Conversion Rate (%)", conversion_rate)
    col3.metric("Average Monthly Income", avg_income)
    col4.metric("Average Age", avg_age)

    st.markdown("---")

    # ---------------- PURCHASE DISTRIBUTION ---------------- #

    st.subheader("🎯 Purchase Distribution")

    fig1, ax1 = plt.subplots()
    sns.countplot(x="ProdTaken", data=filtered_df, ax=ax1, palette="Blues")
    ax1.set_xticklabels(["Not Purchased", "Purchased"])
    st.pyplot(fig1)

    st.markdown("---")

    # ---------------- FEATURE IMPACT ---------------- #

    st.subheader("📈 Feature Impact Analysis")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### 💰 Monthly Income vs Purchase")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x="ProdTaken", y="MonthlyIncome", data=filtered_df, ax=ax2)
        ax2.set_xticklabels(["Not Purchased", "Purchased"])
        st.pyplot(fig2)

    with col6:
        st.markdown("### ⭐ Pitch Satisfaction vs Purchase")
        fig3, ax3 = plt.subplots()
        sns.boxplot(x="ProdTaken", y="PitchSatisfactionScore", data=filtered_df, ax=ax3)
        ax3.set_xticklabels(["Not Purchased", "Purchased"])
        st.pyplot(fig3)

    st.markdown("---")

    col7, col8 = st.columns(2)

    with col7:
        st.markdown("### 🛂 Passport vs Purchase")
        fig4, ax4 = plt.subplots()
        sns.countplot(x="Passport", hue="ProdTaken", data=filtered_df, ax=ax4)
        ax4.set_xticklabels(["No Passport", "Has Passport"])
        st.pyplot(fig4)

    with col8:
        st.markdown("### 📞 Followups vs Purchase")
        fig5, ax5 = plt.subplots()
        sns.countplot(x="NumberOfFollowups", hue="ProdTaken", data=filtered_df, ax=ax5)
        st.pyplot(fig5)

    st.markdown("---")

    # ---------------- PRODUCT CONVERSION ---------------- #

    st.subheader("📦 Product-wise Conversion Rate")

    product_conversion = (
        filtered_df.groupby("ProductPitched")["ProdTaken"]
        .mean()
        .sort_values(ascending=False)
    )

    fig6, ax6 = plt.subplots()
    product_conversion.plot(kind="bar", ax=ax6, color="#1f77ff")
    st.pyplot(fig6)

    st.markdown("---")

    # ---------------- CORRELATION HEATMAP ---------------- #

    st.subheader("🔎 Correlation Analysis")

    numeric_df = filtered_df.select_dtypes(include=['int64','float64'])

    fig7, ax7 = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", ax=ax7)
    st.pyplot(fig7)

    st.markdown("---")

    # ---------------- INSIGHTS ---------------- #

    st.subheader("📌 Key Insights")

    st.markdown("""
    - Higher income customers show stronger purchase behavior.
    - Passport holders are more likely to buy travel packages.
    - Pitch satisfaction significantly impacts conversion.
    - 3–4 follow-ups increase conversion probability.
    - Certain products outperform others in conversion rate.
    """)

    st.markdown("---")

    # ---------------- RECOMMENDATIONS ---------------- #

    st.subheader("🚀 Business Recommendations")

    st.markdown("""
    1. Target high-income customers in Tier 1 cities.
    2. Focus marketing campaigns on passport holders.
    3. Improve pitch quality through sales training.
    4. Ensure minimum 3–4 follow-ups for serious leads.
    5. Promote high-performing products aggressively.
    """)

    st.success("Dashboard Loaded Successfully ✅")
