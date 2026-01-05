import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Precision Agriculture Water Management",
    page_icon="🌱",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🌱 Precision Agriculture Water Management System")
st.subheader("Smart Decision Support for Efficient Irrigation")
st.markdown(
    "Aligned with **SDG 2 (Zero Hunger)** and **SDG 6 (Clean Water & Sanitation)**"
)

st.markdown("---")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🌾 Field Input Parameters")

soil_moisture = st.sidebar.slider(
    "Soil Moisture (%)", min_value=0, max_value=100, value=40
)

temperature = st.sidebar.slider(
    "Temperature (°C)", min_value=10, max_value=50, value=30
)

humidity = st.sidebar.slider(
    "Humidity (%)", min_value=0, max_value=100, value=60
)

crop_type = st.sidebar.selectbox(
    "Crop Type", ["Rice", "Wheat", "Maize", "Cotton"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Prototype for Ecoverse Hackathon")

# ---------------- LOGIC ----------------
def irrigation_decision_logic(moisture, temp, hum):
    """
    Rule-based decision logic.
    ML model will be integrated in later phase.
    """
    if moisture < 30 and temp > 30:
        return "High"
    elif moisture < 40:
        return "Medium"
    else:
        return "Low"

risk_level = irrigation_decision_logic(
    soil_moisture, temperature, humidity
)

# ---------------- OUTPUT DASHBOARD ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Soil Moisture", f"{soil_moisture}%")

with col2:
    st.metric("Temperature", f"{temperature} °C")

with col3:
    st.metric("Humidity", f"{humidity}%")

st.markdown("---")

# ---------------- RECOMMENDATION ----------------
st.subheader("💧 Irrigation Recommendation")

if risk_level == "High":
    st.error("⚠️ High Water Stress Detected")
    st.write(
        f"Crop **{crop_type}** requires immediate irrigation to avoid yield loss."
    )
elif risk_level == "Medium":
    st.warning("⚠️ Moderate Water Requirement")
    st.write(
        f"Monitor soil conditions for **{crop_type}** and plan irrigation soon."
    )
else:
    st.success("✅ Water Level Optimal")
    st.write(
        f"No irrigation required currently for **{crop_type}**."
    )

st.markdown("---")

# ---------------- FOOTNOTE ----------------
st.caption(
    "⚠️ This is a prototype using rule-based decision logic. "
    "Machine Learning models using real agricultural datasets "
    "will be integrated in the next development phase."
)
