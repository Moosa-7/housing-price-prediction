import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os

st.set_page_config(page_title="Housing Intelligence", layout="wide")

# Config
DEFAULT_API = "http://localhost:8000" # Update based on Docker URL
API_URL = os.getenv("API_URL", DEFAULT_API)

st.title("🏡 Housing Intelligence System")

# TABS for multiple tasks
tab1, tab2, tab3 = st.tabs(["💰 Price & Tier", "📈 Future Trends", "❤️ Recommendations"])

# ==========================
# TAB 1: Prediction & Classification
# ==========================
with tab1:
    st.header("Estimate Value & Tier")
    
    # Input Form (simplified example)
    col1, col2, col3 = st.columns(3)
    with col1: beds = st.number_input("Bedrooms", 1, 10, 3)
    with col2: baths = st.number_input("Bathrooms", 1, 10, 2)
    with col3: area = st.number_input("Area (SqFt)", 500, 10000, 1500)
    
    if st.button("Analyze House"):
        # Construct Payload (Must match your model training columns)
        payload = [{"beds": beds, "baths": baths, "area": area}] 
        # Note: You likely need to add 0s for missing columns if your model trained on more features
        
        try:
            # 1. Get Price
            res_price = requests.post(f"{API_URL}/predict", json=payload).json()
            price = res_price["predictions"][0]
            
            # 2. Get Tier
            res_tier = requests.post(f"{API_URL}/classify", json=payload).json()
            tier = res_tier["tiers"][0]
            
            st.metric("Estimated Price", f"${price:,.2f}")
            st.info(f"Market Tier: **{tier}**")
            
        except Exception as e:
            st.error(f"API Error: {e}")

# ==========================
# TAB 2: Market Forecasting
# ==========================
with tab2:
    st.header("Market Trend Forecast")
    months = st.slider("Select Forecast Horizon (Months)", 1, 24, 12)
    
    if st.button("Generate Forecast"):
        try:
            resp = requests.get(f"{API_URL}/forecast/{months}")
            data = resp.json().get("forecast", [])
            
            if data:
                df_fore = pd.DataFrame(data)
                fig = px.line(df_fore, x="date", y="price", markers=True, 
                              title="Predicted Average Market Price")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No forecast data returned.")
                
        except Exception as e:
            st.error(f"Forecasting Error: {e}")

# ==========================
# TAB 3: Recommendations
# ==========================
with tab3:
    st.header("Find Similar Homes")
    st.write("Based on the inputs in Tab 1, find comparable properties.")
    
    if st.button("Find Comps"):
        # Use same payload from Tab 1
        payload = [{"beds": beds, "baths": baths, "area": area}]
        
        try:
            resp = requests.post(f"{API_URL}/recommend", json=payload)
            recs = resp.json()
            
            if recs:
                st.write("### Top 5 Similar Properties")
                st.dataframe(pd.DataFrame(recs))
            else:
                st.warning("No recommendations found.")
                
        except Exception as e:
            st.error(f"Recommendation Error: {e}")