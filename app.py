import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import json
import os

st.set_page_config(
    page_title="Yet Another Budget",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for theming
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #e2999b;
        --secondary-color: #52181e;
        --background-color: #d7d7d7;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #e2999b;
    }
    
    /* Main content */
    .main {
        background-color: #d7d7d7;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #52181e;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("client/public/figmaAssets/cash-5816752-1.png", width=60)
    st.title("Yet Another Budget")
    st.caption("Your spending, analyzed by AI.")
    
    st.divider()
    
    # Navigation
    st.button("📊 Import & View", type="primary", use_container_width=True)
    st.button("📈 Analyze & Understand", disabled=True, use_container_width=True)
    st.button("🎯 Set Targets (WIP)", disabled=True, use_container_width=True)
    
    st.divider()

# Main content
st.title("Upload Your Spending Data")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your bank statement CSV",
    type=['csv'],
    help="CSV should contain columns: date, payee, amount"
)

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Validate columns
    required_cols = ['date', 'payee', 'amount']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {', '.join(required_cols)}")
    else:
        st.success(f"✅ Loaded {len(df)} transactions")
        
        # Display raw data
        with st.expander("📋 View Raw Data"):
            st.dataframe(df, use_container_width=True)
        
        # Categorization placeholder
        st.info("🤖 Click below to categorize transactions with AI")
        
        if st.button("Categorize with AI", type="primary"):
            st.warning("OpenAI integration needed - coming next!")
else:
    # Empty state
    st.info("👆 Upload a CSV file to get started")
    
    # Sample data format
    with st.expander("💡 Expected CSV Format"):
        sample_df = pd.DataFrame({
            'date': ['12/10/2025', '11/10/2025'],
            'payee': ['TESCO STORES', 'RENT PAYMENT'],
            'amount': [35.64, 1776.50]
        })
        st.dataframe(sample_df, use_container_width=True)
