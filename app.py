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
    
    /* Primary button */
    .stButton>button[kind="primary"] {
        background-color: #832632;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = None
if 'categorized' not in st.session_state:
    st.session_state.categorized = False
if 'summary' not in st.session_state:
    st.session_state.summary = None

# Sidebar
with st.sidebar:
    st.title("💰 Yet Another Budget")
    st.caption("Your spending, analyzed by AI.")
    
    st.divider()
    
    # Navigation
    st.button("📊 Import & View", type="primary", use_container_width=True)
    st.button("📈 Analyze & Understand", disabled=True, use_container_width=True)
    st.button("🎯 Set Targets (WIP)", disabled=True, use_container_width=True)
    
    st.divider()
    
    # Import button at bottom
    if st.button("📤 Import New Data", use_container_width=True):
        st.session_state.transactions = None
        st.session_state.categorized = False
        st.session_state.summary = None
        st.rerun()

# Main content
st.title("AI Spending Analyzer")

# File uploader
if st.session_state.transactions is None:
    uploaded_file = st.file_uploader(
        "Upload your bank statement CSV",
        type=['csv'],
        help="CSV should contain columns: date, payee, amount"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = ['date', 'payee', 'amount']
            if not all(col in df.columns for col in required_cols):
                st.error(f"❌ CSV must contain columns: {', '.join(required_cols)}")
            else:
                st.session_state.transactions = df
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")
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

else:
    # Display loaded data
    df = st.session_state.transactions
    
    st.success(f"✅ Loaded {len(df)} transactions")
    
    # Categorization section
    if not st.session_state.categorized:
        st.info("🤖 Ready to categorize transactions with AI")
        
        if st.button("Categorize with AI", type="primary"):
            with st.spinner("Analyzing transactions..."):
                # OpenAI integration will go here
                st.warning("OpenAI integration coming next!")
                
    else:
        # Display results
        st.subheader("📊 Analysis Results")
        
        # Create three columns for widgets
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 💳 Categorized Transactions")
            st.dataframe(df, use_container_width=True, height=400)
        
        with col2:
            st.markdown("### 📈 Spending by Category")
            # Pie chart will go here
            st.info("Pie chart coming next")
        
        # Summary section
        st.markdown("### 📝 AI Summary")
        if st.session_state.summary:
            st.markdown(st.session_state.summary)
        else:
            st.info("Summary will appear here")
    
    # Raw data expander
    with st.expander("📋 View Raw Data"):
        st.dataframe(df, use_container_width=True)
