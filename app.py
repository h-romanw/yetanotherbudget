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

# Check for OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OpenAI API key not found. Please add OPENAI_API_KEY to your Replit Secrets.")
    client = None
else:
    client = OpenAI(api_key=api_key)

# Common spending categories (defined globally for reuse)
CATEGORIES = [
    "Groceries", "Dining & Takeout", "Bills & Utilities", 
    "Transport", "Shopping", "Entertainment", 
    "Health & Fitness", "Travel", "Other"
]

# AI categorization function with summary
def categorize_transactions_with_summary(df):
    """Batch categorize transactions and generate summary in one call"""
    
    if not client:
        return None, None
    
    categories_list = CATEGORIES
    
    # Prepare batch of transactions
    batch_size = 15
    all_results = []
    summary = None
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Create transaction list for prompt
        transactions_text = "\n".join([
            f"{idx}. {row['payee']} - £{row['amount']}" 
            for idx, row in batch.iterrows()
        ])
        
        # Determine if this is the last batch
        is_last_batch = (i + batch_size) >= len(df)
        
        if is_last_batch:
            # Last batch: include summary request
            prompt = f"""Analyze these bank transactions and assign each one to the most appropriate category.

Categories: {', '.join(categories_list)}

Transactions (from final batch of {len(df)} total):
{transactions_text}

Return a JSON object with this exact structure:
{{
  "categorizations": [
    {{"index": <transaction_index>, "category": "<category_name>"}},
    ...
  ],
  "summary": "A brief 2-paragraph friendly summary of overall spending patterns, main areas, and gentle optimization suggestions"
}}"""
        else:
            # Regular batch: just categorizations
            prompt = f"""Analyze these bank transactions and assign each one to the most appropriate category.

Categories: {', '.join(categories_list)}

Transactions:
{transactions_text}

Return a JSON object with this exact structure:
{{
  "categorizations": [
    {{"index": <transaction_index>, "category": "<category_name>"}},
    ...
  ]
}}"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial categorization assistant. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                categorizations = result.get("categorizations", [])
                
                if not isinstance(categorizations, list):
                    st.error(f"Invalid response format from AI")
                    return None, None
                    
                all_results.extend(categorizations)
                
                # Extract summary from last batch if present
                if is_last_batch and "summary" in result:
                    summary = result.get("summary")
            
        except Exception as e:
            st.error(f"Error categorizing batch: {str(e)}")
            return None, None
    
    # Map categories back to dataframe
    category_map = {item.get('index'): item.get('category', 'Other') for item in all_results if isinstance(item, dict)}
    df['category'] = df.index.map(lambda idx: category_map.get(idx, "Other"))
    
    # Fallback if summary wasn't generated
    if not summary:
        total_spent = df['amount'].sum()
        if total_spent == 0:
            summary = "No spending detected in the uploaded transactions."
        else:
            summary = "Summary generation failed. Please try again."
    
    return df, summary

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
            st.dataframe(sample_df, width='stretch')

else:
    # Display loaded data
    df = st.session_state.transactions
    
    st.success(f"✅ Loaded {len(df)} transactions")
    
    # Categorization section
    if not st.session_state.categorized:
        st.info("🤖 Ready to categorize transactions with AI")
        
        if st.button("Categorize with AI", type="primary"):
            with st.spinner("Analyzing transactions with AI..."):
                # Categorize transactions and generate summary
                categorized_df, summary = categorize_transactions_with_summary(df.copy())
                
                if categorized_df is not None:
                    st.session_state.transactions = categorized_df
                    st.session_state.summary = summary
                    st.session_state.categorized = True
                    st.rerun()
                
    else:
        # Display results
        st.subheader("📊 Analysis Results")
        
        # Create three columns for widgets
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 💳 Categorized Transactions")
            
            # Use data_editor to allow manual category changes
            edited_df = st.data_editor(
                df,
                column_config={
                    "category": st.column_config.SelectboxColumn(
                        "Category",
                        help="Select category for this transaction",
                        width="medium",
                        options=CATEGORIES,
                        required=True,
                    )
                },
                width='stretch',
                height=400,
                hide_index=False,
                key="transaction_editor"
            )
            
            # Update session state if data was edited
            if not edited_df.equals(df):
                st.session_state.transactions = edited_df
        
        with col2:
            st.markdown("### 📈 Spending by Category")
            
            # Calculate spending by category (use edited_df to reflect manual changes)
            category_spending = edited_df.groupby('category')['amount'].sum().reset_index()
            category_spending = category_spending.sort_values('amount', ascending=False)
            
            # Pink/burgundy color palette matching the theme
            colors = ['#832632', '#a13344', '#be4056', '#d94d68', '#e2999b', '#f0b8ba', '#f5d0d1', '#fae8e9']
            
            # Create pie chart
            fig = px.pie(
                category_spending, 
                values='amount', 
                names='category',
                title='',
                color_discrete_sequence=colors
            )
            fig.update_traces(
                textposition='auto',
                textinfo='percent+label',
                textfont_size=12,
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            fig.update_layout(
                showlegend=True,
                height=400,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02
                ),
                font=dict(
                    family="Arial, sans-serif",
                    size=12,
                    color="#52181e"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary section
        st.markdown("### 📝 AI Summary")
        if st.session_state.summary:
            st.markdown(st.session_state.summary)
        else:
            st.info("Summary will appear here")
    
    # Raw data expander
    with st.expander("📋 View Raw Data"):
        st.dataframe(df, width='stretch')
