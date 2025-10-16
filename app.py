from typing import Literal
import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import json
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

st.set_page_config(
    page_title="Yet Another Budget",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
    # theme paramater doesn't work in Replit. See .streamlit/config.toml for theme parameters.
    # theme={
    #    "base": "light",
    #    "primaryColor": "#832632",
    #    "backgroundColor": "#f5f5f5",
    #    "secondaryBackgroundColor": "#e2999b",
    #    "textColor": "#52181e"
    #}
)

# Check for OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error(
        "⚠️ OpenAI API key not found. Please add OPENAI_API_KEY to your Replit Secrets."
    )
    client = None
else:
    client = OpenAI(api_key=api_key)

# Common spending categories (defined globally for reuse)
CATEGORIES = [
    "Groceries", "Dining & Takeout", "Bills & Utilities", "Transport",
    "Shopping", "Entertainment", "Health & Fitness", "Travel", "Other"
]

# Default category color mapping
DEFAULT_CATEGORY_COLORS = {
    "Groceries": "#00D084",  # Green
    "Dining & Takeout": "#FF9500",  # Orange
    "Bills & Utilities": "#B040D0",  # Purple
    "Transport": "#007AFF",  # Blue
    "Shopping": "#FF2D55",  # Pink/Red
    "Entertainment": "#FF3B30",  # Red
    "Health & Fitness": "#34C759",  # Light Green
    "Travel": "#5856D6",  # Indigo
    "Other": "#8E8E93"  # Gray
}

# Color palette for auto-assigning to custom categories
COLOR_PALETTE = [
    "#00D084", "#FF9500", "#B040D0", "#007AFF", "#FF2D55",
    "#FF3B30", "#34C759", "#5856D6", "#8E8E93", "#FF6B6B",
    "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7B731",
    "#5F27CD", "#00D2D3", "#FF9FF3", "#54A0FF", "#48DBFB"
]


def get_all_categories():
    """Get combined list of default + custom categories"""
    return CATEGORIES + st.session_state.get('custom_categories', [])


def assign_color_to_category(category):
    """Auto-assign a color to a new category"""
    if 'category_colors' not in st.session_state:
        st.session_state.category_colors = DEFAULT_CATEGORY_COLORS.copy()
    
    # If category already has a color, return it
    if category in st.session_state.category_colors:
        return st.session_state.category_colors[category]
    
    # Find used colors
    used_colors = set(st.session_state.category_colors.values())
    
    # Find first unused color from palette
    for color in COLOR_PALETTE:
        if color not in used_colors:
            st.session_state.category_colors[category] = color
            return color
    
    # If all colors used, cycle back to start
    st.session_state.category_colors[category] = COLOR_PALETTE[len(st.session_state.category_colors) % len(COLOR_PALETTE)]
    return st.session_state.category_colors[category]


def get_category_color(category):
    """Get color for a category"""
    if 'category_colors' not in st.session_state:
        st.session_state.category_colors = DEFAULT_CATEGORY_COLORS.copy()
    
    # If category doesn't have a color, assign one
    if category not in st.session_state.category_colors:
        return assign_color_to_category(category)
    
    return st.session_state.category_colors[category]


# Smart CSV parser using AI to identify columns
def parse_csv_with_ai(uploaded_file):
    """Use AI to intelligently parse any CSV format and extract required columns"""
    
    if not client:
        return None, "OpenAI API key not configured"
    
    try:
        # Read the CSV file
        df_raw = pd.read_csv(uploaded_file)
        
        # Get column names and first 3 rows as sample
        columns = df_raw.columns.tolist()
        sample_rows = df_raw.head(3).to_dict('records')
        
        # Create prompt for AI to identify columns
        sample_data = "\n".join([
            f"Row {i+1}: " + ", ".join([f"{col}={row[col]}" for col in columns])
            for i, row in enumerate(sample_rows)
        ])
        
        prompt = f"""Analyze this CSV structure and identify which columns contain the following information:
- Date: The transaction date
- Payee: The merchant/payee name  
- Amount: The transaction amount (debit/spending amount, as positive number)
- Balance: The account balance after transaction (OPTIONAL - may not exist)

CSV Columns: {', '.join(columns)}

Sample Data:
{sample_data}

Also determine the date format by examining the date values. Common formats:
- "day_first": dd/mm/yyyy or dd-mm-yyyy (e.g., 25/12/2024 for Christmas)
- "month_first": mm/dd/yyyy or mm-dd-yyyy (e.g., 12/25/2024 for Christmas)
- "iso": yyyy-mm-dd (e.g., 2024-12-25)

Return a JSON object with column mappings AND date format:
{{
  "date": "<actual_column_name>",
  "payee": "<actual_column_name>",
  "amount": "<actual_column_name>",
  "balance": "<actual_column_name_or_null>",
  "date_format": "day_first|month_first|iso"
}}

If balance column doesn't exist, set it to null."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "You are a CSV parsing assistant. Analyze column names and data to identify the correct mappings. Respond only with valid JSON."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.1,
            response_format={"type": "json_object"})
        
        content = response.choices[0].message.content
        if not content:
            return None, "AI response was empty"
        
        mapping = json.loads(content)
        
        # Validate required fields
        if not all(key in mapping for key in ['date', 'payee', 'amount']):
            return None, "AI couldn't identify required columns (date, payee, amount)"
        
        # Build the standardized dataframe
        df_standard = pd.DataFrame()
        
        # Parse dates with validation - try AI's suggestion first, then fall back
        date_format = mapping.get('date_format', 'day_first')
        
        # Try parsing with different formats
        parsing_attempts = []
        
        # Try AI's suggested format first
        if date_format == 'month_first':
            dates_suggested = pd.to_datetime(df_raw[mapping['date']], dayfirst=False, errors='coerce')
            parsing_attempts.append(('month_first', dates_suggested))
        elif date_format == 'iso':
            dates_suggested = pd.to_datetime(df_raw[mapping['date']], errors='coerce')
            parsing_attempts.append(('iso', dates_suggested))
        else:
            dates_suggested = pd.to_datetime(df_raw[mapping['date']], dayfirst=True, errors='coerce')
            parsing_attempts.append(('day_first', dates_suggested))
        
        # Always try alternative formats as fallback
        if date_format != 'day_first':
            parsing_attempts.append(('day_first', pd.to_datetime(df_raw[mapping['date']], dayfirst=True, errors='coerce')))
        if date_format != 'month_first':
            parsing_attempts.append(('month_first', pd.to_datetime(df_raw[mapping['date']], dayfirst=False, errors='coerce')))
        
        # Sort by NaT count to find best parsing
        parsing_attempts.sort(key=lambda x: x[1].isna().sum())
        
        # Check if we have a clear winner or ambiguous data
        if len(parsing_attempts) > 1:
            best_nat_count = parsing_attempts[0][1].isna().sum()
            second_best_nat_count = parsing_attempts[1][1].isna().sum()
            
            # If NaT counts are equal, the data is ambiguous - prefer AI's suggestion
            if best_nat_count == second_best_nat_count:
                # Find AI's suggested format in the attempts
                for fmt, dates in parsing_attempts:
                    if fmt == date_format:
                        best_format, best_dates = fmt, dates
                        break
                else:
                    # Fallback if AI's format not in attempts
                    best_format, best_dates = parsing_attempts[0]
            else:
                # Clear winner - use it
                best_format, best_dates = parsing_attempts[0]
        else:
            best_format, best_dates = parsing_attempts[0]
        
        # Validate: if too many NaT values, return error
        nat_percentage = (best_dates.isna().sum() / len(df_raw)) * 100
        if nat_percentage > 20:  # If more than 20% of dates are invalid
            return None, f"Could not parse dates reliably. {nat_percentage:.1f}% of dates were invalid. Please check the date format in your CSV."
        
        # Use the best parsing result
        df_standard['date'] = best_dates.dt.strftime('%d/%m/%Y')
        
        df_standard['payee'] = df_raw[mapping['payee']]
        df_standard['amount'] = df_raw[mapping['amount']].abs()  # Ensure positive amounts
        
        # Add balance if it exists
        if mapping.get('balance') and mapping['balance'] in df_raw.columns:
            df_standard['balance'] = df_raw[mapping['balance']]
        
        return df_standard, None
        
    except Exception as e:
        return None, f"Error parsing CSV: {str(e)}"


# Project management functions
def save_project(project_name, transactions_df):
    """Save categorized transactions and targets to a project file"""
    try:
        # Create projects directory if it doesn't exist
        os.makedirs('projects', exist_ok=True)
        
        # Sanitize project name for filename
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"projects/{safe_name}.json"
        
        # Convert dataframe to records
        data = {
            'project_name': project_name,
            'transactions': transactions_df.to_dict('records'),
            'created_at': pd.Timestamp.now().isoformat(),
            'total_transactions': len(transactions_df),
            'targets': st.session_state.targets  # Save targets
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True, filename
    except Exception as e:
        return False, str(e)


def load_project(project_name):
    """Load project from file including targets"""
    try:
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"projects/{safe_name}.json"
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['transactions'])
        
        # Load targets if they exist
        if 'targets' in data:
            st.session_state.targets = data['targets']
        
        return df, None
    except Exception as e:
        return None, f"Error loading project: {str(e)}"


def list_projects():
    """List all available projects"""
    try:
        if not os.path.exists('projects'):
            return []
        
        projects = []
        for filename in os.listdir('projects'):
            if filename.endswith('.json'):
                with open(f'projects/{filename}', 'r') as f:
                    data = json.load(f)
                    projects.append({
                        'name': data['project_name'],
                        'transactions': data.get('total_transactions', 0),
                        'created_at': data.get('created_at', 'Unknown')
                    })
        return projects
    except Exception as e:
        st.error(f"Error listing projects: {str(e)}")
        return []


def append_to_project(project_name, new_transactions_df):
    """Append new transactions to an existing project"""
    try:
        # Load existing project
        existing_df, error = load_project(project_name)
        if error or existing_df is None:
            return None, error or "Failed to load project"
        
        # Combine transactions
        combined_df = pd.concat([existing_df, new_transactions_df], ignore_index=True)
        
        # Remove duplicates based on date, payee, and amount
        combined_df = combined_df.drop_duplicates(subset=['date', 'payee', 'amount'], keep='first')
        
        # Save back to project
        success, result = save_project(project_name, combined_df)
        if success:
            return combined_df, None
        else:
            return None, result
    except Exception as e:
        return None, f"Error appending to project: {str(e)}"


# AI categorization function (categorization only - no summary)
def categorize_transactions(df):
    """Batch categorize transactions using AI"""

    if not client:
        return None

    categories_list = get_all_categories()

    # Prepare batch of transactions
    batch_size = 15
    all_results = []

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]

        # Create transaction list for prompt
        transactions_text = "\n".join([
            f"{idx}. {row['payee']} - £{row['amount']}"
            for idx, row in batch.iterrows()
        ])

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
                messages=[{
                    "role":
                    "system",
                    "content":
                    "You are a financial categorization assistant. Respond only with valid JSON."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3,
                response_format={"type": "json_object"})

            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                categorizations = result.get("categorizations", [])

                if not isinstance(categorizations, list):
                    st.error(f"Invalid response format from AI")
                    return None

                all_results.extend(categorizations)

        except Exception as e:
            st.error(f"Error categorizing batch: {str(e)}")
            return None

    # Map categories back to dataframe
    category_map = {
        item.get('index'): item.get('category', 'Other')
        for item in all_results if isinstance(item, dict)
    }
    df['category'] = df.index.map(lambda idx: category_map.get(idx, "Other"))

    return df


# AI analysis/summary function (separate from categorization)
def generate_spending_summary(df):
    """Generate AI summary from already categorized transactions"""

    if not client:
        return None

    # Calculate spending by category
    category_spending = df.groupby('category')['amount'].sum().reset_index()
    category_spending = category_spending.sort_values('amount',
                                                      ascending=False)

    # Build summary of categories and amounts
    spending_summary = "\n".join([
        f"- {row['category']}: £{row['amount']:.2f}"
        for _, row in category_spending.iterrows()
    ])

    total_spent = df['amount'].sum()

    prompt = f"""You are a friendly financial advisor. Based on this spending breakdown, provide a brief 2-paragraph summary:

Total Spending: £{total_spent:.2f}

Breakdown by category:
{spending_summary}

Provide:
1. A friendly overview of spending patterns and main areas
2. Gentle, practical optimization suggestions

Keep it conversational and supportive."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role":
                "system",
                "content":
                "You are a friendly financial advisor providing helpful spending insights."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.7)

        summary = response.choices[0].message.content
        return summary if summary else "Summary generation failed. Please try again."

    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None


# Custom CSS for theming
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #e2999b;
        color: #000000;
    }

    /* Main content */
    .main {
        background-color: #f5f5f5;
    }

    /* Sidebar styling 
    [data-testid="stSidebar"] {
        background-color: #e2999b;
        color: #000000;
    } */

    /* Headers */
    h1, h2, h3 {
        color: #000000 !important;
    }
    /* Inactive button */
    .stButton>button[disabled] {
        color: #000000 !important;
    }

    /* Primary button */
    .stButton>button[kind="primary"] {
        background-color: #832632;
        color: white;
    }

    /* Chat input send button icon */
    .stChatInput button svg {
        fill: white !important;
    }

    .stChatInput button path {
        fill: white !important;
    }

    /* Reduce chat message text size */
    .stChatMessage {
        font-size: 14px !important;
    }

    .stChatMessage p {
        font-size: 14px !important;
    }

    /* Burgundy background for user messages */
    .stChatMessage[data-testid="chat-message-user"] div:last-child div div div {
        background-color: #832632 !important; /* Burgundy */
        color: white !important;
    }

</style>
""",
            unsafe_allow_html=True)

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = None
if 'categorized' not in st.session_state:
    st.session_state.categorized = False
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'custom_categories' not in st.session_state:
    st.session_state.custom_categories = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "summarize"
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'category_colors' not in st.session_state:
    st.session_state.category_colors = DEFAULT_CATEGORY_COLORS.copy()
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'projects_list' not in st.session_state:
    st.session_state.projects_list = list_projects()
if 'targets' not in st.session_state:
    st.session_state.targets = {
        'monthly': {},  # e.g., {'July 2025': {'Groceries': 300, 'Transport': 100}}
        'yearly': {},   # e.g., {'2025': {'Groceries': 3600, 'Transport': 1200}}
        'alltime': {}   # e.g., {'Groceries': 10000, 'Transport': 5000}
    }
if 'target_period_type' not in st.session_state:
    st.session_state.target_period_type = 'monthly'  # 'monthly', 'yearly', or 'alltime'
if 'current_target_period' not in st.session_state:
    # Initialize with current month/year
    now = datetime.now()
    st.session_state.current_target_period = f"{now.strftime('%B %Y')}"  # e.g., "January 2025"

# Sidebar
with st.sidebar:
    st.title("💰 Yet Another Budget")
    st.caption("Your spending, analyzed by AI.")

    st.divider()

    # Navigation
    if st.button("📊 Summarize",
                 type="primary" if st.session_state.current_page == "summarize"
                 else "secondary",
                 use_container_width=True):
        st.session_state.current_page = "summarize"
        st.rerun()

    if st.button("📈 Analyze",
                 type="primary" if st.session_state.current_page == "analyze"
                 else "secondary",
                 use_container_width=True,
                 disabled=not st.session_state.categorized):
        st.session_state.current_page = "analyze"
        st.rerun()

    if st.button("🎯 Set Targets",
                 type="primary" if st.session_state.current_page == "targets"
                 else "secondary",
                 use_container_width=True,
                 disabled=not st.session_state.categorized):
        st.session_state.current_page = "targets"
        st.rerun()

    st.divider()

    # Import button at bottom
    if st.button("📤 Import New Data",
                 use_container_width=True,
                 type="secondary"):
        st.session_state.transactions = None
        st.session_state.categorized = False
        st.session_state.analyzed = False
        st.session_state.summary = None
        st.rerun()

    st.divider()

    # Custom categories section
    st.markdown("### 🏷️ Custom Categories")

    # Show existing custom categories
    if st.session_state.custom_categories:
        st.caption("Your custom categories:")
        for cat in st.session_state.custom_categories:
            st.markdown(f"• {cat}")
    else:
        st.caption("No custom categories yet")

    # Add new category
    new_category = st.text_input("New category name",
                                 key="new_category_input",
                                 placeholder="e.g., Pet Care")

    if st.button("➕ Add Category", use_container_width=True):
        if new_category and new_category.strip():
            # Clean the category name
            clean_name = new_category.strip().title()

            # Check if it already exists (case-insensitive)
            all_cats = get_all_categories()
            if clean_name.lower() not in [c.lower() for c in all_cats]:
                st.session_state.custom_categories.append(clean_name)
                # Auto-assign a color to the new category
                color = assign_color_to_category(clean_name)
                st.success(f"✅ Added '{clean_name}' with color {color}")
                st.rerun()
            else:
                st.error(f"❌ Category '{clean_name}' already exists")
        else:
            st.error("❌ Please enter a category name")

# ============================================
# PAGE ROUTING
# ============================================

# PAGE 1: SUMMARIZE (Import & View)
if st.session_state.current_page == "summarize":
    st.title("AI Spending Analyzer")
    
    # Projects section - always visible
    st.subheader("📁 Projects")
    
    # Refresh projects list
    st.session_state.projects_list = list_projects()
    existing_projects = [p['name'] for p in st.session_state.projects_list]
    
    if existing_projects:
        col_proj1, col_proj2 = st.columns([3, 1])
        with col_proj1:
            selected_project = st.selectbox(
                "Load existing project or start new",
                ["-- Start New --"] + existing_projects,
                index=0 if not st.session_state.current_project else (
                    (["-- Start New --"] + existing_projects).index(st.session_state.current_project)
                    if st.session_state.current_project in existing_projects else 0
                ),
                key="project_selector_main"
            )
        
        with col_proj2:
            if selected_project != "-- Start New --":
                if st.button("📂 Load Project", type="primary", use_container_width=True):
                    loaded_df, error = load_project(selected_project)
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.session_state.transactions = loaded_df
                        st.session_state.current_project = selected_project
                        st.session_state.categorized = True
                        st.session_state.analyzed = False
                        st.success(f"✅ Loaded project '{selected_project}'")
                        st.rerun()
    else:
        st.info("💡 No saved projects yet. Upload data below to create your first project.")
    
    if st.session_state.current_project:
        st.success(f"📁 Current project: **{st.session_state.current_project}**")
    
    st.divider()

    # File uploader
    if st.session_state.transactions is None:
        uploaded_file = st.file_uploader(
            "Upload your bank statement CSV",
            type=['csv'],
            help="Upload any CSV bank statement - AI will automatically identify the columns")

        if uploaded_file is not None:
            with st.spinner("🤖 Analyzing CSV format..."):
                df, error = parse_csv_with_ai(uploaded_file)
                
                if error:
                    st.error(f"❌ {error}")
                elif df is not None:
                    st.success(f"✅ Successfully parsed {len(df)} transactions" + 
                              (" (with balance tracking)" if 'balance' in df.columns else ""))
                    st.session_state.transactions = df
                    st.rerun()
        else:
            # Empty state
            st.info("👆 Upload any CSV bank statement - AI will automatically detect the format")

            # Sample data format
            with st.expander("💡 Supported Formats"):
                st.markdown("""
                The AI can parse **any CSV format** that contains:
                - **Date** (transaction date)
                - **Payee/Description** (merchant name)
                - **Amount** (transaction amount)
                - **Balance** *(optional - account balance)*
                
                Column names don't matter - the AI will figure it out!
                """)
                
                sample_df = pd.DataFrame({
                    'date': ['12/10/2025', '11/10/2025'],
                    'payee': ['TESCO STORES', 'RENT PAYMENT'],
                    'amount': [35.64, 1776.50],
                    'balance': [1500.36, 1536.00]
                })
                st.dataframe(sample_df, width='stretch')

    else:
        # Display loaded data
        df = st.session_state.transactions

        st.success(f"✅ Loaded {len(df)} transactions")

        # Step 1: Categorization
        if not st.session_state.categorized:
            st.info("🤖 Ready to categorize transactions with AI")

            if st.button("Categorize with AI", type="primary"):
                with st.spinner("Categorizing transactions..."):
                    # Only categorize (no summary yet)
                    categorized_df = categorize_transactions(df.copy())

                    if categorized_df is not None:
                        st.session_state.transactions = categorized_df
                        st.session_state.categorized = True
                        st.rerun()

        else:
            # Step 2: Review and edit categories
            st.subheader("📊 Categorized Transactions")
            st.caption(
                "Review the AI categorization. You can edit categories or add custom ones in the sidebar."
            )

            # Use data_editor to allow manual category changes
            edited_df = st.data_editor(
                df,
                column_config={
                    "category":
                    st.column_config.SelectboxColumn(
                        "Category",
                        help="Select category for this transaction",
                        width="medium",
                        options=get_all_categories(),
                        required=True,
                    )
                },
                width='stretch',
                height=400,
                hide_index=False,
                key="transaction_editor")

            # Update session state if data was edited
            if not edited_df.equals(df):
                st.session_state.transactions = edited_df
                # Reset analysis if categories were changed
                st.session_state.analyzed = False
                st.session_state.summary = None
            
            # Refresh categorization button
            if st.button("🔄 Refresh Categorization", help="Re-run AI categorization on all transactions"):
                with st.spinner("Re-categorizing transactions..."):
                    categorized_df = categorize_transactions(edited_df.copy())
                    if categorized_df is not None:
                        st.session_state.transactions = categorized_df
                        st.session_state.analyzed = False
                        st.session_state.summary = None
                        st.success("✅ Transactions re-categorized!")
                        st.rerun()

            st.divider()

            # Step 3: Generate analysis and insights
            if not st.session_state.analyzed:
                st.info("📈 Ready to analyze your spending patterns")

                if st.button("🔍 Analyze Now", type="primary"):
                    with st.spinner("Generating insights..."):
                        # Generate summary from categorized data
                        summary = generate_spending_summary(edited_df.copy())

                        if summary:
                            st.session_state.summary = summary
                            st.session_state.analyzed = True
                            st.rerun()

            else:
                # Step 4: Display analysis results
                st.subheader("📈 Spending Analysis")

                # Create two columns for pie chart and summary
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("### 📊 Breakdown by Category")

                    # Calculate spending by category
                    category_spending = edited_df.groupby(
                        'category')['amount'].sum().reset_index()
                    category_spending = category_spending.sort_values(
                        'amount', ascending=False)

                    # Create pie chart with consistent category colors
                    fig = px.pie(category_spending,
                                 values='amount',
                                 names='category',
                                 title='',
                                 color='category',
                                 color_discrete_map={
                                     cat: get_category_color(cat)
                                     for cat in category_spending['category']
                                 })
                    fig.update_traces(
                        textposition='auto',
                        textinfo='percent+label',
                        textfont_size=12,
                        marker=dict(line=dict(color='#FFFFFF', width=2)))
                    fig.update_layout(showlegend=True,
                                      height=400,
                                      legend=dict(orientation="v",
                                                  yanchor="middle",
                                                  y=0.5,
                                                  xanchor="left",
                                                  x=1.02),
                                      font=dict(family="Arial, sans-serif",
                                                size=12,
                                                color="#52181e"))

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("### 💡 AI Insights")
                    if st.session_state.summary:
                        st.markdown(st.session_state.summary)
                    else:
                        st.info("Analysis insights will appear here")
            
            st.divider()
            
            # Save/Append Project Section
            st.subheader("💾 Projects")
            
            col_save1, col_save2 = st.columns([2, 1])
            
            with col_save1:
                # Check if there are existing projects
                existing_projects = [p['name'] for p in st.session_state.projects_list]
                
                # Project save mode selection
                save_mode = st.radio(
                    "Choose action:",
                    ["Save as new project", "Append to existing project"] if existing_projects else ["Save as new project"],
                    horizontal=True
                )
                
                if save_mode == "Save as new project":
                    project_name = st.text_input(
                        "Project name",
                        placeholder="e.g., October 2024",
                        key="new_project_name"
                    )
                    
                    if st.button("💾 Save Project", type="primary"):
                        if project_name and project_name.strip():
                            success, result = save_project(project_name.strip(), edited_df)
                            if success:
                                st.success(f"✅ Project '{project_name}' saved successfully!")
                                st.session_state.current_project = project_name.strip()
                                st.session_state.projects_list = list_projects()
                            else:
                                st.error(f"❌ Error saving project: {result}")
                        else:
                            st.error("❌ Please enter a project name")
                else:
                    # Append to existing project
                    selected_project = st.selectbox(
                        "Select project to append to",
                        existing_projects,
                        key="append_project_select"
                    )
                    
                    if st.button("➕ Append to Project", type="primary"):
                        combined_df, error = append_to_project(selected_project, edited_df)
                        if error:
                            st.error(f"❌ {error}")
                        else:
                            st.success(f"✅ Added {len(edited_df)} transactions to '{selected_project}'!")
                            st.session_state.transactions = combined_df
                            st.session_state.current_project = selected_project
                            st.session_state.projects_list = list_projects()
            
            with col_save2:
                if st.session_state.current_project:
                    st.info(f"📁 Current project:\n**{st.session_state.current_project}**")
                else:
                    st.info("💡 Save your analysis to create a project")

        # Raw data expander
        with st.expander("📋 View Raw Data"):
            st.dataframe(df, width='stretch')

# PAGE 2: ANALYZE & UNDERSTAND
elif st.session_state.current_page == "analyze":
    # Reload current project data if one is selected (ensures appended transactions are shown)
    if st.session_state.current_project and st.session_state.current_project != "Current Session":
        loaded_df, error = load_project(st.session_state.current_project)
        if not error and loaded_df is not None:
            # Only update if data has changed (avoid unnecessary reruns)
            if st.session_state.transactions is None or len(loaded_df) != len(st.session_state.transactions):
                st.session_state.transactions = loaded_df
                st.session_state.categorized = True
    
    # Project selector at top
    col_title, col_project = st.columns([2, 1])
    
    with col_title:
        st.title("Analyze & Understand")
    
    with col_project:
        # Refresh projects list
        st.session_state.projects_list = list_projects()
        saved_projects = [p['name'] for p in st.session_state.projects_list]
        
        if saved_projects:
            # Add option to use current session data
            project_options = ["Current Session"] + saved_projects
            
            selected_option = st.selectbox(
                "📁 Select Project",
                project_options,
                index=0 if not st.session_state.current_project else (
                    project_options.index(st.session_state.current_project) 
                    if st.session_state.current_project in project_options else 0
                ),
                key="analyze_project_selector"
            )
            
            # Load selected project if different from current
            if selected_option != "Current Session" and selected_option != st.session_state.current_project:
                loaded_df, error = load_project(selected_option)
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.session_state.transactions = loaded_df
                    st.session_state.current_project = selected_option
                    st.session_state.categorized = True
                    st.rerun()

    if st.session_state.transactions is None or not st.session_state.categorized:
        st.warning(
            "⚠️ Please upload and categorize transactions first on the Summarize page, or load a saved project above."
        )
    else:
        df = st.session_state.transactions

        # Calculate metrics
        total_spent = df['amount'].sum()

        # Create two columns for main content and chat
        main_col, chat_col = st.columns([2, 1])

        with main_col:
            # Balance card
            st.markdown(f"""
            <div style='background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: #52181E; margin: 0 0 10px 0;'>SPENT</h3>
                <h2 style='color: #000; margin: 0;'>£{total_spent:,.2f}</h2>
            </div>
            """,
                        unsafe_allow_html=True)

            # Donut chart showing spending by category
            category_spending = df.groupby(
                'category')['amount'].sum().reset_index()
            category_spending = category_spending.sort_values('amount',
                                                              ascending=False)

            # Use category colors
            colors = [
                get_category_color(cat)
                for cat in category_spending['category']
            ]

            fig_donut = px.pie(category_spending,
                               values='amount',
                               names='category',
                               hole=0.6,
                               color='category',
                               color_discrete_map={
                                   cat: get_category_color(cat)
                                   for cat in category_spending['category']
                               })

            # Add "Spent" text in center
            fig_donut.add_annotation(text=f"Spent<br>£{total_spent:,.0f}",
                                     x=0.5,
                                     y=0.5,
                                     font_size=20,
                                     showarrow=False)

            fig_donut.update_traces(
                textposition='outside',
                textinfo='percent',
                textfont_size=12,
                marker=dict(line=dict(color='#FFFFFF', width=2)))

            fig_donut.update_layout(showlegend=True,
                                    height=400,
                                    legend=dict(orientation="v",
                                                yanchor="middle",
                                                y=0.5,
                                                xanchor="left",
                                                x=1.02),
                                    font=dict(
                                        family="Manrope, Arial, sans-serif",
                                        size=12,
                                        color="#000000"),
                                    margin=dict(t=0, b=0, l=0, r=0),
                                    paper_bgcolor='white',
                                    plot_bgcolor='white')

            st.plotly_chart(fig_donut, use_container_width=True)

            # Line chart showing spending over time
            st.markdown("### Spending Over Time")

            # Parse dates and group by date and category
            df_chart = df.copy()
            df_chart['date'] = pd.to_datetime(df_chart['date'],
                                              format='%d/%m/%Y', errors='coerce')
            df_chart = df_chart.dropna(subset=['date'])

            # Group by date and category
            timeline_data = df_chart.groupby(['date', 'category'
                                              ])['amount'].sum().reset_index()

            # Create line chart with matching colors from donut chart
            import plotly.graph_objects as go
            
            fig_line = px.line(
                timeline_data,
                x='date',
                y='amount',
                color='category',
                color_discrete_map={
                    cat: get_category_color(cat)
                    for cat in timeline_data['category'].unique()
                })

            fig_line.update_traces(mode='lines', line=dict(width=3))

            fig_line.update_layout(
                height=300,
                showlegend=True,
                legend=dict(orientation="h",
                           yanchor="bottom",
                           y=1.02,
                           xanchor="left",
                           x=0),
                xaxis=dict(showgrid=True,
                          gridcolor='#E0E0E0',
                          zeroline=False),
                yaxis=dict(showgrid=True,
                          gridcolor='#E0E0E0',
                          zeroline=False),
                font=dict(family="Manrope, Arial, sans-serif",
                         size=12,
                         color="#000000"),
                margin=dict(t=50, b=20, l=20, r=20),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )

            st.plotly_chart(fig_line, use_container_width=True)
            
            # Balance tracking chart (if balance column exists)
            if 'balance' in df.columns:
                st.markdown("### Account Balance Over Time")
                
                # Prepare balance data
                balance_data = df_chart[['date', 'balance']].copy()
                balance_data = balance_data.sort_values('date')
                balance_data = balance_data.drop_duplicates(subset=['date'], keep='last')
                
                # Create balance line chart
                fig_balance = px.line(
                    balance_data,
                    x='date',
                    y='balance',
                    markers=True
                )
                
                fig_balance.update_traces(
                    line=dict(color='#007AFF', width=3),
                    marker=dict(size=8, color='#007AFF')
                )
                
                fig_balance.update_layout(
                    height=250,
                    showlegend=False,
                    xaxis=dict(showgrid=True,
                              gridcolor='#E0E0E0',
                              zeroline=False,
                              title="Date"),
                    yaxis=dict(showgrid=True,
                              gridcolor='#E0E0E0',
                              zeroline=False,
                              title="Balance (£)"),
                    font=dict(family="Manrope, Arial, sans-serif",
                             size=12,
                             color="#000000"),
                    margin=dict(t=20, b=40, l=50, r=20),
                    paper_bgcolor='white',
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig_balance, use_container_width=True)

            # Transaction table with color-coded categories
            st.markdown("### Transactions")

            # Check if category column exists
            if 'category' not in df.columns:
                st.error(
                    "❌ Transactions need to be categorized first. Go to the Summarize page."
                )
            else:
                # Display transactions with styled dataframe
                display_df = df.head(10).copy()

                # Display with column configuration
                st.dataframe(
                    display_df[['date', 'payee', 'amount', 'category']],
                    column_config={
                        "date":
                        st.column_config.TextColumn("DATE", width="small"),
                        "payee":
                        st.column_config.TextColumn("PAYEE", width="medium"),
                        "amount":
                        st.column_config.NumberColumn("VALUE",
                                                      format="£%.2f",
                                                      width="small"),
                        "category":
                        st.column_config.TextColumn("CATEGORY",
                                                    width="medium"),
                    },
                    use_container_width=True,
                    hide_index=True)

        with chat_col:
            # Create container for entire chat section
            chat_container = st.container()

            with chat_container:
                # Chat header
                st.markdown("""
                <div style='background: white; padding: 20px 20px 10px 20px; border-radius: 10px 10px 0 0;'>
                    <h3 style='color: #52181E; margin: 0;'>CHAT</h3>
                </div>
                """,
                            unsafe_allow_html=True)

                # Scrollable chat messages container with fixed height
                chat_messages_container = st.container(height=450)

                with chat_messages_container:
                    if st.session_state.chat_messages:
                        for idx, msg in enumerate(
                                st.session_state.chat_messages):
                            if msg['role'] == 'user':
                                with st.chat_message("user"):
                                    st.markdown(msg['content'])
                            else:
                                # AI message with collapsible expander
                                with st.chat_message("assistant"):
                                    with st.expander(
                                            "View response",
                                            expanded=(idx == len(
                                                st.session_state.chat_messages)
                                                      - 1)):
                                        st.markdown(msg['content'])
                    else:
                        st.info("Ask me about your spending patterns!")

                # Chat input at bottom with white background
                st.markdown("""
                <div style='background: white; padding: 0 20px 20px 20px;'>
                </div>
                """,
                            unsafe_allow_html=True)

                user_question = st.chat_input(
                    placeholder="Explore your spending...", key="chat_input")

            if user_question and user_question.strip():
                # Add user message
                st.session_state.chat_messages.append({
                    'role': 'user',
                    'content': user_question
                })

                # Generate AI response
                with st.spinner("Thinking..."):
                    # Prepare detailed context about spending
                    category_summary = df.groupby(
                        'category')['amount'].sum().to_dict()
                    total = df['amount'].sum()

                    # Add daily spending summary
                    df_with_dates = df.copy()
                    df_with_dates['date'] = pd.to_datetime(
                        df_with_dates['date'], format='%d/%m/%Y', errors='coerce')
                    df_with_dates = df_with_dates.dropna(subset=['date'])
                    daily_spending = df_with_dates.groupby(
                        'date')['amount'].sum().sort_values(ascending=False)

                    # Create transaction list for AI
                    transactions_list = []
                    for _, row in df.iterrows():
                        if 'balance' in df.columns:
                            transactions_list.append(
                                f"{row['date']}: {row['payee']} - £{row['amount']:.2f} ({row['category']}) [Balance: £{row['balance']:.2f}]"
                            )
                        else:
                            transactions_list.append(
                                f"{row['date']}: {row['payee']} - £{row['amount']:.2f} ({row['category']})"
                            )
                    
                    # Add balance info if available
                    balance_info = ""
                    if 'balance' in df.columns and len(df_with_dates) > 0:
                        try:
                            latest_balance = df_with_dates.sort_values('date').iloc[-1]['balance']
                            earliest_balance = df_with_dates.sort_values('date').iloc[0]['balance']
                            balance_info = f"""
BALANCE TRACKING:
- Current balance: £{latest_balance:.2f}
- Starting balance: £{earliest_balance:.2f}
- Balance change: £{latest_balance - earliest_balance:.2f}
"""
                        except:
                            balance_info = ""

                    context = f"""User's spending data{f' for project: {st.session_state.current_project}' if st.session_state.current_project else ''}:

SUMMARY:
- Total spent: £{total:.2f}
- Number of transactions: {len(df)}
- Breakdown by category: {', '.join([f'{k}: £{v:.2f}' for k, v in category_summary.items()])}
{balance_info}
DAILY SPENDING (top 5 days):
{chr(10).join([f'- {date.strftime("%d/%m/%Y")}: £{amount:.2f}' for date, amount in daily_spending.head(5).items()])}

ALL TRANSACTIONS:
{chr(10).join(transactions_list[:20])}
{"..." if len(transactions_list) > 20 else ""}

User question: {user_question}

Provide a helpful, specific response using the transaction data above. You can analyze dates, identify patterns, compare days/categories, track balance changes, etc."""

                    if client:
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{
                                    "role":
                                    "system",
                                    "content":
                                    "You are a helpful financial coaching assistant. Analyze the transaction data carefully and provide specific, data-driven insights."
                                }, {
                                    "role": "user",
                                    "content": context
                                }],
                                temperature=0.7,
                                max_tokens=1000)

                            ai_response = response.choices[0].message.content

                            # Add AI message
                            st.session_state.chat_messages.append({
                                'role':
                                'assistant',
                                'content':
                                ai_response
                            })

                            st.rerun()
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
                    else:
                        st.error("OpenAI API key not configured")

# PAGE 3: SET TARGETS
elif st.session_state.current_page == "targets":
    # Helper functions for period navigation
    def get_next_period(current_period, period_type):
        if period_type == 'monthly':
            # Parse "January 2025" format
            date_obj = datetime.strptime(current_period, '%B %Y')
            next_date = date_obj + relativedelta(months=1)
            return next_date.strftime('%B %Y')
        elif period_type == 'yearly':
            # Parse "2025" format
            year = int(current_period)
            return str(year + 1)
        else:  # alltime
            return current_period
    
    def get_prev_period(current_period, period_type):
        if period_type == 'monthly':
            date_obj = datetime.strptime(current_period, '%B %Y')
            prev_date = date_obj - relativedelta(months=1)
            return prev_date.strftime('%B %Y')
        elif period_type == 'yearly':
            year = int(current_period)
            return str(year - 1)
        else:  # alltime
            return current_period
    
    st.title("🎯 Set Spending Targets")
    
    # Create main layout with chat
    main_col, chat_col = st.columns([2, 1])
    
    with main_col:
        # Period type selector
        st.subheader("Target Period")
        
        period_type = st.radio(
            "How often do you want to track targets?",
            ["Monthly", "Yearly", "All-Time"],
            horizontal=True,
            index=0 if st.session_state.target_period_type == 'monthly' else (1 if st.session_state.target_period_type == 'yearly' else 2)
        )
        
        # Update period type if changed
        new_period_type = period_type.lower().replace('-', '')
        if new_period_type != st.session_state.target_period_type:
            st.session_state.target_period_type = new_period_type
            # Update current period based on new type
            now = datetime.now()
            if new_period_type == 'monthly':
                st.session_state.current_target_period = now.strftime('%B %Y')
            elif new_period_type == 'yearly':
                st.session_state.current_target_period = str(now.year)
            else:  # alltime
                st.session_state.current_target_period = 'All Time'
            st.rerun()
        
        # Period navigation (only for monthly/yearly)
        if st.session_state.target_period_type != 'alltime':
            col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])
            
            with col_nav1:
                if st.button("◀ Previous", use_container_width=True):
                    st.session_state.current_target_period = get_prev_period(
                        st.session_state.current_target_period, 
                        st.session_state.target_period_type
                    )
                    st.rerun()
            
            with col_nav2:
                st.markdown(f"<h3 style='text-align: center;'>{st.session_state.current_target_period}</h3>", 
                           unsafe_allow_html=True)
            
            with col_nav3:
                if st.button("Next ▶", use_container_width=True):
                    st.session_state.current_target_period = get_next_period(
                        st.session_state.current_target_period, 
                        st.session_state.target_period_type
                    )
                    st.rerun()
        else:
            st.markdown(f"<h3 style='text-align: center;'>All-Time Targets</h3>", 
                       unsafe_allow_html=True)
        
        st.divider()
        
        # Get all categories
        all_categories = get_all_categories()
        
        # Get current targets for this period
        period_key = st.session_state.current_target_period
        period_targets = st.session_state.targets[st.session_state.target_period_type].get(period_key, {})
        
        # Category targets section
        st.subheader("Category Targets")
        st.caption("Set spending limits for each category")
        
        # Create target inputs for each category
        updated_targets = {}
        
        for category in all_categories:
            col_cat, col_input = st.columns([2, 1])
            
            with col_cat:
                # Get category color
                color = get_category_color(category)
                # Display category with colored badge
                st.markdown(
                    f'<div style="background-color: {color}; color: white; padding: 8px 16px; '
                    f'border-radius: 20px; display: inline-block; margin: 4px 0;">'
                    f'{category}</div>',
                    unsafe_allow_html=True
                )
            
            with col_input:
                # Input field for target amount
                current_value = period_targets.get(category, 0.0)
                target_value = st.number_input(
                    f"£",
                    min_value=0.0,
                    value=float(current_value),
                    step=10.0,
                    key=f"target_{category}_{period_key}",
                    label_visibility="collapsed"
                )
                updated_targets[category] = target_value
        
        # Save button
        if st.button("💾 Save Targets", type="primary", use_container_width=True):
            # Update targets in session state
            if period_key not in st.session_state.targets[st.session_state.target_period_type]:
                st.session_state.targets[st.session_state.target_period_type][period_key] = {}
            
            st.session_state.targets[st.session_state.target_period_type][period_key] = updated_targets
            st.success(f"✅ Targets saved for {period_key}!")
    
    with chat_col:
        # Create container for entire chat section
        chat_container = st.container()

        with chat_container:
            # Chat header
            st.markdown("""
            <div style='background: white; padding: 20px 20px 10px 20px; border-radius: 10px 10px 0 0;'>
                <h3 style='color: #52181E; margin: 0;'>CHAT</h3>
            </div>
            """,
                        unsafe_allow_html=True)

            # Scrollable chat messages container with fixed height
            chat_messages_container = st.container(height=450)

            with chat_messages_container:
                if st.session_state.chat_messages:
                    for idx, msg in enumerate(st.session_state.chat_messages):
                        if msg['role'] == 'user':
                            with st.chat_message("user"):
                                st.markdown(msg['content'])
                        else:
                            # AI message with collapsible expander
                            with st.chat_message("assistant"):
                                with st.expander(
                                        "View response",
                                        expanded=(idx == len(st.session_state.chat_messages) - 1)):
                                    st.markdown(msg['content'])
                else:
                    st.info("Ask me about your spending or set budgets!")

            # Chat input at bottom with white background
            st.markdown("""
            <div style='background: white; padding: 0 20px 20px 20px;'>
            </div>
            """,
                        unsafe_allow_html=True)

            user_question = st.chat_input(
                placeholder="Ask about targets or request changes...", key="targets_chat_input")

        if user_question and user_question.strip():
            # Add user message
            st.session_state.chat_messages.append({
                'role': 'user',
                'content': user_question
            })
            
            # OpenAI function calling for target updates
            if st.session_state.transactions is not None and st.session_state.categorized:
                df = st.session_state.transactions
                total = df['amount'].sum()
                category_summary = df.groupby('category')['amount'].sum().to_dict()
                
                # Build context with targets
                targets_context = f"\nCURRENT TARGETS ({st.session_state.target_period_type} - {st.session_state.current_target_period}):\n"
                current_targets = st.session_state.targets[st.session_state.target_period_type].get(
                    st.session_state.current_target_period, {}
                )
                
                if current_targets:
                    for cat, target in current_targets.items():
                        actual = category_summary.get(cat, 0)
                        targets_context += f"- {cat}: £{target:.2f} target, £{actual:.2f} spent\n"
                else:
                    targets_context += "No targets set for this period yet.\n"
                
                # Get list of all categories
                all_cats = get_all_categories()
                
                context = f"""User's spending data:

SUMMARY:
- Total spent: £{total:.2f}
- Number of transactions: {len(df)}
- Breakdown by category: {', '.join([f'{k}: £{v:.2f}' for k, v in category_summary.items()])}
{targets_context}

Available categories: {', '.join(all_cats)}
Current period type: {st.session_state.target_period_type}
Current period: {st.session_state.current_target_period}

User question: {user_question}

Provide helpful financial coaching. If the user wants to set or modify targets, use the update_targets function to make the changes."""
                
                # Define function for AI to update targets
                update_targets_function = {
                    "name": "update_targets",
                    "description": "Update spending targets for one or more categories. The targets will be set for the current period.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "targets": {
                                "type": "object",
                                "description": "Dictionary of category names to target amounts in GBP",
                                "additionalProperties": {
                                    "type": "number"
                                }
                            }
                        },
                        "required": ["targets"]
                    }
                }
                
                if client:
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{
                                "role": "system",
                                "content": """You are a helpful financial coaching assistant. When users ask you to set, update, or propose budgets/targets, you MUST use the update_targets function to make the changes. 

Examples of when to use the function:
- "Set my groceries to £300" -> Call update_targets with {"Groceries": 300}
- "Propose a budget for all categories" -> Call update_targets with all categories
- "Update my dining budget to £150" -> Call update_targets with {"Dining & Takeout": 150}

Always call the function when setting/updating targets - don't just describe what should be done."""
                            }, {
                                "role": "user",
                                "content": context
                            }],
                            functions=[update_targets_function],
                            function_call="auto",
                            temperature=0.7,
                            max_tokens=1000
                        )
                        
                        response_message = response.choices[0].message
                        
                        # Check if AI wants to call a function
                        if response_message.function_call:
                            function_name = response_message.function_call.name
                            function_args = json.loads(response_message.function_call.arguments)
                            
                            if function_name == "update_targets":
                                # Validate function arguments
                                if 'targets' not in function_args:
                                    ai_response = "❌ Error: Could not update targets. Invalid format returned by AI."
                                else:
                                    # Update the targets
                                    period_key = st.session_state.current_target_period
                                    if period_key not in st.session_state.targets[st.session_state.target_period_type]:
                                        st.session_state.targets[st.session_state.target_period_type][period_key] = {}
                                    
                                    # Update each target
                                    for category, amount in function_args['targets'].items():
                                        st.session_state.targets[st.session_state.target_period_type][period_key][category] = amount
                                        
                                        # Clear the widget state so it updates with new value
                                        widget_key = f"target_{category}_{period_key}"
                                        if widget_key in st.session_state:
                                            del st.session_state[widget_key]
                                    
                                    # Create confirmation message
                                    updates_text = ", ".join([f"{cat}: £{amt:.2f}" for cat, amt in function_args['targets'].items()])
                                    ai_response = f"✅ I've updated your targets for {period_key}:\n\n{updates_text}\n\nYour new targets are now in effect!"
                        else:
                            ai_response = response_message.content if response_message.content else "I can help you set budgets! Just tell me which categories and amounts you'd like."
                        
                        st.session_state.chat_messages.append({
                            'role': 'assistant',
                            'content': ai_response
                        })
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.error("OpenAI API key not configured")
