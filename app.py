from typing import Literal
import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv
import json
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

load_dotenv()

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
        "⚠️ OpenAI API key not found. Please add OPENAI_API_KEY to your .env file in the root directory."
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


def sync_categories_from_dataframe(df):
    """Detect and add any categories from the DataFrame that aren't in the category lists"""
    if df is None or 'category' not in df.columns:
        return
    
    # Get unique categories from the DataFrame
    df_categories = df['category'].unique().tolist()
    
    # Get current category lists
    all_current = get_all_categories()
    
    # Find categories in DataFrame that aren't registered
    new_categories = []
    for cat in df_categories:
        if cat and cat not in all_current:
            new_categories.append(cat)
    
    # Add new categories to custom categories list
    if new_categories:
        if 'custom_categories' not in st.session_state:
            st.session_state.custom_categories = []
        
        for cat in new_categories:
            if cat not in st.session_state.custom_categories:
                st.session_state.custom_categories.append(cat)
                # Auto-assign color
                assign_color_to_category(cat)


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

#Refactoring core helper functions
def get_transactions_for_period(df, period_type: str, period_key: str) -> pd.DataFrame:
    """
    Filter transactions that occurred during a specific period.
    
    Args:
        df: DataFrame with 'date' column in format 'dd/mm/yyyy'
        period_type: 'monthly' | 'yearly' | 'alltime'
        period_key: 'July 2025' | '2025' | 'All Time'
    
    Returns:
        Filtered DataFrame of transactions matching the period
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    df_copy = df.copy()
    
    # Parse dates safely
    df_copy['date'] = pd.to_datetime(df_copy['date'], format='%d/%m/%Y', errors='coerce')
    
    if period_type == 'alltime':
        return df_copy
    
    elif period_type == 'yearly':
        try:
            year = int(period_key)
            return df_copy[df_copy['date'].dt.year == year]
        except (ValueError, TypeError):
            return df_copy
    
    elif period_type == 'monthly':
        try:
            # Parse "July 2025" -> July, 2025
            date_obj = datetime.strptime(period_key, '%B %Y')
            return df_copy[
                (df_copy['date'].dt.month == date_obj.month) &
                (df_copy['date'].dt.year == date_obj.year)
            ]
        except (ValueError, TypeError):
            return df_copy
    
    return df_copy


def get_spending_for_period(df, period_type: str, period_key: str, category: str = None) -> float:
    """
    Get total spending for a period, optionally filtered by category.
    
    Args:
        df: DataFrame with spending data
        period_type: 'monthly' | 'yearly' | 'alltime'
        period_key: Period identifier (e.g., 'July 2025', '2025', 'All Time')
        category: Optional category name to filter by
    
    Returns:
        Total amount spent as float
    """
    period_df = get_transactions_for_period(df, period_type, period_key)
    
    if period_df.empty:
        return 0.0
    
    if category:
        period_df = period_df[period_df['category'] == category]
    
    return float(period_df['amount'].sum())


def get_target_progress(df, period_type: str, period_key: str, targets: dict) -> dict:
    """
    Calculate progress towards targets for a period.
    
    Args:
        df: DataFrame with spending data
        period_type: 'monthly' | 'yearly' | 'alltime'
        period_key: Period identifier
        targets: Dict of {category: target_amount}
    
    Returns:
        Dict of {category: {'spent': X, 'target': Y, 'remaining': Z, 'percent': P, 'over_budget': bool}}
    """
    progress = {}
    
    for category, target_amount in targets.items():
        spent = get_spending_for_period(df, period_type, period_key, category)
        remaining = max(0, target_amount - spent)
        percent = (spent / target_amount * 100) if target_amount > 0 else 0
        
        progress[category] = {
            'spent': round(spent, 2),
            'target': round(target_amount, 2),
            'remaining': round(remaining, 2),
            'percent': min(100, round(percent, 1)),
            'over_budget': spent > target_amount
        }
    
    return progress


def get_prev_period(period_key: str, period_type: str) -> str:
    """
    Get the previous period based on the current period and type.
    
    Args:
        period_key: Current period (e.g., 'July 2025', '2025', 'All Time')
        period_type: 'monthly' | 'yearly' | 'alltime'
    
    Returns:
        Previous period string
    """
    if period_type == 'alltime':
        return 'All Time'
    
    if period_type == 'yearly':
        try:
            year = int(period_key)
            return str(year - 1)
        except (ValueError, TypeError):
            return period_key
    
    if period_type == 'monthly':
        try:
            date_obj = datetime.strptime(period_key, '%B %Y')
            prev_month = date_obj - relativedelta(months=1)
            return prev_month.strftime('%B %Y')
        except (ValueError, TypeError):
            return period_key
    
    return period_key


def get_next_period(period_key: str, period_type: str) -> str:
    """
    Get the next period based on the current period and type.
    
    Args:
        period_key: Current period (e.g., 'July 2025', '2025', 'All Time')
        period_type: 'monthly' | 'yearly' | 'alltime'
    
    Returns:
        Next period string
    """
    if period_type == 'alltime':
        return 'All Time'
    
    if period_type == 'yearly':
        try:
            year = int(period_key)
            return str(year + 1)
        except (ValueError, TypeError):
            return period_key
    
    if period_type == 'monthly':
        try:
            date_obj = datetime.strptime(period_key, '%B %Y')
            next_month = date_obj + relativedelta(months=1)
            return next_month.strftime('%B %Y')
        except (ValueError, TypeError):
            return period_key
    
    return period_key


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
    """Save categorized transactions, targets, and chat history to a project file"""
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
            'targets': st.session_state.targets,
            'custom_categories': st.session_state.get('custom_categories', []),
            'category_colors': st.session_state.get('category_colors', DEFAULT_CATEGORY_COLORS.copy()),
            'chat_messages': st.session_state.get('chat_messages', [])
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True, filename
    except Exception as e:
        return False, str(e)


def load_project(project_name):
    """Load project from file including targets and chat history"""
    try:
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"projects/{safe_name}.json"
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['transactions'])
        
        # Reset to clean state first to ensure project isolation
        st.session_state.targets = {
            'monthly': {},
            'yearly': {},
            'alltime': {}
        }
        st.session_state.chat_messages = []
        st.session_state.custom_categories = []
        st.session_state.category_colors = DEFAULT_CATEGORY_COLORS.copy()
        
        # Load project-specific targets if they exist
        if 'targets' in data:
            st.session_state.targets = data['targets']
        
        # Load project-specific chat history if it exists
        if 'chat_messages' in data:
            st.session_state.chat_messages = data['chat_messages']
        
        # Load project-specific custom categories if they exist
        if 'custom_categories' in data:
            st.session_state.custom_categories = data['custom_categories']
        
        # Load project-specific category colors if they exist
        if 'category_colors' in data:
            st.session_state.category_colors = data['category_colors']
        
        # Sync categories from DataFrame (in case manual edits added new ones)
        sync_categories_from_dataframe(df)
        
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


def delete_project(project_name):
    """Delete a project file"""
    try:
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"projects/{safe_name}.json"
        
        if os.path.exists(filename):
            os.remove(filename)
            return True, None
        else:
            return False, "Project file not found"
    except Exception as e:
        return False, f"Error deleting project: {str(e)}"


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
        
        # Sync categories from the combined dataframe
        sync_categories_from_dataframe(combined_df)
        
        # Save back to project
        success, result = save_project(project_name, combined_df)
        if success:
            return combined_df, None
        else:
            return None, result
    except Exception as e:
        return None, f"Error appending to project: {str(e)}"


def save_targets_to_project(project_name):
    """Save current targets and chat history to the project file without modifying transactions"""
    try:
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"projects/{safe_name}.json"
        
        # Read existing project data
        if not os.path.exists(filename):
            return False, "Project file not found"
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Update targets, chat history, categories, and colors
        data['targets'] = st.session_state.targets
        data['chat_messages'] = st.session_state.get('chat_messages', [])
        data['custom_categories'] = st.session_state.get('custom_categories', [])
        data['category_colors'] = st.session_state.get('category_colors', DEFAULT_CATEGORY_COLORS.copy())
        
        # Write back to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True, None
    except Exception as e:
        return False, f"Error saving targets: {str(e)}"


def update_targets_and_save(period_type, period_key, category, amount):
    """Update a target and automatically save to current project (only for real saved projects)"""
    try:
        # Update the target in session state
        if period_key not in st.session_state.targets[period_type]:
            st.session_state.targets[period_type][period_key] = {}
        
        st.session_state.targets[period_type][period_key][category] = amount
        
        # Save to project file only if a real project (not "Current Session") is loaded
        if st.session_state.current_project and st.session_state.current_project != "Current Session":
            save_targets_to_project(st.session_state.current_project)
        
        return True
    except Exception as e:
        st.error(f"Error updating targets: {str(e)}")
        return False


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
if 'chat_messages_page2' not in st.session_state:
    st.session_state.chat_messages_page2 = []
if 'chat_messages_page3' not in st.session_state:
    st.session_state.chat_messages_page3 = []
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
                 use_container_width=True):
        st.session_state.current_page = "targets"
        st.rerun()

    st.divider()

    # Import button at bottom
    if st.button("📤 Import New Data",
                 use_container_width=True,
                 type="secondary",
                 help="Clear current data and start fresh"):
        st.session_state.transactions = None
        st.session_state.categorized = False
        st.session_state.analyzed = False
        st.session_state.summary = None
        st.session_state.current_project = None
        st.session_state.chat_messages = []
        st.session_state.last_uploaded_file_id = None
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
            # Calculate the correct index for the selectbox
            options = ["-- Start New --"] + existing_projects
            if st.session_state.current_project and st.session_state.current_project in existing_projects:
                default_index = options.index(st.session_state.current_project)
            else:
                default_index = 0
            
            selected_project = st.selectbox(
                "Select project to load or start new",
                options,
                index=default_index,
                key="project_selector_main"
            )
            
            # Handle project selection change
            if selected_project == "-- Start New --":
                # Clear current project if "Start New" is selected
                if st.session_state.current_project is not None:
                    st.session_state.current_project = None
                    st.session_state.transactions = None
                    st.session_state.categorized = False
                    st.session_state.analyzed = False
                    st.session_state.summary = None
                    st.rerun()
            elif selected_project != st.session_state.current_project:
                # User selected a different project - load it
                loaded_df, error = load_project(selected_project)
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.session_state.transactions = loaded_df
                    st.session_state.current_project = selected_project
                    st.session_state.categorized = True
                    st.session_state.analyzed = False
                    st.session_state.summary = None
                    # Chat messages are already loaded by load_project() function
                    st.rerun()
        
        with col_proj2:
            if selected_project != "-- Start New --":
                if st.button("🗑️ Delete", type="secondary", use_container_width=True):
                    success, error = delete_project(selected_project)
                    if success:
                        st.success(f"✅ Deleted project '{selected_project}'")
                        # Clear current project if it was deleted
                        if st.session_state.current_project == selected_project:
                            st.session_state.current_project = None
                            st.session_state.transactions = None
                            st.session_state.categorized = False
                            st.session_state.analyzed = False
                        # Refresh projects list
                        st.session_state.projects_list = list_projects()
                        st.rerun()
                    else:
                        st.error(f"❌ {error}")
    else:
        st.info("💡 No saved projects yet. Upload data below to create your first project.")
    
    # Show current project status
    if st.session_state.current_project:
        st.success(f"📁 Current project: **{st.session_state.current_project}**")
    
    st.divider()

    # File uploader - always visible
    st.subheader("📤 Upload Data")
    
    uploaded_file = st.file_uploader(
        "Upload your bank statement CSV" + (f" (will append to '{st.session_state.current_project}')" if st.session_state.current_project else ""),
        type=['csv'],
        help="Upload any CSV bank statement - AI will automatically identify the columns",
        key="csv_uploader")

    # Track if we've processed this file already
    if 'last_uploaded_file_id' not in st.session_state:
        st.session_state.last_uploaded_file_id = None

    if uploaded_file is not None:
        # Generate a unique ID for this file based on name and size
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Only process if this is a new file
        if current_file_id != st.session_state.last_uploaded_file_id:
            with st.spinner("🤖 Analyzing CSV format..."):
                df, error = parse_csv_with_ai(uploaded_file)
                st.session_state.last_uploaded_file_id = current_file_id
            
            if error:
                st.error(f"❌ {error}")
            elif df is not None:
                # If a project is loaded, append to it
                if st.session_state.current_project:
                    combined_df, append_error = append_to_project(st.session_state.current_project, df)
                    if append_error:
                        st.error(f"❌ {append_error}")
                    else:
                        st.success(f"✅ Added {len(df)} new transactions to '{st.session_state.current_project}'" + 
                                  (" (with balance tracking)" if 'balance' in df.columns else ""))
                        st.session_state.transactions = combined_df
                        st.session_state.categorized = True
                        st.rerun()
                else:
                    st.success(f"✅ Successfully parsed {len(df)} transactions" + 
                              (" (with balance tracking)" if 'balance' in df.columns else ""))
                    st.session_state.transactions = df
                    st.rerun()
    
    # Show helpful info if no data loaded yet
    if st.session_state.transactions is None:
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
    
    # Display loaded data section
    if st.session_state.transactions is not None:
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
                        # Sync any new categories
                        sync_categories_from_dataframe(categorized_df)
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
                # Sync any new categories from the dataframe
                sync_categories_from_dataframe(edited_df)
                # Reset analysis if categories were changed
                st.session_state.analyzed = False
                st.session_state.summary = None
            
            # Refresh categorization button
            if st.button("🔄 Refresh Categorization", help="Re-run AI categorization on all transactions"):
                with st.spinner("Re-categorizing transactions..."):
                    categorized_df = categorize_transactions(edited_df.copy())
                    if categorized_df is not None:
                        st.session_state.transactions = categorized_df
                        # Sync any new categories
                        sync_categories_from_dataframe(categorized_df)
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
            
            # Save Project Section
            st.subheader("💾 Save Project")
            
            # Show project name input only if no project is loaded (-- Start New -- selected)
            if not st.session_state.current_project:
                project_name = st.text_input(
                    "Project name",
                    placeholder="e.g., October 2024",
                    key="new_project_name"
                )
                
                if st.button("💾 Save as New Project", type="primary", use_container_width=True):
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
                # If project is loaded, save updates to it
                st.info(f"📁 Saving to: **{st.session_state.current_project}**")
                
                if st.button("💾 Save Changes to Project", type="primary", use_container_width=True):
                    success, result = save_project(st.session_state.current_project, edited_df)
                    if success:
                        st.success(f"✅ Project '{st.session_state.current_project}' updated successfully!")
                        st.session_state.projects_list = list_projects()
                    else:
                        st.error(f"❌ Error saving project: {result}")

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
                    if st.session_state.chat_messages_page2:
                        for idx, msg in enumerate(
                                st.session_state.chat_messages_page2):
                            if msg['role'] == 'user':
                                with st.chat_message("user"):
                                    st.markdown(msg['content'])
                            else:
                                # AI message with collapsible expander
                                with st.chat_message("assistant"):
                                    with st.expander(
                                            "View response",
                                            expanded=(idx == len(
                                                st.session_state.chat_messages_page2)
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

        # PAGE 2 CHAT RESPONSE HANDLER
        if user_question and user_question.strip():
            # Add user message
            st.session_state.chat_messages_page2.append({
                'role': 'user',
                'content': user_question
            })
            
            # Build context with transaction data
            all_cats = get_all_categories()
            current_targets = st.session_state.targets[st.session_state.target_period_type].get(
                st.session_state.current_target_period, {}
            )
            
            # Build targets context
            targets_context = f"\nCURRENT TARGETS ({st.session_state.target_period_type} - {st.session_state.current_target_period}):\n"
            if current_targets:
                for cat, target in current_targets.items():
                    targets_context += f"- {cat}: £{target:.2f}\n"
            else:
                targets_context += "No targets set for this period yet.\n"
            
            # Build context - with transaction-level data
            if st.session_state.transactions is not None and st.session_state.categorized:
                df = st.session_state.transactions
                total = df['amount'].sum()
                
                # Parse dates for detailed analysis
                df_detailed = df.copy()
                df_detailed['date'] = pd.to_datetime(df_detailed['date'], format='%d/%m/%Y', errors='coerce')
                df_detailed = df_detailed.dropna(subset=['date'])
                
                # Build detailed category breakdown with transaction details
                category_details = []
                for category in sorted(df['category'].unique()):
                    cat_df = df_detailed[df_detailed['category'] == category]
                    cat_total = cat_df['amount'].sum()
                    cat_avg = cat_df['amount'].mean()
                    cat_max = cat_df['amount'].max()
                    
                    # Get top 5 spends in this category
                    top_spends = cat_df.nlargest(5, 'amount')[['date', 'payee', 'amount']]
                    top_spends_text = "; ".join([
                        f"{row['date'].strftime('%d/%m/%Y')}: {row['payee']} £{row['amount']:.2f}"
                        for _, row in top_spends.iterrows()
                    ])
                    
                    # Get the biggest transaction
                    max_transaction = cat_df.loc[cat_df['amount'].idxmax()]
                    
                    category_details.append(
                        f"- {category}: Total £{cat_total:.2f}, Avg £{cat_avg:.2f}, Max single transaction: £{cat_max:.2f} ({max_transaction['payee']} on {max_transaction['date'].strftime('%d/%m/%Y')}) | Top 5 spends: {top_spends_text}"
                    )
                
                # Get spending progress for current period if targets exist
                period_progress = get_target_progress(
                    df,
                    st.session_state.target_period_type,
                    st.session_state.current_target_period,
                    current_targets
                )
                
                progress_text = ""
                if period_progress and any(current_targets.values()):
                    progress_text = "\nPROGRESS VS TARGETS:\n"
                    for cat, data in period_progress.items():
                        status = "✅ ON TRACK" if not data['over_budget'] else "⚠️ OVER"
                        progress_text += f"- {cat}: £{data['spent']:.2f}/£{data['target']:.2f} ({data['percent']:.1f}%) {status}\n"
                
                context = f"""User's spending data and budget targets:

PERIOD: {st.session_state.target_period_type.upper()} - {st.session_state.current_target_period}

SUMMARY:
- Total spent: £{total:.2f}
- Number of transactions: {len(df)}

DETAILED CATEGORY BREAKDOWN (with transaction details):
{chr(10).join(category_details)}

{targets_context}
{progress_text}

User question: {user_question}

CRITICAL INSTRUCTIONS FOR AI:
- You have access to detailed transaction-level data in DETAILED CATEGORY BREAKDOWN above
- Each category shows: Total, Average, Max transaction with EXACT payee name and date
- Example: "Transport: Total £500, Avg £50, Max £100 (UBER on 15/10/2025)"
- When asked "What's my biggest [category] expense?", find that category and report the max transaction with payee and date
- Use top 5 spends to explain patterns
- Compare actual spending vs targets
- Provide specific financial coaching with real transaction details"""
            else:
                context = f"""Setting budget targets (no spending data uploaded yet):

{targets_context}

Available categories: {', '.join(all_cats)}
Current period type: {st.session_state.target_period_type}
Current period: {st.session_state.current_target_period}

User question: {user_question}

Provide helpful budgeting advice."""
            
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
                    # Build message history for multi-turn conversation
                    messages = [
                        {
                            "role": "system",
                            "content": """You are a helpful financial coaching assistant.

CRITICAL: You have detailed transaction-level data in the context above. USE IT.
- Reference specific payee names and dates when answering about spending
- Look at the "Max single transaction" field for biggest expenses
- When updating budgets, use the update_targets function"""
                        }
                    ]
                    
                    # Add conversation history (excluding current question)
                    for msg in st.session_state.chat_messages_page2[:-1]:
                        messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
                    
                    # Add current question with full context
                    messages.append({
                        "role": "user",
                        "content": context
                    })
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        functions=[update_targets_function],
                        function_call="auto",
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    response_message = response.choices[0].message
                    
                    # Check if AI wants to call a function
                    if response_message.function_call:
                        function_name = response_message.function_call.name
                        
                        try:
                            function_args = json.loads(response_message.function_call.arguments)
                        except json.JSONDecodeError:
                            ai_response = "❌ Error parsing response. Please try again."
                            function_args = None
                        
                        if function_name == "update_targets" and function_args:
                            targets_to_update = None
                            
                            if 'targets' in function_args and isinstance(function_args['targets'], dict):
                                targets_to_update = function_args['targets']
                            elif isinstance(function_args, dict) and all(isinstance(v, (int, float)) for v in function_args.values()):
                                targets_to_update = function_args
                            
                            if targets_to_update:
                                period_key = st.session_state.current_target_period
                                if period_key not in st.session_state.targets[st.session_state.target_period_type]:
                                    st.session_state.targets[st.session_state.target_period_type][period_key] = {}
                                
                                # Update targets
                                for category, amount in targets_to_update.items():
                                    st.session_state.targets[st.session_state.target_period_type][period_key][category] = amount
                                    widget_key = f"target_{category}_{period_key}"
                                    if widget_key in st.session_state:
                                        del st.session_state[widget_key]
                                
                                # Save to project
                                if st.session_state.current_project and st.session_state.current_project != "Current Session":
                                    save_targets_to_project(st.session_state.current_project)
                                
                                # Create function result
                                updates_text = ", ".join([f"{cat}: £{amt:.2f}" for cat, amt in targets_to_update.items()])
                                function_result = json.dumps({
                                    "success": True,
                                    "updated_targets": targets_to_update,
                                    "period": period_key,
                                    "message": f"Successfully updated targets: {updates_text}"
                                })
                                
                                # Make follow-up call with better prompt
                                try:
                                    follow_up_messages = messages + [
                                        {
                                            "role": "assistant",
                                            "content": None,
                                            "function_call": {
                                                "name": "update_targets",
                                                "arguments": response_message.function_call.arguments
                                            }
                                        },
                                        {
                                            "role": "function",
                                            "name": "update_targets",
                                            "content": function_result
                                        },
                                        {
                                            "role": "user",
                                            "content": "Great! Can you confirm what you just updated and provide a friendly summary?"
                                        }
                                    ]
                                    
                                    follow_up_response = client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=follow_up_messages,
                                        temperature=0.7,
                                        max_tokens=1000
                                    )
                                    
                                    ai_response = follow_up_response.choices[0].message.content
                                    
                                    # If still no response, create a good default
                                    if not ai_response or not ai_response.strip():
                                        ai_response = f"✅ Perfect! I've updated your {st.session_state.target_period_type} budget targets for {period_key}:\n\n"
                                        for cat, amt in targets_to_update.items():
                                            ai_response += f"• **{cat}**: £{amt:.2f}\n"
                                        ai_response += "\nYour targets are now saved! 📊"
                                        
                                        if not st.session_state.current_project or st.session_state.current_project == "Current Session":
                                            ai_response += "\n\n💡 **Tip**: Save your data as a project to keep these targets permanently."
                                            
                                except Exception as e:
                                    # Fallback if follow-up call fails
                                    ai_response = f"✅ Updated targets:\n\n{updates_text}"
                                    if not st.session_state.current_project or st.session_state.current_project == "Current Session":
                                        ai_response += "\n\n💡 Save as project to persist targets."
                            else:
                                ai_response = "❌ Could not update targets. Please specify amounts (e.g., 'Set Groceries to £300')."
                    else:
                        ai_response = response_message.content if response_message.content else "How can I help with your budget?"
                    
                    st.session_state.chat_messages_page2.append({
                        'role': 'assistant',
                        'content': ai_response
                    })
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.error("OpenAI API key not configured")

# PAGE 3: SET TARGETS
elif st.session_state.current_page == "targets":
    # Reload current project data if one is selected
    if st.session_state.current_project and st.session_state.current_project != "Current Session":
        loaded_df, error = load_project(st.session_state.current_project)
        if not error and loaded_df is not None:
            if st.session_state.transactions is None or len(loaded_df) != len(st.session_state.transactions):
                st.session_state.transactions = loaded_df
                st.session_state.categorized = True
    
    # Project selector at top
    col_title, col_project = st.columns([2, 1])
    
    with col_title:
        st.title("Set Targets")
    
    with col_project:
        # Refresh projects list
        st.session_state.projects_list = list_projects()
        saved_projects = [p['name'] for p in st.session_state.projects_list]
        
        if saved_projects:
            project_options = ["Current Session"] + saved_projects
            
            selected_option = st.selectbox(
                "📁 Select Project",
                project_options,
                index=0 if not st.session_state.current_project else (
                    project_options.index(st.session_state.current_project) 
                    if st.session_state.current_project in project_options else 0
                ),
                key="targets_project_selector"
            )
            
            if selected_option != "Current Session" and selected_option != st.session_state.current_project:
                loaded_df, error = load_project(selected_option)
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.session_state.transactions = loaded_df
                    st.session_state.current_project = selected_option
                    st.session_state.categorized = True
                    st.rerun()
    
    # Create two columns for main content and chat
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
            now = datetime.now()
            if new_period_type == 'monthly':
                st.session_state.current_target_period = now.strftime('%B %Y')
            elif new_period_type == 'yearly':
                st.session_state.current_target_period = str(now.year)
            else:
                st.session_state.current_target_period = 'All Time'
            st.rerun()
        
        # Period navigation
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
                color = get_category_color(category)
                st.markdown(
                    f'<div style="background-color: {color}; color: white; padding: 8px 16px; '
                    f'border-radius: 20px; display: inline-block; margin: 4px 0;">'
                    f'{category}</div>',
                    unsafe_allow_html=True
                )
            
            with col_input:
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
            if period_key not in st.session_state.targets[st.session_state.target_period_type]:
                st.session_state.targets[st.session_state.target_period_type][period_key] = {}
            
            st.session_state.targets[st.session_state.target_period_type][period_key] = updated_targets
            
            if st.session_state.current_project and st.session_state.current_project != "Current Session":
                success, error = save_targets_to_project(st.session_state.current_project)
                if success:
                    st.success(f"✅ Targets saved for {period_key}!")
                else:
                    st.error(f"❌ Failed to save targets: {error}")
            else:
                st.success(f"✅ Targets saved for {period_key}!")
                if not st.session_state.current_project or st.session_state.current_project == "Current Session":
                    st.info("💡 Save your transactions as a project to persist these targets permanently.")
        
        st.divider()
        
        # TARGET PROGRESS SECTION
        st.subheader("📊 Target Progress")
        
        if st.session_state.transactions is not None and st.session_state.categorized:
            df = st.session_state.transactions
            
            progress = get_target_progress(
                df, 
                st.session_state.target_period_type, 
                period_key, 
                period_targets
            )
            
            if progress and any(period_targets.values()):
                for category in all_categories:
                    if category in progress:
                        data = progress[category]
                        spent = data['spent']
                        target = data['target']
                        percent = data['percent']
                        remaining = data['remaining']
                        over_budget = data['over_budget']
                        
                        if target > 0:
                            color = get_category_color(category)
                            
                            if over_budget:
                                status = "⚠️ OVER BUDGET"
                                status_color = "#FF3B30"
                            elif percent >= 80:
                                status = "⚠️ WARNING"
                                status_color = "#FF9500"
                            else:
                                status = "✅ ON TRACK"
                                status_color = "#34C759"
                            
                            col_cat_prog, col_amt_prog, col_stat_prog = st.columns([2, 2, 1])
                            
                            with col_cat_prog:
                                st.markdown(
                                    f'<div style="background-color: {color}; color: white; padding: 8px 16px; '
                                    f'border-radius: 20px; display: inline-block; margin: 4px 0;">'
                                    f'{category}</div>',
                                    unsafe_allow_html=True
                                )
                            
                            with col_amt_prog:
                                st.markdown(f"**£{spent:.2f}** / £{target:.2f} ({percent:.1f}%)")
                            
                            with col_stat_prog:
                                st.markdown(
                                    f'<div style="color: {status_color}; font-weight: bold; text-align: right;">'
                                    f'{status}</div>',
                                    unsafe_allow_html=True
                                )
                            
                            st.progress(min(percent / 100, 1.0))
                            
                            if remaining > 0 and not over_budget:
                                st.caption(f"💰 £{remaining:.2f} remaining")
                            elif over_budget:
                                st.caption(f"⚠️ £{abs(remaining):.2f} over budget")
                            
                            st.markdown("")
            else:
                st.info("💡 Set targets above to see your progress")
        else:
            st.info("📥 Upload and categorize transactions to see progress towards targets")
    
    with chat_col:
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Chat header
            st.markdown("""
            <div style='background: white; padding: 20px 20px 10px 20px; border-radius: 10px 10px 0 0;'>
                <h3 style='color: #52181E; margin: 0;'>CHAT</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Chat messages container
            chat_messages_container = st.container(height=450)
            
            with chat_messages_container:
                if st.session_state.chat_messages_page3:
                    for idx, msg in enumerate(st.session_state.chat_messages_page3):
                        if msg['role'] == 'user':
                            with st.chat_message("user"):
                                st.markdown(msg['content'])
                        else:
                            with st.chat_message("assistant"):
                                with st.expander(
                                    "View response",
                                    expanded=(idx == len(st.session_state.chat_messages_page3) - 1)
                                ):
                                    st.markdown(msg['content'])
                else:
                    st.info("💬 Ask me about your spending or let me help you set budgets!")
            
            # Chat input
            st.markdown("""
            <div style='background: white; padding: 0 20px 20px 20px;'>
            </div>
            """, unsafe_allow_html=True)
            
            user_question = st.chat_input(
                placeholder="Ask about spending or request budget help...", 
                key="targets_chat_input"
            )
    
    # PAGE 3 CHAT HANDLER
    if user_question and user_question.strip():
        # Add user message
        st.session_state.chat_messages_page3.append({
            'role': 'user',
            'content': user_question
        })
        
        # Build context
        all_cats = get_all_categories()
        current_targets = st.session_state.targets[st.session_state.target_period_type].get(
            st.session_state.current_target_period, {}
        )
        
        # Build targets context
        targets_context = f"\nCURRENT TARGETS ({st.session_state.target_period_type} - {st.session_state.current_target_period}):\n"
        if current_targets:
            for cat, target in current_targets.items():
                targets_context += f"- {cat}: £{target:.2f}\n"
        else:
            targets_context += "No targets set for this period yet.\n"
        
        # Build context with transaction data
        if st.session_state.transactions is not None and st.session_state.categorized:
            df = st.session_state.transactions
            total = df['amount'].sum()
            
            # Parse dates
            df_detailed = df.copy()
            df_detailed['date'] = pd.to_datetime(df_detailed['date'], format='%d/%m/%Y', errors='coerce')
            df_detailed = df_detailed.dropna(subset=['date'])
            
            # Build detailed category breakdown
            category_details = []
            for category in sorted(df['category'].unique()):
                cat_df = df_detailed[df_detailed['category'] == category]
                cat_total = cat_df['amount'].sum()
                cat_avg = cat_df['amount'].mean()
                cat_max = cat_df['amount'].max()
                
                # Top 5 spends
                top_spends = cat_df.nlargest(5, 'amount')[['date', 'payee', 'amount']]
                top_spends_text = "; ".join([
                    f"{row['date'].strftime('%d/%m/%Y')}: {row['payee']} £{row['amount']:.2f}"
                    for _, row in top_spends.iterrows()
                ])
                
                # Biggest transaction
                max_transaction = cat_df.loc[cat_df['amount'].idxmax()]
                
                category_details.append(
                    f"- {category}: Total £{cat_total:.2f}, Avg £{cat_avg:.2f}, Max single transaction: £{cat_max:.2f} ({max_transaction['payee']} on {max_transaction['date'].strftime('%d/%m/%Y')}) | Top 5 spends: {top_spends_text}"
                )
            
            # Get progress
            period_progress = get_target_progress(
                df,
                st.session_state.target_period_type,
                st.session_state.current_target_period,
                current_targets
            )
            
            progress_text = ""
            if period_progress and any(current_targets.values()):
                progress_text = "\nPROGRESS VS TARGETS:\n"
                for cat, data in period_progress.items():
                    status = "✅ ON TRACK" if not data['over_budget'] else "⚠️ OVER"
                    progress_text += f"- {cat}: £{data['spent']:.2f}/£{data['target']:.2f} ({data['percent']:.1f}%) {status}\n"
            
            context = f"""You are a helpful financial assistant. The user has uploaded their transaction data, and you can see every single transaction with payee names, dates, and amounts.

PERIOD: {st.session_state.target_period_type.upper()} - {st.session_state.current_target_period}

TRANSACTION DATA SUMMARY:
- Total spent: £{total:.2f}
- Number of transactions: {len(df)}

DETAILED SPENDING BY CATEGORY (with individual transaction details):
{chr(10).join(category_details)}

{targets_context}
{progress_text}

USER QUESTION: {user_question}

CRITICAL INSTRUCTIONS:
- THE USER IS CURRENTLY VIEWING: {st.session_state.target_period_type.upper()} period for {st.session_state.current_target_period}
- When the user says "implement these" or "set these budgets", they mean for the CURRENT PERIOD: {st.session_state.current_target_period}
- If the user asks for "November 2025" targets but you're viewing "Yearly 2025", you need to tell them to switch to Monthly view first
- You have access to detailed transaction data above. Each category shows specific payee names, dates, and amounts.
- When answering questions like "What's my biggest expense?", reference the actual payee name and date from the data.
- You can act like ChatGPT - provide context, ask follow-up questions, and have a natural conversation.
- If the user asks about setting budgets, you can propose realistic targets based on their actual spending patterns.
- Use the update_targets function to actually update budget targets when appropriate - but ONLY for the current period!
- Be conversational and helpful - you're a financial coach, not just a data analyzer."""
        else:
            context = f"""You are a helpful financial assistant.

The user hasn't uploaded transaction data yet, but they can still set budget targets.

{targets_context}

Available categories: {', '.join(all_cats)}
Current period: {st.session_state.target_period_type} - {st.session_state.current_target_period}

USER QUESTION: {user_question}

INSTRUCTIONS:
- Help the user set realistic budget targets.
- If they want to update budgets, use the update_targets function.
- Be conversational like ChatGPT - provide helpful financial advice."""
        
        # Define update targets function
        update_targets_function = {
            "name": "update_targets",
            "description": f"Update spending targets for one or more categories for the current period ({st.session_state.target_period_type} - {st.session_state.current_target_period}). ONLY use this when the user explicitly wants to set or modify budgets for THIS specific period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "description": "Dictionary of category names (e.g., 'Groceries', 'Transport') to target amounts in GBP (must be numbers, not strings)",
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
                # Build message history for multi-turn conversation
                messages = [
                    {
                        "role": "system",
                        "content": f"""You are a helpful, conversational financial assistant. You have access to the user's detailed transaction data and can reference specific transactions.

CRITICAL CONTEXT AWARENESS:
- The user is currently viewing {st.session_state.target_period_type.upper()} targets for {st.session_state.current_target_period}
- When they say "implement these" or "set these budgets", they mean for THIS period: {st.session_state.current_target_period}
- If they ask for a different period (e.g., "November" when viewing "Yearly 2025"), tell them to switch the period selector first
- ALWAYS pass actual target amounts in the targets dictionary, never empty

When the user asks you to set, update, or propose budgets, you MUST use the update_targets function to make the changes. Don't just suggest - actually do it.

FUNCTION CALLING RULES:
- Always include category names and amounts in the targets parameter
- Example: {{"targets": {{"Groceries": 300, "Transport": 150}}}}
- Never call the function with empty parameters

Be natural and conversational like ChatGPT. Ask follow-up questions, provide context, and help the user make good financial decisions."""
                    }
                ]
                
                # Add conversation history (excluding current question)
                for msg in st.session_state.chat_messages_page3[:-1]:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
                
                # Add current question with full context
                messages.append({
                    "role": "user",
                    "content": context
                })
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    functions=[update_targets_function],
                    function_call="auto",
                    temperature=0.7,
                    max_tokens=1500
                )
                
                response_message = response.choices[0].message
                
                # Handle function call
                if response_message.function_call:
                    function_name = response_message.function_call.name
                    
                    # Debug: show that function was called
                    st.write(f"DEBUG: Function called: {function_name}")
                    
                    try:
                        function_args = json.loads(response_message.function_call.arguments)
                        st.write(f"DEBUG: Function args: {function_args}")
                    except json.JSONDecodeError:
                        ai_response = "❌ I had trouble processing that request. Could you try rephrasing?"
                        function_args = None
                    
                    if function_name == "update_targets" and function_args:
                        targets_to_update = None
                        
                        if 'targets' in function_args and isinstance(function_args['targets'], dict):
                            targets_to_update = function_args['targets']
                        elif isinstance(function_args, dict) and all(isinstance(v, (int, float)) for v in function_args.values()):
                            targets_to_update = function_args
                        
                        # Check if targets_to_update is valid and not empty
                        if targets_to_update and len(targets_to_update) > 0:
                            period_key = st.session_state.current_target_period
                            if period_key not in st.session_state.targets[st.session_state.target_period_type]:
                                st.session_state.targets[st.session_state.target_period_type][period_key] = {}
                            
                            # Update targets
                            for category, amount in targets_to_update.items():
                                st.session_state.targets[st.session_state.target_period_type][period_key][category] = amount
                                widget_key = f"target_{category}_{period_key}"
                                if widget_key in st.session_state:
                                    del st.session_state[widget_key]
                            
                            # Save to project
                            if st.session_state.current_project and st.session_state.current_project != "Current Session":
                                save_targets_to_project(st.session_state.current_project)
                            
                            # Create function result
                            updates_text = ", ".join([f"{cat}: £{amt:.2f}" for cat, amt in targets_to_update.items()])
                            function_result = json.dumps({
                                "success": True,
                                "updated_targets": targets_to_update,
                                "period": period_key,
                                "message": f"Successfully updated targets: {updates_text}"
                            })
                            
                            # Make follow-up call with better prompt
                            try:
                                st.write("DEBUG: Making follow-up call...")
                                follow_up_messages = messages + [
                                    {
                                        "role": "assistant",
                                        "content": None,
                                        "function_call": {
                                            "name": "update_targets",
                                            "arguments": response_message.function_call.arguments
                                        }
                                    },
                                    {
                                        "role": "function",
                                        "name": "update_targets",
                                        "content": function_result
                                    },
                                    {
                                        "role": "user",
                                        "content": "Great! Can you confirm what you just updated and provide a friendly summary?"
                                    }
                                ]
                                
                                follow_up_response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=follow_up_messages,
                                    temperature=0.7,
                                    max_tokens=1500
                                )
                                
                                ai_response = follow_up_response.choices[0].message.content
                                st.write(f"DEBUG: Follow-up response: {ai_response}")
                                
                                # If still no response, create a good default
                                if not ai_response or not ai_response.strip():
                                    st.write("DEBUG: No AI response, using fallback")
                                    ai_response = f"✅ Perfect! I've updated your {st.session_state.target_period_type} budget targets for {period_key}:\n\n"
                                    for cat, amt in targets_to_update.items():
                                        ai_response += f"• **{cat}**: £{amt:.2f}\n"
                                    ai_response += "\nYour targets are now saved and you can see the updated progress bars above! 📊"
                                    
                                    if not st.session_state.current_project or st.session_state.current_project == "Current Session":
                                        ai_response += "\n\n💡 **Tip**: Save your data as a project to keep these targets permanently."
                                        
                            except Exception as e:
                                # Fallback if follow-up call fails
                                st.write(f"DEBUG: Follow-up call failed: {str(e)}")
                                ai_response = f"✅ I've updated your budget targets for {period_key}:\n\n{updates_text}\n\nYour targets are now saved! Check the progress bars above to see how you're doing. 📊"
                                if not st.session_state.current_project or st.session_state.current_project == "Current Session":
                                    ai_response += "\n\n💡 Save your data as a project to keep these targets."
                        else:
                            # AI called function but didn't provide targets - it might be confused about the period
                            st.write(f"DEBUG: No valid targets provided. targets_to_update = {targets_to_update}")
                            ai_response = f"❌ I couldn't determine which targets to update. The current period is **{st.session_state.target_period_type} - {st.session_state.current_target_period}**.\n\nPlease be specific, for example:\n- 'Set Groceries to £300 for {st.session_state.current_target_period}'\n- 'Update Transport to £150'\n\nMake sure the period type matches what you're asking for (you're currently viewing **{st.session_state.target_period_type}** targets)."
                    else:
                        ai_response = "❌ I had trouble understanding that request. Could you try rephrasing?"
                else:
                    ai_response = response_message.content if response_message.content else "I'm here to help with your budgets. What would you like to know?"
                
                st.session_state.chat_messages_page3.append({
                    'role': 'assistant',
                    'content': ai_response
                })
                
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("OpenAI API key not configured")

# DEBUG: Test helper functions
with st.expander("🧪 Debug: Test Helper Functions", expanded=False):
    st.warning("⚠️ This section is for development only")
    
    if st.session_state.transactions is not None and st.session_state.categorized:
        df = st.session_state.transactions
        
        st.markdown("### Test Data")
        st.write(f"Loaded {len(df)} transactions")
        st.write(f"Columns: {df.columns.tolist()}")
        
        # Test 1: get_transactions_for_period
        st.markdown("### Test 1: get_transactions_for_period()")
        
        col_t1_1, col_t1_2, col_t1_3 = st.columns(3)
        
        with col_t1_1:
            period_type_test = st.selectbox(
                "Period type",
                ["monthly", "yearly", "alltime"],
                key="test_period_type"
            )
        
        with col_t1_2:
            if period_type_test == "monthly":
                period_key_test = st.text_input(
                    "Period key",
                    "July 2025",
                    key="test_period_key"
                )
            elif period_type_test == "yearly":
                period_key_test = st.text_input(
                    "Period key",
                    "2025",
                    key="test_period_key"
                )
            else:
                period_key_test = "All Time"
        
        with col_t1_3:
            if st.button("🔍 Test", key="test_get_transactions"):
                filtered_df = get_transactions_for_period(df, period_type_test, period_key_test)
                st.success(f"✅ Returned {len(filtered_df)} transactions")
                st.dataframe(filtered_df.head(), use_container_width=True)
        
        st.divider()
        
        # Test 2: get_spending_for_period
        st.markdown("### Test 2: get_spending_for_period()")
        
        col_t2_1, col_t2_2, col_t2_3 = st.columns(3)
        
        with col_t2_1:
            period_type_test2 = st.selectbox(
                "Period type",
                ["monthly", "yearly", "alltime"],
                key="test_period_type2"
            )
        
        with col_t2_2:
            if period_type_test2 == "monthly":
                period_key_test2 = st.text_input(
                    "Period key",
                    "July 2025",
                    key="test_period_key2"
                )
            elif period_type_test2 == "yearly":
                period_key_test2 = st.text_input(
                    "Period key",
                    "2025",
                    key="test_period_key2"
                )
            else:
                period_key_test2 = "All Time"
        
        with col_t2_3:
            category_test = st.selectbox(
                "Category (optional)",
                ["All"] + df['category'].unique().tolist() if 'category' in df.columns else ["All"],
                key="test_category"
            )
        
        if st.button("🔍 Test", key="test_get_spending"):
            cat = None if category_test == "All" else category_test
            spending = get_spending_for_period(df, period_type_test2, period_key_test2, cat)
            st.success(f"✅ Total spending: £{spending:.2f}")
        
        st.divider()
        
        # Test 3: get_target_progress
        st.markdown("### Test 3: get_target_progress()")
        
        col_t3_1, col_t3_2 = st.columns(2)
        
        with col_t3_1:
            period_type_test3 = st.selectbox(
                "Period type",
                ["monthly", "yearly", "alltime"],
                key="test_period_type3"
            )
        
        with col_t3_2:
            if period_type_test3 == "monthly":
                period_key_test3 = st.text_input(
                    "Period key",
                    "July 2025",
                    key="test_period_key3"
                )
            elif period_type_test3 == "yearly":
                period_key_test3 = st.text_input(
                    "Period key",
                    "2025",
                    key="test_period_key3"
                )
            else:
                period_key_test3 = "All Time"
        
        # Create test targets
        test_targets = {
            "Groceries": 300,
            "Transport": 150,
            "Entertainment": 100
        }
        
        if st.button("🔍 Test Progress", key="test_get_progress"):
            progress = get_target_progress(df, period_type_test3, period_key_test3, test_targets)
            st.success("✅ Target progress calculated:")
            
            # Display as a nice table
            progress_df = pd.DataFrame([
                {
                    "Category": cat,
                    "Spent": f"£{data['spent']:.2f}",
                    "Target": f"£{data['target']:.2f}",
                    "Remaining": f"£{data['remaining']:.2f}",
                    "Progress": f"{data['percent']:.1f}%",
                    "Over Budget": "⚠️ YES" if data['over_budget'] else "✅ No"
                }
                for cat, data in progress.items()
            ])
            
            st.dataframe(progress_df, use_container_width=True, hide_index=True)
    else:
        st.info("💡 Upload and categorize transactions first to test helper functions")
