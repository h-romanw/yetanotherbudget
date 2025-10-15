from typing import Literal
import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import json
import os

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

# Category color mapping (consistent across all visualizations)
CATEGORY_COLORS = {
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


def get_all_categories():
    """Get combined list of default + custom categories"""
    return CATEGORIES + st.session_state.get('custom_categories', [])


def get_category_color(category):
    """Get color for a category (with fallback for custom categories)"""
    return CATEGORY_COLORS.get(category,
                               "#832632")  # Fallback to burgundy for custom


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

    st.button("🎯 Set Targets (WIP)", disabled=True, use_container_width=True)

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
                st.success(f"✅ Added '{clean_name}'")
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

    # File uploader
    if st.session_state.transactions is None:
        uploaded_file = st.file_uploader(
            "Upload your bank statement CSV",
            type=['csv'],
            help="CSV should contain columns: date, payee, amount")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # Validate columns
                required_cols = ['date', 'payee', 'amount']
                if not all(col in df.columns for col in required_cols):
                    st.error(
                        f"❌ CSV must contain columns: {', '.join(required_cols)}"
                    )
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

            # Add color indicator column with dot
            df_with_indicator = df.copy()
            df_with_indicator['●'] = df_with_indicator['category'].apply(
                lambda cat: '●')

            # Use data_editor to allow manual category changes
            edited_df = st.data_editor(
                df_with_indicator,
                column_config={
                    "●":
                    st.column_config.TextColumn("",
                                                width="small",
                                                disabled=True),
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
                key="transaction_editor",
                column_order=["date", "payee", "amount", "●", "category"])

            # Show category legend with colored badges
            st.caption("**Category Colors:**")
            if 'category' in edited_df.columns:
                legend_cols = st.columns(
                    min(len(edited_df['category'].unique()), 5))
                for idx, category in enumerate(
                        list(edited_df['category'].unique())[:5]):
                    with legend_cols[idx]:
                        color = get_category_color(category)
                        st.markdown(
                            f'<div style="background: {color}; color: white; padding: 5px 10px; border-radius: 15px; text-align: center; font-size: 12px; margin: 2px;">{category}</div>',
                            unsafe_allow_html=True)

            # Update session state if data was edited
            if not edited_df.equals(df_with_indicator):
                # Remove indicator column before saving (it's computed dynamically)
                edited_df_clean = edited_df.drop(columns=['●'])
                st.session_state.transactions = edited_df_clean
                # Reset analysis if categories were changed
                st.session_state.analyzed = False
                st.session_state.summary = None

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

        # Raw data expander
        with st.expander("📋 View Raw Data"):
            st.dataframe(df, width='stretch')

# PAGE 2: ANALYZE & UNDERSTAND
elif st.session_state.current_page == "analyze":
    st.title("Analyze & Understand")

    if st.session_state.transactions is None or not st.session_state.categorized:
        st.warning(
            "⚠️ Please upload and categorize transactions first on the Summarize page."
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
                                              format='%d/%m/%Y')

            # Group by date and category
            timeline_data = df_chart.groupby(['date', 'category'
                                              ])['amount'].sum().reset_index()

            # Create line chart with gradient fills
            # fig_line = px.line(
            #     timeline_data,
            #     x='date',
            #     y='amount',
            #     color='category',
            #     color_discrete_map={
            #         cat: get_category_color(cat)
            #         for cat in timeline_data['category'].unique()
            #     })
            fig_line = st.line_chart(df_chart,
                                     x='date',
                                     y='amount',
                                     x_label='Date',
                                     y_label='Amount',
                                     color='category')
            # # Update traces to add gradient-like fills
            # for trace in fig_line.data:
            #     trace.update(mode='lines',
            #                  line=dict(width=3),
            #                  fill='tonexty',
            #                  fillcolor=trace.line.color.replace(
            #                      'rgb', 'rgba').replace(')', ', 0.2)'))

            fig_line.update_layout(height=300,
                                   showlegend=False,
                                   xaxis=dict(showgrid=True,
                                              gridcolor='#E0E0E0',
                                              zeroline=False),
                                   yaxis=dict(showgrid=True,
                                              gridcolor='#E0E0E0',
                                              zeroline=False),
                                   font=dict(
                                       family="Manrope, Arial, sans-serif",
                                       size=12,
                                       color="#000000"),
                                   margin=dict(t=20, b=20, l=20, r=20),
                                   paper_bgcolor='white',
                                   plot_bgcolor='white')

            st.plotly_chart(fig_line, use_container_width=True)

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

                # Add color indicator column with colored circle emoji
                display_df['●'] = display_df['category'].apply(lambda cat: '●')

                # Custom CSS for styled table
                st.markdown("""
                <style>
                    .stDataFrame {
                        font-family: 'Manrope', Arial, sans-serif;
                    }
                </style>
                """,
                            unsafe_allow_html=True)

                # Display with column configuration for better styling
                st.dataframe(
                    display_df[['date', 'payee', 'amount', '●', 'category']],
                    column_config={
                        "date":
                        st.column_config.TextColumn("DATE", width="small"),
                        "payee":
                        st.column_config.TextColumn("PAYEE", width="medium"),
                        "amount":
                        st.column_config.NumberColumn("VALUE",
                                                      format="£%.2f",
                                                      width="small"),
                        "●":
                        st.column_config.TextColumn("", width="small"),
                        "category":
                        st.column_config.TextColumn("CATEGORY",
                                                    width="medium"),
                    },
                    use_container_width=True,
                    hide_index=True)

                # Show category legend with colored badges
                st.caption("**Categories:**")
                legend_cols = st.columns(len(df['category'].unique()))
                for idx, category in enumerate(df['category'].unique()):
                    with legend_cols[idx]:
                        color = get_category_color(category)
                        st.markdown(
                            f'<div style="background: {color}; color: white; padding: 3px 8px; border-radius: 12px; text-align: center; font-size: 11px; margin: 2px;">{category}</div>',
                            unsafe_allow_html=True)

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
                        df_with_dates['date'], format='%d/%m/%Y')
                    daily_spending = df_with_dates.groupby(
                        'date')['amount'].sum().sort_values(ascending=False)

                    # Create transaction list for AI
                    transactions_list = []
                    for _, row in df.iterrows():
                        transactions_list.append(
                            f"{row['date']}: {row['payee']} - £{row['amount']:.2f} ({row['category']})"
                        )

                    context = f"""User's spending data:

SUMMARY:
- Total spent: £{total:.2f}
- Number of transactions: {len(df)}
- Breakdown by category: {', '.join([f'{k}: £{v:.2f}' for k, v in category_summary.items()])}

DAILY SPENDING (top 5 days):
{chr(10).join([f'- {date.strftime("%d/%m/%Y")}: £{amount:.2f}' for date, amount in daily_spending.head(5).items()])}

ALL TRANSACTIONS:
{chr(10).join(transactions_list[:20])}
{"..." if len(transactions_list) > 20 else ""}

User question: {user_question}

Provide a helpful, specific response using the transaction data above. You can analyze dates, identify patterns, compare days/categories, etc."""

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
