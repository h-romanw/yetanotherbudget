# Yet Another Budget - AI-Powered Spending Analysis

## Overview

Yet Another Budget is a personal finance application that analyzes spending patterns using AI. The application allows users to import financial data, view transactions, and understand their spending habits through AI-powered categorization and insights.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Stack**
- Streamlit as the primary web framework for rapid UI development
- Python-based reactive components with automatic state management
- Plotly Express for interactive data visualizations
- Pandas for data manipulation and transaction processing

**UI Framework Decision**
- Streamlit's built-in component library for forms, charts, and tables
- Custom theming via `.streamlit/config.toml` for brand colors
- Wide layout mode optimized for data visualization
- Rationale: Streamlit enables rapid development of data-focused applications without frontend framework complexity

**Design System**
- Based on Figma design specifications with Manrope font family
- Custom color scheme: Primary burgundy (#832632), Dark burgundy (#52181E), Light pink (#E2999B)
- Category-specific colors: Groceries (green), Bills (purple), Transport (blue), etc.
- External CSS file (styles.css) for reusable styling across future React migration
- Wide layout configuration for better data visualization
- Expanded sidebar with navigation between pages
- Pill-shaped buttons with shadows and hover effects
- Custom branding through page configuration (icon, title)

**State Management Approach**
- Streamlit's session_state for persistent user data across reruns
- Custom categories stored in session state (`custom_categories`)
- Page navigation managed via `current_page` session state
- Chat messages persisted in `chat_messages` session state
- Automatic re-execution model handles UI updates reactively
- No explicit client-side state management required

**Multi-Page Architecture**
- Page 1: "Summarize" - Import CSV, categorize transactions, view analysis
- Page 2: "Analyze & Understand" - Interactive charts, AI chat coaching, spending insights
- Sidebar navigation with active page highlighting
- Conditional rendering based on `current_page` session state
- Disabled navigation to "Analyze" page until transactions are categorized

### Backend Architecture

**Server Framework**
- Streamlit's built-in server handles HTTP requests and WebSocket connections
- Python backend running on port 5000, bound to 0.0.0.0 for external access
- Single-file application structure (app.py) with modular function organization

**API Structure**
- OpenAI API integration for AI-powered transaction categorization
- Batch processing approach for categorizing multiple transactions
- Environment-based API key management via Replit Secrets
- Error handling for missing API credentials with user-friendly messaging

**Data Processing Pipeline**
- Pandas DataFrames as the core data structure for transactions
- AI-powered categorization with predefined category taxonomy
- Support for custom user-defined categories alongside defaults
- Batch processing to optimize API calls and improve performance

**Category System**
- Predefined categories: Groceries, Dining & Takeout, Bills & Utilities, Transport, Shopping, Entertainment, Health & Fitness, Travel, Other
- Extensible system allowing users to add custom categories
- Combined category list merges defaults with user additions
- Global category constants for consistency across the application

## External Dependencies

**AI Services**
- OpenAI API for transaction categorization and spending analysis
- GPT models used for natural language understanding of transaction descriptions
- API key managed through environment variables (OPENAI_API_KEY)

**Python Libraries**
- streamlit: Web application framework and server
- pandas: Data manipulation and analysis
- plotly: Interactive visualization library (Plotly Express)
- openai: Official OpenAI Python client

**Deployment Environment**
- Replit hosting platform
- Environment secrets management for API keys
- Port 5000 configured for web traffic
- Python 3 runtime environment