# Yet Another Budget - AI-Powered Spending Analysis

## Overview

Yet Another Budget is a personal finance application that analyzes spending patterns using AI. The application allows users to import financial data, view transactions, and understand their spending habits through AI-powered categorization and insights.

## Recent Changes

**2025-10-16: Project-Scoped Target Persistence**
- Implemented automatic saving of category targets to project files
- Targets now persist independently per project (each project maintains its own targets)
- Added guard logic to prevent errors when working with "Current Session" (in-memory data)
- Manual target saves and AI-updated targets both automatically write to project JSON
- User-friendly messaging when working with unsaved sessions

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
- Custom color scheme: Primary (#832632), Background (#f5f5f5), Secondary (#e2999b), Text (#52181e)
- Wide layout configuration for better data table viewing
- Collapsed sidebar by default to maximize content area
- Custom branding through page configuration (icon, title)

**State Management Approach**
- Streamlit's session_state for persistent user data across reruns
- Custom categories stored in session state (`custom_categories`)
- Automatic re-execution model handles UI updates reactively
- No explicit client-side state management required

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