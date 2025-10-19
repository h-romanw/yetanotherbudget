# Yet Another Budget - AI-Powered Spending Management
# Table of Contents
1. [Overview](#overview)
2. [Installation and running the app locally](#installation-and-running-the-app-locally)
- [Troubleshooting](#troubleshooting)
3. 

# Overview

Yet Another Budget is a personal finance application that analyzes spending patterns using AI. The application allows users to import financial data, view transactions, and understand their spending habits through automated categorization and analysis. The platform is built as a full-stack TypeScript application with a React frontend and Express backend.

# Installation and running the app locally
To run the app locally, start by cloning the main branch. Ensure that you are using Python 3.13 and create a virtual environment. Then, after opening the repo,
1. Install all dependencies by running `pip install -r requirements.txt` (alternatively, copy and paste the dependencies from requirements.txt and pip install them in the terminal).
2. Create a file in .streamlit called secrets.toml, with the following code: `OPENAI_API_KEY = <Your key here>` or, if I have provided you with a secrets.toml file separately, place it into .streamlit.
3. Open app.py
4. In the terminal, run the command `streamlit run app.py` and Ctrl/Cmd-Click the URL given in the terminal.

## Troubleshooting
These are some issues I ran into during my time building the app. While I did test to ensure they should not affect local deployment before sharing, here are some common issues and solutions nonetheless:

### 1. Missing API Key
**Error** `streamlit.errors.StreamlitSecretNotFoundError`
**Solution** Ensure that a secrets.toml file with a valid OPENAI_API_KEY is correctly installed in the .streamlit folder.

### 2. Missing dependency
**Error** `ModuleNotFoundError`
**Solution** `pip install -r requirements.txt`

### 3. Port already in use
**Error** Address already in use or port conflict.
**Solution** Specify an unused port for Streamlit- I usually go with `streamlit run app.py --server-port=8502` if port 5000 or 8501 (Streamlit default) aren't available.

