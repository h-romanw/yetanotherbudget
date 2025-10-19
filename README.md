# Yet Another Budget - AI-Powered Spending Management
# Table of Contents
1. [Overview](#1-overview)
2. [Installation and running the app locally](#2-installation-and-running-the-app-locally) 
3. [My Product Development Process](#3-my-product-development-process)
4. [Product Walkthrough](#4-product-walkthrough)
[Troubleshooting](#troubleshooting)

# 1 Overview

Yet Another Budget (YAnB) is project-based, AI-powered spending analyser and budgeting tool for young professionals, small businesses and project planners. YAnB allows users to upload spending data and have their transactions automatically categorised by AI, as well as generate a written summary of their spending behaviour and basic charts. Users can then use an integrated chatbot to query the data further, as well as see visualisations of their spending by category and net balance over time. Finally, users can set spending targets for categories at different time horizons, or have the chatbot suggest and implement budgets in the GUI for them based on uploaded data and expressed preferences.


# 2 Installation and running the app locally

To run the app locally, start by cloning the main branch. Ensure that you are using Python 3.13 and create a virtual environment. Then, after opening the repo,
1. Install all dependencies by running `pip install -r requirements.txt` (alternatively, copy and paste the dependencies from requirements.txt and pip install them in the terminal).
2. Create a file in .streamlit called secrets.toml, with the following code: `OPENAI_API_KEY = <Your key here>` or, if I have provided you with a secrets.toml file separately, place it into .streamlit.
3. Open app.py
4. In the terminal, run the command `streamlit run app.py` and Ctrl/Cmd-Click the URL given in the terminal.
5. Use synthetic debit card transaction data found in dummy_data for any data uploads (though 


# 3 My Product Development Process

<img width="3395" height="1885" alt="IMG_8493" src="https://github.com/user-attachments/assets/e510a627-79e9-46a6-a5da-ce99e589c867" />
Prior to starting the actual build for this app, it was my goal to follow a user-centred product design approach building off of my postgrad education and time working in fintech. Often using pen and paper, I followed the below approach.

## 3.1 The user and the problem
I started this project by 

## 3.2 Other apps and YAnB USPs

## 3.3 MoSCoW feature planning
### 3.3.1 MoSCoW
### 3.3.2 Resources
### 3.3.3 Estimation and Kanban

## 3.4 UX design with Figma

## 3.5 Feature backlog and execution

## 3.6 Stagegating and final decisions

## 3.7 Next steps



# Troubleshooting
## Installation
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

## App features

