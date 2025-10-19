# Yet Another Budget - AI-Powered Spending Management
# Table of Contents
1. [Overview](#1-overview)
2. [Installation and running the app locally](#2-installation-and-running-the-app-locally) 
3. [My Product Development Process](#3-my-product-development-process)
4. [Product Walkthrough](#4-product-walkthrough)
5. [Troubleshooting](#5-troubleshooting)

# 1 Overview

Yet Another Budget (YAnB) is project-based, AI-powered spending analyser and budgeting tool for young professionals, small businesses and project planners. YAnB allows users to upload spending data and have their transactions automatically categorised by AI, as well as generate a written summary of their spending behaviour and basic charts. Users can then use an integrated chatbot to query the data further, as well as see visualisations of their spending by category and net balance over time. Finally, users can set spending targets for categories at different time horizons, or have the chatbot suggest and implement budgets in the GUI for them based on uploaded data and expressed preferences.

At the core of YAnB are projects, which aim to deliver the feature depth of more complex budgeting toos with the flexibility to meet a wider range of real-world use cases. A project is simply a running balance sheet of uploaded transactions, with relevant categories, budget targets/timeframes, and chat history saved alongside. A project could be:
- A bank account.
- A fixed-term event or period with expected spend amounts across different categories.
- A new business venture.
- A dependent or family member's expenses for which the user is responsible.
- A marketing campaign for a product launch.
- A running tally of shared expenses among friends.
- ...or anything else, really!

Projects integrate seamlessly with the AI features, allowing the AI chat to act as a spending summariser and financial coach in setting targets, determining appropriate categories and assigning transactions to them.

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

## 3.1 The user and product strategy
I started this project by identifying a niche within the broader category of people who use budgeting apps. From my personal experience, I found that many spending analysis/budgeting apps struggle to balance flexibility with usability, either offering deep data-based customisation but sacraficing rigidity in UX or opting for ease-of-use but limiting functionality to simple graph-based summarisation. I was particularly influenced by two back-to-back conversations with my partner, with whom I had recently opened a Monzo joint account. In summary, the two conversations were:
- We were beginning to plan for our wedding, but were finding that no budgeting tools currently available allowed us to integrate a fixed-term, project-based budget with our exisiting bank-account-based budgeting.
- Having just started working for an events agency, she was surprised that her business relied predominantly on spreadsheets to manage the budgets and balances available to their clients. This created additional overhead and made cooperation more cumbersome.

Based on these, I felt inspired to create a webapp that could tackle the problems faced by both users. Additionally, my time working in fintech had taught me the value of integrating a B2B/SaaS-first business model with D2C offerings; by working closely with other businesses (hypothetically, this could be my partner's events agency) to build services that meet their needs, the cost of building features for D2C app releases can be subsidised by B2B contracts. 

### 3.1.1 Other apps and YAnB USPs
When designing Yet Another Budget, there were two apps that I used to determine existing pain-points and feature opportunities: Monzo and You Need A Budget (YNAB)*. The former is quite easy to use, making it simple to set budget categories and targets; it is limited, however, in it's ability to integrate multiple accounts at a time, which can become cumbersome when different accounts have different purposes and so need different category targets. This was the case with my partner and I, who rely on a Monzo joint for basic expenses and our personal accounts for transport, etc.

YNAB, on the other hand, has sophisticated, zero-based budgeting tools with a wider range of customisation and tracking over time. The downside? YNAB has a steep learning curve and can become very rigid, with users online (my own experience corroborating) expressing frustration with a perceived lack of control over the system.



*_Yes, Yet Another Budget (YAnB) is indeed my poor attempt at wordplay based off of YNAB. I kindly ask that you forgive this attempt at product-name-based humour._

## 3.2 MoSCoW feature planning
### 3.2.1 MoSCoW
### 3.2.2 Resources
### 3.2.3 Estimation and Kanban

## 3.3 UX design with Figma
<img width="2880" height="2048" alt="image" src="https://github.com/user-attachments/assets/6085bfd2-abc2-46ee-b425-179bc60a61fc" />
[My Figma design file can be found here](https://www.figma.com/design/Q7xI5WTuwnrUUFfF5wCmVp/YAnB-Designs?node-id=0-1&t=NP5KkBi38aNzWolN-1)

## 3.4 Feature backlog and execution

## 3.5 Stagegating and final decisions

## 3.6 Next steps

# 4 Product Walkthrough


# 5 Troubleshooting
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

