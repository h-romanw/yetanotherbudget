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
- ...or anything else you could want them to be, really!

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

The pain-points I personally experienced with both of these combined with those I discovered online led me to develop the project-based budgeting idea further; I realised that zero-based budgeting, while effective, can be difficult to manage across multiple accounts and is challenging to adapt to in real life. Instead, giving users the greatest flexibility at the project level, rather than at the category level, could allow them to more easily reflect their real-world behaviour in their budgets. In doing that, I hoped that effective use of data visualisation and AI-based coaching would make it more likely for users to implement changes to their spending behaviour in the long-run.

*_Yes, Yet Another Budget (YAnB) is indeed my poor attempt at wordplay based off of YNAB. I kindly ask that you forgive this attempt at product-name-based humour._

## 3.2 Planning
### 3.2.1 MoSCoW
| **Must** | **Should** |
|----------|------------|
|Live webapp  | Easy-to-use UX/UI  |
|Dummy data   | Non zero-budgeting |
|Multiple visualisations with different purposes | Natural-language budget generation and implementation |
|AI data organisation from CSV import | AI spending summary |
|Friendly, financial coach AI ToV | Category creation and targets |
|AI spending categorisation | Spending data = golden source-of-truth |
|Projects | Project-level data storage of categories, targets, etc.|
| Streamlit POC | Streamlit Prototype |

| **Could** | **Won't**|
|----------|--------|
|React FE + Flask BE, Node.js                                   | Flutter mobile                                         |
|Login state | Openbanking/Truelayer integration |
|PostgreSQL database | Credit scoring |
|Budget export | Hard-code |

In addition to all of these, there was the consideration of my experience. I am comfortable with Python, using it and Jupyter Notebook frequently in my work for data analysis.
Despite my UX design training, I was less familiar with CSS and HTML, and had no experience with JavaScript. These led me to settle on using Streamlit for my prototype, as it would allow me to build a webapp in Python while learning more about CSS and HTML. Additionally, I relied on Replit (a co-worker's recommendation) and Github Co-Pilot to vibe code much of the final product and deliver the whole thing in a ~1-week period. I set a stretch target to migrate my Streamlit app to a Python backend (via Flask) and React frontend, which would give me greater flexibility over the look-and-feel defined by my designs.

### 3.2.2 Resources
Based on the MoSCoW analysis, I created a short list of resources I would need to be able to execute the project. These included:
- Replit / Github Co-Pilot for agentic AI and vibe-coding.
- OpenAI API, for the core AI functionality in the app itself.
- Figma, for my designs.
- Streamlit, to build a Proof-of-Concept and then feature-complete webapp.
- HuggingFace for any other AI functionality needs (not used in the end).
- Innumerable sticky notes, notebook pages and pen ink cartridges.


### 3.2.3 Estimation and Gantt Chart
Having established what I needed to do and the tools I would need to do so, I created a 1-week Gantt chart with stages,  timelines (estimated) and progress check-ins, allowing me to quickly change focus and reprioritise parts of the project based on what I had achieved throughout different stages of the week. 


## 3.3 UX and product design
### 3.3.1 Flows and Jobs-to-be-Done
Returning to my target users, I identified the core jobs-to-be-done (JTBD) that they would use the app for. At this stage, the features I had defined in the MoSCoW analysis suggested a three-page webapp structure, which I then used to define the key user flows for these JTBD. The jobs and flows were:
1. Importing and categorising data (+ generating a high-level text and visual summary of overall spending behaviour).
2. Querying the data in more detail using both visualisations and chatbot conversations.
3. Appending new data and modifying existing transactions.
4. Creating, renaming and deleting categories.
5. Setting category targets at different time frames, based on the project needs (monthly, yearly or all-time for a fixed-duration project).
6. Reviewing AI analysis and asking for help setting targets based on desired behavioural changes and preferences.

### 3.3.2 Figma designs
<img width="2880" height="2048" alt="image" src="https://github.com/user-attachments/assets/6085bfd2-abc2-46ee-b425-179bc60a61fc" />

[My Figma design file can be found here](https://www.figma.com/design/Q7xI5WTuwnrUUFfF5wCmVp/YAnB-Designs?node-id=0-1&t=NP5KkBi38aNzWolN-1)

These flows suggested a structure to the three pages, which I implemented in Figma wireframes. This stage also allowed me to start defining the look of basic components, their behaviour and a font and colour library.
Crucially, the joint B2B/SaaS and D2C product strategy required a widget-based design. This would allow simple and easy customisation of the product for different businesses' use cases, with widgets being largely self-contained.

## 3.4 Feature backlog and execution
With designs in place, I created a formal Kanban board of features (again on pen and paper), allowing me to track progress across multiple tasks and organise features in order of priority and dependency.

## 3.5 Stagegating and final decisions
As the week of development went by, I included various stagegates to determine whether progress to implementing different groups of features would be appropriate. By the end of the week the Streamlit app was feature complete, but I was faced with the choice between testing and attempting to migrate to my desired React - Flask architecture. I ultimately chose not to proceed with the latter, as I wasn't comfortable with the risk of delivering a poorly-funcitoning product (though the intitial steps towards this can be found in the `experimental` branch of the repo). 
Though Replit offers the ability to host webapps through their service, I opted to do so through Streamlit for simplicity.

## 3.6 Next steps
If you have read this far, thank you! My goal with this app is to implement my target architecture, not just for the additional UX flexibility but so that I could implement some of the features in the "Could" section (e.g. PostgreSQL database, login states, etc.). For now, I am excited to share this feature-complete prototype version of Yett Another Budget with you!


# 4 Product Walkthrough
## Data upload and categorisation
- Drag and drop (or file browse) to one of the synthetic debit statements in dummy_data. Upload these and press "Categorise with AI"

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

