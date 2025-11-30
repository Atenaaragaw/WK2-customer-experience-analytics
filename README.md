# Customer Experience Analytics: Ethiopian Digital Banking Reviews

This project provides a comprehensive analysis of customer satisfaction and thematic feedback for leading digital banking apps in Ethiopia: **Commercial Bank of Ethiopia (CBE)**, **Bank of Abyssinia (BOA)**, and **Dashen Bank**.

The solution covers the full data engineering and data science lifecycle: from scraping public reviews to applying advanced NLP techniques, storing results in a relational database, and generating final business insights.

---

# Project Goals

The primary objectives of this analysis are to:
1. **Collect & Preprocess:** Gather over 1,000 customer reviews and prepare the data for analysis.
2. **Analyze Sentiment:** Quantify customer feelings using state-of-the-art NLP models (**DistilBERT**).
3. **Identify Themes:** Extract key customer pain points and operational themes using **TF-IDF**.
4. **Data Engineering:** Store all analyzed data in a **PostgreSQL** database.
5. **Report & Recommend:** Generate actionable insights for improving the digital banking experience.

---

# Project Structure

The repository is organized to reflect the sequential steps of the data pipeline.

WK2-customer-experience-analytics/ 
├── data/ 
│ ├── raw/ 
│ │ └── cleaned_play_store_reviews.csv # Output of Task 1 
│ └── processed/ 
│ └── analyzed_reviews.csv # Output of Task 2 
├── scripts/ 
│ ├── 1_data_collection/ 
│ ├── 2_nlp_analysis/ 
│ │ └── sentiment_thematic_analysis.py # Core NLP script 
├── .venv/ # Python Virtual Environment 
└── requirements.txt # Project dependencies


---

# Current Status: Tasks 1 & 2 Complete

The core data acquisition and analysis phases have been successfully completed, and the data is ready for storage.

# Task 1: Data Collection and Preprocessing

| Component | Status | Details |
| :--- | :--- | :--- |
| **Tool** | `google-play-scraper` | Used to collect publicly available app reviews. |
| **Data Volume** | **1,014+ unique reviews** | Collected for the three target banks. |
| **Output** | Cleaned, deduplicated, and date-normalized dataset. |

# Task 2: Sentiment and Thematic Analysis

| Component | Status | Details |
| :--- | :--- | :--- |
| **Sentiment Model** | **DistilBERT** | Classified reviews into **POSITIVE** or **NEGATIVE** sentiment labels with confidence scores. |
| **Thematic Analysis** | **TF-IDF** | Extracted top keywords and n-grams for each bank, which were then mapped to specific themes (e.g., 'Transaction Performance', 'Account Access/Bugs'). |
| **Output** | Data enriched with `sentiment_label`, `sentiment_score`, and `identified_themes`. |

---

# Next Step: Task 3 - Database Storage

The project is currently focused on **Task 3: Storing Data in PostgreSQL**. This involves setting up the relational schema (`Banks` and `Reviews` tables) and executing a Python script for efficient batch data insertion.

---

# Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [YOUR-REPO-URL]
    cd WK2-customer-experience-analytics
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```