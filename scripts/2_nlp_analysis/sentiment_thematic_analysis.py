import pandas as pd
import os
import re
import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# Configuration
INPUT_FILE = 'data/raw/cleaned_play_store_reviews.csv'
OUTPUT_FILE = 'data/processed/analyzed_reviews.csv'

# python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# Initialize global NLP components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
word_tokenizer = TreebankWordTokenizer() 

# Sentiment Analysis Function (DistilBERT)

def analyze_sentiment(df):
    """Uses a pre-trained Hugging Face model for sentiment classification."""
    print("Starting DistilBERT Sentiment Analysis...")
    
    # Initialize the sentiment analysis pipeline using the required model
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )

    reviews_list = df['review'].tolist()
    batch_size = 64
    results = []
    
    # Process reviews in batches
    for i in range(0, len(reviews_list), batch_size):
        batch = reviews_list[i:i + batch_size]
        try:
            batch_results = sentiment_pipeline(batch)
            results.extend(batch_results)
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}. Skipping batch.")
            results.extend([{'label': 'NEUTRAL', 'score': 0.0}] * len(batch))
            
    # Extract label and score
    df['sentiment_label'] = [res['label'] for res in results]
    df['sentiment_score'] = [res['score'] for res in results]
    
    # Normalize labels (SST-2 labels are LABEL_1/LABEL_0)
    df['sentiment_label'] = df['sentiment_label'].replace({'LABEL_1': 'POSITIVE', 'LABEL_0': 'NEGATIVE'})
    
    print("Sentiment analysis complete.")
    return df

# Text Preprocessing for Thematic Analysis

def preprocess_text(text):
    """Cleans, tokenizes (using Treebank), removes stop words, and lemmatizes text."""
    # Ensure text is string and strip whitespace
    text = str(text).strip()
    
    # Remove special characters/numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenizer.tokenize(text) 
    
    # Remove stop words and short tokens
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Thematic Analysis (TF-IDF & Keyword Extraction)

def extract_keywords_by_bank(df, bank_name, num_keywords=15):
    """Performs TF-IDF on reviews for a specific bank and extracts top keywords."""
    bank_df = df[df['bank'] == bank_name].copy()
    
    # Apply preprocessing
    bank_df['processed_review'] = bank_df['review'].apply(preprocess_text)
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)
    
    try:
        # Check if there's enough processed text to vectorize
        if bank_df['processed_review'].str.strip().eq('').all():
             print(f"Skipping TF-IDF for {bank_name}: No valid processed text found.")
             return []
             
        tfidf_matrix = vectorizer.fit_transform(bank_df['processed_review'])
    except ValueError:
        print(f"Not enough data for TF-IDF in {bank_name}.")
        return []

    feature_names = vectorizer.get_feature_names_out()
    
    # Sum TF-IDF scores across all reviews to find overall importance
    sums = tfidf_matrix.sum(axis=0)
    scores = [(feature_names[col], sums[0, col]) for col in range(sums.shape[1])]
    
    # Sort by score and extract top keywords/n-grams
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_keywords = [word for word, score in sorted_scores[:num_keywords]]
    
    return top_keywords

def identify_themes(df):
    """Orchestrates thematic analysis and assigns basic themes based on keywords."""
    print("\nStarting Thematic Analysis (TF-IDF Keyword Extraction)...")
    
    all_keywords = {}
    
    for bank in df['bank'].unique():
        keywords = extract_keywords_by_bank(df, bank)
        all_keywords[bank] = keywords
        
        # Rule-Based Theme Assignment
        themes = []
        # Themes based on common banking review topics (Transaction, Access, UI, Support)
        if any(kw in ' '.join(keywords) for kw in ['transfer', 'send', 'money', 'transaction', 'slow']):
            themes.append('Transaction Performance')
        if any(kw in ' '.join(keywords) for kw in ['login', 'crash', 'error', 'bug', 'fingerprint', 'access']):
            themes.append('Account Access/Bugs')
        if any(kw in ' '.join(keywords) for kw in ['interface', 'ui', 'easy', 'layout', 'design', 'user']):
            themes.append('User Interface (UI) & Experience')
        if any(kw in ' '.join(keywords) for kw in ['support', 'customer', 'service', 'call', 'reply']):
            themes.append('Customer Support')

        # Add the identified themes
        df.loc[df['bank'] == bank, 'identified_themes'] = ', '.join(themes)
        
        print(f"Top keywords for {bank}: {keywords[:5]}...")
        print(f"Identified Themes for {bank}: {themes}")
        
    return df, all_keywords

# Main Execution Block

def run_task_2():
    """Loads data, runs analysis, and saves results."""
    
    try:
        # Ensure 'processed' directory exists
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        df = pd.read_csv(INPUT_FILE)
        # Create a unique ID for the DB
        df.reset_index(names=['review_id'], inplace=True) 
        print(f"Loaded {len(df)} reviews from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}. Run Task 1 first.")
        return

    # 1. Sentiment Analysis
    df = analyze_sentiment(df)

    # 2. Thematic Analysis
    df, all_keywords = identify_themes(df)

    # Save the processed data
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nAnalysis complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_task_2()