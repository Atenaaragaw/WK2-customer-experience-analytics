import pandas as pd
from google_play_scraper import Sort, reviews_all

# Configuration
banks_data = [
    {'bank': 'Commercial Bank of Ethiopia', 'app_id': 'com.combanketh.mobilebanking'},
    {'bank': 'Bank of Abyssinia', 'app_id': 'com.boa.boaMobileBanking'},
    {'bank': 'Dashen Bank', 'app_id': 'com.dashen.dashensuperapp'} # Using the SuperApp ID
]
min_reviews_per_bank = 500
output_path = 'data/raw/cleaned_play_store_reviews.csv'
# --- Scraping Function ---

def scrape_reviews(app_id, bank_name, limit):
    """Scrapes a specified number of reviews for a given app ID."""
    print(f"-> Scraping reviews for {bank_name} ({app_id})...")
    
    # Use reviews_all to get a large volume of reviews
    result = reviews_all(
        app_id,
        sleep_milliseconds=0,
        lang='en',             # Focus on English reviews
        country='et',          # Focus on Ethiopia
        sort=Sort.NEWEST,      # Prioritize recent feedback
        filter_score_with=None # Get all ratings
    )
    
    # Extract only the required fields and enforce the minimum limit
    collected_reviews = []
    for review in result:
        collected_reviews.append({
            'review_text': review.get('content'),
            'rating': review.get('score'),
            'date': review.get('at'),
            'bank': bank_name,
            'source': 'Google Play'
        })
        if len(collected_reviews) >= limit:
            break
            
    print(f"   Collected {len(collected_reviews)} reviews.")
    return collected_reviews

# Main 
all_reviews = []
for item in banks_data:
    all_reviews.extend(scrape_reviews(item['app_id'], item['bank'], min_reviews_per_bank))

df = pd.DataFrame(all_reviews)
print(f"\nTotal initial reviews collected: {len(df)}")

# Preprocessing Steps 

print("\n--- Starting Preprocessing ---")

# Drop duplicates (based on review text and bank)
initial_len = len(df)
df.drop_duplicates(subset=['review_text', 'bank'], inplace=True)
print(f"Reviews dropped (duplicates): {initial_len - len(df)}")

# Handle missing data (must have review text and rating)
df.dropna(subset=['review_text', 'rating'], inplace=True)
df['review_text'] = df['review_text'].astype(str).str.strip()
df = df[df['review_text'] != '']
print(f"Reviews remaining after cleaning: {len(df)}")

# Normalize Date
# The 'date' column is a datetime object, convert to YYYY-MM-DD string
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

# Final Clean-up and Column Ordering
df.rename(columns={'review_text': 'review'}, inplace=True)
final_columns = ['review', 'rating', 'date', 'bank', 'source']
df = df[final_columns]

# KPI: Total collected and per bank
reviews_per_bank = df.groupby('bank').size()
print("\nReviews Collected Per Bank:")
print(reviews_per_bank)
print(f"\nFinal total reviews (KPI Check): {len(df)} (Target: 1200+)")

# Save the cleaned CSV
df.to_csv(output_path, index=False)
print(f"Cleaned data saved to {output_path}")