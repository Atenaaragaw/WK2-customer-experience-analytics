import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd
import os
from dotenv import load_dotenv 

load_dotenv() 

DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "database": "bank_reviews",
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "@Bezawit1"),
    "port": os.getenv("PG_PORT", 5432)
}

INPUT_FILE = 'data/processed/analyzed_reviews.csv'
INSERT_BANKS = "INSERT INTO Banks (bank_name) VALUES (%s) ON CONFLICT (bank_name) DO UPDATE SET bank_name = EXCLUDED.bank_name RETURNING bank_id;"
SELECT_BANK_ID = "SELECT bank_id FROM Banks WHERE bank_name = %s;"

INSERT_REVIEWS = """
INSERT INTO Reviews (bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, identified_themes, source)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Successfully connected to PostgreSQL.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"FATAL ERROR: Could not connect to database. Check DB_CONFIG and ensure PostgreSQL server is running: {e}")
        return None

def get_bank_id(cursor, bank_name):
    """Fetches the bank_id for a given bank name, inserting it if it doesn't exist."""
    cursor.execute(INSERT_BANKS, (bank_name,))
    fetch_result = cursor.fetchone()
    
    if fetch_result:
        return fetch_result[0]
    else:
        cursor.execute(SELECT_BANK_ID, (bank_name,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            raise Exception(f"Could not retrieve ID for bank: {bank_name}")

def insert_banks(conn, cursor, bank_names):
    """Inserts bank names into the Banks table and returns a bank_id mapping."""
    bank_id_map = {}
    print("Inserting/verifying bank metadata...")
    for bank in bank_names:
        bank_id = get_bank_id(cursor, bank)
        bank_id_map[bank] = bank_id
        
    conn.commit()
    print(f"Bank ID map created successfully.")
    return bank_id_map

def insert_reviews(conn, cursor, df, bank_id_map):
    """Prepares and efficiently inserts review data into the Reviews table."""
    print("Preparing review data for batch insertion...")
    
    # Map bank names to their Foreign Key IDs
    df['bank_id'] = df['bank'].map(bank_id_map)
    
    # Replace NaN themes with an empty string as the DB field is NOT NULL or TEXT/VARCHAR
    df['identified_themes'] = df['identified_themes'].fillna('')
    
    # Prepare data tuples in the order of the INSERT_REVIEWS SQL command
    data_to_insert = [
        (
            row['bank_id'],
            row['review'],
            row['rating'],
            row['date'],
            row['sentiment_label'],
            row['sentiment_score'],
            row['identified_themes'],
            row['source']
        )
        for index, row in df.iterrows()
    ]

    print(f"Inserting {len(data_to_insert)} records into Reviews table using batch execution...")
    
    # Use execute_batch for fast, efficient bulk insertion (page_size defines the commit frequency)
    execute_batch(cursor, INSERT_REVIEWS, data_to_insert, page_size=1000)
    
    conn.commit()
    print("Reviews insertion complete.")


def run_task_3():
    """Main function to orchestrate the data loading and insertion."""
    
    conn = None
    try:
        # 1. Load data
        if not os.path.exists(INPUT_FILE):
             print(f"FATAL ERROR: Input file not found at {INPUT_FILE}. Ensure Task 2 was completed successfully.")
             return
             
        df = pd.read_csv(INPUT_FILE)
        print(f"Data loaded successfully from {INPUT_FILE}. Total records: {len(df)}")
        
        conn = get_db_connection()
        if not conn:
            return

        with conn.cursor() as cursor:
            # 2. Insert Banks and get mapping
            bank_names = df['bank'].unique().tolist()
            bank_id_map = insert_banks(conn, cursor, bank_names)
            
            # 3. Insert Reviews
            insert_reviews(conn, cursor, df, bank_id_map)
            
            # 4. Verification
            cursor.execute("SELECT COUNT(*) FROM Reviews;")
            review_count = cursor.fetchone()[0]
            print(f"\nVerification: Total records successfully inserted into Reviews table: {review_count}")
            
            if review_count >= 1000:
                print("KPI Success: Database populated with over 1,000 entries.")
            else:
                print("KPI Warning: Less than 1,000 entries found.")

    except Exception as e:
        print(f"An unexpected error occurred during Task 3: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    run_task_3()