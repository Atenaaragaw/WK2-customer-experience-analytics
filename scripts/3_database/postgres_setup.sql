-- 1. Banks Table:
CREATE TABLE IF NOT EXISTS Banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) UNIQUE NOT NULL
);

-- 2. Reviews Table:
CREATE TABLE IF NOT EXISTS Reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INTEGER NOT NULL,
    review_text TEXT NOT NULL,
    rating INTEGER NOT NULL,
    review_date DATE NOT NULL,
    sentiment_label VARCHAR(10) NOT NULL,
    sentiment_score DECIMAL(5, 4) NOT NULL,
    identified_themes VARCHAR(255),
    source VARCHAR(50) NOT NULL,
    -- define foriegn key
    CONSTRAINT fk_bank
        FOREIGN KEY(bank_id) 
        REFERENCES Banks(bank_id)
        ON DELETE CASCADE
);