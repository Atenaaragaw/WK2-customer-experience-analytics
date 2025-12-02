WITH RankedThemes AS (
    SELECT
        b.bank_name,
        -- UNNEST and STRING_TO_ARRAY split the comma-separated theme string into individual rows
        UNNEST(STRING_TO_ARRAY(r.identified_themes, ', ')) AS theme,
        COUNT(r.review_id) AS theme_count,
        -- Assign a rank within each bank's negative themes
        ROW_NUMBER() OVER (PARTITION BY b.bank_name ORDER BY COUNT(r.review_id) DESC) as rank_num
    FROM
        Reviews r
    JOIN
        Banks b ON r.bank_id = b.bank_id
    WHERE
        -- Filter only for negative sentiment reviews
        r.sentiment_label = 'NEGATIVE' 
        -- Ensure themes column is not null or empty
        AND r.identified_themes IS NOT NULL 
        AND r.identified_themes != ''
    GROUP BY
        b.bank_name, theme
)
SELECT
    bank_name,
    theme,
    theme_count
FROM
    RankedThemes
WHERE
    rank_num <= 5
ORDER BY
    bank_name, theme_count DESC;