SELECT
    b.bank_name,
    COUNT(r.review_id) AS total_reviews,
    ROUND(AVG(r.rating), 2) AS average_rating,
    ROUND(
        CAST(SUM(CASE WHEN r.sentiment_label = 'POSITIVE' THEN 1 ELSE 0 END) AS NUMERIC) * 100 / COUNT(r.review_id),
        2
    ) AS percent_positive
FROM
    Reviews r
JOIN
    Banks b ON r.bank_id = b.bank_id
GROUP BY
    b.bank_name
ORDER BY
    percent_positive DESC;