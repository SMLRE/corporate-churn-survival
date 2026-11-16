-- =====================================================================
-- 02  Recency / Frequency / Monetary features over the observation window
-- ---------------------------------------------------------------------
-- One row per client. Both payment directions are profiled separately,
-- because the churn definition depends on both.
-- =====================================================================
WITH win AS (
    SELECT *
    FROM payments_weekly
    WHERE week_idx BETWEEN {snapshot} - {obs} AND {snapshot} - 1
)
SELECT
    client_id,
    -- Recency: weeks since last activity (obs+1 if none in window)
    COALESCE({snapshot} - MAX(CASE WHEN direction = 'OUT' THEN week_idx END), {obs} + 1) AS recency_out_wks,
    COALESCE({snapshot} - MAX(CASE WHEN direction = 'IN'  THEN week_idx END), {obs} + 1) AS recency_in_wks,
    -- Frequency
    SUM(CASE WHEN direction = 'OUT' THEN txn_count ELSE 0 END)                AS freq_out,
    SUM(CASE WHEN direction = 'IN'  THEN txn_count ELSE 0 END)                AS freq_in,
    COUNT(DISTINCT CASE WHEN direction = 'OUT' THEN week_idx END)             AS active_wks_out,
    COUNT(DISTINCT CASE WHEN direction = 'IN'  THEN week_idx END)             AS active_wks_in,
    -- Monetary (log-scaled downstream)
    SUM(CASE WHEN direction = 'OUT' THEN txn_amount ELSE 0 END)               AS monetary_out,
    SUM(CASE WHEN direction = 'IN'  THEN txn_amount ELSE 0 END)               AS monetary_in,
    -- Service quality
    SUM(failed_count) * 1.0
        / NULLIF(SUM(CASE WHEN direction = 'OUT' THEN txn_count ELSE 0 END), 0) AS failed_rate
FROM win
GROUP BY client_id
