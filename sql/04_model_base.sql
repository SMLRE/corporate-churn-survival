-- =====================================================================
-- 04  Model base: population + labels + all candidate features
-- =====================================================================
SELECT
    l.client_id,
    l.snapshot_week,
    l.churn_flag,
    l.surv_event,
    l.surv_duration_weeks,
    -- static attributes
    c.segment,
    c.industry,
    c.tenure_months_at_start + CAST({snapshot} * 7 / 30.4 AS INTEGER) AS tenure_months,
    c.products_held,
    c.has_rm,
    -- RFM
    r.recency_out_wks, r.recency_in_wks,
    r.freq_out, r.freq_in, r.active_wks_out, r.active_wks_in,
    LN(1 + r.monetary_out) AS log_monetary_out,
    LN(1 + r.monetary_in)  AS log_monetary_in,
    COALESCE(r.failed_rate, 0) AS failed_rate,
    (r.freq_in + 1.0) / (r.freq_out + 1.0) AS in_out_ratio,
    -- trend
    t.trend_out_cnt, t.trend_out_amt, t.trend_in_cnt, t.slope_out_cnt,
    COALESCE(t.cv_out_amt, 0) AS cv_out_amt
FROM labels l
JOIN clients c USING (client_id)
JOIN rfm     r USING (client_id)
JOIN trend   t USING (client_id)
ORDER BY client_id   -- deterministic row order (DuckDB is multi-threaded)
