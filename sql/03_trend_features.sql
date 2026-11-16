-- =====================================================================
-- 03  Trend & volatility features
-- ---------------------------------------------------------------------
-- Churners rarely stop overnight - they fade. These features compare the
-- most recent 4 weeks against the earlier part of the window and fit a
-- simple linear slope to weekly outbound volume.
-- A dense client x week grid is built first so that silent weeks count as 0.
-- =====================================================================
WITH weeks AS (
    SELECT UNNEST(range({snapshot} - {obs}, {snapshot})) AS week_idx
),
pop AS (
    SELECT client_id FROM labels      -- scoring population from step 01
),
grid AS (
    SELECT p.client_id, w.week_idx FROM pop p CROSS JOIN weeks w
),
weekly AS (
    SELECT g.client_id, g.week_idx,
           COALESCE(SUM(CASE WHEN pw.direction = 'OUT' THEN pw.txn_count  END), 0) AS out_cnt,
           COALESCE(SUM(CASE WHEN pw.direction = 'OUT' THEN pw.txn_amount END), 0) AS out_amt,
           COALESCE(SUM(CASE WHEN pw.direction = 'IN'  THEN pw.txn_count  END), 0) AS in_cnt
    FROM grid g
    LEFT JOIN payments_weekly pw
           ON pw.client_id = g.client_id AND pw.week_idx = g.week_idx
    GROUP BY g.client_id, g.week_idx
)
SELECT
    client_id,
    -- recent 4 weeks vs earlier weeks (ratio of weekly averages, +1 smoothing)
    (AVG(CASE WHEN week_idx >= {snapshot} - 4 THEN out_cnt END) + 1)
      / (AVG(CASE WHEN week_idx <  {snapshot} - 4 THEN out_cnt END) + 1)       AS trend_out_cnt,
    (AVG(CASE WHEN week_idx >= {snapshot} - 4 THEN out_amt END) + 1)
      / (AVG(CASE WHEN week_idx <  {snapshot} - 4 THEN out_amt END) + 1)       AS trend_out_amt,
    (AVG(CASE WHEN week_idx >= {snapshot} - 4 THEN in_cnt END) + 1)
      / (AVG(CASE WHEN week_idx <  {snapshot} - 4 THEN in_cnt END) + 1)        AS trend_in_cnt,
    -- slope of weekly outbound count, normalised by the client's own mean
    REGR_SLOPE(out_cnt, week_idx) / (AVG(out_cnt) + 1)                          AS slope_out_cnt,
    -- volatility of weekly outbound value
    STDDEV_SAMP(out_amt) / (AVG(out_amt) + 1)                                   AS cv_out_amt
FROM weekly
GROUP BY client_id
