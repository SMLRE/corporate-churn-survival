-- =====================================================================
-- 01  Scoring population + labels for one snapshot
-- ---------------------------------------------------------------------
-- Parameters (filled by src/features.py):
--   {snapshot}   first week AFTER the observation window
--   {obs}        observation window length in weeks  (13 weeks ~ 90 days)
--   {outcome}    outcome window length in weeks      (13 weeks ~ 90 days)
--   {max_week}   last week available in the data
--
-- Population : clients still live at the snapshot, i.e. with activity (in or
--              out) in the last 4 weeks of the observation window. Clients who
--              have already gone quiet are a collections problem, not a
--              retention one, so they are excluded.
-- Churn label: "authentic churn" = the client is dormant in BOTH directions by
--              the end of the outcome window: no outbound AND no inbound
--              activity in its final 8 weeks. A client that stops paying out
--              but keeps receiving funds is still a live relationship, so it is
--              not churn.
-- Survival   : weeks from snapshot until the client goes permanently silent
--              (event), or censored at the end of the data.
-- Written in portable SQL (runs on DuckDB here; production used Oracle).
-- =====================================================================
WITH obs_active AS (
    SELECT DISTINCT client_id
    FROM payments_weekly
    WHERE week_idx BETWEEN {snapshot} - 4 AND {snapshot} - 1
),
outcome_activity AS (
    SELECT client_id,
           SUM(CASE WHEN direction = 'OUT' THEN txn_count ELSE 0 END) AS out_cnt_outcome,
           SUM(CASE WHEN direction = 'IN'  THEN txn_count ELSE 0 END) AS in_cnt_outcome
    FROM payments_weekly
    WHERE week_idx BETWEEN {snapshot} + {outcome} - 8 AND {snapshot} + {outcome} - 1
    GROUP BY client_id
),
last_seen AS (
    SELECT client_id, MAX(week_idx) AS last_active_week
    FROM payments_weekly
    GROUP BY client_id
)
SELECT
    p.client_id,
    {snapshot}                                                    AS snapshot_week,
    CASE WHEN COALESCE(o.out_cnt_outcome, 0) = 0
          AND COALESCE(o.in_cnt_outcome, 0)  = 0 THEN 1 ELSE 0 END AS churn_flag,
    CASE WHEN l.last_active_week <= {max_week} - 4 THEN 1 ELSE 0 END AS surv_event,
    CASE WHEN l.last_active_week <= {max_week} - 4
         THEN GREATEST(l.last_active_week + 1 - {snapshot}, 1)
         ELSE {max_week} + 1 - {snapshot} END                       AS surv_duration_weeks
FROM obs_active p
LEFT JOIN outcome_activity o USING (client_id)
LEFT JOIN last_seen       l USING (client_id)
