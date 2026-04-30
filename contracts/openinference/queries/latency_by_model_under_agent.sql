-- Locked Gate-1 reference query for Mews.
--
-- Question: "p50/p95 latency of LLM-kind spans whose parent is AGENT-kind, grouped by model."
--
-- Why this query forces real Mews methodology:
--   1. Self-join on parent_id   -> exercises parent/child traversal
--   2. parent.span_kind = AGENT -> closes the false-positive hole where a flat
--                                   "latency by model" query could win without
--                                   touching nesting at all
--   3. llm.model_name           -> nested JSON attribute extraction (per OpenInference
--                                   convention; not a flat column in Phoenix)
--   4. tenant_id filter         -> multi-tenant scoping, Mews extension
--
-- DuckDB syntax. quantile_cont gives continuous-interpolated percentile.
-- Latency in milliseconds for human-readable output.

WITH llm_under_agent AS (
    SELECT
        json_extract_string(s.attributes, '$.llm.model_name')        AS model_name,
        EXTRACT(EPOCH FROM (s.end_time - s.start_time)) * 1000.0          AS latency_ms
    FROM spans s
    JOIN spans p
      ON s.parent_id = p.span_id
     AND s.tenant_id = p.tenant_id            -- tenant-scoped self-join
    WHERE s.span_kind = 'LLM'
      AND p.span_kind = 'AGENT'
      AND s.tenant_id = ?                     -- placeholder bound at query time
)
SELECT
    model_name,
    COUNT(*)                              AS n,
    quantile_cont(latency_ms, 0.50)       AS p50_ms,
    quantile_cont(latency_ms, 0.95)       AS p95_ms
FROM llm_under_agent
WHERE model_name IS NOT NULL
GROUP BY model_name
ORDER BY p95_ms DESC;
