-- Mews Gate-1 contract: OpenInference + OpenTelemetry span schema as surfaced by Arize Phoenix,
-- extended with tenant_id for multi-tenant Phoenix deployments.
--
-- Sources:
--   OpenInference semantic conventions:
--     https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
--   Phoenix span model:
--     https://github.com/Arize-ai/phoenix/blob/main/src/phoenix/db/models.py
--
-- Notable: llm.model_name lives inside the JSON attributes blob (per OpenInference
-- convention) rather than as a flat column. This forces the Mews synthesizer to
-- plan a nested attribute extraction in addition to the parent/child self-join,
-- which is the core methodology question of Gate 1.

CREATE TABLE IF NOT EXISTS spans (
    -- OTel core
    trace_id        VARCHAR    NOT NULL,
    span_id         VARCHAR    NOT NULL PRIMARY KEY,
    parent_id       VARCHAR,                              -- NULL for trace root
    name            VARCHAR    NOT NULL,
    start_time      TIMESTAMP  NOT NULL,
    end_time        TIMESTAMP  NOT NULL,
    status_code     VARCHAR    NOT NULL CHECK (status_code IN ('OK','ERROR','UNSET')),
    status_message  VARCHAR,

    -- OpenInference required: openinference.span.kind enum (10 values, no UNKNOWN per spec)
    span_kind       VARCHAR    NOT NULL CHECK (span_kind IN (
        'LLM', 'EMBEDDING', 'CHAIN', 'RETRIEVER', 'RERANKER',
        'TOOL', 'AGENT', 'GUARDRAIL', 'EVALUATOR', 'PROMPT'
    )),

    -- LLM-kind flat columns (Phoenix surfaces these; nullable when span_kind != LLM)
    llm_token_count_prompt      INTEGER,
    llm_token_count_completion  INTEGER,

    -- OpenInference attribute blob; llm.model_name and embedding.model_name live here
    attributes      JSON,

    -- Mews extension: multi-tenant scoping
    tenant_id       VARCHAR    NOT NULL
);

-- Phoenix-style indexes on hierarchy + time + kind, plus tenant for Mews multi-tenancy
CREATE INDEX IF NOT EXISTS idx_spans_parent      ON spans (parent_id);
CREATE INDEX IF NOT EXISTS idx_spans_trace       ON spans (trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_kind        ON spans (span_kind);
CREATE INDEX IF NOT EXISTS idx_spans_tenant_time ON spans (tenant_id, start_time);
