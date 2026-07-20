-- =============================================================================
-- AXIOM Semantic Cache — Version 2.0 Database Migration
-- =============================================================================
-- Run this entire script in your Supabase SQL Editor.
-- It is fully idempotent: safe to run multiple times.
--
-- What this script does:
--   1. Adds hit_count and last_accessed_at columns to shared_llm_cache
--   2. Creates a composite index for fast LFU scoring / eviction sorting
--   3. Creates increment_hit() — called by main.py on every cache hit
--   4. Creates evict_lfu_cache() — triggered by GitHub Actions every 2 days
-- =============================================================================


-- -----------------------------------------------------------------------------
-- STEP 1: Schema Migration — Add Hit Tracking Columns
-- -----------------------------------------------------------------------------

ALTER TABLE shared_llm_cache
  ADD COLUMN IF NOT EXISTS hit_count        INTEGER     NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT now();

-- Index for fast LFU eviction sorting (lowest scores evicted first)
CREATE INDEX IF NOT EXISTS idx_cache_lfu_score
  ON shared_llm_cache (hit_count ASC, last_accessed_at ASC);


-- -----------------------------------------------------------------------------
-- STEP 2: increment_hit() — Called by main.py on every DB Semantic Hit
-- -----------------------------------------------------------------------------
-- Atomically increments hit_count and refreshes last_accessed_at.
-- Called via supabase.rpc("increment_hit", {"p_row_id": <id>})
--
-- Using a stored procedure instead of a Python table.update() call:
--   - Atomic operation (no race condition under concurrent hits)
--   - Single round-trip instead of read-modify-write
--   - Runs even if the Python BackgroundTask is slightly delayed
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION increment_hit(p_row_id UUID)
RETURNS VOID
LANGUAGE sql
AS $$
  UPDATE shared_llm_cache
  SET
    hit_count        = hit_count + 1,
    last_accessed_at = now()
  WHERE id = p_row_id;
$$;


-- -----------------------------------------------------------------------------
-- STEP 3: evict_lfu_cache() — GitHub Actions Scheduled Eviction
-- -----------------------------------------------------------------------------
-- Removes the lowest-scoring rows when the table exceeds p_threshold_pct of
-- p_max_rows, pruning the table back down to p_target_pct of p_max_rows.
--
-- LFU Scoring Formula:
--   base_score     = hit_count
--   recency_bonus  = hit_count AGAIN if last_accessed_at > now() - 30 days
--                    (doubling the score for recently active rows)
--   grace_immune   = rows younger than 7 days are NEVER evicted
--
--   final_score = base_score + recency_bonus
--   Lowest final_score evicted first. Ties broken by oldest last_accessed_at.
--
-- Parameters:
--   p_max_rows      — Hard capacity ceiling (default 50,000)
--   p_threshold_pct — Eviction triggers above this fraction (default 0.80)
--   p_target_pct    — DB is pruned back to this fraction (default 0.70)
--
-- Returns a JSON summary: rows_before, rows_after, rows_evicted, triggered
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION evict_lfu_cache(
  p_max_rows      INTEGER DEFAULT 50000,
  p_threshold_pct FLOAT   DEFAULT 0.80,
  p_target_pct    FLOAT   DEFAULT 0.70
)
RETURNS JSON
LANGUAGE plpgsql
AS $$
DECLARE
  v_current_rows   INTEGER;
  v_threshold      INTEGER;
  v_target         INTEGER;
  v_to_delete      INTEGER;
  v_rows_deleted   INTEGER := 0;
  v_grace_cutoff   TIMESTAMPTZ := now() - INTERVAL '7 days';
  v_recency_cutoff TIMESTAMPTZ := now() - INTERVAL '30 days';
BEGIN
  -- Count current rows
  SELECT COUNT(*) INTO v_current_rows FROM shared_llm_cache;

  v_threshold := FLOOR(p_max_rows * p_threshold_pct);
  v_target    := FLOOR(p_max_rows * p_target_pct);

  -- Only evict if we have crossed the threshold
  IF v_current_rows <= v_threshold THEN
    RETURN json_build_object(
      'triggered',    false,
      'reason',       format('Row count (%s) is below threshold (%s). No eviction needed.', v_current_rows, v_threshold),
      'rows_before',  v_current_rows,
      'rows_after',   v_current_rows,
      'rows_evicted', 0
    );
  END IF;

  -- How many rows do we need to delete to reach the target?
  v_to_delete := v_current_rows - v_target;

  -- Delete lowest-scoring rows (grace period rows are always excluded)
  -- Score = hit_count  +  hit_count (doubled if accessed within last 30 days)
  WITH lfu_ranked AS (
    SELECT
      id,
      (hit_count + CASE
        WHEN last_accessed_at >= v_recency_cutoff THEN hit_count
        ELSE 0
      END) AS lfu_score
    FROM shared_llm_cache
    -- Grace period: protect rows younger than 7 days from eviction
    WHERE created_at < v_grace_cutoff
    ORDER BY lfu_score ASC, last_accessed_at ASC   -- ties: oldest access first
    LIMIT v_to_delete
  )
  DELETE FROM shared_llm_cache
  WHERE id IN (SELECT id FROM lfu_ranked);

  GET DIAGNOSTICS v_rows_deleted = ROW_COUNT;

  RETURN json_build_object(
    'triggered',    true,
    'rows_before',  v_current_rows,
    'rows_evicted', v_rows_deleted,
    'rows_after',   v_current_rows - v_rows_deleted,
    'threshold',    v_threshold,
    'target',       v_target,
    'grace_cutoff', v_grace_cutoff,
    'evicted_at',   now()
  );
END;
$$;


-- -----------------------------------------------------------------------------
-- STEP 4: Update match_shared_cache() to return hit_count in results
-- -----------------------------------------------------------------------------
-- Returns hit_count so main.py can pass the row id to increment_hit()
-- in a BackgroundTask — no extra DB round-trip needed.
--
-- NOTE: PostgreSQL cannot change a function's return type with CREATE OR REPLACE.
-- We must DROP it first, then recreate it with the new signature.
-- -----------------------------------------------------------------------------

DROP FUNCTION IF EXISTS match_shared_cache(vector, double precision, integer);

CREATE OR REPLACE FUNCTION match_shared_cache(
    query_embedding vector(768),
    match_threshold float,
    match_count     int
)
RETURNS TABLE (
    id            uuid,
    query_text    text,
    response_text text,
    similarity    float,
    hit_count     integer
)
LANGUAGE sql STABLE AS $$
    SELECT
        id,
        query_text,
        response_text,
        1 - (embedding <=> query_embedding) AS similarity,
        hit_count
    FROM shared_llm_cache
    WHERE 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;


-- =============================================================================
-- Verification Queries (uncomment and run to confirm migration succeeded)
-- =============================================================================

-- 1. Check new columns exist
-- SELECT column_name, data_type, column_default
-- FROM information_schema.columns
-- WHERE table_name = 'shared_llm_cache'
--   AND column_name IN ('hit_count', 'last_accessed_at');

-- 2. Check the index exists
-- SELECT indexname, indexdef
-- FROM pg_indexes
-- WHERE tablename = 'shared_llm_cache' AND indexname = 'idx_cache_lfu_score';

-- 3. Test increment_hit (replace the uuid below with a real row id from your table)
-- SELECT increment_hit('00000000-0000-0000-0000-000000000000'::uuid);
-- SELECT id, hit_count, last_accessed_at FROM shared_llm_cache WHERE id = '00000000-0000-0000-0000-000000000000'::uuid;

-- 4. Test evict_lfu_cache — safe to run even when table is under threshold
-- SELECT evict_lfu_cache(50000, 0.80, 0.70);
