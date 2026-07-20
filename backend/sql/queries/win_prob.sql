-- name: ListWinProbCells :many
SELECT table_kind, inning_bucket, half, outs, base_code, score_bucket, win_rate, n
FROM win_prob_cells
WHERE table_kind = ?;
