-- name: ListModelCoefficients :many
SELECT feature_name, sort_order, coefficient FROM model_coefficients ORDER BY sort_order;

-- name: GetModelMeta :one
SELECT intercept, median_runner_sprint_speed, median_runner_age, median_catcher_pop_time,
       trained_at, train_rows
FROM model_meta WHERE id = 1;
