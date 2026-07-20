-- name: SearchRunners :many
SELECT player_id, full_name, last_team, bats_lhb, prior_sr, prior_att,
       sprint_speed, sprint_speed_missing, age, age_missing
FROM runners
WHERE search_name LIKE ?1
ORDER BY full_name
LIMIT ?2;

-- name: SearchPitchers :many
SELECT player_id, full_name, last_team, throws_lhp, prior_sr_allowed
FROM pitchers
WHERE search_name LIKE ?1
ORDER BY full_name
LIMIT ?2;

-- name: SearchCatchers :many
SELECT player_id, full_name, last_team, prior_cs_rate, pop_time, pop_time_missing
FROM catchers
WHERE search_name LIKE ?1
ORDER BY full_name
LIMIT ?2;
