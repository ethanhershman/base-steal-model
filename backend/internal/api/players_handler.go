package api

import (
	"database/sql"
	"net/http"
	"strconv"
	"strings"

	"basestealmodel/backend/internal/db"
)

const (
	defaultSearchLimit = 10
	maxSearchLimit      = 25
)

func (s *Server) handlePlayerSearch(w http.ResponseWriter, r *http.Request) {
	role := r.URL.Query().Get("role")
	q := r.URL.Query().Get("q")

	if q == "" {
		writeError(w, http.StatusBadRequest, "q is required")
		return
	}

	limit := defaultSearchLimit
	if raw := r.URL.Query().Get("limit"); raw != "" {
		if v, err := strconv.Atoi(raw); err == nil && v > 0 {
			limit = v
			if limit > maxSearchLimit {
				limit = maxSearchLimit
			}
		}
	}

	pattern := "%" + strings.ToLower(q) + "%"
	ctx := r.Context()

	switch role {
	case "runner":
		rows, err := s.Queries.SearchRunners(ctx, db.SearchRunnersParams{SearchName: pattern, Limit: int64(limit)})
		if err != nil {
			writeError(w, http.StatusInternalServerError, "search failed")
			return
		}
		results := make([]playerSearchResult, len(rows))
		for i, row := range rows {
			results[i] = playerSearchResult{
				ID: row.PlayerID, Name: row.FullName, Team: nullStringPtr(row.LastTeam),
				Stats: map[string]any{
					"bats_lhb":             row.BatsLhb != 0,
					"prior_sr":             row.PriorSr,
					"prior_att":            row.PriorAtt,
					"sprint_speed":         nullFloatPtr(row.SprintSpeed),
					"sprint_speed_missing": row.SprintSpeedMissing != 0,
					"age":                  nullFloatPtr(row.Age),
					"age_missing":          row.AgeMissing != 0,
				},
			}
		}
		writeJSON(w, http.StatusOK, results)

	case "pitcher":
		rows, err := s.Queries.SearchPitchers(ctx, db.SearchPitchersParams{SearchName: pattern, Limit: int64(limit)})
		if err != nil {
			writeError(w, http.StatusInternalServerError, "search failed")
			return
		}
		results := make([]playerSearchResult, len(rows))
		for i, row := range rows {
			results[i] = playerSearchResult{
				ID: row.PlayerID, Name: row.FullName, Team: nullStringPtr(row.LastTeam),
				Stats: map[string]any{
					"throws_lhp":       row.ThrowsLhp != 0,
					"prior_sr_allowed": row.PriorSrAllowed,
				},
			}
		}
		writeJSON(w, http.StatusOK, results)

	case "catcher":
		rows, err := s.Queries.SearchCatchers(ctx, db.SearchCatchersParams{SearchName: pattern, Limit: int64(limit)})
		if err != nil {
			writeError(w, http.StatusInternalServerError, "search failed")
			return
		}
		results := make([]playerSearchResult, len(rows))
		for i, row := range rows {
			results[i] = playerSearchResult{
				ID: row.PlayerID, Name: row.FullName, Team: nullStringPtr(row.LastTeam),
				Stats: map[string]any{
					"prior_cs_rate":    row.PriorCsRate,
					"pop_time":         nullFloatPtr(row.PopTime),
					"pop_time_missing": row.PopTimeMissing != 0,
				},
			}
		}
		writeJSON(w, http.StatusOK, results)

	default:
		writeError(w, http.StatusBadRequest, "role must be runner, pitcher, or catcher")
	}
}

func nullStringPtr(ns sql.NullString) *string {
	if !ns.Valid {
		return nil
	}
	return &ns.String
}

func nullFloatPtr(nf sql.NullFloat64) *float64 {
	if !nf.Valid {
		return nil
	}
	return &nf.Float64
}
