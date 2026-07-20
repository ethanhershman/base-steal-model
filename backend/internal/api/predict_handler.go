package api

import (
	"encoding/json"
	"net/http"

	"basestealmodel/backend/internal/decision"
)

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	var req predictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if msg := req.validate(); msg != "" {
		writeError(w, http.StatusBadRequest, msg)
		return
	}

	result := decision.PredictStealDecision(s.Tables, s.Model, s.Medians, req.toSituation())
	writeJSON(w, http.StatusOK, result)
}
