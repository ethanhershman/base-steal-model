package api

import (
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"

	"basestealmodel/backend/internal/db"
	"basestealmodel/backend/internal/decision"
)

// Server holds everything a request handler needs -- the in-memory
// decision tables/model (loaded once at startup, see decision.LoadAll)
// plus a live DB handle for player search.
type Server struct {
	Tables  decision.Tables
	Model   *decision.Model
	Medians decision.Medians
	Queries db.Querier
}

func NewRouter(s *Server, corsOrigin string) http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins: []string{corsOrigin},
		AllowedMethods: []string{"GET", "POST"},
		AllowedHeaders: []string{"Content-Type"},
	}))

	r.Get("/api/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	r.Get("/api/players/search", s.handlePlayerSearch)
	r.Post("/api/predict", s.handlePredict)

	return r
}
