// Command server runs the steal-decision web app's backend: loads the
// precomputed RE24/win-probability/model artifacts from SQLite once at
// startup (see decision.LoadAll), then serves /api/players/search and
// /api/predict over HTTP. No Python process involved at runtime -- see
// /Users/colin/.claude/plans/go-with-chi-and-soft-grove.md.
package main

import (
	"context"
	"database/sql"
	"log"
	"net/http"

	_ "modernc.org/sqlite"

	"basestealmodel/backend/internal/api"
	"basestealmodel/backend/internal/config"
	"basestealmodel/backend/internal/db"
	"basestealmodel/backend/internal/decision"
)

func main() {
	cfg := config.Load()

	conn, err := sql.Open("sqlite", cfg.DBPath)
	if err != nil {
		log.Fatalf("open %s: %v", cfg.DBPath, err)
	}
	defer conn.Close()

	queries := db.New(conn)

	log.Printf("loading RE24/win-probability/model artifacts from %s...", cfg.DBPath)
	tables, model, medians, err := decision.LoadAll(context.Background(), queries)
	if err != nil {
		log.Fatalf("load artifacts: %v", err)
	}
	log.Printf("loaded: %d RE24 cells, %d after-cells, %d hold-cells, %d model features",
		len(tables.RE24), len(tables.WPTable), len(tables.WPHold), len(model.FeatureOrder))

	server := &api.Server{Tables: tables, Model: model, Medians: medians, Queries: queries}
	router := api.NewRouter(server, cfg.CORSOrigin)

	log.Printf("listening on :%s (CORS origin: %s)", cfg.Port, cfg.CORSOrigin)
	log.Fatal(http.ListenAndServe(":"+cfg.Port, router))
}
