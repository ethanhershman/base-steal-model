package config

import "os"

type Config struct {
	DBPath     string
	Port       string
	CORSOrigin string
}

func Load() Config {
	return Config{
		DBPath:     getEnv("DB_PATH", "data/app.db"),
		Port:       getEnv("PORT", "8080"),
		CORSOrigin: getEnv("CORS_ORIGIN", "http://localhost:5173"),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
