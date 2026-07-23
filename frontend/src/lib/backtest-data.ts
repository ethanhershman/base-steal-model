// Sourced from README.md, "Backtest" section -- XGBoost model, 2,714 held-out
// attempts (2025-06-01 to 2025-11-01), full held-out test set (not a sample).
export const BACKTEST_LAYERS = [
  {
    key: "re24",
    label: "RE24 (runs)",
    description:
      "Most of the game -- decisions valued in expected runs added.",
    attempts: 2039,
    goSuccessRate: 0.811,
    holdSuccessRate: 0.695,
    actualPolicy: 0.0076,
    modelPolicy: 0.0309,
  },
  {
    key: "win-probability",
    label: "Win probability (late/close)",
    description:
      "High-leverage late/close situations, where RE24's run-based math understates the cost of a caught stealing that ends a trailing team's last chance.",
    attempts: 675,
    goSuccessRate: 0.822,
    holdSuccessRate: 0.807,
    actualPolicy: 0.0069,
    modelPolicy: 0.0211,
  },
] as const;

export const BACKTEST_TOTAL_ATTEMPTS = 2714;

export const BACKTEST_HOLD_STORY = {
  totalHeld: 819,
  caughtIfAttempted: 250,
  missedOpportunities: 569,
};
