import type {
  PlayerRole,
  PlayerSearchResult,
  PredictResponse,
  Situation,
} from "@/lib/types"

export async function searchPlayers(
  role: PlayerRole,
  q: string,
  limit = 10,
): Promise<PlayerSearchResult[]> {
  const params = new URLSearchParams({ role, q, limit: String(limit) })
  const res = await fetch(`/api/players/search?${params}`)
  if (!res.ok) {
    throw new Error(`player search failed (${res.status})`)
  }
  return res.json()
}

export async function predictStealDecision(
  situation: Situation,
): Promise<PredictResponse> {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(situation),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.error ?? `prediction failed (${res.status})`)
  }
  return res.json()
}
