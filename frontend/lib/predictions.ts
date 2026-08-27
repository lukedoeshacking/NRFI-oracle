export type Prediction = {
  home_team: string;
  away_team: string;
  home_pitcher: string;
  away_pitcher: string;
  nrfi_probability: number;
  prediction: "NRFI" | "YRFI" | string;
  confidence: "High" | "Medium" | "Low" | string;
  game_date: string;
};

export function apiUrl(path: string) {
  const base = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";
  return `${base.replace(/\/$/, "")}${path}`;
}

/** API may send 0–1 or 0–100. Always return 0–1. */
export function asUnitProbability(value: number) {
  if (Number.isNaN(value)) return 0;
  return value > 1 ? value / 100 : value;
}
