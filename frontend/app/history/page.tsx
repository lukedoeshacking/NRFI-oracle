"use client";

import { useEffect, useState } from "react";
import { apiUrl, type Prediction } from "@/lib/predictions";

type HistoryResponse = {
  predictions: Prediction[];
  accuracy_pct: number | null;
  total_predictions: number;
};

const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

function formatDate(value: string) {
  // Parse YYYY-MM-DD without Date(), which would shift the day in US timezones.
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(value);
  if (!match) return value;
  const year = match[1];
  const month = Number(match[2]);
  const day = Number(match[3]);
  return `${MONTHS[month - 1]} ${day}, ${year}`;
}

function confidenceClass(confidence: string) {
  const key = confidence.toLowerCase();
  if (key === "high") return "text-emerald-300";
  if (key === "medium") return "text-amber-300";
  return "text-slate-400";
}

function predictionClass(prediction: string) {
  return prediction.toUpperCase() === "NRFI" ? "text-emerald-300" : "text-rose-300";
}

export default function HistoryPage() {
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const response = await fetch(apiUrl("/api/predictions/history?days=30"));
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }
        const json = await response.json();
        if (!cancelled) {
          setData({
            predictions: Array.isArray(json.predictions) ? json.predictions : [],
            accuracy_pct:
              json.accuracy_pct === null || json.accuracy_pct === undefined
                ? null
                : Number(json.accuracy_pct),
            total_predictions: Number(json.total_predictions ?? 0),
          });
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Could not load history");
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const loading = data === null && !error;

  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-10 sm:px-6">
      <h1 className="text-2xl font-semibold tracking-tight text-white">History</h1>
      <p className="mt-1 text-sm text-slate-400">Past first-inning picks and model accuracy</p>

      {loading && (
        <div className="flex justify-center py-24" role="status">
          <div className="h-10 w-10 animate-spin rounded-full border-2 border-slate-600 border-t-emerald-400" />
        </div>
      )}

      {error && (
        <p className="mt-8 rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          Could not reach the API. {error}
        </p>
      )}

      {data && (
        <>
          <div className="mt-8 flex flex-wrap gap-4">
            {data.accuracy_pct !== null && !Number.isNaN(data.accuracy_pct) && (
              <div className="rounded-xl border border-slate-700/70 bg-[#1e293b] px-5 py-4">
                <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
                  Overall accuracy
                </p>
                <p className="mt-1 text-3xl font-semibold tabular-nums text-white">
                  {data.accuracy_pct.toFixed(1)}%
                </p>
              </div>
            )}
            <div className="rounded-xl border border-slate-700/70 bg-[#1e293b] px-5 py-4">
              <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
                Predictions
              </p>
              <p className="mt-1 text-3xl font-semibold tabular-nums text-white">
                {data.total_predictions || data.predictions.length}
              </p>
            </div>
          </div>

          <div className="mt-8 overflow-x-auto rounded-xl border border-slate-700/70 bg-[#1e293b]">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead className="border-b border-slate-700 text-xs uppercase tracking-wider text-slate-400">
                <tr>
                  <th className="px-4 py-3 font-medium">Date</th>
                  <th className="px-4 py-3 font-medium">Matchup</th>
                  <th className="px-4 py-3 font-medium">Prediction</th>
                  <th className="px-4 py-3 font-medium">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {data.predictions.length === 0 && (
                  <tr>
                    <td colSpan={4} className="px-4 py-10 text-center text-slate-400">
                      No prediction history yet
                    </td>
                  </tr>
                )}
                {data.predictions.map((row, index) => (
                  <tr
                    key={`${row.game_date}-${row.home_team}-${row.away_team}-${index}`}
                    className="border-b border-slate-800 last:border-0"
                  >
                    <td className="whitespace-nowrap px-4 py-3 text-slate-300">
                      {formatDate(row.game_date)}
                    </td>
                    <td className="px-4 py-3 text-white">
                      {row.away_team} @ {row.home_team}
                    </td>
                    <td
                      className={`px-4 py-3 font-semibold ${predictionClass(row.prediction)}`}
                    >
                      {row.prediction}
                    </td>
                    <td className={`px-4 py-3 ${confidenceClass(row.confidence)}`}>
                      {row.confidence}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </main>
  );
}
