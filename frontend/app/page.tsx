"use client";

import { useEffect, useState } from "react";
import GameCard from "@/components/GameCard";
import { apiUrl, type Prediction } from "@/lib/predictions";

function todayLabel() {
  return new Date().toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

export default function HomePage() {
  const [games, setGames] = useState<Prediction[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dateLabel, setDateLabel] = useState("");

  useEffect(() => {
    setDateLabel(todayLabel());
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const response = await fetch(apiUrl("/api/predictions/today"));
        if (!response.ok) {
          throw new Error(`API returned ${response.status}`);
        }
        const data = await response.json();
        const list = Array.isArray(data) ? data : data.predictions;
        if (!cancelled) {
          setGames(Array.isArray(list) ? list : []);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Could not load predictions");
          setGames([]);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const loading = games === null;

  return (
    <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-10 sm:px-6">
      <div className="mb-10">
        <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
          ⚾ NRFI Oracle
        </h1>
        <p className="mt-2 text-base text-slate-400">MLB First Inning Predictions</p>
        <p className="mt-4 text-sm font-medium text-slate-300">{dateLabel}</p>
      </div>

      {loading && (
        <div className="flex flex-col items-center justify-center py-24" role="status">
          <div className="h-10 w-10 animate-spin rounded-full border-2 border-slate-600 border-t-emerald-400" />
          <span className="mt-4 text-sm text-slate-400">Loading today&apos;s slate…</span>
        </div>
      )}

      {!loading && error && (
        <p className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          Could not reach the API. {error}
        </p>
      )}

      {!loading && !error && games.length === 0 && (
        <p className="rounded-lg border border-slate-700 bg-[#1e293b] px-4 py-8 text-center text-slate-300">
          No predictions yet for today
        </p>
      )}

      {!loading && games.length > 0 && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {games.map((game, index) => (
            <GameCard
              key={`${game.game_date}-${game.home_team}-${game.away_team}-${index}`}
              game={game}
            />
          ))}
        </div>
      )}
    </main>
  );
}
