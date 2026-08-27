import type { Prediction } from "@/lib/predictions";
import { asUnitProbability } from "@/lib/predictions";

function confidenceStyles(confidence: string) {
  const key = confidence.toLowerCase();
  if (key === "high") {
    return "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30";
  }
  if (key === "medium") {
    return "bg-amber-400/15 text-amber-300 ring-1 ring-amber-400/30";
  }
  return "bg-slate-500/20 text-slate-300 ring-1 ring-slate-500/30";
}

type GameCardProps = {
  game: Prediction;
};

export default function GameCard({ game }: GameCardProps) {
  const nrfiProb = asUnitProbability(Number(game.nrfi_probability));
  const isNrfi = String(game.prediction).toUpperCase() === "NRFI";
  const displayProb = isNrfi ? nrfiProb : 1 - nrfiProb;
  const percent = Math.round(displayProb * 100);
  const barColor = isNrfi ? "bg-emerald-500" : "bg-rose-500";
  const label = `${percent}% ${isNrfi ? "NRFI" : "YRFI"}`;

  return (
    <article className="rounded-xl border border-slate-700/70 bg-[#1e293b] p-5 shadow-sm shadow-black/20">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-white">
            {game.away_team} @ {game.home_team}
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            {game.away_pitcher} vs {game.home_pitcher}
          </p>
        </div>
        <span
          className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-semibold ${confidenceStyles(game.confidence)}`}
        >
          {game.confidence}
        </span>
      </div>

      <div className="mt-5">
        <div className="relative h-8 overflow-hidden rounded-md bg-slate-900/80 ring-1 ring-inset ring-slate-700/80">
          <div
            className={`h-full ${barColor} transition-[width] duration-500`}
            style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
          />
          <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold tracking-wide text-white drop-shadow">
            {label}
          </span>
        </div>
      </div>
    </article>
  );
}
