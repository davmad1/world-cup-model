"""
build_ratings.py — Pipeline: Kaggle CSVs → dynamic team ratings in teams.py

Downloads are NOT included in the repo (data/ is gitignored).
Place the Kaggle CSV at data/results.csv before running.

Kaggle dataset
--------------
Search: "International Soccer Results 1872 to 2024"
URL   : https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
File  : results.csv
Cols  : date, home_team, away_team, home_score, away_score,
        tournament, city, country, neutral

Usage
-----
    python build_ratings.py                  # update teams.py with Elo ratings
    python build_ratings.py --calibrate      # also fit OVERDISPERSION from data
    python build_ratings.py --show-top 20    # print top-N rated teams + don't write
    python build_ratings.py --history ARG    # print Argentina Elo over time
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

import pandas as pd

import config
from elo import compute_elo, elo_to_off_def, calibrate_overdispersion

DATA_DIR   = Path(__file__).parent / "data"
RESULTS_CSV = DATA_DIR / "results.csv"
TEAMS_PY   = Path(__file__).parent / "teams.py"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(path: Path = RESULTS_CSV) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"\nERROR: {path} not found.\n\n"
            "Download the Kaggle dataset:\n"
            "  https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017\n"
            "Place results.csv in the data/ directory, then re-run.\n"
        )
    df = pd.read_csv(path)
    # Drop rows with missing scores
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    print(f"Loaded {len(df):,} matches from {path.name} "
          f"({df['date'].min()} → {df['date'].max()})")
    return df


# ── Current teams in teams.py ─────────────────────────────────────────────────

def load_current_teams() -> dict[str, dict]:
    """
    Parse the TEAMS dict from teams.py without importing it
    (avoids circular issues when we overwrite the file).
    """
    src = TEAMS_PY.read_text()
    # Extract the TEAMS = { ... } block
    m = re.search(r"TEAMS[^=]*=\s*(\{.*?\})\s*\n\n", src, re.DOTALL)
    if not m:
        sys.exit("Could not parse TEAMS dict from teams.py")
    return ast.literal_eval(m.group(1))


# ── Rating computation ────────────────────────────────────────────────────────

def compute_ratings(df: pd.DataFrame) -> dict[str, float]:
    """Run Elo on the full historical dataset and return final ratings."""
    print("Computing Elo ratings …")
    ratings, history = compute_elo(df, verbose=True)
    n = len(ratings)
    print(f"Done. Rated {n} teams.")
    return ratings, history


# ── Patch teams.py ────────────────────────────────────────────────────────────

def patch_teams_py(
    current_teams: dict[str, dict],
    new_ratings: dict[str, float],
    dry_run: bool = False,
) -> None:
    """
    Update the off/def values in teams.py for every team that has an Elo rating.
    Preserves group assignments and any fields not related to ratings.
    """
    src = TEAMS_PY.read_text()
    changed = []

    for team, data in current_teams.items():
        elo = new_ratings.get(team)
        if elo is None:
            print(f"  WARNING: no Elo found for '{team}' — keeping existing rating")
            continue

        new_off, new_def = elo_to_off_def(elo)
        old_off = data.get("off", 0)
        old_def = data.get("def", 0)

        if abs(new_off - old_off) < 0.001 and abs(new_def - old_def) < 0.001:
            continue  # no change

        # Replace "off": X.XX  and "def": Y.YY for this team
        # Pattern: find the team's dict entry and swap off/def values
        pattern = rf'("{re.escape(team)}":\s*\{{[^}}]*?"off":\s*)[\d.]+([^}}]*?"def":\s*)[\d.]+'
        replacement = rf'\g<1>{new_off}\g<2>{new_def}'
        new_src, n = re.subn(pattern, replacement, src)
        if n > 0:
            src = new_src
            changed.append((team, old_off, new_off, old_def, new_def, elo))

    if not changed:
        print("teams.py already up to date — no changes needed.")
        return

    print(f"\n{'DRY RUN — ' if dry_run else ''}Rating changes ({len(changed)} teams):")
    print(f"  {'Team':<25} {'Elo':>6}  {'old off':>7} → {'new off':>7}  "
          f"{'old def':>7} → {'new def':>7}")
    for team, oo, no, od, nd, elo in sorted(changed, key=lambda r: -r[5]):
        print(f"  {team:<25} {elo:>6.0f}  {oo:>7.3f} → {no:>7.3f}  {od:>7.3f} → {nd:>7.3f}")

    if not dry_run:
        TEAMS_PY.write_text(src)
        print(f"\nteams.py updated.")


# ── Top-N display ─────────────────────────────────────────────────────────────

def show_top(ratings: dict[str, float], n: int = 20) -> None:
    top = sorted(ratings.items(), key=lambda kv: -kv[1])[:n]
    print(f"\nTop-{n} Elo ratings (all-time history):")
    print(f"  {'Rank':>4}  {'Team':<25} {'Elo':>7}  {'off':>5}  {'def':>5}  {'SPI':>5}")
    for rank, (team, elo) in enumerate(top, 1):
        off, def_ = elo_to_off_def(elo)
        spi = 100 * off / (off + def_)
        print(f"  {rank:>4}  {team:<25} {elo:>7.1f}  {off:>5.3f}  {def_:>5.3f}  {spi:>5.1f}")


# ── History plot (ASCII) ──────────────────────────────────────────────────────

def show_history(history: pd.DataFrame, team: str) -> None:
    # Gather all appearances (home or away)
    home_mask = history["home"] == team
    away_mask = history["away"] == team
    pts = pd.concat([
        history[home_mask][["date", "elo_home"]].rename(columns={"elo_home": "elo"}),
        history[away_mask][["date", "elo_away"]].rename(columns={"elo_away": "elo"}),
    ]).sort_values("date")

    if pts.empty:
        print(f"No history found for '{team}'.")
        return

    # Annual snapshots
    pts["year"] = pd.to_datetime(pts["date"]).dt.year
    annual = pts.groupby("year")["elo"].last()

    lo, hi = annual.min(), annual.max()
    width = 50

    print(f"\nElo history — {team}")
    print(f"  Range: {lo:.0f} – {hi:.0f}")
    for year, elo in annual.items():
        bar_len = int((elo - lo) / max(1, hi - lo) * width)
        bar = "█" * bar_len
        print(f"  {year}  {elo:>7.1f}  {bar}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dynamic Elo ratings from Kaggle historical match data"
    )
    parser.add_argument("--calibrate", action="store_true",
                        help="Fit and print the OVERDISPERSION parameter from goal data")
    parser.add_argument("--show-top", type=int, default=0, metavar="N",
                        help="Print top-N rated teams and exit (no file write)")
    parser.add_argument("--history", metavar="TEAM",
                        help="Print ASCII Elo history for one team and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing to teams.py")
    args = parser.parse_args()

    df = load_results()

    if args.calibrate:
        all_goals = pd.concat([df["home_score"], df["away_score"]])
        disp = calibrate_overdispersion(all_goals)
        print(f"\nFitted OVERDISPERSION = {disp:.4f}  "
              f"(current config: {config.OVERDISPERSION})")
        if abs(disp - config.OVERDISPERSION) > 0.01:
            print("  → Consider updating config.OVERDISPERSION")

    ratings, history = compute_ratings(df)

    if args.history:
        show_history(history, args.history)
        return

    if args.show_top:
        show_top(ratings, args.show_top)
        return

    current_teams = load_current_teams()
    patch_teams_py(current_teams, ratings, dry_run=args.dry_run)

    # Summary stats for WC teams
    wc_elos = {t: ratings[t] for t in current_teams if t in ratings}
    if wc_elos:
        best  = max(wc_elos, key=lambda t: wc_elos[t])
        worst = min(wc_elos, key=lambda t: wc_elos[t])
        print(f"\nWC team Elo summary:")
        print(f"  Best:   {best:<20} {wc_elos[best]:.1f}")
        print(f"  Worst:  {worst:<20} {wc_elos[worst]:.1f}")
        print(f"  Mean:   {sum(wc_elos.values())/len(wc_elos):.1f}")
        missing = [t for t in current_teams if t not in ratings]
        if missing:
            print(f"  No Elo data: {', '.join(missing)}")


if __name__ == "__main__":
    main()
