"""
build_ratings.py — Pipeline: historical match results → dynamic team ratings in teams.py

Reads data/results.csv (populated automatically by refresh.py).
Run `python refresh.py` first if the file is missing.

Usage
-----
    python build_ratings.py                  # update teams.py with Elo ratings
    python build_ratings.py --calibrate      # also fit OVERDISPERSION from data
    python build_ratings.py --show-top 20    # print top-N rated teams + don't write
    python build_ratings.py --history ARG    # print Argentina Elo over time
"""

from __future__ import annotations

import argparse
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
            "Run `python refresh.py --data-only` to download it automatically.\n"
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
    """Load the TEAMS dict from teams.py via direct module import."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("teams", TEAMS_PY)
    mod  = importlib.util.module_from_spec(spec)       # type: ignore[arg-type]
    spec.loader.exec_module(mod)                        # type: ignore[union-attr]
    return dict(mod.TEAMS)


# ── Rating computation ────────────────────────────────────────────────────────

def compute_ratings(df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
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


# ── Tilt computation ─────────────────────────────────────────────────────────

def compute_tilt(
    df: pd.DataFrame,
    ratings: dict[str, float],
    current_teams: dict[str, dict],
    n_recent: int = 60,
) -> dict[str, float]:
    """
    Compute tactical tilt for each WC team from goal residuals.

    For each team, look at its last *n_recent* matches. For each match compute
    the ratio of actual total goals to expected total goals (based on current
    Elo). Average these ratios, subtract 1, then shrink toward zero.

    Positive tilt → team tends to be in higher-scoring games than expected.
    Negative tilt → team tends to be in lower-scoring games than expected.
    """
    from elo import elo_to_off_def, normalise_name

    tilts: dict[str, float] = {}

    for team in current_teams:
        elo_team = ratings.get(team)
        if elo_team is None:
            tilts[team] = 0.0
            continue

        # Filter matches where this team appeared under any of its names
        def _is_team(raw: str) -> bool:
            return normalise_name(raw) == team

        mask = (
            df["home_team"].apply(_is_team) | df["away_team"].apply(_is_team)
        )
        team_matches = df[mask].sort_values("date").tail(n_recent)

        if len(team_matches) < 15:
            tilts[team] = 0.0
            continue

        residuals = []
        for _, row in team_matches.iterrows():
            home_name = normalise_name(str(row["home_team"]))
            away_name = normalise_name(str(row["away_team"]))
            if home_name is None or away_name is None:
                continue
            elo_h = ratings.get(home_name, 1500.0)
            elo_a = ratings.get(away_name, 1500.0)
            off_h, def_h = elo_to_off_def(elo_h)
            off_a, def_a = elo_to_off_def(elo_a)
            xg_h = off_h * (def_a / config.LEAGUE_AVG)
            xg_a = off_a * (def_h / config.LEAGUE_AVG)
            expected_total = xg_h + xg_a
            actual_total   = int(row["home_score"]) + int(row["away_score"])
            if expected_total > 0.1:
                residuals.append(actual_total / expected_total - 1.0)

        if residuals:
            raw = sum(residuals) / len(residuals)
            tilts[team] = round(raw * config.TILT_TACTICAL_SHRINK, 4)
        else:
            tilts[team] = 0.0

    return tilts


def patch_tilt(
    current_teams: dict[str, dict],
    tilts: dict[str, float],
    src: str,
) -> str:
    """Apply tilt values into the teams.py source string; return updated src."""
    for team, tilt_val in tilts.items():
        if team not in current_teams:
            continue
        pattern     = rf'("{re.escape(team)}":\s*\{{[^}}]*?"tilt":\s*)-?[\d.]+'
        replacement = rf'\g<1>{tilt_val}'
        src = re.sub(pattern, replacement, src)
    return src


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
        description="Build dynamic Elo ratings from historical match data (data/results.csv)"
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

    # Tilt computation
    print("\nComputing tactical tilt …")
    tilts = compute_tilt(df, ratings, current_teams)
    non_zero = {t: v for t, v in tilts.items() if abs(v) > 0.001}
    print(f"  {len(non_zero)} teams with non-zero tilt.")
    if not args.dry_run:
        src = TEAMS_PY.read_text()
        src = patch_tilt(current_teams, tilts, src)
        TEAMS_PY.write_text(src)
        print("  Tilt values written to teams.py.")
    else:
        top_tilts = sorted(non_zero.items(), key=lambda kv: -abs(kv[1]))[:10]
        print("  Top tilt values (dry run):")
        for t, v in top_tilts:
            print(f"    {t:<25} {v:+.4f}")

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
