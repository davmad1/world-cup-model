"""
2026 FIFA World Cup Monte Carlo simulator — 538-style SPI model.

Usage
-----
    python simulate.py                   # 10 000 simulations (default)
    python simulate.py --n 50000         # custom iteration count
    python simulate.py --group-probs     # also print group-stage win probs
    python simulate.py --matchup ARG BRA # print head-to-head win probability

The model recreates and extends FiveThirtyEight's Soccer Power Index:
  • Dynamic Elo ratings from 49 000+ historical matches (harmonic MOV,
    importance weights, time decay, altitude/distance HFA)
  • Multiplicative xG model with tactical tilt (off × def / avg × goal_scalar)
  • Negative Binomial goal draws + Dixon-Coles low-score correction
  • Matchday-aware group schedule with final-day incentive modeling
  • Within-tournament hot-simulation Elo updates
  • Fair play (card simulation) as tiebreaker
  • Extra time + penalty shootout for knockout ties
  • 12-group / 48-team 2026 format with 8 best third-place qualifiers
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict

from tabulate import tabulate
from tqdm import tqdm

import config
from model import win_probability
from teams import TEAMS, spi, groups as get_groups
from tournament import simulate_tournament


ROUNDS = ["group", "r32", "r16", "qf", "sf", "final", "winner"]
ROUND_LABELS = {
    "group":  "Group",
    "r32":    "R32",
    "r16":    "R16",
    "qf":     "QF",
    "sf":     "SF",
    "final":  "Final",
    "winner": "Win",
}


def run_simulation(n: int) -> dict[str, dict[str, float]]:
    """Run *n* Monte Carlo tournaments; return averaged probabilities."""
    totals: dict[str, dict[str, int]] = {
        t: {r: 0 for r in ROUNDS} for t in TEAMS
    }

    for _ in tqdm(range(n), desc="Simulating", unit="sim", ncols=72):
        result = simulate_tournament()
        for team, rounds in result.items():
            for rnd, val in rounds.items():
                totals[team][rnd] += val

    probs: dict[str, dict[str, float]] = {
        t: {r: totals[t][r] / n for r in ROUNDS} for t in TEAMS
    }
    return probs


def _win_ci(p: float, n: int) -> float:
    """95 % binomial confidence interval half-width (percentage points)."""
    return 1.96 * math.sqrt(p * (1.0 - p) / n) * 100.0


def print_results(
    probs: dict[str, dict[str, float]],
    n_sims: int,
    winner_odds: dict[str, float] | None = None,
) -> None:
    """
    Print a sorted results table.

    Columns always shown: Team, Grp, SPI, Win%±CI, Final%, SF%, QF%, R16%, Adv%
    Extra columns when winner odds are available: Mkt%, Edge
    """
    has_market = bool(winner_odds)
    rows = []

    for team, p in probs.items():
        p_win = p["winner"]
        ci    = _win_ci(p_win, n_sims)
        win_str = f"{p_win*100:.1f}±{ci:.1f}%"

        row = [
            team,
            TEAMS[team]["group"],
            f"{spi(team):.1f}",
            win_str,
            f"{p['final']*100:.1f}%",
            f"{p['sf']*100:.1f}%",
            f"{p['qf']*100:.1f}%",
            f"{p['r16']*100:.1f}%",
            f"{p['r32']*100:.1f}%",
        ]

        if has_market:
            mkt = (winner_odds or {}).get(team)
            if mkt is not None:
                edge = (p_win - mkt) * 100
                row += [f"{mkt*100:.1f}%", f"{edge:+.1f}pp"]
            else:
                row += ["—", "—"]

        rows.append(row)

    rows.sort(key=lambda r: float(r[3].split("±")[0].rstrip("%")), reverse=True)

    headers = ["Team", "Grp", "SPI", "Win%±CI", "Final%", "SF%", "QF%", "R16%", "Adv%"]
    if has_market:
        headers += ["Mkt%", "Edge"]

    print("\n" + tabulate(rows, headers=headers, tablefmt="simple"))
    print(f"  CI = 95% binomial interval (sampling uncertainty, n={n_sims:,})")
    if has_market:
        print("  Mkt% = bookmaker implied win probability (overround removed)")
        print("  Edge = model − market (positive → we're higher than market)")


def print_group_probs(probs: dict[str, dict[str, float]]) -> None:
    """Print per-group tables showing group-stage advance probability."""
    groups = get_groups()
    for grp, teams in sorted(groups.items()):
        print(f"\n── Group {grp} ───────────────────────────")
        rows = [
            [t, f"{spi(t):.1f}", f"{probs[t]['r32']*100:.1f}%"]
            for t in sorted(teams, key=lambda t: probs[t]["r32"], reverse=True)
        ]
        print(tabulate(rows, headers=["Team", "SPI", "Advance%"], tablefmt="simple"))


def print_matchup(team_a: str, team_b: str) -> None:
    """Print analytical win probabilities for a single matchup."""
    if team_a not in TEAMS:
        sys.exit(f"Unknown team: {team_a}")
    if team_b not in TEAMS:
        sys.exit(f"Unknown team: {team_b}")

    off_a, def_a = TEAMS[team_a]["off"], TEAMS[team_a]["def"]
    off_b, def_b = TEAMS[team_b]["off"], TEAMS[team_b]["def"]

    p_win, p_draw, p_loss = win_probability(off_a, def_a, off_b, def_b)

    print(f"\n{team_a} vs {team_b} (neutral site, 90 min)")
    print(f"  {team_a} win : {p_win*100:.1f}%")
    print(f"  Draw        : {p_draw*100:.1f}%")
    print(f"  {team_b} win : {p_loss*100:.1f}%")

    # Knockout (extra time + pens)
    p_ko_a = p_win + p_draw * 0.5  # rough approximation for display
    print(f"\nKnockout (including ET/pens):")
    print(f"  {team_a} advances: ~{p_ko_a*100:.1f}%")
    print(f"  {team_b} advances: ~{(1-p_ko_a)*100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2026 World Cup SPI prediction model (538-style)"
    )
    parser.add_argument("-n", "--n", type=int, default=config.N_SIMS,
                        help=f"Number of Monte Carlo simulations (default: {config.N_SIMS})")
    parser.add_argument("--group-probs", action="store_true",
                        help="Print per-group advance probabilities")
    parser.add_argument("--matchup", nargs=2, metavar=("TEAM_A", "TEAM_B"),
                        help="Print head-to-head win probability for two teams")
    args = parser.parse_args()

    if args.matchup:
        print_matchup(args.matchup[0], args.matchup[1])
        return

    print(f"\n2026 FIFA World Cup — SPI Prediction Model (n={args.n:,})")
    print("=" * 60)
    print("Model: NB xG + Dixon-Coles + tilt + decay + incentives + ET + penalties")
    print("Groups sourced from openfootball/worldcup.json (official draw).\n")

    # Load bookmaker winner odds if available
    try:
        from odds import load_winner_odds
        winner_odds: dict[str, float] | None = load_winner_odds() or None
        if winner_odds:
            print(f"  Market odds loaded for {len(winner_odds)} teams (data/winner_odds.csv)")
    except Exception:
        winner_odds = None

    probs = run_simulation(args.n)
    print_results(probs, args.n, winner_odds=winner_odds)

    if args.group_probs:
        print_group_probs(probs)

    # Summary stats
    top5 = sorted(TEAMS, key=lambda t: probs[t]["winner"], reverse=True)[:5]
    print("\nTop-5 favourites to win:")
    for i, t in enumerate(top5, 1):
        p = probs[t]["winner"]
        ci = _win_ci(p, args.n)
        print(f"  {i}. {t:20s}  {p*100:.1f}% ± {ci:.1f}pp")


if __name__ == "__main__":
    main()
