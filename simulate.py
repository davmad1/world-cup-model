"""
2026 FIFA World Cup Monte Carlo simulator — 538-style SPI model.

Usage
-----
    python simulate.py                   # 10 000 simulations (default)
    python simulate.py --n 50000         # custom iteration count
    python simulate.py --group-probs     # also print group-stage win probs
    python simulate.py --matchup ARG BRA # print head-to-head win probability

The model recreates the core logic of FiveThirtyEight's Soccer Power Index:
  • Multiplicative xG model (off × def / avg)
  • Independent Poisson goal distributions with Dixon-Coles low-score fix
  • Full group-stage simulation with FIFA tiebreaker rules
  • Extra time + penalty shootout for knockout ties
  • 12-group / 48-team 2026 format with 8 best third-place qualifiers
"""

from __future__ import annotations

import argparse
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


def print_results(probs: dict[str, dict[str, float]]) -> None:
    """Print a sorted results table."""
    rows = []
    for team, p in probs.items():
        rows.append([
            team,
            TEAMS[team]["group"],
            f"{spi(team):.1f}",
            f"{p['winner']*100:.1f}%",
            f"{p['final']*100:.1f}%",
            f"{p['sf']*100:.1f}%",
            f"{p['qf']*100:.1f}%",
            f"{p['r16']*100:.1f}%",
            f"{p['r32']*100:.1f}%",
        ])

    rows.sort(key=lambda r: float(r[3].rstrip("%")), reverse=True)

    headers = ["Team", "Grp", "SPI", "Win%", "Final%", "SF%", "QF%", "R16%", "Adv%"]
    print("\n" + tabulate(rows, headers=headers, tablefmt="simple"))


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
    print("Model: Poisson xG + Dixon-Coles + ET + penalties")
    print("Groups sourced from openfootball/worldcup.json (official draw).\n")

    probs = run_simulation(args.n)
    print_results(probs)

    if args.group_probs:
        print_group_probs(probs)

    # Summary stats
    top5 = sorted(TEAMS, key=lambda t: probs[t]["winner"], reverse=True)[:5]
    print("\nTop-5 favourites to win:")
    for i, t in enumerate(top5, 1):
        print(f"  {i}. {t:15s}  {probs[t]['winner']*100:.1f}%")


if __name__ == "__main__":
    main()
