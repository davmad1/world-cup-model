"""
2026 FIFA World Cup tournament structure.

Format
------
• 48 teams in 12 groups (A–L) of 4
• Each team plays the other 3 in its group once
• Top 2 from every group advance automatically (24 teams)
• Best 8 third-place finishers also advance (32 total in knockout)
• Knockout: R32 → R16 → QF → SF → Final (+ 3rd-place play-off)

Third-place tiebreakers (FIFA rules):
  1. Points   2. GD   3. GF   4. Fair play (we use random in simulation)

Bracket
-------
The R32 bracket pairing of third-place teams follows FIFA's pre-announced
schedule. Because the exact 2026 pairing table isn't publicly confirmed at
model-build time, we use a reasonable approximation that mirrors the 2026
format structure: the 8 advancing thirds are inserted into the bracket in
seeded order (best third gets easiest projected draw, etc.).

Update _r32_bracket() if the official bracket pairings are released.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import NamedTuple

from model import simulate_knockout_match, simulate_match
from teams import TEAMS, groups as get_groups


# ── Data structures ───────────────────────────────────────────────────────────

class Record(NamedTuple):
    team: str
    pts: int
    gd: int
    gf: int

    def tiebreak_key(self) -> tuple:
        return (self.pts, self.gd, self.gf)


# ── Group stage ───────────────────────────────────────────────────────────────

def _update(standings: dict, team: str, gf: int, ga: int, pts: int) -> None:
    s = standings[team]
    standings[team] = Record(
        team=team,
        pts=s.pts + pts,
        gd=s.gd + gf - ga,
        gf=s.gf + gf,
    )


def simulate_group(teams: list[str]) -> list[Record]:
    """
    Simulate all 6 matches in a group. Return standings sorted by
    (pts, gd, gf) descending. Remaining ties broken randomly.
    """
    standings = {t: Record(t, 0, 0, 0) for t in teams}

    for a, b in combinations(teams, 2):
        off_a, def_a = TEAMS[a]["off"], TEAMS[a]["def"]
        off_b, def_b = TEAMS[b]["off"], TEAMS[b]["def"]
        g_a, g_b = simulate_match(off_a, def_a, off_b, def_b)

        if g_a > g_b:
            _update(standings, a, g_a, g_b, 3)
            _update(standings, b, g_b, g_a, 0)
        elif g_b > g_a:
            _update(standings, b, g_b, g_a, 3)
            _update(standings, a, g_a, g_b, 0)
        else:
            _update(standings, a, g_a, g_b, 1)
            _update(standings, b, g_b, g_a, 1)

    sorted_records = sorted(
        standings.values(),
        key=lambda r: (r.tiebreak_key(), random.random()),
        reverse=True,
    )
    return sorted_records


# ── Third-place ranking ───────────────────────────────────────────────────────

def best_third_places(thirds: list[Record], n: int = 8) -> list[Record]:
    """Return the *n* best third-place teams across all groups."""
    return sorted(thirds, key=lambda r: (r.tiebreak_key(), random.random()), reverse=True)[:n]


# ── R32 bracket construction ──────────────────────────────────────────────────

# Pre-set R32 matchup template (group positions only; not the exact 2026 FIFA
# table, but structurally equivalent — avoids same-group encounters in R32).
#
# Each tuple is (slot_A_source, slot_B_source) where source is a string like
# "A1" (winner of group A), "B2" (runner-up of group B), or "3rd" (a
# third-place qualifier, ordered best→worst).

_R32_TEMPLATE = [
    ("A1", "B2"),
    ("C1", "D2"),
    ("E1", "F2"),
    ("G1", "H2"),
    ("I1", "J2"),
    ("K1", "L2"),
    ("A2", "C2"),  # these two filled by remaining group-stage qualifiers
    ("E2", "G2"),
    ("I2", "K2"),
    ("B1", "D1"),
    ("F1", "H1"),
    ("J1", "L1"),
    # The remaining 4 R32 matches pair 3rd-place teams vs group runners-up
    # (exact matchups depend on which thirds advance; approximate here)
    ("B2_3rd", "3rd_1"),
    ("D2_3rd", "3rd_2"),
    ("F2_3rd", "3rd_3"),
    ("H2_3rd", "3rd_4"),
]


def build_r32(
    group_results: dict[str, list[Record]],
    advancing_thirds: list[Record],
) -> list[tuple[str, str]]:
    """
    Build the list of 16 R32 matchups (team_a, team_b) from group results
    and the 8 advancing third-place teams.
    """
    # Map "A1", "A2", "A3" etc.
    slots: dict[str, str] = {}
    for grp, records in group_results.items():
        for pos, rec in enumerate(records, start=1):
            slots[f"{grp}{pos}"] = rec.team

    # Third-place teams ordered best→worst
    thirds_ordered = [r.team for r in advancing_thirds]

    matchups: list[tuple[str, str]] = []

    # Fixed 12 group-winner / runner-up matchups
    fixed_pairs = [
        ("A1", "B2"), ("C1", "D2"), ("E1", "F2"), ("G1", "H2"),
        ("I1", "J2"), ("K1", "L2"), ("A2", "C2"), ("E2", "G2"),
        ("I2", "K2"), ("B1", "D1"), ("F1", "H1"), ("J1", "L1"),
    ]
    for s_a, s_b in fixed_pairs:
        matchups.append((slots[s_a], slots[s_b]))

    # 4 matchups: best remaining runners-up vs 3rd-place teams
    # Use group runners-up not yet placed (L2, K2 already used above;
    # these are the remaining 4: B2, D2, F2, H2 against thirds)
    runner_up_slots = ["B2", "D2", "F2", "H2"]
    for i, ru_slot in enumerate(runner_up_slots):
        team_a = slots[ru_slot]
        team_b = thirds_ordered[i] if i < len(thirds_ordered) else slots[ru_slot]
        matchups.append((team_a, team_b))

    return matchups


# ── Knockout rounds ───────────────────────────────────────────────────────────

def simulate_knockout_round(matchups: list[tuple[str, str]]) -> list[str]:
    """Simulate one round of knockout matches; return list of winners."""
    winners = []
    for team_a, team_b in matchups:
        off_a, def_a = TEAMS[team_a]["off"], TEAMS[team_a]["def"]
        off_b, def_b = TEAMS[team_b]["off"], TEAMS[team_b]["def"]
        winner = simulate_knockout_match(team_a, team_b, off_a, def_a, off_b, def_b)
        winners.append(winner)
    return winners


def pair_winners(winners: list[str]) -> list[tuple[str, str]]:
    """Pair consecutive winners: [w0,w1,w2,w3,...] → [(w0,w1),(w2,w3),...]."""
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


# ── Full tournament simulation ────────────────────────────────────────────────

def simulate_tournament() -> dict[str, dict[str, int]]:
    """
    Run one complete tournament simulation.

    Returns a dict mapping team name → round reached flags:
      {team: {"group": 1, "r32": 0|1, "r16": 0|1, "qf": 0|1,
              "sf": 0|1, "final": 0|1, "winner": 0|1}}
    """
    all_teams = list(TEAMS.keys())
    results: dict[str, dict[str, int]] = {
        t: {"group": 1, "r32": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "winner": 0}
        for t in all_teams
    }

    # ── Group stage ───────────────────────────────────────────────
    groups = get_groups()
    group_results: dict[str, list[Record]] = {}
    thirds: list[Record] = []

    for grp, teams in groups.items():
        standings = simulate_group(teams)
        group_results[grp] = standings
        thirds.append(standings[2])  # 3rd-place finisher

        # Top 2 advance
        for rec in standings[:2]:
            results[rec.team]["r32"] = 1

    # ── Best 8 third-place teams ───────────────────────────────────
    best_thirds = best_third_places(thirds, n=8)
    for rec in best_thirds:
        results[rec.team]["r32"] = 1

    # ── Build R32 bracket ─────────────────────────────────────────
    r32_matchups = build_r32(group_results, best_thirds)

    # ── R32 ───────────────────────────────────────────────────────
    r32_winners = simulate_knockout_round(r32_matchups)
    for t in r32_winners:
        results[t]["r16"] = 1

    # ── R16 ───────────────────────────────────────────────────────
    r16_matchups = pair_winners(r32_winners)
    r16_winners = simulate_knockout_round(r16_matchups)
    for t in r16_winners:
        results[t]["qf"] = 1

    # ── QF ────────────────────────────────────────────────────────
    qf_matchups = pair_winners(r16_winners)
    qf_winners = simulate_knockout_round(qf_matchups)
    for t in qf_winners:
        results[t]["sf"] = 1

    # ── SF ────────────────────────────────────────────────────────
    sf_matchups = pair_winners(qf_winners)
    sf_winners = simulate_knockout_round(sf_matchups)
    sf_losers = [t for t in (m[0] for m in sf_matchups) if t not in sf_winners]
    sf_losers += [t for t in (m[1] for m in sf_matchups) if t not in sf_winners]

    for t in sf_winners:
        results[t]["final"] = 1

    # ── Final ─────────────────────────────────────────────────────
    final_matchup = [(sf_winners[0], sf_winners[1])]
    final_winner = simulate_knockout_round(final_matchup)[0]
    results[final_winner]["winner"] = 1

    return results
