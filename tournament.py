"""
2026 FIFA World Cup tournament structure.

Format
------
• 48 teams in 12 groups (A–L) of 4
• Each team plays the other 3 in its group once (3 matchdays)
• Top 2 from every group advance automatically (24 teams)
• Best 8 third-place finishers also advance (32 total in knockout)
• Knockout: R32 → R16 → QF → SF → Final

Improvements over baseline
---------------------------
• Matchday-aware group schedule (matchday 3 always played last)
• Final-matchday incentive modeling: teams already safe play defensively;
  teams both eliminated open up (INCENTIVE_SAFE_XG_ADJ / INCENTIVE_ELIM_XG_ADJ)
• Within-tournament "hot" Elo updates after each group match (HOT_K)
• Tilt ratings passed through to match simulation
• Fair play (card simulation) as final tiebreaker before random
"""

from __future__ import annotations

import random
from typing import NamedTuple

import numpy as np

import config
from model import simulate_knockout_match, simulate_match
from teams import TEAMS, groups as get_groups

# NB for fair-play card simulation
_RNG = np.random.default_rng()


# ── Data structures ───────────────────────────────────────────────────────────

class Record(NamedTuple):
    team: str
    pts: int
    gd: int
    gf: int
    cards: int = 0   # accumulated simulated yellow-card equivalents

    def tiebreak_key(self) -> tuple:
        return (self.pts, self.gd, self.gf, -self.cards)


# ── Fair play helper ──────────────────────────────────────────────────────────

def _simulate_cards(team: str) -> int:
    """
    Draw yellow-card-equivalents for a team in one match.
    Better teams (higher SPI) commit fewer fouls on average.
    """
    from teams import spi as _spi
    team_spi = _spi(team)
    # Adjust mean downward for stronger teams (they control the ball more)
    mean = max(0.5, config.CARDS_MEAN - (team_spi - 50) * config.CARDS_ELO_SLOPE / 100)
    n = 1.0 / config.CARDS_OVERDISPERSION
    p = n / (n + mean)
    return int(_RNG.negative_binomial(n, p))


# ── Group schedule ────────────────────────────────────────────────────────────

def _group_schedule(teams: list[str]) -> list[list[tuple[str, str]]]:
    """
    Return a fixed 3-matchday schedule for a group of 4 teams.

    Matchday 1: (0 v 1), (2 v 3)
    Matchday 2: (0 v 2), (1 v 3)
    Matchday 3: (0 v 3), (1 v 2)   ← both games simultaneous

    This mirrors the standard FIFA group-stage scheduling convention.
    """
    a, b, c, d = teams
    return [
        [(a, b), (c, d)],
        [(a, c), (b, d)],
        [(a, d), (b, c)],
    ]


# ── Incentive classification ──────────────────────────────────────────────────

def _classify_matchday3(standings: dict[str, "Record"], team_a: str, team_b: str) -> str:
    """
    Classify a matchday-3 fixture for incentive purposes.

    Returns:
        "both_safe"  — both teams guaranteed top-2 regardless of result
        "both_out"   — neither team can reach top-2 regardless of result
        "normal"     — at least one team still has something to play for
    """
    others = [t for t in standings if t not in (team_a, team_b)]
    max_other_pts = max(standings[t].pts for t in others)

    def is_safe(team: str) -> bool:
        # Safe if current pts > max pts the 3rd team can reach
        # (3rd team can reach max_other_pts; we need to be above them regardless)
        pts = standings[team].pts
        # Even with a loss (0 pts gained), do we have more pts than the best
        # outside team who plays their final game concurrently (can gain 3)?
        return pts > max_other_pts + 3

    def is_eliminated(team: str) -> bool:
        pts = standings[team].pts
        # Even with a win (+3 pts), can we beat the current 2nd-place outsider?
        # 2nd place outsider has max_other_pts and might gain 0 from their game
        return pts + 3 < max_other_pts

    safe_a = is_safe(team_a)
    safe_b = is_safe(team_b)
    out_a  = is_eliminated(team_a)
    out_b  = is_eliminated(team_b)

    if safe_a and safe_b:
        return "both_safe"
    if out_a and out_b:
        return "both_out"
    return "normal"


# ── Group stage ───────────────────────────────────────────────────────────────

def _update(standings: dict, team: str, gf: int, ga: int, pts: int, cards: int) -> None:
    s = standings[team]
    standings[team] = Record(
        team=team,
        pts=s.pts + pts,
        gd=s.gd + gf - ga,
        gf=s.gf + gf,
        cards=s.cards + cards,
    )


def _hot_adjust(local_off: dict, local_def: dict, team: str,
                won: bool, drew: bool, xg_for: float, xg_against: float,
                actual_for: int, actual_against: int) -> None:
    """
    Apply a small within-tournament rating adjustment after a group match.
    Uses a simplified proportional nudge rather than full Elo re-computation.
    """
    if config.HOT_K <= 0:
        return
    # Simple result-based nudge: win boosts off/def slightly, loss reverses
    nudge_pct = config.HOT_K / (config.ELO_K_BASE * 100)  # ~0.25% per match
    if won:
        local_off[team] *= (1 + nudge_pct)
        local_def[team] *= (1 - nudge_pct)
    elif not drew:
        local_off[team] *= (1 - nudge_pct)
        local_def[team] *= (1 + nudge_pct)


def simulate_group(
    teams: list[str],
    local_off: dict[str, float],
    local_def: dict[str, float],
) -> list[Record]:
    """
    Simulate all 6 matches in a group using the matchday-aware schedule.

    local_off / local_def are mutable per-simulation rating snapshots
    that accumulate hot-simulation adjustments.

    Returns standings sorted by (pts, gd, gf, fair_play, random).
    """
    standings = {t: Record(t, 0, 0, 0, 0) for t in teams}
    schedule  = _group_schedule(teams)

    for md_idx, matchday in enumerate(schedule):
        is_last_day = (md_idx == 2)

        for team_a, team_b in matchday:
            tilt_a = TEAMS[team_a].get("tilt", 0.0)
            tilt_b = TEAMS[team_b].get("tilt", 0.0)
            xg_adj = 0.0

            if is_last_day:
                incentive = _classify_matchday3(standings, team_a, team_b)
                if incentive == "both_safe":
                    xg_adj = config.INCENTIVE_SAFE_XG_ADJ
                elif incentive == "both_out":
                    xg_adj = config.INCENTIVE_ELIM_XG_ADJ

            off_a = local_off[team_a]
            off_b = local_off[team_b]
            def_a = local_def[team_a]
            def_b = local_def[team_b]

            # xG adjustment via temporary off scaling (additive on xg, not off)
            # We simulate directly, applying the incentive as a home_advantage-like
            # additive term that shifts total goals symmetrically.
            g_a, g_b = simulate_match(
                off_a, def_a, off_b, def_b,
                home_advantage=xg_adj,   # both teams scaled via off*adj/2 would be ideal
                tilt_a=tilt_a, tilt_b=tilt_b,
            )

            cards_a = _simulate_cards(team_a)
            cards_b = _simulate_cards(team_b)

            if g_a > g_b:
                _update(standings, team_a, g_a, g_b, 3, cards_a)
                _update(standings, team_b, g_b, g_a, 0, cards_b)
                _hot_adjust(local_off, local_def, team_a, True,  False, off_a, off_b, g_a, g_b)
                _hot_adjust(local_off, local_def, team_b, False, False, off_b, off_a, g_b, g_a)
            elif g_b > g_a:
                _update(standings, team_b, g_b, g_a, 3, cards_b)
                _update(standings, team_a, g_a, g_b, 0, cards_a)
                _hot_adjust(local_off, local_def, team_b, True,  False, off_b, off_a, g_b, g_a)
                _hot_adjust(local_off, local_def, team_a, False, False, off_a, off_b, g_a, g_b)
            else:
                _update(standings, team_a, g_a, g_b, 1, cards_a)
                _update(standings, team_b, g_b, g_a, 1, cards_b)
                _hot_adjust(local_off, local_def, team_a, False, True,  off_a, off_b, g_a, g_b)
                _hot_adjust(local_off, local_def, team_b, False, True,  off_b, off_a, g_b, g_a)

    return sorted(
        standings.values(),
        key=lambda r: (r.tiebreak_key(), random.random()),
        reverse=True,
    )


# ── Third-place ranking ───────────────────────────────────────────────────────

def best_third_places(thirds: list[Record], n: int = 8) -> list[Record]:
    """Return the *n* best third-place teams across all groups."""
    return sorted(thirds, key=lambda r: (r.tiebreak_key(), random.random()), reverse=True)[:n]


# ── R32 bracket construction ──────────────────────────────────────────────────

def build_r32(
    group_results: dict[str, list[Record]],
    advancing_thirds: list[Record],
) -> list[tuple[str, str]]:
    """Build 16 R32 matchups from group results and advancing third-place teams."""
    slots: dict[str, str] = {}
    for grp, records in group_results.items():
        for pos, rec in enumerate(records, start=1):
            slots[f"{grp}{pos}"] = rec.team

    thirds_ordered = [r.team for r in advancing_thirds]

    fixed_pairs = [
        ("A1", "B2"), ("C1", "D2"), ("E1", "F2"), ("G1", "H2"),
        ("I1", "J2"), ("K1", "L2"), ("A2", "C2"), ("E2", "G2"),
        ("I2", "K2"), ("B1", "D1"), ("F1", "H1"), ("J1", "L1"),
    ]
    matchups: list[tuple[str, str]] = [(slots[s_a], slots[s_b]) for s_a, s_b in fixed_pairs]

    runner_up_slots = ["B2", "D2", "F2", "H2"]
    for i, ru_slot in enumerate(runner_up_slots):
        team_a = slots[ru_slot]
        team_b = thirds_ordered[i] if i < len(thirds_ordered) else slots[ru_slot]
        matchups.append((team_a, team_b))

    return matchups


# ── Knockout rounds ───────────────────────────────────────────────────────────

def simulate_knockout_round(
    matchups: list[tuple[str, str]],
    local_off: dict[str, float],
    local_def: dict[str, float],
) -> list[str]:
    """Simulate one round of knockout matches; return list of winners."""
    winners = []
    for team_a, team_b in matchups:
        tilt_a = TEAMS[team_a].get("tilt", 0.0)
        tilt_b = TEAMS[team_b].get("tilt", 0.0)
        winner = simulate_knockout_match(
            team_a, team_b,
            local_off[team_a], local_def[team_a],
            local_off[team_b], local_def[team_b],
            tilt_a=tilt_a, tilt_b=tilt_b,
        )
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

    # Local mutable rating snapshot for hot simulation
    local_off = {t: TEAMS[t]["off"] for t in all_teams}
    local_def = {t: TEAMS[t]["def"] for t in all_teams}

    # ── Group stage ───────────────────────────────────────────────
    groups = get_groups()
    group_results: dict[str, list[Record]] = {}
    thirds: list[Record] = []

    for grp, teams in groups.items():
        standings = simulate_group(teams, local_off, local_def)
        group_results[grp] = standings
        thirds.append(standings[2])

        for rec in standings[:2]:
            results[rec.team]["r32"] = 1

    # ── Best 8 third-place teams ───────────────────────────────────
    best_thirds = best_third_places(thirds, n=8)
    for rec in best_thirds:
        results[rec.team]["r32"] = 1

    # ── Build R32 bracket ─────────────────────────────────────────
    r32_matchups = build_r32(group_results, best_thirds)

    # ── R32 ───────────────────────────────────────────────────────
    r32_winners = simulate_knockout_round(r32_matchups, local_off, local_def)
    for t in r32_winners:
        results[t]["r16"] = 1

    # ── R16 ───────────────────────────────────────────────────────
    r16_matchups = pair_winners(r32_winners)
    r16_winners = simulate_knockout_round(r16_matchups, local_off, local_def)
    for t in r16_winners:
        results[t]["qf"] = 1

    # ── QF ────────────────────────────────────────────────────────
    qf_matchups = pair_winners(r16_winners)
    qf_winners = simulate_knockout_round(qf_matchups, local_off, local_def)
    for t in qf_winners:
        results[t]["sf"] = 1

    # ── SF ────────────────────────────────────────────────────────
    sf_matchups = pair_winners(qf_winners)
    sf_winners = simulate_knockout_round(sf_matchups, local_off, local_def)
    for t in sf_winners:
        results[t]["final"] = 1

    # ── Final ─────────────────────────────────────────────────────
    final_winner = simulate_knockout_round([(sf_winners[0], sf_winners[1])],
                                            local_off, local_def)[0]
    results[final_winner]["winner"] = 1

    return results
