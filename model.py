"""
Core match-prediction model — a close recreation of 538's Soccer Power Index approach.

Match model
-----------
Expected goals are computed multiplicatively:

    xG_A = off_A × (def_B / LEAGUE_AVG)
    xG_B = off_B × (def_A / LEAGUE_AVG)

Goals are then drawn from independent Poisson distributions (xG_A) and (xG_B).
This is the standard Dixon-Robinson formulation; we also offer the Dixon-Coles
low-score correction (ρ ≈ −0.13) which adjusts joint probabilities for
scorelines (0-0), (1-0), (0-1), and (1-1).

Penalty shootouts
-----------------
Each team's per-kick success rate is calibrated to the empirical international
average (~75 %) with a small SPI-based adjustment (±5 pp max). The shootout
runs 5 rounds then sudden death.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson as sp_poisson

from teams import LEAGUE_AVG, spi

RNG = np.random.default_rng()

# Dixon-Coles inflation parameter (negative → draws/0-0 slightly suppressed)
RHO = -0.13

# Penalty base success rate and maximum SPI-based adjustment
PEN_BASE = 0.745
PEN_MAX_ADJ = 0.05


# ── Expected goals ────────────────────────────────────────────────────────────

def expected_goals(
    off_a: float,
    def_a: float,
    off_b: float,
    def_b: float,
    home_advantage: float = 0.0,
    avg: float = LEAGUE_AVG,
) -> tuple[float, float]:
    """Return (xg_a, xg_b) for a match between two teams."""
    xg_a = off_a * (def_b / avg) + home_advantage
    xg_b = off_b * (def_a / avg)
    return max(xg_a, 0.05), max(xg_b, 0.05)


# ── Dixon-Coles correction ────────────────────────────────────────────────────

def _dc_tau(x: int, y: int, lam: float, mu: float, rho: float = RHO) -> float:
    """Joint-probability correction factor for low-scoring cells."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


# ── Analytical win probabilities ──────────────────────────────────────────────

def win_probability(
    off_a: float,
    def_a: float,
    off_b: float,
    def_b: float,
    home_advantage: float = 0.0,
    dixon_coles: bool = True,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    """
    Return (p_win, p_draw, p_loss) for team A vs team B analytically.

    Sums over the Poisson joint PMF up to *max_goals* goals per side.
    With dixon_coles=True the low-score correction is applied.
    """
    xg_a, xg_b = expected_goals(off_a, def_a, off_b, def_b, home_advantage)
    p_win = p_draw = p_loss = 0.0

    for g_a in range(max_goals + 1):
        p_a = sp_poisson.pmf(g_a, xg_a)
        for g_b in range(max_goals + 1):
            p_b = sp_poisson.pmf(g_b, xg_b)
            tau = _dc_tau(g_a, g_b, xg_a, xg_b) if dixon_coles else 1.0
            p = p_a * p_b * tau
            if g_a > g_b:
                p_win += p
            elif g_a == g_b:
                p_draw += p
            else:
                p_loss += p

    # Normalise rounding error
    total = p_win + p_draw + p_loss
    return p_win / total, p_draw / total, p_loss / total


# ── Match simulation ──────────────────────────────────────────────────────────

def simulate_match(
    off_a: float,
    def_a: float,
    off_b: float,
    def_b: float,
    home_advantage: float = 0.0,
    dixon_coles: bool = True,
) -> tuple[int, int]:
    """
    Simulate a single 90-minute match and return (goals_a, goals_b).

    When dixon_coles=True the low-score outcomes are adjusted via rejection
    sampling using the tau correction. In practice the adjustment is tiny and
    a direct Poisson draw is used as the fast path.
    """
    xg_a, xg_b = expected_goals(off_a, def_a, off_b, def_b, home_advantage)

    if dixon_coles:
        # Rejection sampling (rarely needs more than 1 iteration)
        while True:
            g_a = int(RNG.poisson(xg_a))
            g_b = int(RNG.poisson(xg_b))
            tau = _dc_tau(g_a, g_b, xg_a, xg_b)
            threshold = max(
                1 - abs(xg_a * xg_b * RHO),
                1 - abs(xg_a * RHO),
                1 - abs(xg_b * RHO),
                1 - abs(RHO),
            )
            u = RNG.uniform(0, 1 / threshold)
            if u <= tau:
                return g_a, g_b
    else:
        return int(RNG.poisson(xg_a)), int(RNG.poisson(xg_b))


def simulate_extra_time(
    off_a: float,
    def_a: float,
    off_b: float,
    def_b: float,
) -> tuple[int, int]:
    """Simulate 30 minutes of extra time (scaled from a 90-min match)."""
    scale = 30.0 / 90.0
    xg_a, xg_b = expected_goals(off_a, def_a, off_b, def_b)
    return int(RNG.poisson(xg_a * scale)), int(RNG.poisson(xg_b * scale))


# ── Penalty shootout ──────────────────────────────────────────────────────────

def simulate_penalty_shootout(team_a: str, team_b: str) -> bool:
    """
    Simulate a penalty shootout. Returns True if team A wins.

    Per-kick success rates are calibrated to ~74.5 % international average
    with a small SPI-based adjustment (max ±5 pp).
    """
    spi_a, spi_b = spi(team_a), spi(team_b)
    spi_diff = (spi_a - spi_b) / 100.0  # roughly in [-0.5, 0.5]
    adj = np.clip(spi_diff * PEN_MAX_ADJ / 0.3, -PEN_MAX_ADJ, PEN_MAX_ADJ)
    rate_a = np.clip(PEN_BASE + adj, 0.60, 0.88)
    rate_b = np.clip(PEN_BASE - adj, 0.60, 0.88)

    # First five rounds
    goals_a = int(np.sum(RNG.random(5) < rate_a))
    goals_b = int(np.sum(RNG.random(5) < rate_b))

    if goals_a != goals_b:
        return goals_a > goals_b

    # Sudden death (max 20 extra rounds — astronomically enough)
    for _ in range(20):
        a = RNG.random() < rate_a
        b = RNG.random() < rate_b
        if a != b:
            return bool(a)

    return bool(RNG.random() < 0.5)


# ── Knockout match (90 min + ET + pens) ──────────────────────────────────────

def simulate_knockout_match(
    team_a: str,
    team_b: str,
    off_a: float,
    def_a: float,
    off_b: float,
    def_b: float,
) -> str:
    """Simulate a knockout match; return the name of the winner."""
    g_a, g_b = simulate_match(off_a, def_a, off_b, def_b)

    if g_a != g_b:
        return team_a if g_a > g_b else team_b

    # Extra time
    et_a, et_b = simulate_extra_time(off_a, def_a, off_b, def_b)
    total_a, total_b = g_a + et_a, g_b + et_b

    if total_a != total_b:
        return team_a if total_a > total_b else team_b

    return team_a if simulate_penalty_shootout(team_a, team_b) else team_b
