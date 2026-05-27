"""
Core match-prediction model — a close recreation of 538's Soccer Power Index approach.

Match model
-----------
Expected goals are computed multiplicatively:

    xG_A = off_A × (def_B / LEAGUE_AVG) × goal_scalar
    xG_B = off_B × (def_A / LEAGUE_AVG) × goal_scalar

    goal_scalar = 1 + (tilt_A + tilt_B) × TILT_GOAL_IMPACT

Goals are drawn from independent Negative Binomial distributions which
add overdispersion relative to Poisson (more 0-goal and high-scoring outcomes).
When OVERDISPERSION ≈ 0 the distribution converges to Poisson.

The Dixon-Coles low-score correction (ρ ≈ −0.13) adjusts joint probabilities
for scorelines (0-0), (1-0), (0-1), and (1-1).

Penalty shootouts
-----------------
Each team's per-kick success rate is calibrated to the empirical international
average (~74.5 %) with a small SPI-based adjustment (±5 pp max).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson as sp_poisson
from scipy.stats import nbinom as sp_nbinom

import config
from teams import spi

LEAGUE_AVG = config.LEAGUE_AVG

RNG = np.random.default_rng(config.RANDOM_SEED)

RHO         = config.RHO
PEN_BASE    = config.PEN_BASE
PEN_MAX_ADJ = config.PEN_MAX_ADJ


# ── Expected goals ────────────────────────────────────────────────────────────

def expected_goals(
    off_a: float,
    def_a: float,
    off_b: float,
    def_b: float,
    home_advantage: float = 0.0,
    avg: float = LEAGUE_AVG,
    tilt_a: float = 0.0,
    tilt_b: float = 0.0,
) -> tuple[float, float]:
    """
    Return (xg_a, xg_b) for a match between two teams.

    Tilt captures a team's tendency to produce more/fewer total goals than
    the xG formula predicts. Positive = more open style; negative = defensive.
    The combined tilt of both teams scales total expected goals.
    """
    goal_scalar = 1.0 + (tilt_a + tilt_b) * config.TILT_GOAL_IMPACT
    goal_scalar = max(0.5, min(1.5, goal_scalar))
    xg_a = (off_a * (def_b / avg) + home_advantage) * goal_scalar
    xg_b = (off_b * (def_a / avg)) * goal_scalar
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


# ── NB helpers ────────────────────────────────────────────────────────────────

def _nb_params(xg: float) -> tuple[float, float]:
    """NB (n, p) for scipy/numpy given mean xg and config OVERDISPERSION."""
    n = 1.0 / config.OVERDISPERSION
    p = n / (n + xg)
    return n, p


def _draw_goals(xg: float) -> int:
    """Sample goals from NB (falls back to Poisson when OVERDISPERSION≈0)."""
    if config.OVERDISPERSION < 1e-6:
        return int(RNG.poisson(xg))
    n, p = _nb_params(xg)
    return int(RNG.negative_binomial(n, p))


# ── Analytical win probabilities ──────────────────────────────────────────────

def win_probability(
    off_a: float,
    def_a: float,
    off_b: float,
    def_b: float,
    home_advantage: float = 0.0,
    dixon_coles: bool = True,
    max_goals: int = 10,
    tilt_a: float = 0.0,
    tilt_b: float = 0.0,
) -> tuple[float, float, float]:
    """
    Return (p_win, p_draw, p_loss) for team A vs B analytically.

    Sums over the NB joint PMF up to *max_goals* goals per side.
    With dixon_coles=True the low-score correction is applied.
    """
    xg_a, xg_b = expected_goals(off_a, def_a, off_b, def_b, home_advantage,
                                 tilt_a=tilt_a, tilt_b=tilt_b)
    p_win = p_draw = p_loss = 0.0

    use_nb = config.OVERDISPERSION >= 1e-6
    if use_nb:
        n_a, p_a = _nb_params(xg_a)
        n_b, p_b = _nb_params(xg_b)

    for g_a in range(max_goals + 1):
        pmf_a = sp_nbinom.pmf(g_a, n_a, p_a) if use_nb else sp_poisson.pmf(g_a, xg_a)
        for g_b in range(max_goals + 1):
            pmf_b = sp_nbinom.pmf(g_b, n_b, p_b) if use_nb else sp_poisson.pmf(g_b, xg_b)
            tau = _dc_tau(g_a, g_b, xg_a, xg_b) if dixon_coles else 1.0
            p = pmf_a * pmf_b * tau
            if g_a > g_b:
                p_win += p
            elif g_a == g_b:
                p_draw += p
            else:
                p_loss += p

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
    tilt_a: float = 0.0,
    tilt_b: float = 0.0,
) -> tuple[int, int]:
    """
    Simulate a single 90-minute match; return (goals_a, goals_b).

    Uses Negative Binomial goal draws. When dixon_coles=True the low-score
    outcomes are adjusted via rejection sampling (rarely iterates more than once).
    """
    xg_a, xg_b = expected_goals(off_a, def_a, off_b, def_b, home_advantage,
                                 tilt_a=tilt_a, tilt_b=tilt_b)

    if dixon_coles:
        dc_threshold = max(
            1 - abs(xg_a * xg_b * RHO),
            1 - abs(xg_a * RHO),
            1 - abs(xg_b * RHO),
            1 - abs(RHO),
        )
        while True:
            g_a = _draw_goals(xg_a)
            g_b = _draw_goals(xg_b)
            tau = _dc_tau(g_a, g_b, xg_a, xg_b)
            if RNG.uniform(0, 1 / dc_threshold) <= tau:
                return g_a, g_b
    else:
        return _draw_goals(xg_a), _draw_goals(xg_b)


def simulate_extra_time(
    off_a: float,
    def_a: float,
    off_b: float,
    def_b: float,
    tilt_a: float = 0.0,
    tilt_b: float = 0.0,
) -> tuple[int, int]:
    """Simulate 30 minutes of extra time (scaled from a 90-min match)."""
    scale = 30.0 / 90.0
    xg_a, xg_b = expected_goals(off_a, def_a, off_b, def_b,
                                 tilt_a=tilt_a, tilt_b=tilt_b)
    return _draw_goals(xg_a * scale), _draw_goals(xg_b * scale)


# ── Penalty shootout ──────────────────────────────────────────────────────────

def simulate_penalty_shootout(team_a: str, team_b: str) -> bool:
    """Simulate a penalty shootout. Returns True if team A wins."""
    spi_a, spi_b = spi(team_a), spi(team_b)
    spi_diff = (spi_a - spi_b) / 100.0
    adj = np.clip(spi_diff * PEN_MAX_ADJ / 0.3, -PEN_MAX_ADJ, PEN_MAX_ADJ)
    rate_a = np.clip(PEN_BASE + adj, 0.60, 0.88)
    rate_b = np.clip(PEN_BASE - adj, 0.60, 0.88)

    goals_a = int(np.sum(RNG.random(5) < rate_a))
    goals_b = int(np.sum(RNG.random(5) < rate_b))
    if goals_a != goals_b:
        return goals_a > goals_b

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
    tilt_a: float = 0.0,
    tilt_b: float = 0.0,
) -> str:
    """Simulate a knockout match; return the name of the winner."""
    g_a, g_b = simulate_match(off_a, def_a, off_b, def_b,
                               tilt_a=tilt_a, tilt_b=tilt_b)
    if g_a != g_b:
        return team_a if g_a > g_b else team_b

    et_a, et_b = simulate_extra_time(off_a, def_a, off_b, def_b,
                                      tilt_a=tilt_a, tilt_b=tilt_b)
    total_a, total_b = g_a + et_a, g_b + et_b
    if total_a != total_b:
        return team_a if total_a > total_b else team_b

    return team_a if simulate_penalty_shootout(team_a, team_b) else team_b
