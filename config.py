"""
config.py — Single source of truth for every tunable model parameter.

This is our key advantage over PELE: every knob is here, documented,
and easy to experiment with. Change a value, re-run simulate.py, see
the effect immediately. Use explain.py --sensitivity to sweep a range.

All parameters are grouped by subsystem with inline commentary.
"""

import os

# ── Elo engine ────────────────────────────────────────────────────────────────

# Base K-factor — how much each match moves ratings.
# Higher = faster adaptation to recent form, lower = more stable long-term.
# 538's club SPI used 40; PELE is slightly higher for WC weight.
ELO_K_BASE: float = 40.0

# Number of matches before a team leaves the "provisional" period.
# During provisional, K is doubled to allow fast convergence from the 1500 prior.
PROVISIONAL_MATCHES: int = 100

# Time-decay half-life for Elo K-factor (Rue & Salvesen 2000).
# A match played N days ago has its K scaled by exp(-ln(2) * N / HALFLIFE).
# 0 = disabled (no decay). Any positive integer = halflife in days.
#
# Calibrated via `python calibrate.py --decay` across 2006-2022 WC group stages:
#   halflife=   0d → RPS 0.20909  ← optimal (no decay)
#   halflife= 730d → RPS 0.22718
#   halflife=1460d → RPS 0.22335
#   halflife=2920d → RPS 0.21880
#   halflife=3650d → RPS 0.21740
#   halflife=∞     → RPS 0.20943
#
# Conclusion: chronological Elo accumulation already handles recency adequately
# for international football (far fewer matches per year than club football).
# Explicit decay hurts at all tested halflives. Set > 0 only to experiment.
ELO_DECAY_HALFLIFE_DAYS: int = 0

# ── Home advantage components ─────────────────────────────────────────────────

# Base Elo-point advantage for the home team on flat ground at sea level.
# 75 → home team wins ~60% when teams are otherwise equal (empirical average).
HOME_ADV_BASE: float = 75.0

# Altitude (metres) below which the home bonus is negligible.
# Above this, a nonlinear exponential kicker is applied.
ALTITUDE_KNEE: float = 1_500.0

# Altitude boost coefficient. Scales the quadratic term above ALTITUDE_KNEE.
# At COEFF=60 the actual values produced are:
#   Mexico City (2 240 m) → ~33 pts,  Bogotá (2 600 m) → ~73 pts
#   Quito       (2 850 m) → ~109 pts, La Paz (3 640 m) → ~275 pts
# Lower to ~38 for a softer altitude effect (La Paz → ~174 pts).
ALTITUDE_COEFF: float = 60.0

# Distance-based HFA: each 1 000 km the away team must travel adds this many
# extra Elo points to the home advantage (capped at DISTANCE_CAP).
DISTANCE_PER_1000KM: float = 10.0
DISTANCE_CAP: float = 60.0

# ── Match importance multipliers ──────────────────────────────────────────────
# Applied to K when updating ratings.
#
# Calibration note (python calibrate.py --weights):
#   The unconstrained optimizer always finds a corner solution: WC weight → max,
#   everything else → min. This is a biased result — the 5-WC backtest sample
#   is dominated by ~15 historically dominant teams (Spain, Germany, Argentina…)
#   that appear in every WC AND every continental tournament. The optimizer can't
#   cleanly separate "WC history is informative" from "strong teams stay strong
#   regardless of match type." Applying the extreme weights (wc=4.5, others=0.1)
#   would also hurt 2026 prediction: many of the 48 qualifiers have limited WC
#   history, and their best signal comes from continental / qualifying form.
#
# Approach: use domain-knowledge weights that reflect the user's semantic bands,
#   nudged modestly toward the optimizer's direction (WC slightly up, qualifying
#   slightly down). Re-run calibrate.py --weights after adding more WC tournaments
#   to the backtest set for a less biased estimate.
#
# Semantic bands (calibrate.py --weights):
#   Consequential  (WC, Euros, Copa…):  high    — championship-level opposition
#   Competitive qualifying:             medium  — real opposition but noisy path
#   Minor / low-stakes:                 low
#   Friendly:                           low     — preparation, but uninformative
IMPORTANCE: dict[str, float] = {
    "friendly":                        0.50,
    "nations league":                  0.75,
    "nations league qualification":    0.65,
    "confederation cup":               1.10,
    "confederations cup":              1.10,
    "olympic":                         0.65,
    "olympics":                        0.65,
    # Regional qualifying
    "qualification":                   0.90,   # default qualifier catch-all
    "uefa euro qualification":         1.00,
    "fifa world cup qualification":    1.10,
    # Continental tournaments
    "african cup of nations":          1.10,
    "afc asian cup":                   1.10,
    "concacaf gold cup":               1.00,
    "copa america":                    1.20,
    "copa américa":                    1.20,
    "uefa euro":                       1.30,
    "uefa european":                   1.30,
    # World Cup — nudged up from 1.60; optimizer consistently pushes this higher
    "fifa world cup":                  2.00,
}
# Fallback when no keyword matches
IMPORTANCE_DEFAULT: float = 0.75

# ── Elo-to-goals conversion ───────────────────────────────────────────────────

# Global average goals per team per match (used as the baseline in xG formulas).
LEAGUE_AVG: float = 1.40

# Scale factor controlling how steeply Elo differences translate into
# goal-scoring gaps. Lower = more upsets; higher = more deterministic.
# 0.40 means a 400-Elo gap → e^0.40 ≈ 1.49× more goals scored than allowed.
# Lowered from 0.50 after Phase 1 review showed too-wide SPI spread.
ELO_GOALS_SCALE: float = 0.40

# Average Elo around which the off/def conversion is centred.
# 1 500 = the initialisation prior for every team.
ELO_AVG: float = 1_500.0

# ── Match distribution ────────────────────────────────────────────────────────

# Dixon-Coles joint-probability correction for low-scoring outcomes.
# Negative = fewer 0-0 draws than pure Poisson; −0.13 fits international data.
RHO: float = -0.13

# Negative-binomial overdispersion parameter (0 = Poisson; higher = more
# variance → more blowouts and more scoreless draws).
# Marginal estimate from data: 0.54 (inflated by cross-team variation).
# Conditional estimate for international football: ~0.30.
# Calibrate via:  python build_ratings.py --calibrate
OVERDISPERSION: float = 0.30

# ── Player market values ──────────────────────────────────────────────────────

# Transfermarkt systematically over-values players at UEFA clubs relative to
# actual performance (PELE found ~30 %). Applied as a haircut to their listed
# values before summing squad strength.
UEFA_DISCOUNT: float = 0.30

# Weight of starter vs. bench contribution to squad value.
# Players ranked 1-11 by value get STARTER_WEIGHT; players 12-23 get a
# linearly decreasing share down to BENCH_MIN_WEIGHT at slot 23.
STARTER_WEIGHT: float = 1.00
BENCH_MIN_WEIGHT: float = 0.10

# Squad value mean-reversion blend with Elo (0 = pure Elo, 1 = pure squad value).
# Phase 2 will tune this; 0.35 matches PELE's empirical ~25-point mean reversion.
ELO_SQUAD_BLEND: float = 0.35

# ── Age trajectory ────────────────────────────────────────────────────────────

# Peak age for international players (market-value weighted).
AGE_PEAK: float = 26.5

# Annual Elo-point change per year away from peak.
# A squad with weighted-avg age of 30 (3.5 years past peak) loses 14 points.
AGE_ELO_PER_YEAR: float = 4.0

# ── Tilt ratings ─────────────────────────────────────────────────────────────

# Tactical tilt (residual goals vs. expectation) is very noisy in soccer.
# Shrink it toward zero by this factor before adding to personnel tilt.
TILT_TACTICAL_SHRINK: float = 0.20

# How much a team's combined tilt scales the total expected goals in a match.
# tilt_scalar = 1 + (tilt_a + tilt_b) × TILT_GOAL_IMPACT
# Tilt values range roughly −0.15 to +0.15; impact of 0.5 → up to ±15 % total goals.
TILT_GOAL_IMPACT: float = 0.50

# ── Penalty shootout ──────────────────────────────────────────────────────────

# Base success rate per kick (empirical international average ~74.5 %).
PEN_BASE: float = 0.745

# Maximum SPI-based adjustment (±pp). Better team kicks slightly more reliably.
PEN_MAX_ADJ: float = 0.05

# ── Simulation ───────────────────────────────────────────────────────────────

# Default number of Monte Carlo tournament simulations.
# 10 000 runs in ~6 s; 50 000 in ~30 s; 100 000 in ~60 s.
N_SIMS: int = 10_000

# Random seed (None = different results every run; set an int for reproducibility).
RANDOM_SEED = None

# ── Within-tournament "hot" simulation ───────────────────────────────────────

# Mini K-factor applied after each simulated group-stage match within a single
# tournament run. Captures momentum / form.  Set to 0 to disable.
HOT_K: float = 10.0

# ── Group-stage incentive modelling ──────────────────────────────────────────

# xG adjustment (per team) on the final group matchday when both teams are
# already safe (mutual-advance scenario) or both already eliminated.
INCENTIVE_SAFE_XG_ADJ: float  = -0.50   # both safe   → fewer goals
INCENTIVE_ELIM_XG_ADJ: float  =  0.50   # both out    → more goals

# ── Fair play tiebreaker ──────────────────────────────────────────────────────

# Expected yellow cards per team per match (negative binomial).
CARDS_MEAN: float = 1.8
CARDS_OVERDISPERSION: float = 0.8

# Better teams (higher SPI) commit fewer fouls. Elo-point reduction per card.
CARDS_ELO_SLOPE: float = 0.10

# ── Bookmaker odds ────────────────────────────────────────────────────────────

# The Odds API key (https://the-odds-api.com — free tier, 500 req/month).
# Set the ODDS_API_KEY environment variable before running refresh.py.
# If empty, the odds fetch step is silently skipped.
ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
