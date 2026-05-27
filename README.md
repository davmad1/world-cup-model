# 2026 FIFA World Cup Prediction Model

A Python recreation — and deliberate improvement — of FiveThirtyEight's Soccer Power Index (SPI) approach, targeting the 2026 FIFA World Cup (USA / Canada / Mexico, 48 teams, 12 groups).

The model is **transparent and tweakable**: every parameter lives in [`config.py`](config.py) with inline documentation. We also implement features from Nate Silver's [PELE model](https://www.natesilver.net/p/pele-international-football-rankings-soccer-ratings-projections) that 538's original SPI lacked — and add calibration and explainability tools that PELE doesn't expose at all.

---

## Quick start

```bash
git clone https://github.com/davmad1/world-cup-model.git
cd world-cup-model
pip install -r requirements.txt

# Download all data + compute Elo ratings (no manual steps ever, no API keys)
python refresh.py

# Run the full simulation (10 000 iterations by default)
python simulate.py

# Head-to-head matchup breakdown
python explain.py matchup Argentina France

# Sensitivity sweep: how does Spain's win% change as OVERDISPERSION varies?
python explain.py sensitivity --team Spain --param OVERDISPERSION --min 0.05 --max 0.55

# During the tournament: poll for live results every 15 min
python refresh.py --watch 15
```

---

## How the model works

### 1 · Team ratings (dynamic Elo)

Ratings are computed automatically from ~49 000 historical international matches via a modified Elo engine ([`elo.py`](elo.py)), then converted to off/def via logistic scaling. Run `python refresh.py` to recompute from the latest data.

Each team is assigned two numbers:

| Rating | Meaning |
|---|---|
| **off** | Expected goals scored vs. an average opponent (baseline ≈ 1.40) |
| **def** | Expected goals conceded vs. an average opponent (baseline ≈ 1.40) |

From these we derive the **Soccer Power Index**:

```
SPI = 100 × off / (off + def)
```

An average team has SPI = 50. The best 2026 teams sit in the 78–82 range.

**Elo engine features:**
- Harmonic margin of victory: `h = Σ(1/k for k=1..abs(diff))` — diminishing returns for blowouts
- Match importance weights: WC matches count 1.6×, friendlies 0.5×
- Home advantage: base Elo boost + altitude exponential (La Paz ≈ +275 pts) + travel distance penalty
- Provisional period: 2× K-factor for a team's first 100 matches

### 2 · Match prediction

For a match between team A and team B:

```
goal_scalar = 1 + (tilt_A + tilt_B) × TILT_GOAL_IMPACT   # default: 0.50
xG_A = off_A × (def_B / LEAGUE_AVG) × goal_scalar
xG_B = off_B × (def_A / LEAGUE_AVG) × goal_scalar
```

**Tactical tilt** is computed from goal residuals over a team's last 60 matches: positive tilt → team produces more total goals than the Elo formula expects; negative → fewer.

Goals are drawn from **Negative Binomial distributions** (OVERDISPERSION = 0.30), which adds more 0-goal and high-scoring outcomes than Poisson — matching empirical international football data. A **Dixon-Coles** joint-probability correction (ρ = −0.13) adjusts the likelihood of (0-0), (1-0), (0-1), and (1-1) scorelines.

Win/draw/loss probabilities are also computed analytically (summing the joint PMF over an integer grid) — used by `explain.py matchup` and `simulate.py --matchup`.

### 3 · Tournament structure

**Group stage** — 12 groups of 4, every team plays 3 matches.  
Advancement: top 2 from each group (24 teams) + 8 best third-place finishers = 32 teams in the Round of 32.

The schedule is matchday-aware (matchday 3 always last). For the final matchday, incentive modeling detects:
- *Both safe* → each team's xG reduced by 0.50 (defensive positioning)
- *Both eliminated* → each team's xG increased by 0.50 (nothing to lose)

After each simulated group match, a mini Elo update (HOT_K = 10) adjusts the within-simulation ratings so teams carry momentum into knockout rounds.

Third-place ranking: points → goal difference → goals scored → simulated fair-play cards → random tiebreak.

**Knockout rounds** — R32 → R16 → QF → SF → Final.  
Ties after 90 minutes go to **extra time** (30 min, scaled xG), then **penalty shootout** (5 kicks + sudden death, SPI-based success-rate adjustment).

### 4 · Monte Carlo simulation

The full tournament is simulated 10 000 times (configurable via `N_SIMS`). Probabilities are the fraction of simulations in which each team reaches each round.

---

## Data sources

All sources are downloaded automatically by `refresh.py` — **no manual downloads, no API keys required.**

| Source | Used for |
|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) | ~49 000 historical matches → dynamic Elo ratings |
| [ESPN scoreboard API](https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard) | Live/completed 2026 WC scores (real-time) |
| [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) | Group draw, schedule, cross-check/fallback |

---

## Explainability tools

```bash
# Full matchup breakdown: ratings, xG formula, scoreline matrix, goal distribution
python explain.py matchup Spain Brazil

# What if Mbappé is injured and France's attack drops?
python explain.py what-if France --off 1.9 --def 0.85

# Sensitivity sweep: how much does OVERDISPERSION matter for Argentina?
python explain.py sensitivity --team Argentina --param OVERDISPERSION --min 0.05 --max 0.55 --steps 6

# In-sample calibration: are predicted win probabilities well-calibrated?
python explain.py calibration --last 2000
```

Sweepable parameters: `OVERDISPERSION`, `RHO`, `TILT_GOAL_IMPACT`, `INCENTIVE_SAFE_XG_ADJ`, `INCENTIVE_ELIM_XG_ADJ`, `HOT_K`, `PEN_BASE`, `HOME_ADV_BASE`, `ALTITUDE_COEFF`, `ELO_GOALS_SCALE`.

---

## Tweaking the model

All parameters are in **[`config.py`](config.py)**. Key knobs:

```python
# How aggressively ratings update after each match
ELO_K_BASE = 40.0          # higher = faster adaptation

# Importance multipliers — make WC qualifiers count more or less
IMPORTANCE["fifa world cup qualification"] = 1.30

# Goal distribution shape
OVERDISPERSION = 0.30      # 0 = Poisson; higher = more upsets + more blowouts
RHO = -0.13                # Dixon-Coles correction strength

# Home advantage components
HOME_ADV_BASE = 75.0       # flat advantage in Elo points
ALTITUDE_COEFF = 60.0      # exponential altitude multiplier (La Paz ≈ +275 pts)

# Tilt
TILT_GOAL_IMPACT = 0.50    # how much tilt scales total expected goals
TILT_TACTICAL_SHRINK = 0.20  # shrinks raw tilt estimate toward zero (noise control)

# Simulation precision
N_SIMS = 10_000            # more sims = tighter confidence intervals
```

After changing any parameter, re-run `python simulate.py` to see the updated probabilities.

---

## File map

```
refresh.py         One-command data pipeline: download → merge → Elo → tilt → teams.py
simulate.py        Main CLI — run Monte Carlo, print results table
tournament.py      Group stage + knockout bracket logic (incentives, hot sim, fair play)
model.py           Match model: NB xG, Dixon-Coles, tilt, extra time, penalties
explain.py         Transparency layer: matchup breakdown, what-if, sensitivity, calibration
teams.py           48-team roster with off/def/tilt/group (auto-updated by refresh.py)
fetch_data.py      Pull live draw + schedule from openfootball; squad data (optional)
config.py          Every tunable parameter with documentation
elo.py             Elo engine: harmonic margin, importance weights, altitude/distance HFA
build_ratings.py   Pipeline: results.csv → Elo → tilt → patch teams.py ratings
requirements.txt   Python dependencies
data/              Gitignored — populated automatically by refresh.py
```

---

## Comparison to PELE (Nate Silver's model)

| Feature | PELE | This model |
|---|---|---|
| Match distribution | Negative binomial | ✅ Negative binomial + Dixon-Coles |
| Team ratings | Dynamic Elo + player values | ✅ Dynamic Elo (from 49k matches) |
| Tilt ratings | ✅ | ✅ Tactical tilt from goal residuals |
| Home advantage | Altitude + distance + per-team coefficient | ✅ Altitude + distance |
| Match importance weighting | ✅ | ✅ |
| Group incentive modeling | ✅ | ✅ Final-matchday xG adjustment |
| Within-tournament form | ✅ | ✅ Hot-simulation Elo nudges |
| Age trajectory | ✅ | — (planned; no free automated data source) |
| Player market values | ✅ | — (planned; no free automated data source) |
| Parameter transparency | ❌ proprietary | ✅ Everything in config.py |
| Sensitivity analysis | ❌ | ✅ `explain.py sensitivity` |
| What-if scenarios | ❌ | ✅ `explain.py what-if` |
| Calibration reports | ❌ not published | ✅ `explain.py calibration` |
| Open source | ❌ | ✅ |

---

## Example output

```
2026 FIFA World Cup — SPI Prediction Model (n=10,000)
============================================================
Model: NB xG + Dixon-Coles + tilt + incentives + ET + penalties
Groups sourced from openfootball/worldcup.json (official draw).

Team                  Grp      SPI  Win%    Final%    SF%    QF%    R16%    Adv%
--------------------  -----  -----  ------  --------  -----  -----  ------  ------
Spain                 H       81.6  15.4%   23.2%     34.0%  47.8%  68.0%   94.3%
Argentina             J       79.9   7.2%   11.9%     19.3%  30.6%  51.4%   89.1%
France                I       78.8   6.9%   11.8%     19.4%  32.0%  50.7%   86.0%
England               L       77.3   5.1%    8.8%     14.8%  25.1%  45.5%   87.2%
Netherlands           F       74.5   4.7%    8.4%     16.1%  26.9%  45.6%   81.5%
Brazil                C       75.7   4.3%    8.7%     15.8%  29.4%  47.4%   84.2%
...
Qatar                 B       52.3   0.0%    0.1%      0.8%   3.4%  11.9%   38.8%
```

---

## Contributing

Pull requests welcome. If you update `config.py` parameters, please include the before/after simulation output so the change can be evaluated.
