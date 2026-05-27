# 2026 FIFA World Cup Prediction Model

A Python recreation — and planned improvement — of FiveThirtyEight's Soccer Power Index (SPI) approach, targeting the 2026 FIFA World Cup (USA / Canada / Mexico, 48 teams, 12 groups).

The model is deliberately **transparent and tweakable**: every parameter lives in [`config.py`](config.py) with inline documentation. We also implement features from Nate Silver's [PELE model](https://www.natesilver.net/p/pele-international-football-rankings-soccer-ratings-projections) that 538's original SPI lacked — and add our own calibration and explainability tools that PELE doesn't expose.

---

## Quick start

```bash
git clone https://github.com/davmad1/world-cup-model.git
cd world-cup-model
pip install -r requirements.txt

# Download all data + compute Elo ratings (no manual steps ever)
python refresh.py

# Run the full simulation (10 000 iterations by default)
python simulate.py

# Head-to-head matchup probability
python simulate.py --matchup Argentina France

# Per-group advance probabilities
python simulate.py --group-probs

# During the tournament: poll for live results every 15 min
python refresh.py --watch 15
```

---

## How the model works

### 1 · Team ratings

Each team is assigned two numbers:

| Rating | Meaning |
|---|---|
| **off** | Expected goals scored vs. an average opponent (avg ≈ 1.40) |
| **def** | Expected goals allowed vs. an average opponent (avg ≈ 1.40) |

From these we derive the **Soccer Power Index**:

```
SPI = 100 × off / (off + def)
```

An average team has SPI = 50. The best 2026 teams sit in the 70–78 range.

Ratings are computed automatically from ~49 000 historical international matches via a modified Elo engine (`elo.py`), then converted to off/def via a logistic scaling. Run `python refresh.py` to recompute from the latest data.

### 2 · Match prediction

For a match between team A and team B at a neutral venue:

```
xG_A = off_A × (def_B / LEAGUE_AVG)
xG_B = off_B × (def_A / LEAGUE_AVG)
```

Goals are drawn from **Poisson distributions** with parameters xG_A and xG_B (currently), with a **Dixon-Coles** joint-probability correction (ρ = −0.13) that adjusts the likelihood of low-scoring scorelines (0-0, 1-0, 0-1, 1-1). Phase 3 upgrades the distribution to **Negative Binomial** for better overdispersion handling.

Win/draw/loss probabilities are also computed analytically (no simulation required) by summing the joint PMF over an integer grid — useful for the `--matchup` flag.

### 3 · Tournament structure

**Group stage** — 12 groups of 4, every team plays 3 matches.  
Advancement: top 2 from each group (24 teams) + 8 best third-place finishers = 32 teams in the Round of 32.

Third-place ranking: points → goal difference → goals scored → random tiebreak.

**Knockout rounds** — R32 → R16 → QF → SF → Final.  
Ties after 90 minutes go to **extra time** (30 min, scaled xG), then **penalty shootout** (5 kicks + sudden death, with a small SPI-based success-rate adjustment).

### 4 · Monte Carlo simulation

The full tournament is simulated 10 000 times (configurable in `config.py` via `N_SIMS`). Probabilities are the fraction of simulations in which each team reaches each round.

---

## Data sources

| Source | Used for | Auth |
|---|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) | ~49 000 historical matches → dynamic Elo ratings | None |
| [ESPN scoreboard API](https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard) | Live/completed 2026 WC scores (real-time) | None |
| [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) | Group draw, schedule, cross-check backup | None |
| [football-data.org v4 API](https://www.football-data.org) | Squad rosters, live standings | Free API key |

All three primary sources are downloaded automatically by `refresh.py` — no manual steps required.

---

## Tweaking the model

All parameters are in **[`config.py`](config.py)**. Key knobs:

```python
# How aggressively ratings update after each match
ELO_K_BASE = 40.0          # higher = faster adaptation

# Importance multipliers — make WC qualifiers count more or less
IMPORTANCE["fifa world cup qualification"] = 1.30

# Goal distribution shape
OVERDISPERSION = 0.12      # 0 = Poisson; higher = more upsets + more blowouts
RHO = -0.13                # Dixon-Coles correction strength

# Home advantage components
HOME_ADV_BASE = 75.0       # flat advantage in Elo points
ALTITUDE_COEFF = 60.0      # how much altitude matters

# Simulation precision
N_SIMS = 50_000            # more sims = tighter confidence intervals
```

After changing any parameter, re-run `python simulate.py` to see the updated probabilities.

---

## File map

```
refresh.py         One-command data pipeline: download → merge → Elo → teams.py
simulate.py        Main CLI — run Monte Carlo, print results table
tournament.py      Group stage + knockout bracket logic
model.py           Match simulation: xG formula, Poisson/NegBin, Dixon-Coles, pens
teams.py           48-team roster with off/def/group (auto-updated by build_ratings.py)
fetch_data.py      Pull live draw + schedule from openfootball; squad data from football-data.org
config.py          Every tunable parameter with documentation
elo.py             Elo engine: harmonic margin, importance weights, altitude/distance HFA
build_ratings.py   Pipeline: results.csv → Elo → patch teams.py ratings
requirements.txt   Python dependencies
data/              Gitignored — populated automatically by refresh.py
```

---

## Roadmap

| Phase | Feature | Status |
|---|---|---|
| Baseline | Poisson xG model, Dixon-Coles, group + knockout sim, Monte Carlo | ✅ Done |
| Data | openfootball live feed, football-data.org squad API | ✅ Done |
| Phase 1 | Dynamic Elo from 45k historical matches (harmonic margin, importance weights, altitude/HFA) | 🔧 In progress |
| Phase 2 | Transfermarkt player values, tilt ratings, age trajectory | 📋 Planned |
| Phase 3 | Negative binomial distribution, data-calibrated overdispersion | 📋 Planned |
| Phase 4 | Final-matchday incentive modeling, within-tournament form, fair play tiebreaker | 📋 Planned |
| Phase 5 | `explain.py` — sensitivity sweeps, what-if scenarios, calibration report | 📋 Planned |

---

## Comparison to PELE (Nate Silver's model)

| Feature | PELE | This model |
|---|---|---|
| Match distribution | Negative binomial | Poisson + Dixon-Coles (NB in Phase 3) |
| Team ratings | Dynamic Elo + player values | Dynamic Elo (Phase 1) + player values (Phase 2) |
| Tilt ratings | ✅ | Phase 2 |
| Home advantage | Altitude + distance + per-team coefficient | Altitude + distance (Phase 1) |
| Age trajectory | ✅ | Phase 2 |
| Match importance weighting | ✅ | Phase 1 |
| Group incentive modeling | ✅ | Phase 4 |
| Parameter transparency | ❌ proprietary | ✅ everything in config.py |
| Sensitivity analysis | ❌ | Phase 5 |
| Calibration reports | ❌ not published | Phase 5 |
| Open source | ❌ | ✅ |

---

## Example output

```
2026 FIFA World Cup — SPI Prediction Model (n=10,000)
============================================================

Team                  Grp    SPI   Win%   Final%    SF%    QF%   R16%   Adv%
--------------------  ----  ----  -----  -------  -----  -----  -----  -----
Argentina             J     77.6  16.5%   24.5%   34.0%  46.3%  66.8%  96.6%
Spain                 H     73.4  12.1%   19.6%   30.3%  42.6%  67.1%  93.5%
France                I     74.6  11.3%   18.4%   28.3%  43.1%  63.1%  93.0%
Brazil                C     72.4   8.9%   16.1%   26.9%  44.6%  63.2%  92.4%
Germany               E     71.4   7.7%   13.7%   23.6%  39.2%  60.9%  93.5%
...
```

---

## Contributing

Pull requests welcome. If you update `config.py` parameters, please include the before/after simulation output so the change can be evaluated.
