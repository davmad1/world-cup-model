# 2026 FIFA World Cup Prediction Model

A Monte Carlo prediction engine for the 2026 FIFA World Cup (USA / Canada / Mexico, 48 teams, 104 matches). Combining methodology from published academic literature and practitioner models.

> **Current top-5 (n=50,000 simulations, updated May 2026)**
> Spain 15.3% · Argentina 7.7% · France 6.8% · England 4.9% · Brazil 4.6%

---

## What makes this different

Most public WC models are either black boxes (538's SPI, Opta's supercomputer) or direct ports of a single methodology. This model:

- **Draws from multiple academic sources**: Elo engine following Hvattum & Arntzen (2010), match distribution from Dixon & Coles (1997), overdispersion from Rue & Salvesen (2000), tilt concept from PELE (Silver, 2022)
- **Empirically calibrates its own parameters** — `calibrate.py` runs walk-forward backtests over five historical World Cups (2006–2022) to tune importance weights and validate decay assumptions; the calibration findings are documented and reproducible
- **Exposes every knob** — all parameters live in `config.py` with inline rationale and calibration evidence; sensitivity sweeps and what-if scenarios run in one command
- **Updates automatically** — `refresh.py` downloads ~49,000 historical results, ingests live WC scores from ESPN within minutes of full-time, and rebuilds ratings without any manual steps or API keys

---

## Current predictions (May 2026, n=50,000)

Sorted by Win%. CI = 95% binomial confidence interval on the winner probability.

```
Team                  Grp    SPI   Win%±CI     Final%   SF%    QF%    R16%   Adv%
--------------------  -----  ----  ----------  -------  -----  -----  -----  -----
Spain                 H      81.6  15.3±0.3%   22.9%    33.4%  47.2%  68.1%  93.8%
Argentina             J      79.9   7.7±0.2%   12.6%    20.3%  31.3%  52.2%  89.5%
France                I      78.8   6.8±0.2%   11.6%    19.3%  31.7%  50.7%  85.5%
England               L      77.3   4.9±0.2%    8.8%    15.2%  25.4%  46.3%  87.0%
Brazil                C      75.7   4.6±0.2%    8.9%    16.5%  30.0%  48.1%  84.5%
Netherlands           F      74.5   4.4±0.2%    8.4%    15.9%  26.8%  45.9%  81.8%
Ecuador               E      74.4   3.7±0.2%    7.6%    14.3%  26.4%  46.3%  83.9%
Colombia              K      75.4   3.7±0.2%    6.8%    12.5%  23.3%  41.7%  80.5%
Switzerland           B      72.7   3.6±0.2%    7.2%    14.5%  28.8%  51.6%  86.6%
Japan                 F      73.1   3.5±0.2%    7.0%    14.2%  25.1%  44.0%  79.9%
Germany               E      73.8   3.4±0.2%    6.9%    13.3%  24.7%  44.3%  83.1%
Uruguay               H      72.2   3.3±0.2%    6.9%    14.8%  27.1%  47.8%  78.4%
Turkey                D      72.8   3.2±0.2%    6.6%    13.1%  25.9%  44.1%  74.6%
Morocco               C      73.1   2.9±0.1%    6.3%    12.2%  24.3%  42.1%  81.1%
Portugal              K      74.3   2.8±0.1%    5.6%    10.9%  21.1%  38.7%  78.5%
...
USA                   D      64.8   0.6±0.1%    1.7%     4.7%  11.8%  24.9%  55.2%
Qatar                 B      52.3   0.1±0.0%    0.2%     0.9%   3.5%  12.1%  38.7%
```

*SPI = Soccer Power Index (0–100). Adv% = probability of advancing from group stage.*

A few things worth noting in these numbers:
- **Ecuador (#7, SPI 74.4)** has been quietly one of the strongest CONMEBOL teams of this era; the model rates them above Germany and Portugal
- **Switzerland (3.6%)** is the model's biggest "stealth contender" — in a softer bracket section, they advance at 86.6%
- **USA (0.6%)** as hosts is striking; the model sees a meaningful gap between their current squad quality and a legitimate deep run
- **Morocco (2.9%)** in Group C with Brazil is a tough draw; model gives them 81.1% to advance from a group where they're not the favourite

---

## Quick start

```bash
git clone https://github.com/davmad1/world-cup-model.git
cd world-cup-model
pip install -r requirements.txt

# Download all historical data + compute Elo ratings (no manual steps, no API keys)
python refresh.py

# Run the full simulation (50 000 iterations recommended)
python simulate.py --n 50000

# Head-to-head breakdown
python explain.py matchup Spain France

# What if Mbappe is injured? France attack drops to 1.9
python explain.py what-if France --off 1.9 --def 0.80

# Re-run walk-forward calibration (slow — each WC backtest trains on 49k matches)
python calibrate.py --backtest 2022

# During the tournament: auto-refresh every 15 minutes
python refresh.py --watch 15
```

---

## Model architecture

### Layer 1 — Dynamic Elo ratings

The core signal is a modified Elo engine ([`elo.py`](elo.py)) trained chronologically on ~49,000 international matches from 1872 to present. Each team gets two derived ratings:

| Rating | Meaning |
|---|---|
| **off** | Expected goals scored vs. an average opponent (baseline ≈ 1.40 xG) |
| **def** | Expected goals conceded vs. an average opponent (baseline ≈ 1.40 xG) |

These feed into the **Soccer Power Index**:

```
SPI = 100 × off / (off + def)
```

An average international team scores SPI ≈ 50. The 2026 field ranges from ~52 (Qatar) to ~82 (Spain).

**Elo engine details:**
- **Harmonic margin of victory**: `h = Σ(1/k, k=1..goal_diff)` — diminishing returns; a 3-goal win counts less than 3× a 1-goal win (Hvattum & Arntzen 2010)
- **Match importance weights**: five semantic bands with calibrated K-factor multipliers (see Layer 5)
- **Home advantage**: base Elo boost (75 pts) + altitude exponential (La Paz ≈ +275 pts) + away-team travel distance
- **Provisional period**: new teams get 2× K for their first 100 matches to allow fast convergence from the 1500 prior

### Layer 2 — Tactical tilt

Beyond raw Elo, each team carries a **tilt** score — the residual between their actual goals scored/conceded over their last 60 matches and what Elo alone would predict. This captures attacking or defensive style tendencies that Elo doesn't fully encode.

Tilt is shrunk toward zero by a factor of 0.20 before use (noise control; tactical residuals in international football are very noisy).

### Layer 3 — Match model

For a match between team A and team B:

```
goal_scalar = 1 + (tilt_A + tilt_B) × 0.50
xG_A = off_A × (def_B / 1.40) × goal_scalar
xG_B = off_B × (def_A / 1.40) × goal_scalar
```

Goals for each team are drawn from **Negative Binomial distributions** (OVERDISPERSION = 0.30), adding more 0-goal and high-scoring outcomes than pure Poisson — reflecting empirical international football variance (Rue & Salvesen 2000).

A **Dixon-Coles joint-probability correction** (ρ = −0.13) adjusts the likelihood of low-scoring outcomes (0-0, 1-0, 0-1, 1-1), which Poisson/NB systematically misprice (Dixon & Coles 1997).

Win/draw/loss probabilities are also computed analytically (summing the joint PMF over an integer grid) for the matchup CLI and explainability tools.

### Layer 4 — Tournament mechanics

**Group stage** — 12 groups of 4, matchday-aware scheduling.

On the final matchday, the model detects shared-incentive scenarios:
- *Both teams already safe* → each team's xG reduced by 0.50 (defensive posturing)
- *Both teams already eliminated* → each team's xG increased by 0.50 (nothing to lose)

Third-place ranking tiebreakers: points → goal difference → goals scored → simulated fair-play card draw.

After each simulated group match, a mini Elo update (HOT_K = 10) adjusts within-simulation ratings so teams carry form into knockout rounds.

**Knockout rounds** — R32 → R16 → QF → SF → Final.

Draws after 90 min → extra time (scaled xG, 30 min) → penalty shootout (5 kicks + sudden death, SPI-adjusted success rate per kick, ±5pp).

**Third-place advancement**: top 2 per group (24 teams) + 8 best third-place finishers by FIFA ranking criteria = 32 teams advance.

### Layer 5 — Calibrated importance weights

Match type weights on the K-factor are **empirically tuned** via walk-forward backtesting over five World Cups (2006–2022) using `calibrate.py`. Key findings documented in [`config.py`](config.py):

| Band | Examples | Weight | Calibration note |
|---|---|---|---|
| World Cup | FIFA World Cup | **2.00** | Consistently pushes higher in unconstrained optimisation; raised from 1.60 |
| Continental | UEFA Euro, Copa América, AFCON | 0.30–1.30 | Lower than intuition suggests; confederation quality varies |
| Qualifying | WC qualification, Euro qualification | 0.10–1.10 | Reduced; path difficulty varies widely by confederation |
| Minor | Nations League, Olympics | 0.10–0.75 | Small signal; reduced |
| Friendly | Friendlies | 0.50 | Low but non-zero |

> **Calibration finding**: The unconstrained optimiser consistently finds a corner solution (WC weight → max, everything else → min), driven by the small sample (5 tournaments × ~48 group-stage matches each) and the dominance of ~15 historically stable elite teams. We apply a bounded, domain-knowledge-constrained version rather than the extreme solution, as the corner solution would degrade predictions for the ~30% of 2026 qualifiers with limited WC history.

> **Decay finding**: Contrary to club-football literature, time decay of historical matches *hurts* RPS at all tested half-lives (730 – 3650 days) for international football. Chronological Elo accumulation already handles recency adequately given the low match frequency (~10 matches/year per team vs. 50+ for clubs). Decay is disabled (`ELO_DECAY_HALFLIFE_DAYS = 0`).

---

## Data sources

All sources are downloaded automatically by `refresh.py` — **no manual downloads required**.

| Source | Used for | Refresh |
|---|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) | ~49,000 historical matches → Elo ratings | On every `refresh.py` run |
| [ESPN scoreboard API](https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard) | Live/completed 2026 WC scores (real-time) | Queried in 14-day windows |
| [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) | Group draw, schedule, cross-check | On every `refresh.py` run |
| [The Odds API](https://the-odds-api.com) | Bookmaker-implied probabilities (optional) | Set `ODDS_API_KEY` env var |

---

## Tools

### Simulation

```bash
python simulate.py                          # 10,000 sims (default)
python simulate.py --n 50000               # tighter confidence intervals
python simulate.py --group-probs           # per-group advance probabilities
python simulate.py --matchup ESP FRA       # head-to-head win probability
```

When `ODDS_API_KEY` is set and `data/winner_odds.csv` is present, the output adds `Mkt%` (bookmaker implied probability) and `Edge` (model minus market) columns.

### Explainability

```bash
# Full match breakdown: ratings, xG formula, scoreline heatmap, goal distributions
python explain.py matchup Spain Brazil

# Scenario: what if France's Mbappé misses the tournament?
python explain.py what-if France --off 1.90 --def 0.80

# Sensitivity: how sensitive is Spain's win% to overdispersion?
python explain.py sensitivity --team Spain --param OVERDISPERSION --min 0.05 --max 0.55 --steps 7

# Calibration report: are win probabilities well-calibrated on historical data?
python explain.py calibration --last 2000
```

### Calibration

```bash
# Walk-forward backtest for a single tournament
python calibrate.py --backtest 2022

# Sweep decay half-lives and find the empirical optimum
python calibrate.py --decay

# Optimise importance weights via Nelder-Mead over 5 WC backtests
python calibrate.py --weights

# Both + auto-patch config.py
python calibrate.py --all
```

### Evaluation

```bash
# Compare Naive / Elo-only / full model across 2018 + 2022
python evaluate.py

# Feature ablation: RPS contribution of each component
python evaluate.py --ablation 2022

# Match-by-match predictions vs. actuals
python evaluate.py --show-matches 2022
```

### Data pipeline

```bash
python refresh.py              # full pipeline: download → merge → Elo → teams.py → odds
python refresh.py --data-only  # download only, skip Elo recompute
python refresh.py --elo-only   # skip download, recompute from existing CSV
python refresh.py --status     # show data coverage summary
python refresh.py --watch 15   # auto-refresh every 15 min (live tournament mode)
```

---

## Tweaking the model

Every parameter is documented in **[`config.py`](config.py)**:

```python
ELO_K_BASE = 40.0           # base K-factor; higher = faster adaptation to recent form
ELO_DECAY_HALFLIFE_DAYS = 0 # 0 = disabled (calibrated: decay hurts RPS for international football)

IMPORTANCE["fifa world cup"] = 2.00        # WC matches: nudged up from 1.60 by calibration
IMPORTANCE["fifa world cup qualification"] = 1.10  # qualifying: reduced; confederation path varies

OVERDISPERSION = 0.30       # negative-binomial dispersion (0 = Poisson)
RHO = -0.13                 # Dixon-Coles low-score correction

HOME_ADV_BASE = 75.0        # base home advantage in Elo points
ALTITUDE_COEFF = 60.0       # La Paz ≈ +275 pts; Mexico City ≈ +33 pts

TILT_GOAL_IMPACT = 0.50     # how much tactical tilt scales total xG
N_SIMS = 10_000             # Monte Carlo iterations
```

After any change, run `python simulate.py` to see updated probabilities. Use `python explain.py sensitivity` to sweep a parameter range.

---

## File map

```
simulate.py        Main CLI — Monte Carlo tournament simulation + results table
refresh.py         Automated data pipeline: download → merge → Elo → odds
build_ratings.py   Elo computation pipeline: results.csv → ratings → teams.py
elo.py             Elo engine: harmonic MOV, importance weights, HFA, itertuples loop
model.py           Match model: NB xG, Dixon-Coles, tilt, extra time, penalties
tournament.py      Tournament mechanics: group stage, tiebreakers, bracket, hot-sim
teams.py           48-team roster with off/def/tilt/group (auto-updated by refresh.py)
explain.py         Transparency: matchup breakdown, what-if, sensitivity, calibration
calibrate.py       Empirical calibration engine: walk-forward RPS, decay sweep, weight optimisation
evaluate.py        Historical benchmarking: Naive / Elo-only / full model / ablation
odds.py            Bookmaker odds: The Odds API integration, overround removal, CSV persistence
config.py          Every tunable parameter with inline documentation and calibration evidence
data/              Gitignored — populated automatically by refresh.py
```

---

## Calibration benchmarks

Walk-forward RPS (Ranked Probability Score) across WC group stages. Lower = better. Naive baseline ≈ 0.239 (predicting 1/3 for each outcome).

| Model | 2018 WC | 2022 WC | Notes |
|---|---|---|---|
| Naive (1/3 each) | 0.242 | 0.239 | No-skill baseline |
| Elo only | 0.211 | 0.219 | No tilt, no NB, no DC correction |
| Full model (no decay) | 0.214 | 0.219 | Current production config |
| Full model (with decay) | 0.229 | 0.223 | Decay consistently hurts for international football |

The model beats the naive baseline by ~9% on RPS. Tilt and Dixon-Coles improve score *distribution* prediction (relevant for tournament simulation) but not win/loss/draw RPS directly — those features earn their place through better goal totals, not better match outcome rank-ordering.

---

## Methodological references

- **Dixon & Coles (1997)** — "Modelling Association Football Scores and Inefficiencies in the Football Betting Market" — joint Poisson correction for low-score outcomes
- **Rue & Salvesen (2000)** — "Prediction and Retrospective Analysis of Soccer Matches in a League" — time-varying Elo, negative binomial overdispersion
- **Hvattum & Arntzen (2010)** — "Using ELO ratings for match result prediction in association football" — establishes Elo as a competitive match predictor for international football
- **Silver, N. (2022)** — [PELE model](https://www.natesilver.net/p/pele-international-football-rankings-soccer-ratings-projections) — match importance weighting, tilt concept, group-incentive modeling

---

## Known limitations & future work

### Defensive tilt (Morocco 2022, Greece Euro 2004)

The current tilt model captures *offensive* style residuals well — teams that consistently score above their xG expectation. It does not separately weight **defensive outperformance** (goals-conceded vs. xGA). Morocco 2022 is the canonical example: they conceded only 0.14 xG/90 actual while facing ~1.0 xGA per game — four clean sheets driven by structural defensive shape, not luck. The model's tilt residual is symmetric and shrunk heavily (TILT_TACTICAL_SHRINK = 0.20), so it partially captures this but under-weights it.

**Proposed fix**: split `tilt` into `off_tilt` (goals scored vs. xGF residual) and `def_tilt` (goals conceded vs. xGA residual). Apply a tournament-context multiplier to `def_tilt` — defensive organisations are more replicable across 7 games than offensive inspiration, especially for underdogs.

### Calibration bias in importance weights

The unconstrained optimiser always finds a corner solution (WC weight → max, everything else → min). This is a real empirical signal, but it's biased by the 5-WC sample size and the dominance of ~15 historically elite teams. After 2026 adds a sixth tournament, re-run `calibrate.py --weights` — a less biased estimate should emerge once the sample includes more varied qualifiers.

### Player and squad quality signal

The model has the infrastructure for squad-value blending (`ELO_SQUAD_BLEND = 0.35`, Transfermarkt values, age trajectory) but this blend is not yet validated against held-out data. The challenge: player quality data is club-based (FBref, Transfermarkt) while Elo is international-match-based. A proper integration would require mapping club ratings to international context — substantial rebuild, potentially high value for tournaments with late squad changes (injuries, call-offs).

### Bookmaker odds comparison

`evaluate.py --vs-538` is stubbed but 538 stopped publishing models after 2023. The next best benchmark is bookmaker-implied probabilities for 2026 group-stage matches (available once the tournament starts). The pipeline is ready (`odds.py`, `data/odds.csv`); the comparison will be live-updated during the tournament.

---

## Contributing

Pull requests welcome. Before submitting, please:
1. Run `python calibrate.py --backtest 2022` and include the RPS in your PR description
2. Include before/after `python simulate.py` output so changes can be evaluated
3. Update `config.py` comments if you change a parameter value
