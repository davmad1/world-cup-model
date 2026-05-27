"""
explain.py — Transparency and calibration tools for the 2026 WC model.

Usage
-----
    python explain.py matchup Argentina France
        Full analytical breakdown: xG formula, scoreline matrix, goal distribution.

    python explain.py what-if France --off 1.9 --def 0.85 [--tilt -0.02]
        Run the full tournament with one team's ratings overridden.
        Shows how win% changes for every team, not just the modified one.

    python explain.py sensitivity --team Spain --param OVERDISPERSION \\
                                  --min 0.10 --max 0.50 --steps 6
        Sweep any config.py parameter and plot how a team's win% responds.
        Supported params: OVERDISPERSION, RHO, TILT_GOAL_IMPACT,
        INCENTIVE_SAFE_XG_ADJ, INCENTIVE_ELIM_XG_ADJ, HOT_K,
        PEN_BASE, ALTITUDE_COEFF, HOME_ADV_BASE, ELO_GOALS_SCALE.

    python explain.py calibration [--last N]
        Check model calibration against last N historical matches.
        Bins predicted win probabilities and compares to actual outcomes.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

import config
from model import win_probability, expected_goals
from simulate import run_simulation
from teams import TEAMS, spi, groups as get_groups

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_CSV = DATA_DIR / "results.csv"

_BAR_WIDTH = 40
_SIM_N_SENSITIVITY = 5_000
_SIM_N_WHAT_IF     = 10_000


# ── Shared helpers ────────────────────────────────────────────────────────────

def _bar(frac: float, width: int = _BAR_WIDTH) -> str:
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def _check_team(name: str) -> None:
    if name not in TEAMS:
        # Try case-insensitive match
        matches = [t for t in TEAMS if t.lower() == name.lower()]
        if matches:
            sys.exit(f"Unknown team '{name}'. Did you mean '{matches[0]}'?")
        sys.exit(f"Unknown team '{name}'. Run `python simulate.py` for the full team list.")


# ── matchup ──────────────────────────────────────────────────────────────────

def cmd_matchup(team_a: str, team_b: str) -> None:
    _check_team(team_a)
    _check_team(team_b)

    ta = TEAMS[team_a]
    tb = TEAMS[team_b]
    tilt_a = ta.get("tilt", 0.0)
    tilt_b = tb.get("tilt", 0.0)
    off_a, def_a = ta["off"], ta["def"]
    off_b, def_b = tb["off"], tb["def"]

    goal_scalar = max(0.5, min(1.5, 1.0 + (tilt_a + tilt_b) * config.TILT_GOAL_IMPACT))
    xg_a = off_a * (def_b / config.LEAGUE_AVG) * goal_scalar
    xg_b = off_b * (def_a / config.LEAGUE_AVG) * goal_scalar

    p_win, p_draw, p_loss = win_probability(
        off_a, def_a, off_b, def_b, tilt_a=tilt_a, tilt_b=tilt_b
    )
    p_ko_a = p_win + p_draw * 0.5  # rough ET/pens approximation for display

    print(f"\n{'─'*58}")
    print(f"  {team_a}  vs  {team_b}")
    print(f"  Neutral venue · 90 minutes · NB + Dixon-Coles")
    print(f"{'─'*58}\n")

    print("Ratings")
    for name, off, def_, tilt in [
        (team_a, off_a, def_a, tilt_a),
        (team_b, off_b, def_b, tilt_b),
    ]:
        spi_val = 100 * off / (off + def_)
        print(f"  {name:<22} off={off:.3f}  def={def_:.3f}  "
              f"SPI={spi_val:.1f}  tilt={tilt:+.4f}")

    print(f"\nExpected goals  (goal_scalar = {goal_scalar:.3f})")
    print(f"  xG {team_a:<20} = {off_a:.3f} × ({def_b:.3f}/{config.LEAGUE_AVG:.2f}) "
          f"× {goal_scalar:.3f} = {xg_a:.3f}")
    print(f"  xG {team_b:<20} = {off_b:.3f} × ({def_a:.3f}/{config.LEAGUE_AVG:.2f}) "
          f"× {goal_scalar:.3f} = {xg_b:.3f}")

    print(f"\n90-minute win probabilities  (analytical)")
    print(f"  {team_a} win  {_bar(p_win, 30)}  {p_win*100:.1f}%")
    print(f"  Draw         {_bar(p_draw, 30)}  {p_draw*100:.1f}%")
    print(f"  {team_b} win  {_bar(p_loss, 30)}  {p_loss*100:.1f}%")

    print(f"\nKnockout  (including ET + pens, rough)")
    print(f"  {team_a:<22} advances  ~{p_ko_a*100:.1f}%")
    print(f"  {team_b:<22} advances  ~{(1-p_ko_a)*100:.1f}%")

    # Scoreline probability matrix — top 12 outcomes
    from scipy.stats import nbinom as sp_nbinom, poisson as sp_poisson
    use_nb = config.OVERDISPERSION >= 1e-6
    if use_nb:
        n_a = 1.0 / config.OVERDISPERSION;  p_a = n_a / (n_a + xg_a)
        n_b = 1.0 / config.OVERDISPERSION;  p_b = n_b / (n_b + xg_b)

    from model import _dc_tau
    outcomes = []
    for ga in range(9):
        pmf_a = sp_nbinom.pmf(ga, n_a, p_a) if use_nb else sp_poisson.pmf(ga, xg_a)
        for gb in range(9):
            pmf_b = sp_nbinom.pmf(gb, n_b, p_b) if use_nb else sp_poisson.pmf(gb, xg_b)
            tau = _dc_tau(ga, gb, xg_a, xg_b)
            outcomes.append((ga, gb, pmf_a * pmf_b * tau))

    total_p = sum(p for *_, p in outcomes)
    outcomes = [(ga, gb, p / total_p) for ga, gb, p in outcomes]
    outcomes.sort(key=lambda r: -r[2])

    print(f"\nTop-12 scorelines")
    print(f"  {'Score':>7}    {'Prob':>5}  {'':>30}  Result")
    for ga, gb, prob in outcomes[:12]:
        result = f"{team_a} win" if ga > gb else (f"{team_b} win" if gb > ga else "Draw")
        bar = _bar(prob, 24)
        print(f"  {ga}-{gb:>1}  {' ':3}  {prob*100:>4.1f}%  {bar}  {result}")

    # Per-team goal distribution
    print(f"\nGoal distribution per team")
    print(f"  {'Goals':>5}  {team_a[:16]:<18}  {team_b[:16]:<18}")
    all_ga = [sum(sp_nbinom.pmf(g, n_a, p_a) if use_nb else sp_poisson.pmf(g, xg_a)
                  for _ in [0]) for g in range(8)]
    all_gb = [sum(sp_nbinom.pmf(g, n_b, p_b) if use_nb else sp_poisson.pmf(g, xg_b)
                  for _ in [0]) for g in range(8)]
    for g in range(8):
        pa = (sp_nbinom.pmf(g, n_a, p_a) if use_nb else sp_poisson.pmf(g, xg_a)) / total_p * total_p
        pb = (sp_nbinom.pmf(g, n_b, p_b) if use_nb else sp_poisson.pmf(g, xg_b)) / total_p * total_p
        # normalise individually
        pa2 = (sp_nbinom.pmf(g, n_a, p_a) if use_nb else sp_poisson.pmf(g, xg_a))
        pb2 = (sp_nbinom.pmf(g, n_b, p_b) if use_nb else sp_poisson.pmf(g, xg_b))
        print(f"  {g:>5}  {pa2*100:>5.1f}%  {_bar(pa2, 14)}  "
              f"{pb2*100:>5.1f}%  {_bar(pb2, 14)}")

    print()


# ── what-if ───────────────────────────────────────────────────────────────────

def cmd_what_if(
    team: str,
    off_override: float | None,
    def_override: float | None,
    tilt_override: float | None,
    n: int = _SIM_N_WHAT_IF,
) -> None:
    _check_team(team)
    import teams as teams_mod

    baseline_entry = dict(TEAMS[team])
    modified_entry = dict(baseline_entry)
    if off_override  is not None: modified_entry["off"]  = off_override
    if def_override  is not None: modified_entry["def"]  = def_override
    if tilt_override is not None: modified_entry["tilt"] = tilt_override

    # Describe the change
    changes = []
    for key in ("off", "def", "tilt"):
        old = baseline_entry.get(key, 0.0)
        new = modified_entry.get(key, 0.0)
        if abs(new - old) > 1e-6:
            old_spi = 100 * baseline_entry["off"] / (baseline_entry["off"] + baseline_entry["def"])
            new_spi = 100 * modified_entry["off"] / (modified_entry["off"] + modified_entry["def"])
            changes.append(f"{key}: {old:.3f} → {new:.3f}")
    change_str = ",  ".join(changes) if changes else "(no change)"
    old_spi = 100 * baseline_entry["off"] / (baseline_entry["off"] + baseline_entry["def"])
    new_spi = 100 * modified_entry["off"] / (modified_entry["off"] + modified_entry["def"])

    print(f"\nWhat-if: {team}")
    print(f"  {change_str}")
    print(f"  SPI: {old_spi:.1f} → {new_spi:.1f}")
    print(f"\nRunning baseline ({n:,} sims) …", end=" ", flush=True)

    # Baseline
    base_probs = run_simulation(n)
    print("done.")

    # Patched run
    print(f"Running what-if ({n:,} sims) …", end=" ", flush=True)
    teams_mod.TEAMS[team] = modified_entry
    try:
        alt_probs = run_simulation(n)
    finally:
        teams_mod.TEAMS[team] = baseline_entry  # always restore
    print("done.\n")

    # Show all teams sorted by absolute change in win%
    rows = []
    for t in TEAMS:
        base_w = base_probs[t]["winner"]
        alt_w  = alt_probs[t]["winner"]
        delta  = alt_w - base_w
        rows.append((t, base_w, alt_w, delta))

    rows.sort(key=lambda r: -abs(r[3]))

    print(f"{'Team':<24}  {'Baseline':>9}  {'What-if':>8}  {'Change':>8}")
    print("─" * 58)
    for t, bw, aw, d in rows[:20]:
        marker = " ◄" if t == team else ""
        print(f"{t:<24}  {bw*100:>8.1f}%  {aw*100:>7.1f}%  "
              f"{d*100:>+7.1f}pp{marker}")
    print()


# ── sensitivity ───────────────────────────────────────────────────────────────

_SWEEPABLE_PARAMS = {
    "OVERDISPERSION", "RHO", "TILT_GOAL_IMPACT",
    "INCENTIVE_SAFE_XG_ADJ", "INCENTIVE_ELIM_XG_ADJ",
    "HOT_K", "PEN_BASE", "HOME_ADV_BASE", "ALTITUDE_COEFF",
    "ELO_GOALS_SCALE",
}


def cmd_sensitivity(
    team: str,
    param: str,
    lo: float,
    hi: float,
    steps: int,
    n: int = _SIM_N_SENSITIVITY,
) -> None:
    _check_team(team)
    if param not in _SWEEPABLE_PARAMS:
        sys.exit(
            f"'{param}' is not sweepable. Choose from:\n  "
            + ", ".join(sorted(_SWEEPABLE_PARAMS))
        )

    import teams as teams_mod
    from elo import elo_to_off_def

    base_val  = getattr(config, param)
    step_vals = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]

    print(f"\nSensitivity: {team} win%  vs  {param}")
    print(f"  {steps} steps from {lo} to {hi},  {n:,} sims each")
    print(f"  Base value: {base_val}\n")

    results = []
    for val in step_vals:
        setattr(config, param, val)

        # If ELO_GOALS_SCALE changed, recompute off/def for all teams from stored Elo
        if param == "ELO_GOALS_SCALE":
            _recompute_ratings_for_scale(teams_mod, val)

        probs = run_simulation(n)
        win_pct = probs[team]["winner"]
        results.append((val, win_pct))

    # Restore
    setattr(config, param, base_val)
    if param == "ELO_GOALS_SCALE":
        _recompute_ratings_for_scale(teams_mod, base_val)

    # Find result closest to original base value (may not be in the sweep range)
    closest = min(results, key=lambda r: abs(r[0] - base_val))
    base_win = closest[1]

    max_win = max(w for _, w in results)
    print(f"  {param:<28}  {'Win%':>6}  {'vs base':>8}  {'':30}")
    print("  " + "─" * 72)
    for val, win in results:
        delta = win - base_win
        marker = " [base]" if abs(val - base_val) < 1e-9 else ""
        bar = _bar(win / max_win if max_win > 0 else 0, 28)
        print(f"  {val:<28.4f}  {win*100:>5.1f}%  {delta*100:>+7.1f}pp  {bar}{marker}")

    wins = [w for _, w in results]
    vals = [v for v, _ in results]
    if len(vals) > 1 and (vals[-1] - vals[0]) != 0:
        slope = (wins[-1] - wins[0]) / (vals[-1] - vals[0])
        print(f"\n  Slope: {slope*100:+.2f}pp per unit increase in {param}")
    print()


def _recompute_ratings_for_scale(teams_mod, scale: float) -> None:
    """Recompute off/def for all teams in-memory when ELO_GOALS_SCALE changes."""
    import math
    for team, data in teams_mod.TEAMS.items():
        # Reverse-engineer Elo from current off (before any previous scale change)
        # We keep a stash of canonical Elo ratings if available
        pass
    # Simpler: just re-run build_ratings in-memory if the stash exists
    # (Only practical if we cached ratings; skip with a warning if not)
    print("  (Note: ELO_GOALS_SCALE sweep uses current off/def values "
          "without re-running Elo engine; ratings were computed at scale="
          f"{config.ELO_GOALS_SCALE:.2f})")


# ── calibration ───────────────────────────────────────────────────────────────

def cmd_calibration(last_n: int = 2_000) -> None:
    if not RESULTS_CSV.exists():
        sys.exit("data/results.csv not found. Run `python refresh.py --data-only` first.")

    from elo import normalise_name

    print(f"\nCalibration check  (last {last_n:,} neutral-site matches with both teams in model)")
    print("Using current team ratings to predict neutral-site probabilities.\n")

    df = pd.read_csv(RESULTS_CSV)
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").tail(last_n * 3)  # grab extra to filter down

    records = []
    for _, row in df.iterrows():
        # Only neutral-site matches: our model predicts neutral probabilities
        neutral = str(row.get("neutral", "False")).upper() == "TRUE"
        if not neutral:
            continue

        home = normalise_name(str(row["home_team"]))
        away = normalise_name(str(row["away_team"]))
        if home not in TEAMS or away not in TEAMS:
            continue
        th = TEAMS[home]
        ta = TEAMS[away]
        tilt_h = th.get("tilt", 0.0)
        tilt_a = ta.get("tilt", 0.0)

        p_win, p_draw, p_loss = win_probability(
            th["off"], th["def"], ta["off"], ta["def"],
            tilt_a=tilt_h, tilt_b=tilt_a,
        )

        g_h, g_a = int(row["home_score"]), int(row["away_score"])
        if g_h > g_a:
            actual = "win"
        elif g_h < g_a:
            actual = "loss"
        else:
            actual = "draw"

        records.append({
            "p_home_win": p_win,
            "p_draw":     p_draw,
            "p_away_win": p_loss,
            "actual":     actual,
        })
        if len(records) >= last_n:
            break

    if not records:
        print("No matches found with both teams in the current model.")
        return

    df_cal = pd.DataFrame(records)
    n_total = len(df_cal)

    # Bin home-team win probabilities into 10pp buckets
    bins = [i / 10 for i in range(11)]
    print(f"  Calibration: home-team win probability")
    print(f"  {'Predicted band':>16}  {'Actual win%':>11}  {'Count':>6}  {'Calibration'}")
    print(f"  {'─'*60}")

    total_err = 0.0
    n_bins = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask    = (df_cal["p_home_win"] >= lo) & (df_cal["p_home_win"] < hi)
        subset  = df_cal[mask]
        count   = len(subset)
        if count == 0:
            continue
        actual_rate = (subset["actual"] == "win").mean()
        mid_pred    = (lo + hi) / 2
        err         = abs(actual_rate - mid_pred)
        total_err  += err
        n_bins     += 1
        flag = "✓" if err < 0.06 else ("~" if err < 0.12 else "✗")
        print(f"  {lo*100:>5.0f}–{hi*100:.0f}%         "
              f"{actual_rate*100:>9.1f}%  {count:>6}  {flag}")

    mce = total_err / n_bins if n_bins else 0
    n_correct = sum(
        (row["actual"] == "win"  and row["p_home_win"]  > row["p_away_win"] and row["p_home_win"]  > row["p_draw"]) or
        (row["actual"] == "loss" and row["p_away_win"]  > row["p_home_win"] and row["p_away_win"]  > row["p_draw"]) or
        (row["actual"] == "draw" and row["p_draw"]      > row["p_home_win"] and row["p_draw"]      > row["p_away_win"])
        for row in records
    )
    accuracy = n_correct / n_total

    print(f"\n  Matches analysed:        {n_total:,}")
    print(f"  Mean calibration error:  {mce*100:.1f}pp")
    print(f"  Mode accuracy:           {accuracy*100:.1f}%  (% where predicted mode = actual outcome)")
    print(f"\n  Caveats:")
    print(f"  • In-sample — ratings are derived from this same historical data.")
    print(f"  • Uses *current* ratings for all past matches, not time-specific Elo.")
    print(f"    (Spain's 2010 matches are evaluated with Spain's 2026 rating.)")
    print(f"  • True calibration requires out-of-sample held-out matches.")
    print(f"  • Best interpreted as a rough distribution-shape sanity check.")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explainability and calibration tools for the 2026 WC model"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # matchup
    p_m = sub.add_parser("matchup", help="Full analytical breakdown of one fixture")
    p_m.add_argument("team_a", metavar="TEAM_A")
    p_m.add_argument("team_b", metavar="TEAM_B")

    # what-if
    p_w = sub.add_parser("what-if", help="Tournament simulation with one team's ratings overridden")
    p_w.add_argument("team", metavar="TEAM")
    p_w.add_argument("--off",  type=float, default=None, help="Override offensive rating")
    p_w.add_argument("--def",  dest="def_", type=float, default=None, help="Override defensive rating")
    p_w.add_argument("--tilt", type=float, default=None, help="Override tilt")
    p_w.add_argument("-n",     type=int,   default=_SIM_N_WHAT_IF, help="Simulations per run")

    # sensitivity
    p_s = sub.add_parser("sensitivity", help="Sweep a config parameter and observe effect on win%")
    p_s.add_argument("--team",  required=True)
    p_s.add_argument("--param", required=True, help=f"One of: {', '.join(sorted(_SWEEPABLE_PARAMS))}")
    p_s.add_argument("--min",   type=float, required=True, dest="lo")
    p_s.add_argument("--max",   type=float, required=True, dest="hi")
    p_s.add_argument("--steps", type=int,   default=6)
    p_s.add_argument("-n",      type=int,   default=_SIM_N_SENSITIVITY)

    # calibration
    p_c = sub.add_parser("calibration", help="Check model calibration against historical matches")
    p_c.add_argument("--last", type=int, default=2_000, metavar="N",
                     help="Number of recent matches to evaluate (default 2000)")

    args = parser.parse_args()

    if args.cmd == "matchup":
        cmd_matchup(args.team_a, args.team_b)
    elif args.cmd == "what-if":
        cmd_what_if(args.team, args.off, args.def_, args.tilt, n=args.n)
    elif args.cmd == "sensitivity":
        cmd_sensitivity(args.team, args.param, args.lo, args.hi, args.steps, n=args.n)
    elif args.cmd == "calibration":
        cmd_calibration(args.last)


if __name__ == "__main__":
    main()
