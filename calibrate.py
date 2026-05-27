"""
calibrate.py — Empirical tuning of Elo hyper-parameters via walk-forward backtesting.

For each historical World Cup (2006 → 2022) we:
  1. Train Elo on all matches before the tournament (with a 30-day buffer)
  2. Predict every group-stage match (win/draw/loss probabilities)
  3. Score with Ranked Probability Score (RPS) against actual outcomes

We then optimise:
  a) IMPORTANCE weights (friendly / continental / wc_qualifier / wc)
  b) ELO_DECAY_HALFLIFE_DAYS

Usage
-----
    python calibrate.py --decay            # sweep halflife, print optimal
    python calibrate.py --weights          # optimise importance weights
    python calibrate.py --all              # both, auto-patch config.py
    python calibrate.py --backtest 2022    # show match-by-match for one WC
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.optimize import minimize, Bounds  # noqa: F401  (Bounds used inside fn)

import config
from elo import compute_elo, elo_to_off_def, normalise_name
from model import win_probability

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_CSV = DATA_DIR / "results.csv"
CONFIG_PY   = Path(__file__).parent / "config.py"

# ── WC tournament dates ───────────────────────────────────────────────────────

WC_YEARS: dict[int, date] = {
    2006: date(2006, 6, 9),    # Germany
    2010: date(2010, 6, 11),   # South Africa
    2014: date(2014, 6, 12),   # Brazil
    2018: date(2018, 6, 14),   # Russia
    2022: date(2022, 11, 20),  # Qatar
}

CUTOFF_DAYS_BEFORE = 30   # train on matches at least this many days before WC start


# ── Scoring metrics ───────────────────────────────────────────────────────────

def rps(p_win: float, p_draw: float, p_loss: float,
        actual_home: int, actual_away: int) -> float:
    """
    Ranked Probability Score for a three-outcome (win/draw/loss) prediction.

    RPS = 0.5 × [(p_win − a_win)² + (p_win + p_draw − a_win_or_draw)²]

    Lower is better. A naive model (1/3 each) scores ≈ 0.222.
    """
    if actual_home > actual_away:
        a_win, a_wd = 1.0, 1.0   # home win
    elif actual_home == actual_away:
        a_win, a_wd = 0.0, 1.0   # draw
    else:
        a_win, a_wd = 0.0, 0.0   # away win

    c1 = (p_win - a_win) ** 2
    c2 = (p_win + p_draw - a_wd) ** 2
    return 0.5 * (c1 + c2)


def brier(p_win: float, actual_home: int, actual_away: int) -> float:
    """Binary Brier score for the home-win outcome."""
    actual = 1.0 if actual_home > actual_away else 0.0
    return (p_win - actual) ** 2


# ── Walk-forward backtest for one WC ─────────────────────────────────────────

def backtest_wc(
    year: int,
    full_df: pd.DataFrame,
    importance: dict[str, float] | None = None,
    decay_halflife: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Backtest our model against one historical WC.

    Parameters
    ----------
    year          : WC year (must be in WC_YEARS)
    full_df       : complete results DataFrame (from results.csv).
                    Should already have date as datetime64 and be sorted by date.
    importance    : override for config.IMPORTANCE (None = use config)
    decay_halflife: override for config.ELO_DECAY_HALFLIFE_DAYS (None = use config)
    verbose       : print match-by-match predictions

    Returns
    -------
    dict with keys: year, n_matches, mean_rps, mean_brier, matches (DataFrame)
    """
    if year not in WC_YEARS:
        raise ValueError(f"year must be one of {list(WC_YEARS)}")

    wc_start = WC_YEARS[year]
    cutoff   = pd.Timestamp(wc_start) - pd.Timedelta(days=CUTOFF_DAYS_BEFORE)

    # full_df is expected to already have datetime dates
    train_df = full_df[full_df["date"] < cutoff]
    wc_df    = full_df[
        (full_df["date"] >= pd.Timestamp(wc_start)) &
        (full_df["date"].dt.year == year) &
        (full_df["tournament"].str.contains("FIFA World Cup", case=False, na=False))
    ]

    # Only group stage: matches before the R32 cutoff (first ~20 days)
    group_end = pd.Timestamp(wc_start) + pd.Timedelta(days=20)
    wc_df = wc_df[wc_df["date"] <= group_end]

    if wc_df.empty:
        print(f"  No WC group-stage matches found for {year}.")
        return {"year": year, "n_matches": 0, "mean_rps": float("nan"),
                "mean_brier": float("nan"), "matches": pd.DataFrame()}

    # Temporarily override config if requested
    orig_importance = deepcopy(config.IMPORTANCE)
    orig_decay      = config.ELO_DECAY_HALFLIFE_DAYS

    try:
        if importance is not None:
            for k, v in importance.items():
                config.IMPORTANCE[k] = v
        if decay_halflife is not None:
            config.ELO_DECAY_HALFLIFE_DAYS = decay_halflife

        use_decay = config.ELO_DECAY_HALFLIFE_DAYS > 0
        # return_history=False: skip building the history DataFrame (2-3× faster for calibration)
        ratings, _ = compute_elo(train_df, decay=use_decay, return_history=False)
    finally:
        config.IMPORTANCE.clear()
        config.IMPORTANCE.update(orig_importance)
        config.ELO_DECAY_HALFLIFE_DAYS = orig_decay

    rows = []
    for _, match in wc_df.iterrows():
        home_raw = str(match["home_team"])
        away_raw = str(match["away_team"])
        home = normalise_name(home_raw) or home_raw
        away = normalise_name(away_raw) or away_raw

        elo_h = ratings.get(home, 1500.0)
        elo_a = ratings.get(away, 1500.0)
        off_h, def_h = elo_to_off_def(elo_h)
        off_a, def_a = elo_to_off_def(elo_a)

        p_win, p_draw, p_loss = win_probability(off_h, def_h, off_a, def_a)

        actual_h = int(match["home_score"])
        actual_a = int(match["away_score"])

        r = rps(p_win, p_draw, p_loss, actual_h, actual_a)
        b = brier(p_win, actual_h, actual_a)

        rows.append({
            "date":       match["date"].date(),
            "home":       home,
            "away":       away,
            "elo_h":      round(elo_h, 0),
            "elo_a":      round(elo_a, 0),
            "p_win":      round(p_win, 3),
            "p_draw":     round(p_draw, 3),
            "p_loss":     round(p_loss, 3),
            "actual":     f"{actual_h}-{actual_a}",
            "outcome":    "W" if actual_h > actual_a else ("D" if actual_h == actual_a else "L"),
            "rps":        round(r, 4),
            "brier":      round(b, 4),
        })

        if verbose:
            outcome_label = "W" if actual_h > actual_a else ("D" if actual_h == actual_a else "L")
            print(f"  {match['date'].date()}  {home:20} vs {away:20}  "
                  f"P(W/D/L)={p_win:.2f}/{p_draw:.2f}/{p_loss:.2f}  "
                  f"actual={actual_h}-{actual_a} [{outcome_label}]  RPS={r:.4f}")

    matches_df = pd.DataFrame(rows)
    mean_rps   = matches_df["rps"].mean()
    mean_brier = matches_df["brier"].mean()

    return {
        "year":       year,
        "n_matches":  len(matches_df),
        "mean_rps":   mean_rps,
        "mean_brier": mean_brier,
        "matches":    matches_df,
    }


# ── Multi-WC aggregate score ──────────────────────────────────────────────────

def _aggregate_rps(
    full_df: pd.DataFrame,
    importance: dict[str, float] | None = None,
    decay_halflife: int | None = None,
    years: list[int] | None = None,
) -> float:
    """Mean RPS across all selected WC tournaments (lower is better)."""
    if years is None:
        years = list(WC_YEARS.keys())
    scores = []
    for y in years:
        result = backtest_wc(y, full_df, importance=importance, decay_halflife=decay_halflife)
        if result["n_matches"] > 0:
            scores.append(result["mean_rps"])
    return sum(scores) / len(scores) if scores else float("nan")


# ── Decay halflife sweep ──────────────────────────────────────────────────────

def sweep_decay_halflife(
    full_df: pd.DataFrame,
    candidates: list[int] | None = None,
) -> list[tuple[int, float]]:
    """
    Sweep ELO_DECAY_HALFLIFE_DAYS from 365 to 3 650 days and return (days, rps) pairs.
    """
    if candidates is None:
        candidates = list(range(365, 3_651, 365))

    results = []
    n = len(candidates)
    for i, days in enumerate(candidates):
        print(f"  [{i+1}/{n}] halflife={days}d …", end=" ", flush=True)
        rps_val = _aggregate_rps(full_df, decay_halflife=days)
        print(f"mean RPS={rps_val:.5f}")
        results.append((days, rps_val))

    return results


# ── Importance weight optimisation ───────────────────────────────────────────
#
# We tune 4 semantic bands with explicit bounds via L-BFGS-B:
#
#   CONSEQUENTIAL  — high-stakes championships where best teams play each other
#                    (WC, Euros, Copa America, AFCON, Asian Cup, Gold Cup)
#   QUALIFYING     — competitive qualifiers with real opposition
#                    (WC qualification, Euro qualification, general qualifiers)
#   MINOR          — lower-stakes or weaker-signal competitive matches
#                    (Nations League, Olympics)
#   FRIENDLY       — preparation / low-stakes matches
#
# The bands encode the user's conceptual hierarchy: a WC group-stage match
# (England vs USA) is far more informative than England vs Luxembourg in
# WC qualifying.  The K-factor already discounts lopsided expected results;
# these weights control the *marginal* learning rate per match type.

_PARAM_MAP: dict[str, list[str]] = {
    # The biggest championship: uniquely high-stakes, opponent quality is top-tier.
    # Empirically the strongest predictor of WC performance, allowed to go high.
    "wc": [
        "fifa world cup",
    ],
    # Other major finals: important but weaker WC-signal than WC matches themselves.
    # (Teams field different squads; schedule varies; confederation quality varies.)
    "continental": [
        "copa america", "copa américa",
        "uefa euro", "uefa european",
        "african cup of nations",
        "afc asian cup",
        "concacaf gold cup",
        "confederation cup", "confederations cup",
    ],
    # Competitive qualifiers — real opposition, high stakes, but noisy:
    # easy confederation paths (e.g., OFC, CONCACAF) dilute the signal.
    "qualifying": [
        "fifa world cup qualification",
        "uefa euro qualification",
        "qualification",   # catch-all qualifier
    ],
    # Lower-stakes or weaker-signal competitive
    "minor": [
        "nations league",
        "nations league qualification",
        "olympic", "olympics",
    ],
    # Preparation / non-competitive
    "friendly": [
        "friendly",
    ],
}

# Bounds for L-BFGS-B: (lb, ub) per band.
# Designed so each band stays in a physically sensible range:
#   wc          → can go as high as 4.5; floor at 1.0 so WC always matters most
#   continental → should be significant but below WC; floor at 0.3
#   qualifying  → meaningful signal but noisy; floor at 0.1
#   minor       → quite small, at most 1.0
#   friendly    → low; at most 0.8
_BOUNDS = [
    (1.00, 4.50),  # wc
    (0.30, 2.50),  # continental
    (0.10, 1.50),  # qualifying
    (0.10, 1.00),  # minor
    (0.05, 0.80),  # friendly
]

_BAND_NAMES   = ["wc", "continental", "qualifying", "minor", "friendly"]
_BAND_DISPLAY = ["wc          ", "continental ", "qualifying  ", "minor       ", "friendly    "]


def _params_to_importance(params: list[float]) -> dict[str, float]:
    """Convert [consequential, qualifying, minor, friendly] → full IMPORTANCE dict."""
    out = deepcopy(config.IMPORTANCE)
    for band, weight in zip(_BAND_NAMES, params):
        for key in _PARAM_MAP[band]:
            out[key] = float(weight)
    return out


def optimize_importance_weights(full_df: pd.DataFrame) -> dict[str, float]:
    """
    Find IMPORTANCE weights that minimise mean RPS across 2006–2022 WC group stages.

    Uses L-BFGS-B with explicit per-band bounds to prevent degenerate corner
    solutions (e.g., all weights collapsing to zero).

    Starting point: average of the current config values in each band.

    Returns the best-fit {key: weight} dict (same structure as config.IMPORTANCE).
    """
    from scipy.optimize import Bounds

    def _band_start(band: str) -> float:
        vals = [config.IMPORTANCE.get(k, config.IMPORTANCE_DEFAULT)
                for k in _PARAM_MAP[band]]
        return sum(vals) / len(vals)

    x0 = [_band_start(b) for b in _BAND_NAMES]
    lb  = [lo for lo, _ in _BOUNDS]
    ub  = [hi for _, hi in _BOUNDS]
    # Clip starting point to its band's bounds
    x0  = [max(lo, min(hi, v)) for v, (lo, hi) in zip(x0, _BOUNDS)]

    call_count = [0]

    def objective(params: list[float]) -> float:
        call_count[0] += 1
        # Hard-clip to bounds inside the objective so Nelder-Mead can't escape
        clipped = [max(lo, min(hi, v)) for v, (lo, hi) in zip(params, _BOUNDS)]
        imp = _params_to_importance(clipped)
        score = _aggregate_rps(full_df, importance=imp)
        if call_count[0] % 5 == 0:
            parts = "  ".join(
                f"{d}={c:.2f}"
                for d, c in zip(_BAND_DISPLAY, clipped)
            )
            print(f"    iter {call_count[0]:>4}: {parts}  →  RPS={score:.5f}",
                  flush=True)
        return score

    print("Optimising importance weights (Nelder-Mead + clipped bounds) …",
          flush=True)
    print(f"  Bands   : {' | '.join(_BAND_NAMES)}", flush=True)
    print(f"  Bounds  : {' | '.join(f'[{lo:.2f},{hi:.2f}]' for lo, hi in _BOUNDS)}",
          flush=True)
    print(f"  Starting: {' | '.join(f'{v:.2f}' for v in x0)}", flush=True)
    print(f"  Starting RPS: {objective(x0):.5f}", flush=True)

    result = minimize(
        objective, x0,
        method="Nelder-Mead",
        options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-5, "disp": False},
    )

    raw_params  = result.x.tolist()
    best_params = [max(lo, min(hi, v)) for v, (lo, hi) in zip(raw_params, _BOUNDS)]
    best_imp    = _params_to_importance(best_params)

    print(f"\nOptimisation complete ({call_count[0]} evaluations).", flush=True)
    print(f"  Best RPS : {result.fun:.5f}")
    print(f"\n  Band weights (start → optimal  [lo, hi]):")
    for name, start, opt, (lo, hi) in zip(_BAND_NAMES, x0, best_params, _BOUNDS):
        at_bound = " ← AT BOUND" if abs(opt - lo) < 0.01 or abs(opt - hi) < 0.01 else ""
        print(f"    {name:<14}  {start:.3f} → {opt:.3f}  [{lo:.2f}, {hi:.2f}]{at_bound}")

    return best_imp


# ── Auto-patch config.py ──────────────────────────────────────────────────────

def _patch_config_param(src: str, param: str, value: int | float) -> str:
    """Replace a scalar assignment in config.py source."""
    if isinstance(value, int):
        pattern = rf"^({re.escape(param)}\s*:\s*\w+\s*=\s*)[\d_]+"
        replacement = rf"\g<1>{value}"
    else:
        pattern = rf"^({re.escape(param)}\s*:\s*\w+\s*=\s*)[\d.]+"
        replacement = rf"\g<1>{value:.4f}"
    return re.sub(pattern, replacement, src, flags=re.MULTILINE)


def patch_config_decay(optimal_days: int) -> None:
    src = CONFIG_PY.read_text()
    new_src = _patch_config_param(src, "ELO_DECAY_HALFLIFE_DAYS", optimal_days)
    if new_src != src:
        CONFIG_PY.write_text(new_src)
        print(f"  config.py: ELO_DECAY_HALFLIFE_DAYS updated to {optimal_days}")
    else:
        print(f"  config.py: ELO_DECAY_HALFLIFE_DAYS already {optimal_days} — no change")


def patch_config_importance(best_imp: dict[str, float]) -> None:
    """
    Update the IMPORTANCE dict entries in config.py.
    Only patches keys that exist in the file; preserves comments.
    """
    src = CONFIG_PY.read_text()
    changed = []
    for key, val in best_imp.items():
        current = config.IMPORTANCE.get(key)
        if current is None or abs(current - val) < 0.001:
            continue
        pattern = rf'("{re.escape(key)}":\s*)[\d.]+'
        new_src, n = re.subn(pattern, rf'\g<1>{val:.3f}', src)
        if n > 0:
            src = new_src
            changed.append((key, current, val))

    if changed:
        CONFIG_PY.write_text(src)
        print(f"  config.py: {len(changed)} IMPORTANCE weights updated:")
        for k, old, new in changed:
            print(f"    {k:<40} {old:.3f} → {new:.3f}")
    else:
        print("  config.py: IMPORTANCE already optimal — no change")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_df() -> pd.DataFrame:
    """
    Load and pre-process results CSV once.

    Pre-processing here (date conversion, sort, neutral normalisation) means
    backtest_wc() can skip these steps on every optimiser call — significant
    speedup when the optimiser runs hundreds of objective evaluations.
    """
    if not RESULTS_CSV.exists():
        sys.exit("data/results.csv not found — run `python refresh.py --data-only` first.")
    df = pd.read_csv(RESULTS_CSV)
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # Pre-process once so every backtest_wc / compute_elo call reuses these
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if df["neutral"].dtype == object:
        df["neutral"] = df["neutral"].str.upper() == "TRUE"
    print(f"Loaded {len(df):,} matches from {RESULTS_CSV.name}\n", flush=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empirical calibration of Elo hyper-parameters via WC backtesting"
    )
    parser.add_argument("--decay", action="store_true",
                        help="Sweep decay halflife and print optimal value")
    parser.add_argument("--weights", action="store_true",
                        help="Optimise importance weights via Nelder-Mead")
    parser.add_argument("--all", dest="do_all", action="store_true",
                        help="Run both --decay and --weights, auto-patch config.py")
    parser.add_argument("--backtest", type=int, metavar="YEAR",
                        help="Show match-by-match predictions for a single WC year")
    args = parser.parse_args()

    if not any([args.decay, args.weights, args.do_all, args.backtest]):
        parser.print_help()
        return

    df = _load_df()

    # ── Single backtest ───────────────────────────────────────────────────────
    if args.backtest:
        year = args.backtest
        if year not in WC_YEARS:
            sys.exit(f"Year must be one of {sorted(WC_YEARS)}")
        print(f"{'─'*70}")
        print(f"Walk-forward backtest: {year} FIFA World Cup (group stage)")
        print(f"{'─'*70}")
        result = backtest_wc(year, df, verbose=True)
        print(f"\n{'─'*70}")
        print(f"  Matches evaluated : {result['n_matches']}")
        print(f"  Mean RPS          : {result['mean_rps']:.5f}")
        print(f"  Mean Brier        : {result['mean_brier']:.5f}")
        print(f"  (Naive RPS ≈ 0.222)")
        return

    # ── Decay halflife sweep ──────────────────────────────────────────────────
    optimal_days = None
    if args.decay or args.do_all:
        print("=" * 60)
        print("Sweep: ELO_DECAY_HALFLIFE_DAYS")
        print("=" * 60)
        sweep_results = sweep_decay_halflife(df)
        optimal_days, best_rps = min(sweep_results, key=lambda t: t[1])
        print(f"\nResults:")
        print(f"  {'Halflife':>12}  {'Mean RPS':>10}")
        for days, rps_val in sweep_results:
            marker = " ← optimal" if days == optimal_days else ""
            print(f"  {days:>10}d  {rps_val:>10.5f}{marker}")
        print(f"\n  → Optimal halflife: {optimal_days} days (RPS={best_rps:.5f})")
        current_rps = _aggregate_rps(df, decay_halflife=config.ELO_DECAY_HALFLIFE_DAYS)
        print(f"  → Current config ({config.ELO_DECAY_HALFLIFE_DAYS}d) RPS: {current_rps:.5f}")

        if args.do_all and optimal_days != config.ELO_DECAY_HALFLIFE_DAYS:
            patch_config_decay(optimal_days)

    # ── Weight optimisation ───────────────────────────────────────────────────
    if args.weights or args.do_all:
        print("\n" + "=" * 60)
        print("Optimise: IMPORTANCE weights")
        print("=" * 60)
        best_imp = optimize_importance_weights(df)

        if args.do_all:
            patch_config_importance(best_imp)

    if args.do_all:
        print("\n✓ config.py patched. Run `python refresh.py --elo-only` to rebuild ratings.")


if __name__ == "__main__":
    main()
