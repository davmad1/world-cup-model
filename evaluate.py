"""
evaluate.py — Walk-forward backtesting and multi-model comparison.

Quantifies how our model performs vs. benchmarks on historical World Cups
using the Ranked Probability Score (RPS) and Brier score on group-stage matches.

Benchmarks compared:
  1. Naive          — 1/3 win, 1/3 draw, 1/3 loss (no-skill lower bound)
  2. Elo-only       — our Elo ratings, but Poisson goals, no decay, no tilt, no DC
  3. Our model (no decay) — full model minus time-decay (pre-Phase-6 baseline)
  4. Our model (full)     — current model with time decay

FiveThirtyEight's WC data was removed from their GitHub when they were shut down.
If you obtain the SPI CSV manually, save it to data/538_YEAR.csv and re-run
with --vs-538 to add it as a fifth benchmark.

Usage
-----
    python evaluate.py                     # full comparison 2018 + 2022
    python evaluate.py --year 2022         # single tournament breakdown
    python evaluate.py --ablation 2022     # feature-by-feature contribution
    python evaluate.py --vs-538            # head-to-head vs 538 ratings
    python evaluate.py --show-matches 2022 # match-level predictions table
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from copy import deepcopy
from pathlib import Path

import pandas as pd
from tabulate import tabulate

import config
from calibrate import backtest_wc, WC_YEARS, rps as _rps, brier as _brier
from elo import normalise_name, elo_to_off_def
from model import win_probability

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_CSV = DATA_DIR / "results.csv"

# ── FiveThirtyEight data ──────────────────────────────────────────────────────

F38_URLS = {
    2018: (
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
        "world-cup-2018/wc-2018-forecasts.csv"
    ),
    2022: (
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
        "world-cup-2022/wc-2022-forecasts.csv"
    ),
}

# Columns we expect in 538 forecast CSVs.
# 2018: team, group, spi, off, def, global_o, global_d, ...
# 2022: similar (team, group, spi, off, def)
_F38_TEAM_COL = "team"
_F38_OFF_COL  = "off"
_F38_DEF_COL  = "def"


def load_538_predictions(year: int) -> dict[str, dict[str, float]]:
    """
    Download (and cache) FiveThirtyEight WC forecast CSV.

    Returns {team_name: {"off": float, "def": float, "spi": float}}
    using the ratings published by 538 just before the tournament.
    We take the earliest snapshot (pre-tournament forecast).
    """
    if year not in F38_URLS:
        return {}

    cache_path = DATA_DIR / f"538_{year}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        print(f"  Downloading 538 {year} forecasts …", end=" ", flush=True)
        try:
            req = urllib.request.Request(
                F38_URLS[year], headers={"User-Agent": "WCModel/1.0"}
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
            df = pd.read_csv(io.BytesIO(raw))
            DATA_DIR.mkdir(exist_ok=True)
            df.to_csv(cache_path, index=False)
            print(f"{len(df)} rows cached to {cache_path.name}")
        except Exception as exc:
            print(f"FAILED: {exc}")
            return {}

    # 538 CSVs have one row per team per forecast date; take the first (pre-WC) snapshot
    if "forecast_timestamp" in df.columns:
        df = df.sort_values("forecast_timestamp")
        df = df.groupby(_F38_TEAM_COL).first().reset_index()
    elif "date" in df.columns:
        df = df.sort_values("date")
        df = df.groupby(_F38_TEAM_COL).first().reset_index()

    result: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        team = str(row.get(_F38_TEAM_COL, ""))
        try:
            off_val = float(row[_F38_OFF_COL])
            def_val = float(row[_F38_DEF_COL])
            spi_val = float(row.get("spi", 100 * off_val / (off_val + def_val)))
        except (KeyError, TypeError, ValueError):
            continue
        result[team] = {"off": off_val, "def": def_val, "spi": spi_val}

    return result


# ── Naive baseline ────────────────────────────────────────────────────────────

def _naive_rps(actual_home: int, actual_away: int) -> float:
    return _rps(1 / 3, 1 / 3, 1 / 3, actual_home, actual_away)


# ── 538-ratings match-level predictions ──────────────────────────────────────

def _f38_team_match(team_raw: str, f38: dict[str, dict[str, float]]) -> tuple[float, float] | None:
    """Look up a team in 538's ratings; try both raw name and normalised name."""
    if team_raw in f38:
        d = f38[team_raw]
        return d["off"], d["def"]
    norm = normalise_name(team_raw)
    if norm and norm in f38:
        d = f38[norm]
        return d["off"], d["def"]
    # Try case-insensitive match
    t_lower = team_raw.lower()
    for k, v in f38.items():
        if k.lower() == t_lower:
            return v["off"], v["def"]
    return None


def backtest_f38(year: int, full_df: pd.DataFrame, f38: dict[str, dict[str, float]]) -> dict:
    """
    Predict WC group-stage matches using 538's ratings + our Poisson model.
    Returns same format as calibrate.backtest_wc().
    """
    from datetime import date as _date
    wc_start = WC_YEARS[year]

    full_df = full_df.copy()
    full_df["date"] = pd.to_datetime(full_df["date"])

    wc_df = full_df[
        (full_df["date"] >= pd.Timestamp(wc_start)) &
        (full_df["date"].dt.year == year) &
        (full_df["tournament"].str.contains("FIFA World Cup", case=False, na=False))
    ]
    group_end = pd.Timestamp(wc_start) + pd.Timedelta(days=20)
    wc_df = wc_df[wc_df["date"] <= group_end]

    rows = []
    missing = set()
    for _, match in wc_df.iterrows():
        home_raw = str(match["home_team"])
        away_raw = str(match["away_team"])

        h_ratings = _f38_team_match(home_raw, f38)
        a_ratings = _f38_team_match(away_raw, f38)

        if h_ratings is None or a_ratings is None:
            missing.update(
                [home_raw if h_ratings is None else "",
                 away_raw if a_ratings is None else ""]
            )
            continue

        off_h, def_h = h_ratings
        off_a, def_a = a_ratings

        # Use Poisson (OVERDISPERSION=0) for a clean comparison of ratings only
        orig_od = config.OVERDISPERSION
        config.OVERDISPERSION = 0.0
        p_win, p_draw, p_loss = win_probability(off_h, def_h, off_a, def_a)
        config.OVERDISPERSION = orig_od

        actual_h = int(match["home_score"])
        actual_a = int(match["away_score"])
        r = _rps(p_win, p_draw, p_loss, actual_h, actual_a)
        b = _brier(p_win, actual_h, actual_a)

        rows.append({"home": home_raw, "away": away_raw,
                     "p_win": p_win, "p_draw": p_draw, "p_loss": p_loss,
                     "actual": f"{actual_h}-{actual_a}",
                     "rps": r, "brier": b})

    if missing - {""}:
        print(f"    [538/{year}] No rating found for: {', '.join(missing - {''})}")

    df_out = pd.DataFrame(rows)
    return {
        "year":       year,
        "n_matches":  len(df_out),
        "mean_rps":   df_out["rps"].mean() if not df_out.empty else float("nan"),
        "mean_brier": df_out["brier"].mean() if not df_out.empty else float("nan"),
        "matches":    df_out,
    }


# ── Elo-only baseline (no NB, no DC, no tilt, no decay) ──────────────────────

def backtest_elo_only(year: int, full_df: pd.DataFrame) -> dict:
    """Run backtest with OVERDISPERSION=0 (Poisson) and decay disabled."""
    orig_od    = config.OVERDISPERSION
    orig_rho   = config.RHO
    orig_tilt  = config.TILT_GOAL_IMPACT
    config.OVERDISPERSION  = 0.0
    config.RHO             = 0.0
    config.TILT_GOAL_IMPACT = 0.0
    try:
        result = backtest_wc(year, full_df, decay_halflife=0)
    finally:
        config.OVERDISPERSION  = orig_od
        config.RHO             = orig_rho
        config.TILT_GOAL_IMPACT = orig_tilt
    return result


# ── Naive baseline score for a WC ────────────────────────────────────────────

def _naive_score(year: int, full_df: pd.DataFrame) -> dict:
    wc_start = WC_YEARS[year]
    full_df  = full_df.copy()
    full_df["date"] = pd.to_datetime(full_df["date"])
    wc_df = full_df[
        (full_df["date"] >= pd.Timestamp(wc_start)) &
        (full_df["date"].dt.year == year) &
        (full_df["tournament"].str.contains("FIFA World Cup", case=False, na=False))
    ]
    group_end = pd.Timestamp(wc_start) + pd.Timedelta(days=20)
    wc_df = wc_df[wc_df["date"] <= group_end]
    rps_vals = [
        _naive_rps(int(r["home_score"]), int(r["away_score"]))
        for _, r in wc_df.iterrows()
    ]
    br_vals = [
        _brier(1/3, int(r["home_score"]), int(r["away_score"]))
        for _, r in wc_df.iterrows()
    ]
    return {
        "year": year,
        "n_matches": len(rps_vals),
        "mean_rps":   sum(rps_vals) / len(rps_vals) if rps_vals else float("nan"),
        "mean_brier": sum(br_vals)  / len(br_vals)  if br_vals  else float("nan"),
    }


# ── Comparison table ──────────────────────────────────────────────────────────

def compare_models(
    full_df: pd.DataFrame,
    years: list[int] | None = None,
    include_538: bool = True,
) -> None:
    """Print a side-by-side RPS and Brier comparison across models and years."""
    if years is None:
        years = [y for y in [2018, 2022] if y in WC_YEARS]

    # Load 538 data
    f38_data: dict[int, dict] = {}
    if include_538:
        for y in years:
            if y in F38_URLS:
                f38_data[y] = load_538_predictions(y)

    models = ["Naive", "Elo-only", "Our model (no decay)", "Our model (full)", "538 ratings"]
    rps_table: dict[str, dict[int, str]] = {m: {} for m in models}
    bri_table: dict[str, dict[int, str]] = {m: {} for m in models}

    for y in years:
        print(f"\n  Backtesting {y} … ", end="", flush=True)

        naive    = _naive_score(y, full_df)
        elo_r    = backtest_elo_only(y, full_df)
        no_decay = backtest_wc(y, full_df, decay_halflife=0)
        full_r   = backtest_wc(y, full_df)
        print("done.")

        for m, res in [("Naive", naive), ("Elo-only", elo_r),
                       ("Our model (no decay)", no_decay), ("Our model (full)", full_r)]:
            rps_table[m][y] = f"{res['mean_rps']:.5f}"
            bri_table[m][y] = f"{res['mean_brier']:.5f}"

        if include_538 and y in f38_data and f38_data[y]:
            f38_r = backtest_f38(y, full_df, f38_data[y])
            rps_table["538 ratings"][y] = f"{f38_r['mean_rps']:.5f}"
            bri_table["538 ratings"][y] = f"{f38_r['mean_brier']:.5f}"
        else:
            rps_table["538 ratings"][y] = "N/A (data unavailable)"
            bri_table["538 ratings"][y] = "N/A (data unavailable)"

    # Print tables
    def _build_rows(table: dict) -> list:
        rows = []
        for model in models:
            row = [model] + [table[model].get(y, "—") for y in years]
            rows.append(row)
        return rows

    headers = ["Model"] + [str(y) for y in years]
    print("\n" + "=" * 60)
    print("Ranked Probability Score (RPS) — lower is better")
    print("=" * 60)
    print(tabulate(_build_rows(rps_table), headers=headers, tablefmt="simple"))
    print("  Naive = no-skill baseline; improvement = (Naive − Model) / Naive × 100%")

    print("\n" + "=" * 60)
    print("Brier Score — lower is better")
    print("=" * 60)
    print(tabulate(_build_rows(bri_table), headers=headers, tablefmt="simple"))


# ── Ablation study ────────────────────────────────────────────────────────────

def ablation(year: int, full_df: pd.DataFrame) -> None:
    """Show RPS impact of each feature by disabling them one at a time."""

    def _run(label: str, **overrides) -> float:
        orig = {k: getattr(config, k) for k in overrides}
        try:
            for k, v in overrides.items():
                setattr(config, k, v)
            use_decay = overrides.get("ELO_DECAY_HALFLIFE_DAYS", config.ELO_DECAY_HALFLIFE_DAYS) > 0
            r = backtest_wc(year, full_df, decay_halflife=overrides.get("ELO_DECAY_HALFLIFE_DAYS"))
            return r["mean_rps"]
        finally:
            for k, v in orig.items():
                setattr(config, k, v)

    baseline = backtest_wc(year, full_df)["mean_rps"]
    naive    = _naive_score(year, full_df)["mean_rps"]

    configs = [
        ("Full model (baseline)",          {}),
        ("− Time decay",                   {"ELO_DECAY_HALFLIFE_DAYS": 0}),
        ("− Tilt",                         {"TILT_GOAL_IMPACT": 0.0}),
        ("− Dixon-Coles (RHO=0)",          {"RHO": 0.0}),
        ("− Neg. Binomial (Poisson)",      {"OVERDISPERSION": 0.0}),
        ("− DC and NB (pure Poisson+tilt)",""),  # handled separately
        ("Elo-only (no NB/DC/tilt/decay)", ""),
        ("Naive (1/3 each)",               ""),
    ]

    print(f"\n{'─'*60}")
    print(f"Ablation study — {year} WC group stage")
    print(f"{'─'*60}")
    print(f"  {'Model':<42} {'RPS':>8}  {'vs full':>9}")

    rows = []

    for label, override in configs:
        if override == "":
            # Special pre-computed cases
            if "Elo-only" in label:
                rps_val = backtest_elo_only(year, full_df)["mean_rps"]
            elif "Naive" in label:
                rps_val = naive
            else:
                orig_od  = config.OVERDISPERSION
                orig_rho = config.RHO
                config.OVERDISPERSION = 0.0
                config.RHO            = 0.0
                try:
                    rps_val = backtest_wc(year, full_df)["mean_rps"]
                finally:
                    config.OVERDISPERSION = orig_od
                    config.RHO            = orig_rho
        else:
            rps_val = _run(label, **override)

        delta = rps_val - baseline
        delta_str = f"{delta:+.5f}" if label != "Full model (baseline)" else "—"
        rows.append([label, f"{rps_val:.5f}", delta_str])

    print(tabulate(rows, headers=["Model", "RPS", "Δ vs full"], tablefmt="simple"))
    print(f"\n  Lower RPS = better. Δ > 0 means removing that feature hurts.")
    print(f"  (Naive ≈ {naive:.5f} is the no-skill lower bound)")


# ── Show match-level predictions ──────────────────────────────────────────────

def show_matches(year: int, full_df: pd.DataFrame) -> None:
    """Print match-by-match predictions vs actual outcomes."""
    result = backtest_wc(year, full_df, verbose=False)
    df = result["matches"]

    if df.empty:
        print(f"No matches found for {year}.")
        return

    print(f"\n{year} WC group stage — our model vs actual outcomes")
    print(f"{'─'*80}")
    rows = []
    for _, r in df.iterrows():
        correct = "✓" if (
            (r["outcome"] == "W" and r["p_win"] > max(r["p_draw"], r["p_loss"])) or
            (r["outcome"] == "D" and r["p_draw"] > max(r["p_win"], r["p_loss"])) or
            (r["outcome"] == "L" and r["p_loss"] > max(r["p_win"], r["p_draw"]))
        ) else "✗"
        rows.append([
            str(r["date"]),
            f"{r['home'][:16]:16}",
            f"{r['away'][:16]:16}",
            f"{r['p_win']:.2f}/{r['p_draw']:.2f}/{r['p_loss']:.2f}",
            r["actual"],
            r["outcome"],
            correct,
            f"{r['rps']:.4f}",
        ])
    headers = ["Date", "Home", "Away", "P(W/D/L)", "Score", "Result", "✓/✗", "RPS"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    correct_count = sum(1 for r in rows if r[6] == "✓")
    print(f"\n  {correct_count}/{len(rows)} outcomes correctly ranked as most likely")
    print(f"  Mean RPS: {result['mean_rps']:.5f}  (Naive ≈ 0.222)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_df() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        sys.exit("data/results.csv not found — run `python refresh.py --data-only` first.")
    df = pd.read_csv(RESULTS_CSV)
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    print(f"Loaded {len(df):,} matches\n")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtesting and multi-model benchmark comparison"
    )
    parser.add_argument("--year", type=int, metavar="YEAR",
                        help="Single tournament year (default: 2018 + 2022)")
    parser.add_argument("--ablation", type=int, metavar="YEAR",
                        help="Feature ablation study for a specific WC year")
    parser.add_argument("--vs-538", action="store_true",
                        help="Include FiveThirtyEight ratings as a benchmark")
    parser.add_argument("--show-matches", type=int, metavar="YEAR",
                        help="Print match-by-match predictions for a WC year")
    args = parser.parse_args()

    if not any([args.year, args.ablation, args.vs_538, args.show_matches]):
        # Default: full comparison with 538
        args.vs_538 = True

    df = _load_df()

    if args.show_matches:
        show_matches(args.show_matches, df)
        return

    if args.ablation:
        ablation(args.ablation, df)
        return

    years = [args.year] if args.year else [2018, 2022]
    years = [y for y in years if y in WC_YEARS]
    if not years:
        sys.exit(f"Year(s) must be from {sorted(WC_YEARS)}")

    compare_models(df, years=years, include_538=args.vs_538)


if __name__ == "__main__":
    main()
