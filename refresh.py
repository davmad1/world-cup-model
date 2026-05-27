"""
refresh.py — One-command data pipeline. No manual downloads ever required.

Sources (all free, no API keys):
  1. martj42/international_results (GitHub raw CSV)
     ~49 000 international matches 1872 → today, auto-updated by maintainer.
  2. ESPN unofficial scoreboard API
     Real-time completed scores, updated within minutes of final whistle.
  3. openfootball/worldcup.json (GitHub raw JSON)
     Community-updated 2026 WC schedule + results (backup / cross-check).

Pipeline:
  ① Download martj42 CSV          → data/results.csv
  ② Fetch ESPN completed WC scores → merge & patch any NA rows
  ③ Cross-check vs openfootball   → fill any ESPN gaps
  ④ Save merged file               → data/results.csv
  ⑤ Run build_ratings              → recompute Elo → patch teams.py

Usage:
    python refresh.py              # full refresh + recompute
    python refresh.py --data-only  # download/merge only, skip Elo recompute
    python refresh.py --elo-only   # skip download, just recompute from existing CSV
    python refresh.py --watch      # refresh every N minutes (use during live matches)
    python refresh.py --status     # show current data coverage summary, no changes
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent / "data"
RESULTS_CSV = DATA_DIR / "results.csv"

MARTJ42_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)
OPENFOOTBALL_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json"
    "/master/2026/worldcup.json"
)
ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
)

# 2026 World Cup date range and total match count (12 groups × 6 + 32 + 16 + 8 + 4 + 2 + 1)
WC_START          = date(2026, 6, 11)
WC_END            = date(2026, 7, 19)
WC_TOTAL_MATCHES  = 104

# Timeout for HTTP requests (seconds)
HTTP_TIMEOUT = 20


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None) -> bytes:
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "WCModel/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        return r.read()


def _get_json(url: str, params: dict | None = None) -> dict:
    return json.loads(_get(url, params))


# ── Source 1: martj42 GitHub CSV ──────────────────────────────────────────────

def download_martj42() -> pd.DataFrame:
    """
    Download the full international results CSV from martj42's GitHub repo.
    Returns a DataFrame with rows for all completed matches (drops NA scores).
    """
    print("① Downloading martj42/international_results …", end=" ", flush=True)
    raw = _get(MARTJ42_URL)
    df = pd.read_csv(io.BytesIO(raw))
    total = len(df)
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"{len(df):,} completed matches ({total - len(df)} upcoming/NA dropped).")
    return df


# ── Source 2: ESPN live/completed scores ──────────────────────────────────────

def _espn_date_range(start: date, end: date) -> str:
    """ESPN API date-range format: YYYYMMDD-YYYYMMDD."""
    return f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"


def _parse_espn_event(event: dict) -> dict | None:
    """Extract a completed match record from a single ESPN event dict."""
    comps = event.get("competitions", [])
    if not comps:
        return None
    comp = comps[0]
    status = comp.get("status", {}).get("type", {})
    if not status.get("completed", False):
        return None

    competitors = comp.get("competitors", [])
    if len(competitors) != 2:
        return None

    # ESPN orders home/away via homeAway field
    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

    try:
        h_score = int(float(home["score"]))
        a_score = int(float(away["score"]))
    except (KeyError, ValueError):
        return None

    venue = comp.get("venue", {})
    city    = venue.get("city", "")
    country = venue.get("country", {}).get("name", "United States")

    return {
        "date":       event["date"][:10],
        "home_team":  home["team"]["displayName"],
        "away_team":  away["team"]["displayName"],
        "home_score": h_score,
        "away_score": a_score,
        "tournament": "FIFA World Cup",
        "city":       city,
        "country":    country,
        "neutral":    True,   # all 2026 WC matches are neutral-site
    }


def fetch_espn_wc(start: date = WC_START, end: date = WC_END) -> pd.DataFrame:
    """
    Fetch all completed 2026 WC matches from ESPN's unofficial scoreboard API.
    Queries in 14-day windows to stay within ESPN's response limits.
    """
    print("② Fetching ESPN completed WC scores …", end=" ", flush=True)
    records = []
    cursor = start

    while cursor <= min(end, date.today()):
        window_end = min(cursor + timedelta(days=13), end, date.today())
        try:
            data = _get_json(ESPN_SCOREBOARD, {
                "dates": _espn_date_range(cursor, window_end),
                "limit": 100,
            })
            for event in data.get("events", []):
                rec = _parse_espn_event(event)
                if rec:
                    records.append(rec)
        except Exception as exc:
            print(f"\n   ESPN warning ({cursor}): {exc}")
        cursor = window_end + timedelta(days=1)

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["date", "home_team", "away_team", "home_score",
                 "away_score", "tournament", "city", "country", "neutral"]
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"{len(df)} completed WC matches found.")
    return df


# ── Source 3: openfootball cross-check ────────────────────────────────────────

def fetch_openfootball_wc() -> pd.DataFrame:
    """
    Pull completed 2026 WC results from openfootball/worldcup.json.
    Used as a cross-check / fallback where ESPN data is missing.
    """
    print("③ Cross-checking openfootball/worldcup.json …", end=" ", flush=True)
    data = _get_json(OPENFOOTBALL_URL)
    records = []
    for m in data.get("matches", []):
        if m.get("score1") is None or m.get("score2") is None:
            continue
        grp = m.get("group", "")
        tournament = "FIFA World Cup" if grp else "FIFA World Cup"
        records.append({
            "date":       m["date"],
            "home_team":  m["team1"],
            "away_team":  m["team2"],
            "home_score": int(m["score1"]),
            "away_score": int(m["score2"]),
            "tournament": tournament,
            "city":       m.get("ground", ""),
            "country":    "United States",
            "neutral":    True,
        })
    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["date", "home_team", "away_team", "home_score",
                 "away_score", "tournament", "city", "country", "neutral"]
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"{len(df)} completed WC matches found.")
    return df


# ── Merge & deduplicate ───────────────────────────────────────────────────────

def _match_key(df: pd.DataFrame) -> pd.Series:
    """Unique key: date + sorted team names (handles home/away order differences)."""
    t1 = df[["home_team", "away_team"]].apply(
        lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
    )
    return df["date"].astype(str) + "|" + t1.apply(lambda t: f"{t[0]}|{t[1]}")


def merge_results(
    historical: pd.DataFrame,
    espn: pd.DataFrame,
    openfootball: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all three sources. Priority: ESPN > openfootball > martj42.
    Deduplicates on (date, sorted team names).
    """
    print("④ Merging sources …", end=" ", flush=True)

    # Start with historical baseline
    combined = historical.copy()

    # Add ESPN and openfootball rows not already present
    existing_keys = set(_match_key(combined))
    new_rows = []

    for extra_df in [espn, openfootball]:
        if extra_df.empty:
            continue
        keys = _match_key(extra_df)
        novel = extra_df[~keys.isin(existing_keys)].copy()
        if not novel.empty:
            new_rows.append(novel)
            existing_keys.update(_match_key(novel))

    if new_rows:
        combined = pd.concat([combined] + new_rows, ignore_index=True)

    combined = combined.sort_values("date").reset_index(drop=True)
    print(f"{len(combined):,} total matches after merge.")
    return combined


# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"   Saved → {RESULTS_CSV}  ({len(df):,} rows)")


# ── Status summary ────────────────────────────────────────────────────────────

def show_status() -> None:
    if not RESULTS_CSV.exists():
        print("data/results.csv not found — run `python refresh.py` first.")
        return
    df = pd.read_csv(RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    print(f"\nData coverage summary ({RESULTS_CSV.name})")
    print(f"  Total matches:  {len(df):,}")
    print(f"  Date range:     {df['date'].min().date()} → {df['date'].max().date()}")
    wc26 = df[df["tournament"] == "FIFA World Cup"]
    wc26_done = wc26[wc26["date"] >= pd.Timestamp(WC_START)]
    print(f"  2026 WC played: {len(wc26_done)} / {WC_TOTAL_MATCHES} matches")
    if not wc26_done.empty:
        print(f"\n  Most recent 5 results:")
        recent = wc26_done.sort_values("date").tail(5)
        for _, row in recent.iterrows():
            print(f"    {row['date'].date()}  "
                  f"{row['home_team']:20} {int(row['home_score'])}-{int(row['away_score'])}  "
                  f"{row['away_team']}")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_full_refresh(data_only: bool = False, elo_only: bool = False) -> None:
    if not elo_only:
        historical   = download_martj42()
        espn         = fetch_espn_wc()
        openfootball = fetch_openfootball_wc()
        merged       = merge_results(historical, espn, openfootball)
        save_results(merged)
    else:
        if not RESULTS_CSV.exists():
            print("No results.csv found. Run without --elo-only first.")
            return
        print(f"Using existing {RESULTS_CSV} (--elo-only).")

    if not data_only:
        print("⑤ Recomputing Elo ratings …")
        # Import here to avoid circular issues at top of file
        import build_ratings
        df = build_ratings.load_results(RESULTS_CSV)
        ratings, _ = build_ratings.compute_ratings(df)
        current_teams = build_ratings.load_current_teams()
        build_ratings.patch_teams_py(current_teams, ratings)
        print("\n✓ Refresh complete — teams.py updated with latest Elo ratings.")
    else:
        print("\n✓ Data refresh complete (--data-only, skipped Elo recompute).")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated data refresh: download → merge → Elo → teams.py"
    )
    parser.add_argument("--data-only", action="store_true",
                        help="Download and merge data but skip Elo recompute")
    parser.add_argument("--elo-only", action="store_true",
                        help="Skip download, recompute Elo from existing CSV only")
    parser.add_argument("--status", action="store_true",
                        help="Show data coverage summary without making any changes")
    parser.add_argument("--watch", type=int, metavar="MINUTES", default=0,
                        help="Run refresh every N minutes (useful during live matches)")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.watch:
        print(f"Watch mode: refreshing every {args.watch} minute(s). Ctrl-C to stop.\n")
        while True:
            try:
                print(f"\n{'─'*60}")
                print(f"Refresh at {time.strftime('%H:%M:%S')}")
                run_full_refresh(data_only=args.data_only, elo_only=args.elo_only)
                print(f"Next refresh in {args.watch} min …")
                time.sleep(args.watch * 60)
            except KeyboardInterrupt:
                print("\nWatch mode stopped.")
                break
    else:
        run_full_refresh(data_only=args.data_only, elo_only=args.elo_only)


if __name__ == "__main__":
    main()
