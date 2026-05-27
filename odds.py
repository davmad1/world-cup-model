"""
odds.py — Automated bookmaker odds for the 2026 FIFA World Cup.

Source: The Odds API (https://the-odds-api.com)
  • Free tier: 500 requests/month — more than enough for 104 WC matches
  • Register free at https://the-odds-api.com to get an API key
  • Set the env var ODDS_API_KEY before running refresh.py

If ODDS_API_KEY is empty, all functions silently return empty results
and the rest of the model continues unaffected.

Data stored at:  data/odds.csv
Columns:  date, home_team, away_team, p_home_mkt, p_draw_mkt, p_away_mkt,
          bookmaker, fetched_at

Usage
-----
    from odds import fetch_wc_odds, load_odds, save_odds
    df = fetch_wc_odds()                # fetch from API
    lookup = load_odds()                # {(team_a, team_b): (p_h, p_d, p_a)}
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config

DATA_DIR = Path(__file__).parent / "data"
ODDS_CSV = DATA_DIR / "odds.csv"

_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
_SPORT         = "soccer_fifa_world_cup"
_MARKETS       = "h2h"          # head-to-head: home / draw / away
_REGIONS       = "uk"           # decimal odds
_HTTP_TIMEOUT  = 15

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "WCModel/1.0"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as r:
        return json.loads(r.read())


def _remove_overround(h: float, d: float, a: float) -> tuple[float, float, float]:
    """Convert decimal odds → implied probabilities, normalised to remove juice."""
    raw = [1.0 / h, 1.0 / d, 1.0 / a]
    total = sum(raw)
    return tuple(r / total for r in raw)  # type: ignore[return-value]


def _match_key(team_a: str, team_b: str) -> tuple[str, str]:
    """Canonical (sorted) team pair for deduplication."""
    return tuple(sorted([team_a.strip(), team_b.strip()]))  # type: ignore[return-value]


# ── Fetch from The Odds API ───────────────────────────────────────────────────

def fetch_wc_odds() -> pd.DataFrame:
    """
    Fetch all available 2026 WC match odds from The Odds API.

    Returns a DataFrame with columns:
        date, home_team, away_team,
        p_home_mkt, p_draw_mkt, p_away_mkt,
        bookmaker, fetched_at

    Returns an empty DataFrame if ODDS_API_KEY is not set or API fails.
    The implied probabilities (p_*_mkt) already have the overround removed.
    Averages across bookmakers when multiple are returned.
    """
    empty = pd.DataFrame(columns=[
        "date", "home_team", "away_team",
        "p_home_mkt", "p_draw_mkt", "p_away_mkt",
        "bookmaker", "fetched_at",
    ])

    key = config.ODDS_API_KEY.strip()
    if not key:
        return empty

    url = (
        f"{_ODDS_API_BASE}/sports/{_SPORT}/odds/"
        f"?apiKey={key}&regions={_REGIONS}&markets={_MARKETS}&oddsFormat=decimal"
    )

    try:
        data = _get_json(url)
    except Exception as exc:
        print(f"   Odds API warning: {exc}")
        return empty

    if not isinstance(data, list):
        # API may return error dict
        print(f"   Odds API unexpected response: {data}")
        return empty

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict] = []

    for event in data:
        try:
            commence = event.get("commence_time", "")[:10]   # YYYY-MM-DD

            # Identify home / away from the bookmakers
            bookmaker_probs: list[tuple[float, float, float, str]] = []

            for book in event.get("bookmakers", []):
                h2h = next(
                    (m for m in book.get("markets", []) if m["key"] == "h2h"),
                    None,
                )
                if h2h is None:
                    continue
                outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}

                # The API lists home_team and away_team on the event level
                home_name = event.get("home_team", "")
                away_name = event.get("away_team", "")
                draw_name = "Draw"

                if home_name not in outcomes or away_name not in outcomes:
                    continue
                if draw_name not in outcomes:
                    continue

                p_h, p_d, p_a = _remove_overround(
                    outcomes[home_name],
                    outcomes[draw_name],
                    outcomes[away_name],
                )
                bookmaker_probs.append((p_h, p_d, p_a, book["key"]))

            if not bookmaker_probs:
                continue

            # Average across bookmakers
            n = len(bookmaker_probs)
            avg_ph = sum(t[0] for t in bookmaker_probs) / n
            avg_pd = sum(t[1] for t in bookmaker_probs) / n
            avg_pa = sum(t[2] for t in bookmaker_probs) / n
            books_used = ",".join(t[3] for t in bookmaker_probs[:3])  # first 3 names

            rows.append({
                "date":       commence,
                "home_team":  event.get("home_team", ""),
                "away_team":  event.get("away_team", ""),
                "p_home_mkt": round(avg_ph, 4),
                "p_draw_mkt": round(avg_pd, 4),
                "p_away_mkt": round(avg_pa, 4),
                "bookmaker":  books_used,
                "fetched_at": fetched_at,
            })

        except Exception as exc:
            print(f"   Odds parse warning ({event.get('id', '?')}): {exc}")
            continue

    df = pd.DataFrame(rows) if rows else empty
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ── Persist / load ────────────────────────────────────────────────────────────

def save_odds(df: pd.DataFrame) -> None:
    """
    Upsert new odds rows into data/odds.csv.
    Deduplication key: (date, sorted team pair).
    Existing rows are preserved; newer fetches overwrite for the same match.
    """
    if df.empty:
        return

    DATA_DIR.mkdir(exist_ok=True)

    if ODDS_CSV.exists():
        existing = pd.read_csv(ODDS_CSV)
        existing["date"] = pd.to_datetime(existing["date"]).dt.date
        # Build key sets
        def _key(row: pd.Series) -> str:
            a, b = sorted([str(row["home_team"]), str(row["away_team"])])
            return f"{row['date']}|{a}|{b}"
        existing["_key"] = existing.apply(_key, axis=1)
        df["_key"]       = df.apply(_key, axis=1)
        # Drop existing rows that appear in new data (new data wins)
        keep = existing[~existing["_key"].isin(df["_key"])].drop(columns=["_key"])
        df = df.drop(columns=["_key"])
        combined = pd.concat([keep, df], ignore_index=True)
    else:
        combined = df

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(ODDS_CSV, index=False)
    print(f"   Saved → {ODDS_CSV}  ({len(combined):,} rows)")


def load_odds() -> dict[tuple[str, str], tuple[float, float, float]]:
    """
    Load odds from data/odds.csv.

    Returns a dict keyed by sorted (team_a, team_b) tuple:
        {("Argentina", "France"): (p_home, p_draw, p_away), ...}

    Where p_home corresponds to the team listed first in the CSV (home_team).
    Note: the canonical key uses sorted names, so callers should sort too.
    """
    if not ODDS_CSV.exists():
        return {}

    df = pd.read_csv(ODDS_CSV)
    lookup: dict[tuple[str, str], tuple[float, float, float]] = {}

    for _, row in df.iterrows():
        key = _match_key(str(row["home_team"]), str(row["away_team"]))
        lookup[key] = (
            float(row["p_home_mkt"]),
            float(row["p_draw_mkt"]),
            float(row["p_away_mkt"]),
        )

    return lookup


def load_winner_odds() -> dict[str, float]:
    """
    Load pre-tournament winner odds if stored.
    Returns {team_name: implied_probability}.
    Looks for a separate data/winner_odds.csv file (manually seeded or fetched).
    """
    winner_csv = DATA_DIR / "winner_odds.csv"
    if not winner_csv.exists():
        return {}
    df = pd.read_csv(winner_csv)
    if "team" not in df.columns or "p_win_mkt" not in df.columns:
        return {}
    return dict(zip(df["team"], df["p_win_mkt"]))
