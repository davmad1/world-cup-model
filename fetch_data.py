"""
Data fetcher for 2026 FIFA World Cup official data.

Two sources — one free with no key, one free with a key:

1. openfootball/worldcup.json (GitHub raw, no auth)
   → Groups, match schedule, results as they come in
   → https://github.com/openfootball/worldcup.json

2. football-data.org v4 API (free tier, API key required)
   → Teams, squad rosters, standings, fixtures
   → Register free at https://www.football-data.org/client/register
   → Set env var:  export FOOTBALL_DATA_API_KEY="your_key_here"

Usage
-----
    python fetch_data.py                  # fetch + print groups and schedule
    python fetch_data.py --squads         # also fetch squad rosters (needs API key)
    python fetch_data.py --update-teams   # rewrite teams.py groups from live draw data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pprint import pprint

# ── Constants ─────────────────────────────────────────────────────────────────

OPENFOOTBALL_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json"
    "/master/2026/worldcup.json"
)

FD_BASE = "https://api.football-data.org/v4"
FD_WC_CODE = "WC"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_json(url: str, headers: dict | None = None) -> dict | list:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _fd_headers() -> dict:
    key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not key:
        sys.exit(
            "\nERROR: FOOTBALL_DATA_API_KEY not set.\n"
            "Get a free key at https://www.football-data.org/client/register\n"
            "Then:  export FOOTBALL_DATA_API_KEY='your_key'\n"
        )
    return {"X-Auth-Token": key}


# ── openfootball ──────────────────────────────────────────────────────────────

def fetch_groups() -> dict[str, list[str]]:
    """
    Fetch the official group draw from openfootball/worldcup.json.
    Returns {group_letter: [team_name, ...]} — no API key required.

    JSON structure: {"name": ..., "matches": [{..., "group": "Group A", ...}]}
    """
    print("Fetching groups from openfootball …")
    data = _fetch_json(OPENFOOTBALL_URL)

    groups: dict[str, list[str]] = {}
    for match in data.get("matches", []):
        grp_str = match.get("group", "")          # e.g. "Group A"
        if not grp_str or not grp_str.startswith("Group "):
            continue
        letter = grp_str.split()[-1].upper()       # "A" … "L"
        for side in ("team1", "team2"):
            name = match[side]                     # plain string in this schema
            groups.setdefault(letter, [])
            if name not in groups[letter]:
                groups[letter].append(name)

    return dict(sorted(groups.items()))


def fetch_schedule() -> list[dict]:
    """
    Fetch the full match schedule (groups + knockout).
    Returns a list of match dicts with date, teams, group, round.
    """
    print("Fetching schedule from openfootball …")
    data = _fetch_json(OPENFOOTBALL_URL)

    matches = []
    for m in data.get("matches", []):
        grp_str = m.get("group", "")
        grp = grp_str.split()[-1].upper() if grp_str.startswith("Group ") else None
        matches.append({
            "round":  m.get("round"),
            "date":   m.get("date"),
            "time":   m.get("time"),
            "group":  grp,
            "team1":  m["team1"],
            "team2":  m["team2"],
            "score1": m.get("score1"),
            "score2": m.get("score2"),
            "venue":  m.get("ground"),
        })
    return matches


# ── football-data.org ─────────────────────────────────────────────────────────

def fetch_fd_teams() -> list[dict]:
    """
    Fetch all WC 2026 teams from football-data.org.
    Returns list of {id, name, shortName, tla, crestUrl}.
    Requires FOOTBALL_DATA_API_KEY env var.
    """
    print("Fetching teams from football-data.org …")
    url = f"{FD_BASE}/competitions/{FD_WC_CODE}/teams"
    data = _fetch_json(url, _fd_headers())
    return data.get("teams", [])


def fetch_fd_squad(team_id: int) -> list[dict]:
    """
    Fetch the squad for a given team ID from football-data.org.
    Returns list of {id, name, position, dateOfBirth, nationality}.
    Requires FOOTBALL_DATA_API_KEY env var.
    """
    url = f"{FD_BASE}/teams/{team_id}"
    data = _fetch_json(url, _fd_headers())
    return data.get("squad", [])


def fetch_fd_standings() -> list[dict]:
    """
    Fetch current group standings from football-data.org.
    Returns list of standing tables (one per group).
    Requires FOOTBALL_DATA_API_KEY env var.
    """
    print("Fetching standings from football-data.org …")
    url = f"{FD_BASE}/competitions/{FD_WC_CODE}/standings"
    data = _fetch_json(url, _fd_headers())
    return data.get("standings", [])


def fetch_all_squads() -> dict[str, list[dict]]:
    """
    Fetch squads for every WC team. Respects the 10 req/min free-tier
    rate limit by sleeping between requests.
    Returns {team_name: [player_dict, ...]}.
    """
    import time
    teams = fetch_fd_teams()
    squads: dict[str, list[dict]] = {}
    for i, team in enumerate(teams):
        name = team["name"]
        tid = team["id"]
        print(f"  [{i+1}/{len(teams)}] {name} …")
        try:
            squads[name] = fetch_fd_squad(tid)
        except Exception as e:
            print(f"    WARNING: {e}")
            squads[name] = []
        time.sleep(6.5)  # stay under 10 req/min (includes /teams call above)
    return squads


# ── teams.py patcher ──────────────────────────────────────────────────────────

def update_teams_groups(groups: dict[str, list[str]]) -> None:
    """
    Patch the "group" field in teams.py to match the official draw.
    Prints a diff of any changes made.
    """
    import re
    teams_path = os.path.join(os.path.dirname(__file__), "teams.py")
    with open(teams_path) as f:
        source = f.read()

    # Build a reverse map: team_name → group_letter
    name_to_group: dict[str, str] = {}
    for grp, members in groups.items():
        for m in members:
            name_to_group[m] = grp

    # Alternate name mappings where openfootball name ≠ teams.py key
    aliases: dict[str, str] = {
        "Bosnia & Herzegovina": "Bosnia & Herzegovina",  # kept as-is in teams.py
    }

    # Merge aliases into name_to_group
    for openfb_name, our_name in aliases.items():
        if openfb_name in name_to_group and our_name not in name_to_group:
            name_to_group[our_name] = name_to_group[openfb_name]

    changed = []
    for team_name, new_grp in name_to_group.items():
        # Match lines like:  "Argentina":    {..., "group": "D"},
        pattern = rf'("{re.escape(team_name)}":\s*{{[^}}]*"group":\s*)"([A-L])"'
        replacement = rf'\1"{new_grp}"'
        new_source, n = re.subn(pattern, replacement, source)
        if n > 0 and new_source != source:
            changed.append((team_name, new_grp))
            source = new_source

    if changed:
        with open(teams_path, "w") as f:
            f.write(source)
        print(f"\nUpdated teams.py group assignments ({len(changed)} changes):")
        for name, grp in sorted(changed):
            print(f"  {name:20s} → Group {grp}")
    else:
        print("\nteams.py already matches official draw — no changes needed.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch official 2026 WC data from openfootball + football-data.org"
    )
    parser.add_argument("--squads", action="store_true",
                        help="Fetch squad rosters via football-data.org (needs API key)")
    parser.add_argument("--standings", action="store_true",
                        help="Fetch current group standings via football-data.org (needs API key)")
    parser.add_argument("--update-teams", action="store_true",
                        help="Patch teams.py group assignments from official draw data")
    parser.add_argument("--schedule", action="store_true",
                        help="Print the full match schedule")
    args = parser.parse_args()

    # Always fetch groups (free, no key)
    groups = fetch_groups()
    print(f"\n── Official 2026 World Cup Groups ({'openfootball'}) ──")
    for grp, teams in groups.items():
        print(f"  Group {grp}: {', '.join(teams)}")

    if args.schedule:
        matches = fetch_schedule()
        print(f"\n── Schedule ({len(matches)} matches) ──")
        for m in matches:
            score = ""
            if m["score1"] is not None:
                score = f" {m['score1']}-{m['score2']}"
            grp = f"[Grp {m['group']}] " if m["group"] else ""
            print(f"  {m['date']} {grp}{m['team1']} vs {m['team2']}{score}")

    if args.update_teams:
        update_teams_groups(groups)

    if args.squads:
        squads = fetch_all_squads()
        print(f"\n── Squads (first 3 players per team) ──")
        for team, players in squads.items():
            preview = [p["name"] for p in players[:3]]
            print(f"  {team}: {', '.join(preview)} …")

    if args.standings:
        tables = fetch_fd_standings()
        print("\n── Current Standings ──")
        for table in tables:
            grp = table.get("group", "?")
            print(f"\n  Group {grp}")
            for row in table.get("table", []):
                t = row["team"]["name"]
                print(f"    {row['position']}. {t:20s}  "
                      f"Pts:{row['points']}  GD:{row['goalDifference']}")


if __name__ == "__main__":
    main()
