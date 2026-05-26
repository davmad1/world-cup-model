"""
elo.py — Dynamic Elo rating engine for international football.

Implements a PELE-style modified Elo system:
  • Harmonic margin of victory (diminishing returns per goal)
  • Match importance weighting (friendlies → WC, 0.5× → 1.6×)
  • Home advantage: base HFA + altitude boost + travel distance
  • Provisional period: 2× K for first ~100 matches
  • Chronological processing of ~45 000 historical matches

Usage
-----
    from elo import compute_elo, elo_to_off_def
    ratings, history = compute_elo(matches_df)
    off, def_ = elo_to_off_def(ratings["Argentina"])
"""

from __future__ import annotations

import math
from typing import NamedTuple

import pandas as pd

import config

# ── Team name normalisation ───────────────────────────────────────────────────
# Maps Kaggle CSV names → canonical names used in teams.py.
# Historical predecessor states map to their modern successors.

TEAM_NAME_MAP: dict[str, str] = {
    # CONCACAF
    "United States":            "USA",
    # CONMEBOL
    # (most names match)
    # UEFA
    "Czech Republic":           "Czech Republic",
    "Czechia":                  "Czech Republic",
    "Bosnia-Herzegovina":       "Bosnia & Herzegovina",
    "Bosnia Herzegovina":       "Bosnia & Herzegovina",
    "Türkiye":                  "Turkey",
    # Africa
    "Ivory Coast":              "Ivory Coast",
    "Côte d'Ivoire":            "Ivory Coast",
    "Cote d'Ivoire":            "Ivory Coast",
    "Congo DR":                 "DR Congo",
    "Congo, DR":                "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Cape Verde Islands":       "Cape Verde",
    # Asia
    "Korea Republic":           "South Korea",
    "Republic of Korea":        "South Korea",
    "IR Iran":                  "Iran",
    "Curacao":                  "Curaçao",
    # Historical predecessors → modern successors
    "West Germany":             "Germany",
    "German DR":                "Germany",       # East Germany → minor contrib
    "Swaziland":                "Eswatini",
    "Macedonia":                "North Macedonia",
    "FYR Macedonia":            "North Macedonia",
    "North Macedonia":          "North Macedonia",
}

# Teams to skip entirely (B-teams, defunct states we don't track).
SKIP_TEAMS: set[str] = {
    "Soviet Union",      # defunct; Russia starts fresh in 1992
    "Yugoslavia",        # defunct; successors start fresh
    "Czechoslovakia",    # defunct; successors start fresh
    "Unified Team",      # 1992 Olympics only
    "Serbia and Montenegro",
}


def normalise_name(name: str) -> str | None:
    """Return canonical team name, or None to skip this team."""
    if name in SKIP_TEAMS:
        return None
    return TEAM_NAME_MAP.get(name, name)


# ── Harmonic margin ───────────────────────────────────────────────────────────

def harmonic_margin(goal_diff: int) -> float:
    """
    Convert a goal margin to a diminishing-returns score (PELE-style).

        margin  → h
        0       → 0.00   (draw)
        1       → 1.00
        2       → 1.50
        3       → 1.83
        4       → 2.08
        5       → 2.28

    Sign is preserved: negative means team B won by that margin.
    """
    abs_diff = abs(goal_diff)
    h = sum(1.0 / k for k in range(1, abs_diff + 1))
    return h if goal_diff >= 0 else -h


# ── Match importance ──────────────────────────────────────────────────────────

def importance_weight(tournament: str) -> float:
    """
    Map a tournament name string to an importance multiplier.
    Matches case-insensitively; longest matching keyword wins.
    Falls back to config.IMPORTANCE_DEFAULT.
    """
    t_lower = tournament.lower()
    best_key, best_weight = "", config.IMPORTANCE_DEFAULT
    for keyword, weight in config.IMPORTANCE.items():
        if keyword in t_lower and len(keyword) > len(best_key):
            best_key, best_weight = keyword, weight
    return best_weight


# ── Altitude lookup ───────────────────────────────────────────────────────────
# city (lower-case) → altitude in metres
# Covers all 2026 WC venues + major high-altitude football cities worldwide.

CITY_ALTITUDE: dict[str, float] = {
    # 2026 WC host cities
    "mexico city":          2_240, "ciudad de mexico":    2_240,
    "guadalajara":          1_566, "zapopan":             1_566,
    "monterrey":              513, "guadalupe":             513,
    "new york":                10, "east rutherford":       10,
    "los angeles":             85, "inglewood":             85,
    "dallas":                 195, "arlington":            195,
    "san francisco":           15, "santa clara":           15,
    "seattle":                 10,
    "miami":                    2, "miami gardens":          2,
    "kansas city":            268,
    "philadelphia":            10, "foxborough":            25,
    "atlanta":                320,
    "houston":                 15,
    "boston":                  25,
    "toronto":                 76,
    "vancouver":                5,
    # High-altitude South America
    "la paz":               3_640, "ciudad de la paz":   3_640,
    "quito":                2_850,
    "bogota":               2_600, "bogotá":             2_600,
    "cusco":                3_399,
    "potosi":               3_967,
    # High-altitude Central America / Caribbean
    "san jose":             1_161, "san josé":           1_161,
    "tegucigalpa":            994,
    "guatemala city":       1_530,
    # High-altitude Africa
    "addis ababa":          2_355,
    "nairobi":              1_795,
    "johannesburg":         1_753,
    "pretoria":             1_338,
    "harare":               1_490,
    "lusaka":               1_280,
    "kampala":              1_190,
    "kigali":               1_567,
    # Other significant altitudes
    "kathmandu":            1_400,
    "sana'a":               2_250, "sanaa":              2_250,
    "tehran":               1_191,
    "ankara":                 938,
    "madrid":                 657,
    "bern":                   540,
    "zurich":                 408,
    "munich":                 519,
    "vienna":                 171,
    "rome":                    13,
    "paris":                   35,
    "london":                  11,
    "amsterdam":               -2,
}


def get_altitude(city: str) -> float:
    """Return altitude (m) for a city, or 0 if unknown."""
    return CITY_ALTITUDE.get(city.lower().strip(), 0.0)


def altitude_boost(altitude_m: float) -> float:
    """
    Extra HFA Elo points due to altitude.

    Below ALTITUDE_KNEE (1 500 m) — negligible.
    Above that, exponential growth calibrated so:
        La Paz  (3 640 m) → ~175 points
        Quito   (2 850 m) → ~ 90 points
        Bogotá  (2 600 m) → ~ 55 points
        Mexico City (2 240 m) → ~ 25 points
    """
    excess = max(0.0, altitude_m - config.ALTITUDE_KNEE)
    if excess == 0:
        return 0.0
    # Exponential: coeff × (excess / 1000)^2
    return config.ALTITUDE_COEFF * (excess / 1_000) ** 2


# ── Country → capital coordinates ─────────────────────────────────────────────
# Used to estimate travel distance for the away team.
# lat, lon in decimal degrees.

COUNTRY_COORDS: dict[str, tuple[float, float]] = {
    "Argentina":          (-34.6, -58.4),
    "Australia":          (-35.3, 149.1),
    "Austria":            (48.2, 16.4),
    "Algeria":            (36.7, 3.2),
    "Belgium":            (50.8, 4.4),
    "Bosnia & Herzegovina": (43.8, 18.4),
    "Brazil":             (-15.8, -47.9),
    "Canada":             (45.4, -75.7),
    "Cape Verde":         (14.9, -23.5),
    "Colombia":           (4.7, -74.1),
    "Costa Rica":         (9.9, -84.1),
    "Croatia":            (45.8, 16.0),
    "Czech Republic":     (50.1, 14.4),
    "Curaçao":            (12.1, -69.0),
    "Denmark":            (55.7, 12.6),
    "DR Congo":           (-4.3, 15.3),
    "Ecuador":            (-0.2, -78.5),
    "Egypt":              (30.1, 31.2),
    "England":            (51.5, -0.1),
    "France":             (48.9, 2.3),
    "Germany":            (52.5, 13.4),
    "Ghana":              (5.6, -0.2),
    "Haiti":              (18.5, -72.3),
    "Iran":               (35.7, 51.4),
    "Iraq":               (33.3, 44.4),
    "Italy":              (41.9, 12.5),
    "Ivory Coast":        (5.3, -4.0),
    "Japan":              (35.7, 139.7),
    "Jordan":             (31.9, 35.9),
    "Mexico":             (19.4, -99.1),
    "Morocco":            (34.0, -6.8),
    "Netherlands":        (52.4, 4.9),
    "New Zealand":        (-41.3, 174.8),
    "Nigeria":            (9.1, 7.2),
    "Norway":             (59.9, 10.8),
    "Panama":             (9.0, -79.5),
    "Paraguay":           (-25.3, -57.6),
    "Portugal":           (38.7, -9.1),
    "Qatar":              (25.3, 51.5),
    "Saudi Arabia":       (24.7, 46.7),
    "Scotland":           (55.9, -3.2),
    "Senegal":            (14.7, -17.4),
    "South Africa":       (-25.7, 28.2),
    "South Korea":        (37.6, 127.0),
    "Spain":              (40.4, -3.7),
    "Sweden":             (59.3, 18.1),
    "Switzerland":        (46.9, 7.4),
    "Tunisia":            (36.8, 10.2),
    "Turkey":             (39.9, 32.9),
    "Uruguay":            (-34.9, -56.2),
    "USA":                (38.9, -77.0),
    "Uzbekistan":         (41.3, 69.2),
    # Additional commonly-seen teams
    "Chile":              (-33.5, -70.7),
    "China PR":           (39.9, 116.4),
    "Cameroon":           (3.9, 11.5),
    "Cuba":               (23.1, -82.4),
    "Greece":             (38.0, 23.7),
    "Honduras":           (14.1, -87.2),
    "Hungary":            (47.5, 19.1),
    "India":              (28.6, 77.2),
    "Indonesia":          (-6.2, 106.8),
    "Israel":             (31.8, 35.2),
    "Jamaica":            (18.0, -76.8),
    "Kenya":              (-1.3, 36.8),
    "Mali":               (12.7, -8.0),
    "Peru":               (-12.1, -77.0),
    "Poland":             (52.2, 21.0),
    "Romania":            (44.4, 26.1),
    "Russia":             (55.8, 37.6),
    "Serbia":             (44.8, 20.5),
    "Slovakia":           (48.1, 17.1),
    "Ukraine":            (50.5, 30.5),
    "Venezuela":          (10.5, -66.9),
    "Wales":              (51.5, -3.2),
    "Bolivia":            (-16.5, -68.1),
    "Ecuador":            (-0.2, -78.5),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def travel_distance_km(home_country: str, away_country: str) -> float:
    """Approximate travel distance from away country capital to home country capital."""
    h = COUNTRY_COORDS.get(home_country)
    a = COUNTRY_COORDS.get(away_country)
    if h is None or a is None:
        return 0.0
    return haversine_km(h[0], h[1], a[0], a[1])


def home_advantage_elo(
    home_city: str,
    home_country: str,
    away_country: str,
    neutral: bool,
) -> float:
    """
    Return the Elo-point home advantage to add to the home team's effective rating.

    Components:
      1. Base HFA (flat, sea-level home game)
      2. Altitude bonus (exponential above ALTITUDE_KNEE)
      3. Travel distance penalty on the away side (capped)

    Returns 0 for neutral-site matches.
    """
    if neutral:
        return 0.0
    base = config.HOME_ADV_BASE
    alt  = altitude_boost(get_altitude(home_city))
    dist_km = travel_distance_km(home_country, away_country)
    travel = min(config.DISTANCE_CAP, dist_km / 1_000 * config.DISTANCE_PER_1000KM)
    return base + alt + travel


# ── Core Elo computation ──────────────────────────────────────────────────────

class MatchRecord(NamedTuple):
    date: str
    home: str
    away: str
    home_score: int
    away_score: int
    tournament: str
    city: str
    country: str
    neutral: bool


def compute_elo(
    matches_df: pd.DataFrame,
    init_elo: float = 1_500.0,
    verbose: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Compute Elo ratings by processing *matches_df* chronologically.

    Expected columns: date, home_team, away_team, home_score, away_score,
                      tournament, city, country, neutral (bool or "TRUE"/"FALSE")

    Returns
    -------
    ratings : dict[team_name → current Elo]
    history : DataFrame with columns [date, team, elo] for plotting
    """
    ratings: dict[str, float] = {}
    match_count: dict[str, int] = {}
    history_rows: list[dict] = []

    # Ensure date-sorted
    df = matches_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Normalise neutral column
    if df["neutral"].dtype == object:
        df["neutral"] = df["neutral"].str.upper() == "TRUE"

    for _, row in df.iterrows():
        home_raw = str(row["home_team"])
        away_raw = str(row["away_team"])

        home = normalise_name(home_raw)
        away = normalise_name(away_raw)

        if home is None or away is None:
            continue  # skip defunct / B-team entries

        # Initialise unseen teams at the prior
        for team in (home, away):
            if team not in ratings:
                ratings[team] = init_elo
                match_count[team] = 0

        r_home = ratings[home]
        r_away = ratings[away]

        # Home advantage (0 for neutral sites)
        hfa = home_advantage_elo(
            home_city=str(row.get("city", "")),
            home_country=str(row.get("country", home)),
            away_country=away,
            neutral=bool(row["neutral"]),
        )

        # Expected scores (logistic)
        diff = (r_home + hfa - r_away) / 400.0
        e_home = 1.0 / (1.0 + 10.0 ** (-diff))
        e_away = 1.0 - e_home

        # Actual score via harmonic margin
        goal_diff = int(row["home_score"]) - int(row["away_score"])
        h = harmonic_margin(goal_diff)

        if goal_diff > 0:
            actual_home, actual_away = 1.0, 0.0
        elif goal_diff < 0:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        # K-factor: base × importance × provisional multiplier × harmonic scale
        imp  = importance_weight(str(row.get("tournament", "")))
        k_base = config.ELO_K_BASE * imp

        k_home = k_base * (2.0 if match_count[home] < config.PROVISIONAL_MATCHES else 1.0)
        k_away = k_base * (2.0 if match_count[away] < config.PROVISIONAL_MATCHES else 1.0)

        # Scale K by harmonic margin (minimum scale = 1.0)
        h_scale = max(1.0, abs(h))

        ratings[home] = r_home + k_home * h_scale * (actual_home - e_home)
        ratings[away] = r_away + k_away * h_scale * (actual_away - e_away)

        match_count[home] += 1
        match_count[away] += 1

        if verbose and len(history_rows) % 5_000 == 0:
            print(f"  processed {len(history_rows):,} matches …")

        history_rows.append({"date": row["date"], "home": home, "away": away,
                              "elo_home": ratings[home], "elo_away": ratings[away]})

    history = pd.DataFrame(history_rows)
    return ratings, history


# ── Elo → off / def conversion ────────────────────────────────────────────────

def elo_to_off_def(elo: float) -> tuple[float, float]:
    """
    Convert a single Elo rating to (off, def) for use in the xG model.

    Both ratings are derived symmetrically; Phase 2 tilt will break
    the symmetry for attack/defense specialists.

    Calibration (scale = 0.50, avg_elo = 1 500):
        Elo   off    def    SPI
        1500  1.40   1.40   50.0   (average international team)
        1700  1.72   1.14   60.2
        1900  2.09   0.94   69.0
        2070  2.48   0.79   75.8   (≈ Argentina post-2022)
        1300  1.05   1.87   36.0   (very weak)
    """
    norm = (elo - config.ELO_AVG) / 400.0 * config.ELO_GOALS_SCALE
    off  = config.LEAGUE_AVG * math.exp(norm)
    def_ = config.LEAGUE_AVG * math.exp(-norm)
    return round(off, 3), round(def_, 3)


def calibrate_overdispersion(goals_series: pd.Series) -> float:
    """
    Fit the negative-binomial overdispersion parameter to a series of
    per-team-per-match goal counts (from historical data).

    Uses method-of-moments: p = mean / variance → dispersion = (1 - p) / p.
    Returns the fitted dispersion value for use as config.OVERDISPERSION.
    """
    mu  = goals_series.mean()
    var = goals_series.var()
    if var <= mu:
        return 0.01   # variance ≤ mean → effectively Poisson
    # negative binomial: var = mu + mu² × dispersion
    dispersion = (var - mu) / (mu ** 2)
    return round(dispersion, 4)
