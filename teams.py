"""
2026 FIFA World Cup team SPI ratings.

Ratings are calibrated so that an average international team has
off ≈ 1.40 and def ≈ 1.40 (expected goals scored/allowed vs an
average opponent per 90 minutes).

SPI = 100 * off / (off + def)   →  average team = 50, best ≈ 77+

NOTE: Group assignments below are best estimates based on the
December 5 2024 draw. Verify against the official FIFA draw and
update the "group" field for any corrections.
"""

LEAGUE_AVG = 1.40  # average xG per team per match in international football

# fmt: off
TEAMS: dict[str, dict] = {
    # ── Group A ──────────────────────────────────────────────────
    "USA":          {"off": 1.68, "def": 1.05, "group": "A"},
    "Turkey":       {"off": 1.62, "def": 1.10, "group": "A"},
    "Ecuador":      {"off": 1.52, "def": 1.15, "group": "A"},
    "Nigeria":      {"off": 1.48, "def": 1.20, "group": "A"},
    # ── Group B ──────────────────────────────────────────────────
    "Mexico":       {"off": 1.65, "def": 1.10, "group": "B"},
    "Belgium":      {"off": 1.80, "def": 1.00, "group": "B"},
    "Morocco":      {"off": 1.62, "def": 0.95, "group": "B"},
    "South Korea":  {"off": 1.58, "def": 1.12, "group": "B"},
    # ── Group C ──────────────────────────────────────────────────
    "Canada":       {"off": 1.62, "def": 1.12, "group": "C"},
    "Denmark":      {"off": 1.62, "def": 1.05, "group": "C"},
    "Cameroon":     {"off": 1.45, "def": 1.22, "group": "C"},
    "Iran":         {"off": 1.45, "def": 1.18, "group": "C"},
    # ── Group D ──────────────────────────────────────────────────
    "Argentina":    {"off": 2.35, "def": 0.68, "group": "D"},
    "Croatia":      {"off": 1.70, "def": 1.02, "group": "D"},
    "Tunisia":      {"off": 1.38, "def": 1.22, "group": "D"},
    "Jordan":       {"off": 1.28, "def": 1.35, "group": "D"},
    # ── Group E ──────────────────────────────────────────────────
    "Brazil":       {"off": 2.10, "def": 0.80, "group": "E"},
    "Switzerland":  {"off": 1.65, "def": 1.02, "group": "E"},
    "Ivory Coast":  {"off": 1.58, "def": 1.18, "group": "E"},
    "Australia":    {"off": 1.45, "def": 1.20, "group": "E"},
    # ── Group F ──────────────────────────────────────────────────
    "Spain":        {"off": 2.15, "def": 0.78, "group": "F"},
    "Netherlands":  {"off": 1.88, "def": 0.95, "group": "F"},
    "Senegal":      {"off": 1.58, "def": 1.10, "group": "F"},
    "Honduras":     {"off": 1.28, "def": 1.35, "group": "F"},
    # ── Group G ──────────────────────────────────────────────────
    "France":       {"off": 2.20, "def": 0.75, "group": "G"},
    "Uruguay":      {"off": 1.72, "def": 1.00, "group": "G"},
    "Austria":      {"off": 1.62, "def": 1.10, "group": "G"},
    "Egypt":        {"off": 1.48, "def": 1.15, "group": "G"},
    # ── Group H ──────────────────────────────────────────────────
    "England":      {"off": 2.00, "def": 0.85, "group": "H"},
    "Japan":        {"off": 1.65, "def": 1.05, "group": "H"},
    "Serbia":       {"off": 1.58, "def": 1.12, "group": "H"},
    "South Africa": {"off": 1.35, "def": 1.28, "group": "H"},
    # ── Group I ──────────────────────────────────────────────────
    "Germany":      {"off": 2.05, "def": 0.82, "group": "I"},
    "Italy":        {"off": 1.72, "def": 0.95, "group": "I"},
    "Ghana":        {"off": 1.45, "def": 1.22, "group": "I"},
    "Panama":       {"off": 1.30, "def": 1.32, "group": "I"},
    # ── Group J ──────────────────────────────────────────────────
    "Portugal":     {"off": 2.05, "def": 0.88, "group": "J"},
    "Ukraine":      {"off": 1.55, "def": 1.15, "group": "J"},
    "Algeria":      {"off": 1.48, "def": 1.18, "group": "J"},
    "Indonesia":    {"off": 1.12, "def": 1.48, "group": "J"},
    # ── Group K ──────────────────────────────────────────────────
    "Colombia":     {"off": 1.75, "def": 1.02, "group": "K"},
    "Scotland":     {"off": 1.52, "def": 1.18, "group": "K"},
    "Saudi Arabia": {"off": 1.42, "def": 1.22, "group": "K"},
    "New Zealand":  {"off": 1.18, "def": 1.42, "group": "K"},
    # ── Group L ──────────────────────────────────────────────────
    "Chile":        {"off": 1.52, "def": 1.18, "group": "L"},
    "Venezuela":    {"off": 1.45, "def": 1.22, "group": "L"},
    "Costa Rica":   {"off": 1.38, "def": 1.25, "group": "L"},
    "Uzbekistan":   {"off": 1.28, "def": 1.35, "group": "L"},
}
# fmt: on


def spi(team: str) -> float:
    """Soccer Power Index: 100 * off / (off + def). Average team = 50."""
    t = TEAMS[team]
    return 100.0 * t["off"] / (t["off"] + t["def"])


def groups() -> dict[str, list[str]]:
    """Return {group_letter: [team, ...]} mapping."""
    result: dict[str, list[str]] = {}
    for name, data in TEAMS.items():
        result.setdefault(data["group"], []).append(name)
    return dict(sorted(result.items()))
