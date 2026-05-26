"""
2026 FIFA World Cup team SPI ratings.

Groups sourced directly from openfootball/worldcup.json (official draw,
December 5 2025, Kennedy Center, Washington D.C.).

Ratings are calibrated so that an average international team has
off ≈ 1.40 and def ≈ 1.40 (expected goals scored/allowed vs an
average opponent per 90 minutes).

SPI = 100 * off / (off + def)   →  average team = 50, best ≈ 77+

To refresh groups/schedule from the live data source run:
    python fetch_data.py --update-teams
"""

try:
    from config import LEAGUE_AVG
except ImportError:
    LEAGUE_AVG = 1.40  # fallback if config not yet present

# fmt: off
TEAMS: dict[str, dict] = {
    # ── Group A: Mexico, South Africa, South Korea, Czech Republic ──
    "Mexico":           {"off": 1.65, "def": 1.10, "group": "A", "tilt": 0.0},
    "South Africa":     {"off": 1.35, "def": 1.28, "group": "A", "tilt": 0.0},
    "South Korea":      {"off": 1.58, "def": 1.12, "group": "A", "tilt": 0.0},
    "Czech Republic":   {"off": 1.55, "def": 1.15, "group": "A", "tilt": 0.0},
    # ── Group B: Canada, Bosnia & Herzegovina, Qatar, Switzerland ───
    "Canada":           {"off": 1.62, "def": 1.12, "group": "B", "tilt": 0.0},
    "Bosnia & Herzegovina": {"off": 1.48, "def": 1.20, "group": "B", "tilt": 0.0},
    "Qatar":            {"off": 1.28, "def": 1.35, "group": "B", "tilt": 0.0},
    "Switzerland":      {"off": 1.65, "def": 1.02, "group": "B", "tilt": 0.0},
    # ── Group C: Brazil, Morocco, Haiti, Scotland ────────────────────
    "Brazil":           {"off": 2.10, "def": 0.80, "group": "C", "tilt": 0.0},
    "Morocco":          {"off": 1.62, "def": 0.95, "group": "C", "tilt": 0.0},
    "Haiti":            {"off": 1.22, "def": 1.40, "group": "C", "tilt": 0.0},
    "Scotland":         {"off": 1.52, "def": 1.18, "group": "C", "tilt": 0.0},
    # ── Group D: USA, Paraguay, Australia, Turkey ────────────────────
    "USA":              {"off": 1.68, "def": 1.05, "group": "D", "tilt": 0.0},
    "Paraguay":         {"off": 1.48, "def": 1.18, "group": "D", "tilt": 0.0},
    "Australia":        {"off": 1.45, "def": 1.20, "group": "D", "tilt": 0.0},
    "Turkey":           {"off": 1.62, "def": 1.10, "group": "D", "tilt": 0.0},
    # ── Group E: Germany, Curaçao, Ivory Coast, Ecuador ─────────────
    "Germany":          {"off": 2.05, "def": 0.82, "group": "E", "tilt": 0.0},
    "Curaçao":          {"off": 1.15, "def": 1.45, "group": "E", "tilt": 0.0},
    "Ivory Coast":      {"off": 1.58, "def": 1.18, "group": "E", "tilt": 0.0},
    "Ecuador":          {"off": 1.52, "def": 1.15, "group": "E", "tilt": 0.0},
    # ── Group F: Netherlands, Japan, Sweden, Tunisia ─────────────────
    "Netherlands":      {"off": 1.88, "def": 0.95, "group": "F", "tilt": 0.0},
    "Japan":            {"off": 1.65, "def": 1.05, "group": "F", "tilt": 0.0},
    "Sweden":           {"off": 1.68, "def": 1.05, "group": "F", "tilt": 0.0},
    "Tunisia":          {"off": 1.38, "def": 1.22, "group": "F", "tilt": 0.0},
    # ── Group G: Belgium, Egypt, Iran, New Zealand ───────────────────
    "Belgium":          {"off": 1.80, "def": 1.00, "group": "G", "tilt": 0.0},
    "Egypt":            {"off": 1.48, "def": 1.15, "group": "G", "tilt": 0.0},
    "Iran":             {"off": 1.45, "def": 1.18, "group": "G", "tilt": 0.0},
    "New Zealand":      {"off": 1.18, "def": 1.42, "group": "G", "tilt": 0.0},
    # ── Group H: Spain, Cape Verde, Saudi Arabia, Uruguay ───────────
    "Spain":            {"off": 2.15, "def": 0.78, "group": "H", "tilt": 0.0},
    "Cape Verde":       {"off": 1.28, "def": 1.35, "group": "H", "tilt": 0.0},
    "Saudi Arabia":     {"off": 1.42, "def": 1.22, "group": "H", "tilt": 0.0},
    "Uruguay":          {"off": 1.72, "def": 1.00, "group": "H", "tilt": 0.0},
    # ── Group I: France, Senegal, Iraq, Norway ───────────────────────
    "France":           {"off": 2.20, "def": 0.75, "group": "I", "tilt": 0.0},
    "Senegal":          {"off": 1.58, "def": 1.10, "group": "I", "tilt": 0.0},
    "Iraq":             {"off": 1.28, "def": 1.35, "group": "I", "tilt": 0.0},
    "Norway":           {"off": 1.82, "def": 1.05, "group": "I", "tilt": 0.0},
    # ── Group J: Argentina, Algeria, Austria, Jordan ─────────────────
    "Argentina":        {"off": 2.35, "def": 0.68, "group": "J", "tilt": 0.0},
    "Algeria":          {"off": 1.48, "def": 1.18, "group": "J", "tilt": 0.0},
    "Austria":          {"off": 1.62, "def": 1.10, "group": "J", "tilt": 0.0},
    "Jordan":           {"off": 1.28, "def": 1.35, "group": "J", "tilt": 0.0},
    # ── Group K: Portugal, DR Congo, Uzbekistan, Colombia ───────────
    "Portugal":         {"off": 2.05, "def": 0.88, "group": "K", "tilt": 0.0},
    "DR Congo":         {"off": 1.42, "def": 1.25, "group": "K", "tilt": 0.0},
    "Uzbekistan":       {"off": 1.28, "def": 1.35, "group": "K", "tilt": 0.0},
    "Colombia":         {"off": 1.75, "def": 1.02, "group": "K", "tilt": 0.0},
    # ── Group L: England, Croatia, Ghana, Panama ─────────────────────
    "England":          {"off": 2.00, "def": 0.85, "group": "L", "tilt": 0.0},
    "Croatia":          {"off": 1.70, "def": 1.02, "group": "L", "tilt": 0.0},
    "Ghana":            {"off": 1.45, "def": 1.22, "group": "L", "tilt": 0.0},
    "Panama":           {"off": 1.30, "def": 1.32, "group": "L", "tilt": 0.0},
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
