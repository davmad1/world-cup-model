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
    "Mexico":           {"off": 2.155, "def": 0.909, "group": "A", "tilt": -0.0264},
    "South Africa":     {"off": 1.618, "def": 1.211, "group": "A", "tilt": -0.0418},
    "South Korea":      {"off": 2.024, "def": 0.969, "group": "A", "tilt": 0.0005},
    "Czech Republic":   {"off": 1.857, "def": 1.056, "group": "A", "tilt": -0.0026},
    # ── Group B: Canada, Bosnia & Herzegovina, Qatar, Switzerland ───
    "Canada":           {"off": 2.063, "def": 0.95, "group": "B", "tilt": -0.0297},
    "Bosnia & Herzegovina": {"off": 1.666, "def": 1.177, "group": "B", "tilt": -0.0248},
    "Qatar":            {"off": 1.466, "def": 1.337, "group": "B", "tilt": -0.005},
    "Switzerland":      {"off": 2.285, "def": 0.858, "group": "B", "tilt": -0.0013},
    # ── Group C: Brazil, Morocco, Haiti, Scotland ────────────────────
    "Brazil":           {"off": 2.47, "def": 0.794, "group": "C", "tilt": -0.0152},
    "Morocco":          {"off": 2.309, "def": 0.849, "group": "C", "tilt": -0.0496},
    "Haiti":            {"off": 1.635, "def": 1.199, "group": "C", "tilt": 0.0294},
    "Scotland":         {"off": 1.974, "def": 0.993, "group": "C", "tilt": -0.0077},
    # ── Group D: USA, Paraguay, Australia, Turkey ────────────────────
    "USA":              {"off": 1.898, "def": 1.033, "group": "D", "tilt": 0.0043},
    "Paraguay":         {"off": 2.21, "def": 0.887, "group": "D", "tilt": -0.0556},
    "Australia":        {"off": 2.116, "def": 0.926, "group": "D", "tilt": -0.0329},
    "Turkey":           {"off": 2.29, "def": 0.856, "group": "D", "tilt": 0.0165},
    # ── Group E: Germany, Curaçao, Ivory Coast, Ecuador ─────────────
    "Germany":          {"off": 2.35, "def": 0.834, "group": "E", "tilt": 0.0353},
    "Curaçao":          {"off": 1.54, "def": 1.273, "group": "E", "tilt": -0.0016},
    "Ivory Coast":      {"off": 1.875, "def": 1.045, "group": "E", "tilt": -0.0282},
    "Ecuador":          {"off": 2.387, "def": 0.821, "group": "E", "tilt": -0.0844},
    # ── Group F: Netherlands, Japan, Sweden, Tunisia ─────────────────
    "Netherlands":      {"off": 2.392, "def": 0.82, "group": "F", "tilt": 0.0282},
    "Japan":            {"off": 2.308, "def": 0.849, "group": "F", "tilt": 0.0068},
    "Sweden":           {"off": 1.864, "def": 1.052, "group": "F", "tilt": 0.0125},
    "Tunisia":          {"off": 1.814, "def": 1.08, "group": "F", "tilt": -0.0616},
    # ── Group G: Belgium, Egypt, Iran, New Zealand ───────────────────
    "Belgium":          {"off": 2.204, "def": 0.889, "group": "G", "tilt": 0.0025},
    "Egypt":            {"off": 1.881, "def": 1.042, "group": "G", "tilt": -0.0492},
    "Iran":             {"off": 2.036, "def": 0.963, "group": "G", "tilt": -0.0118},
    "New Zealand":      {"off": 1.791, "def": 1.094, "group": "G", "tilt": -0.0092},
    # ── Group H: Spain, Cape Verde, Saudi Arabia, Uruguay ───────────
    "Spain":            {"off": 2.952, "def": 0.664, "group": "H", "tilt": 0.0073},
    "Cape Verde":       {"off": 1.632, "def": 1.201, "group": "H", "tilt": -0.0501},
    "Saudi Arabia":     {"off": 1.661, "def": 1.18, "group": "H", "tilt": -0.0442},
    "Uruguay":          {"off": 2.258, "def": 0.868, "group": "H", "tilt": -0.0498},
    # ── Group I: France, Senegal, Iraq, Norway ───────────────────────
    "France":           {"off": 2.701, "def": 0.726, "group": "I", "tilt": 0.0006},
    "Senegal":          {"off": 2.154, "def": 0.91, "group": "I", "tilt": -0.0293},
    "Iraq":             {"off": 1.767, "def": 1.109, "group": "I", "tilt": -0.0433},
    "Norway":           {"off": 2.308, "def": 0.849, "group": "I", "tilt": 0.011},
    # ── Group J: Argentina, Algeria, Austria, Jordan ─────────────────
    "Argentina":        {"off": 2.79, "def": 0.702, "group": "J", "tilt": -0.0305},
    "Algeria":          {"off": 2.005, "def": 0.977, "group": "J", "tilt": -0.017},
    "Austria":          {"off": 2.072, "def": 0.946, "group": "J", "tilt": -0.0028},
    "Jordan":           {"off": 1.853, "def": 1.058, "group": "J", "tilt": 0.0052},
    # ── Group K: Portugal, DR Congo, Uzbekistan, Colombia ───────────
    "Portugal":         {"off": 2.38, "def": 0.823, "group": "K", "tilt": 0.0149},
    "DR Congo":         {"off": 1.841, "def": 1.065, "group": "K", "tilt": -0.072},
    "Uzbekistan":       {"off": 1.945, "def": 1.008, "group": "K", "tilt": -0.0327},
    "Colombia":         {"off": 2.451, "def": 0.8, "group": "K", "tilt": -0.0307},
    # ── Group L: England, Croatia, Ghana, Panama ─────────────────────
    "England":          {"off": 2.584, "def": 0.759, "group": "L", "tilt": -0.0122},
    "Croatia":          {"off": 2.308, "def": 0.849, "group": "L", "tilt": -0.0126},
    "Ghana":            {"off": 1.613, "def": 1.215, "group": "L", "tilt": -0.0319},
    "Panama":           {"off": 1.942, "def": 1.009, "group": "L", "tilt": -0.0099},
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
