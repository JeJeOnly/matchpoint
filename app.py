"""
app.py
------
Flask web app around tennis_model.py.

Loads a curated player-stats dataset (data/players.csv) as the baseline,
overlaid with live per-player stats from live_stats.py when RAPIDAPI_KEY is
set and the player has recent match data. Exposes:
  GET  /                -> the UI
  GET  /api/players      -> list of players for the pickers
  POST /api/predict      -> win-probability prediction for two players

Run locally:
    .venv\\Scripts\\python.exe -m pip install -r requirements.txt
    .venv\\Scripts\\python.exe app.py
Then open http://127.0.0.1:5000
"""

import csv
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

import live_stats
from tennis_model import prob_win_match, serve_probs

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYERS_CSV = os.path.join(BASE_DIR, "data", "players.csv")
H2H_CSV = os.path.join(BASE_DIR, "data", "head_to_head.csv")

app = Flask(__name__)


def load_players():
    players = {}
    with open(PLAYERS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            players[row["name"]] = {
                "name": row["name"],
                "tour": row["tour"],
                "serve_pts_won": float(row["serve_pts_won"]),
                "return_pts_won": float(row["return_pts_won"]),
                "country_code": row["country_code"],
                "hand": row["hand"],
                "style": row["style"],
            }
    return players


def load_head_to_head():
    """Key: frozenset of the two names -> (name_a, name_b, wins_a, wins_b) as recorded."""
    h2h = {}
    with open(H2H_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = frozenset((row["player_a"], row["player_b"]))
            h2h[key] = (row["player_a"], row["player_b"], int(row["wins_a"]), int(row["wins_b"]))
    return h2h


PLAYERS = load_players()
HEAD_TO_HEAD = load_head_to_head()

# ---------------------------------------------------------------------------
# Surface model
# ---------------------------------------------------------------------------
# Hard court is treated as the neutral baseline (it's the tour's most common
# surface, so each player's base serve/return numbers already skew hard-court-
# like). Grass and clay get a (serve_delta, return_delta) in percentage points,
# modeled from playing style -- NOT measured per-surface stats, since we don't
# have real surface splits for this dataset. First keyword match wins.
SURFACE_ARCHETYPES = [
    ("clay", ("clay",),
        {"grass": (-1.5, -1.0), "clay": (2.0, 1.5)}),
    ("fast_surface", ("serve", "server", "flat", "hitter"),
        {"grass": (2.0, -0.8), "clay": (-2.0, 0.8)}),
    ("grinder", ("grinder", "counter-punch", "defensive", "speedster", "crafty", "relentless", "veteran"),
        {"grass": (-1.0, -0.5), "clay": (0.5, 1.0)}),
    ("all_court", ("all-court", "versatile", "elegant", "dynamo", "tactician", "compact"),
        {"grass": (1.0, 0.3), "clay": (-0.3, 0.0)}),
    ("one_handed", ("one-handed",),
        {"grass": (0.5, 0.3), "clay": (-0.3, 0.0)}),
    ("topspin", ("topspin", "power", "ball-striker"),
        {"grass": (-0.8, -0.3), "clay": (1.2, 0.5)}),
    ("flair", ("shotmaker", "flair", "showman", "unpredictable", "flashy"),
        {"grass": (0.7, 0.2), "clay": (-0.5, 0.0)}),
]

# A handful of players whose real-world surface dominance (or struggles) is
# unmistakable enough to name explicitly, rather than relying on the generic
# style heuristic above. Overrides replace the heuristic entirely for that
# player. (serve_delta, return_delta) per surface, in percentage points.
SURFACE_OVERRIDES = {
    "Rafael Nadal": {"grass": (-3.0, -2.0), "clay": (6.0, 4.0)},
    "Roger Federer": {"grass": (4.0, 1.0), "clay": (-2.0, -1.0)},
    "Pete Sampras": {"grass": (5.0, 1.0), "clay": (-3.0, -1.5)},
    "John Isner": {"grass": (3.5, -1.5), "clay": (-4.0, 1.0)},
    "Reilly Opelka": {"grass": (3.5, -1.5), "clay": (-4.0, 1.0)},
    "Nick Kyrgios": {"grass": (2.5, -1.0), "clay": (-2.5, 0.5)},
    "Casper Ruud": {"grass": (-2.5, -1.5), "clay": (3.5, 2.0)},
    "Diego Schwartzman": {"grass": (-2.5, -1.5), "clay": (3.5, 2.5)},
    "Alexander Zverev": {"grass": (-1.0, -0.5), "clay": (2.0, 1.0)},
    "Stefanos Tsitsipas": {"grass": (0.0, 0.0), "clay": (2.0, 1.0)},
    "Iga Swiatek": {"grass": (-2.0, -1.5), "clay": (4.0, 3.0)},
    "Elena Rybakina": {"grass": (3.0, -1.0), "clay": (-2.0, 1.0)},
    "Petra Kvitova": {"grass": (3.0, -0.5), "clay": (-1.5, 0.5)},
    "Ashleigh Barty": {"grass": (2.5, 1.0), "clay": (0.5, 0.5)},
    "Serena Williams": {"grass": (2.5, 0.0), "clay": (-1.0, 0.0)},
    "Venus Williams": {"grass": (2.5, 0.0), "clay": (-1.0, 0.0)},
    "Justine Henin": {"grass": (-0.5, 0.0), "clay": (3.0, 2.0)},
    "Maria Sharapova": {"grass": (0.5, 0.0), "clay": (1.5, 0.5)},
    "Simona Halep": {"grass": (0.0, 0.5), "clay": (2.0, 1.5)},
}


def surface_delta(player, surface):
    """(serve_delta, return_delta) in percentage points for this player on
    this surface, relative to their hard-court baseline. Hard is always
    (0, 0)."""
    if surface == "hard":
        return (0.0, 0.0)
    override = SURFACE_OVERRIDES.get(player["name"])
    if override:
        return override[surface]
    style = player["style"].lower()
    for _label, keywords, deltas in SURFACE_ARCHETYPES:
        if any(kw in style for kw in keywords):
            return deltas[surface]
    return (0.0, 0.0)


def apply_surface(player, surface):
    """Player dict with serve_pts_won/return_pts_won adjusted for surface."""
    serve_delta, return_delta = surface_delta(player, surface)
    if serve_delta == 0.0 and return_delta == 0.0:
        return player
    adjusted = dict(player)
    adjusted["serve_pts_won"] = min(max(player["serve_pts_won"] + serve_delta, 1.0), 99.0)
    adjusted["return_pts_won"] = min(max(player["return_pts_won"] + return_delta, 1.0), 99.0)
    return adjusted


def resolve_player(name):
    """CSV record for `name`, overlaid with live stats when the API has
    enough recent-match data to trust (see live_stats.get_live_stats).
    Retired players and anyone the live API doesn't recognize keep the
    curated CSV numbers -- 'estimated' data_source, same as before."""
    player = dict(PLAYERS[name])
    live = live_stats.get_live_stats(name)
    if live:
        player["serve_pts_won"] = live["serve_pts_won"]
        player["return_pts_won"] = live["return_pts_won"]
        player["country_code"] = live.get("country_code", player["country_code"])
        player["hand"] = live.get("hand", player["hand"])
        player["data_source"] = "live"
        player["live_matches_used"] = live["matches_used"]
    else:
        player["data_source"] = "estimated"
    return player


def lookup_h2h(name_a, name_b):
    entry = HEAD_TO_HEAD.get(frozenset((name_a, name_b)))
    if not entry:
        return None
    rec_a, _rec_b, wins_rec_a, wins_rec_b = entry
    if rec_a == name_a:
        return {"wins_a": wins_rec_a, "wins_b": wins_rec_b}
    return {"wins_a": wins_rec_b, "wins_b": wins_rec_a}


def tour_avg_return(tour):
    vals = [p["return_pts_won"] for p in PLAYERS.values() if p["tour"] == tour]
    return (sum(vals) / len(vals)) / 100.0 if vals else 0.35


def point_win_probs(a, b):
    """Per-point serve-win probabilities for two player records, opponent-adjusted."""
    avg_rpw = (tour_avg_return(a["tour"]) + tour_avg_return(b["tour"])) / 2.0
    return serve_probs(
        a["serve_pts_won"] / 100.0,
        b["serve_pts_won"] / 100.0,
        rpw_A=a["return_pts_won"] / 100.0,
        rpw_B=b["return_pts_won"] / 100.0,
        tour_avg_rpw=avg_rpw,
    )


SURFACES = ("hard", "clay", "grass")


def parse_surface(body):
    surface = body.get("surface", "hard")
    return surface if surface in SURFACES else "hard"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/players")
def api_players():
    players = sorted(PLAYERS.values(), key=lambda p: p["name"])
    return jsonify(players)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True) or {}
    name_a = body.get("player_a", "").strip()
    name_b = body.get("player_b", "").strip()
    best_of = int(body.get("best_of", 3))

    if name_a not in PLAYERS or name_b not in PLAYERS:
        return jsonify({"error": "Unknown player. Pick from the suggested list."}), 400
    if name_a == name_b:
        return jsonify({"error": "Choose two different players."}), 400
    if best_of not in (3, 5):
        best_of = 3
    surface = parse_surface(body)

    a = apply_surface(resolve_player(name_a), surface)
    b = apply_surface(resolve_player(name_b), surface)
    pA, pB = point_win_probs(a, b)
    prob_a = prob_win_match(pA, pB, best_of=best_of)

    return jsonify({
        "player_a": a,
        "player_b": b,
        "best_of": best_of,
        "surface": surface,
        "prob_a": prob_a,
        "prob_b": 1 - prob_a,
        "point_win_pct_a": pA,
        "point_win_pct_b": pB,
        "head_to_head": lookup_h2h(name_a, name_b),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
