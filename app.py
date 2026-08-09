"""
app.py
------
Flask web app around tennis_model.py.

Loads a curated player-stats dataset (data/players.csv) and exposes:
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

from flask import Flask, jsonify, render_template, request

from tennis_model import prob_win_match, serve_probs

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

    a, b = PLAYERS[name_a], PLAYERS[name_b]
    avg_rpw = (tour_avg_return(a["tour"]) + tour_avg_return(b["tour"])) / 2.0

    pA, pB = serve_probs(
        a["serve_pts_won"] / 100.0,
        b["serve_pts_won"] / 100.0,
        rpw_A=a["return_pts_won"] / 100.0,
        rpw_B=b["return_pts_won"] / 100.0,
        tour_avg_rpw=avg_rpw,
    )

    prob_a = prob_win_match(pA, pB, best_of=best_of)

    return jsonify({
        "player_a": a,
        "player_b": b,
        "best_of": best_of,
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
