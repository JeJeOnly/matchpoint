"""
live_stats.py
--------------
Live ATP/WTA player data via the "TennisApi" RapidAPI product
(tennisapi1.p.rapidapi.com).

There's no single "season serve/return %" endpoint on this API, so for a
player it finds, get_live_stats() pulls their last several FINISHED matches
and aggregates the real point-by-point service/return counts into the same
serve_pts_won / return_pts_won percentages tennis_model.py expects --
computed fresh from real results, not looked up from a static table.

Players the API doesn't recognize (retired legends, obscure names) or
matches with too little data return None; the caller falls back to the
curated CSV. Results are cached in memory (a few hours) since these
percentages don't meaningfully change point-to-point, and the slow-moving
name -> player-id mapping is cached to disk so it survives restarts.

Requires RAPIDAPI_KEY in the environment (see .env, gitignored). With no
key set, every call is a no-op (returns None) and the app runs on the CSV
alone.
"""

import json
import os
import re
import time

import requests

HOST = "tennisapi1.p.rapidapi.com"
BASE_URL = f"https://{HOST}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
ID_CACHE_PATH = os.path.join(CACHE_DIR, "player_ids.json")

STATS_TTL_SECONDS = 6 * 60 * 60  # refresh a player's live aggregate every few hours
MATCHES_TO_SAMPLE = 8            # how many recent finished matches to aggregate
MIN_SERVICE_POINTS = 60          # need a decent sample before trusting the aggregate
MIN_MATCHES = 3
MAX_MATCH_AGE_DAYS = 270         # ignore matches older than this -- an old/retired
                                  # player's last-ever match shouldn't pass as "current form"
REQUEST_TIMEOUT = 6

UNAVAILABLE = object()  # sentinel: the API call itself failed (network/quota/etc.),
                         # as opposed to succeeding and confirming "nothing here" --
                         # only the latter is safe to cache as a negative result

_stats_cache = {}   # player_id -> {"expires": epoch, "data": {...} or None}
_id_cache = None    # lower(name) -> player_id or None, lazy-loaded from disk


def _headers():
    key = os.environ.get("RAPIDAPI_KEY")
    if not key:
        return None
    return {"x-rapidapi-host": HOST, "x-rapidapi-key": key}


def _get(path):
    headers = _headers()
    if not headers:
        return None
    try:
        resp = requests.get(f"{BASE_URL}{path}", headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def _load_id_cache():
    global _id_cache
    if _id_cache is not None:
        return _id_cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(ID_CACHE_PATH):
        try:
            with open(ID_CACHE_PATH, encoding="utf-8") as f:
                _id_cache = json.load(f)
                return _id_cache
        except (json.JSONDecodeError, OSError):
            pass
    _id_cache = {}
    return _id_cache


def _save_id_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(ID_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_id_cache, f)


def _resolve_player_id(name):
    """Player id for `name`, or None if the API genuinely doesn't have them.
    Only a confirmed (successful) search response gets cached to disk --
    a failed request (quota exhausted, network blip, etc.) must NOT be
    written down as "not found", or an outage would permanently blacklist
    every player looked up during it."""
    cache = _load_id_cache()
    key = name.strip().lower()
    if key in cache:
        return cache[key]

    data = _get(f"/api/tennis/search/{name}")
    if data is None:
        return None  # request failed -- leave unresolved, retry next time

    candidates = [r["entity"] for r in data.get("results", []) if "entity" in r]
    match = next((c for c in candidates if c.get("name", "").strip().lower() == key), None)
    if not match and candidates:
        match = candidates[0]

    player_id = match.get("id") if match else None
    cache[key] = player_id
    _save_id_cache()
    return player_id


def _parse_fraction(text):
    """'22/35 (63%)' -> (22, 35). None if the string doesn't have that shape."""
    if not text:
        return None
    m = re.search(r"(\d+)\s*/\s*(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _side_of(event, player_id):
    if event.get("homeTeam", {}).get("id") == player_id:
        return "home"
    if event.get("awayTeam", {}).get("id") == player_id:
        return "away"
    return None


def _extract_match_points(stats_json, side):
    """(serve_won, serve_played, return_won, return_played) for `side` in one
    match's /event/{id}/statistics response, or None if the shape is missing."""
    periods = stats_json.get("statistics", [])
    all_period = next((p for p in periods if p.get("period") == "ALL"), None)
    if not all_period:
        return None

    groups = {g.get("groupName"): g for g in all_period.get("groups", [])}
    service, ret = groups.get("Service"), groups.get("Return")
    if not service or not ret:
        return None

    def find(group, item_name):
        item = next((i for i in group.get("statisticsItems", []) if i.get("name") == item_name), None)
        return _parse_fraction(item.get(side)) if item else None

    first_serve = find(service, "First serve points")
    second_serve = find(service, "Second serve points")
    first_return = find(ret, "First serve return points")
    second_return = find(ret, "Second serve return points")
    if not (first_serve and second_serve and first_return and second_return):
        return None

    return (
        first_serve[0] + second_serve[0], first_serve[1] + second_serve[1],
        first_return[0] + second_return[0], first_return[1] + second_return[1],
    )


def _aggregate_recent_form(player_id):
    """Aggregated recent-form dict, UNAVAILABLE if the API call itself
    failed (so the caller knows not to cache a false negative), or None if
    it succeeded but there isn't enough recent data to trust (e.g. a
    retired player whose last matches are years old)."""
    events_data = _get(f"/api/tennis/player/{player_id}/events/previous/0")
    if events_data is None:
        return UNAVAILABLE

    cutoff = time.time() - MAX_MATCH_AGE_DAYS * 86400
    finished = [
        e for e in events_data.get("events", [])
        if e.get("status", {}).get("type") == "finished" and e.get("startTimestamp", 0) >= cutoff
    ]
    finished.sort(key=lambda e: e.get("startTimestamp", 0), reverse=True)

    serve_won = serve_played = return_won = return_played = 0
    matches_used = 0

    for event in finished[:MATCHES_TO_SAMPLE]:
        side = _side_of(event, player_id)
        if not side:
            continue
        stats = _get(f"/api/tennis/event/{event['id']}/statistics")
        if not stats:
            continue
        parsed = _extract_match_points(stats, side)
        if not parsed:
            continue
        sw, sp, rw, rp = parsed
        serve_won += sw
        serve_played += sp
        return_won += rw
        return_played += rp
        matches_used += 1

    if matches_used < MIN_MATCHES or serve_played < MIN_SERVICE_POINTS or return_played < MIN_SERVICE_POINTS:
        return None

    return {
        "serve_pts_won": round(100 * serve_won / serve_played, 1),
        "return_pts_won": round(100 * return_won / return_played, 1),
        "matches_used": matches_used,
    }


def _fetch_profile_bio(player_id):
    data = _get(f"/api/tennis/player/{player_id}")
    team = (data or {}).get("team", {})
    info = team.get("playerTeamInfo", {})
    country = team.get("country") or {}
    plays = info.get("plays", "")
    bio = {}
    if country.get("alpha2"):
        bio["country_code"] = country["alpha2"]
    if plays:
        bio["hand"] = "L" if plays.lower().startswith("left") else "R"
    if info.get("currentRanking"):
        bio["current_ranking"] = info["currentRanking"]
    return bio or None


def get_live_stats(name):
    """Live (serve_pts_won, return_pts_won, bio, matches_used) for a player
    name, or None if the API doesn't have them (not found, retired, or too
    little recent-match data to trust). Cached a few hours per player."""
    if not os.environ.get("RAPIDAPI_KEY"):
        return None

    player_id = _resolve_player_id(name)
    if player_id is None:
        return None

    cached = _stats_cache.get(player_id)
    if cached and cached["expires"] > time.time():
        return cached["data"]

    form = _aggregate_recent_form(player_id)
    if form is UNAVAILABLE:
        return None  # API call failed -- don't cache a false negative, just retry next time

    result = None
    if form:
        bio = _fetch_profile_bio(player_id) or {}
        result = {**form, **bio}

    _stats_cache[player_id] = {"expires": time.time() + STATS_TTL_SECONDS, "data": result}
    return result
