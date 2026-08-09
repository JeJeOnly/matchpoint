"""
tennis_model.py
----------------
Real-time tennis win-probability engine.

Core idea: tennis is hierarchical (points -> games -> sets -> match).
If you know each player's probability of winning a single point ON THEIR
OWN SERVE, you can compute the EXACT probability of winning the match from
ANY score state, just by recursion. Re-run it after every point and you get
a live win-probability that updates in real time.

Convention: every function returns the probability that PLAYER A wins.
    pA = P(A wins a point when A is serving)
    pB = P(B wins a point when B is serving)

No external libraries required. Just run it.
"""

from functools import lru_cache


# ---------------------------------------------------------------------------
# 1. GAME  -- first to 4 points, win by 2 (0/15/30/40 counted as 0/1/2/3/4)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def prob_win_game(p, a=0, b=0):
    """P(the SERVER wins the game), given the server currently has `a` points,
    the returner has `b`, and the server wins each point with probability `p`."""
    if a >= 4 and a - b >= 2:          # server already won
        return 1.0
    if b >= 4 and b - a >= 2:          # returner already won
        return 0.0
    # Deuce region (both >= 3): collapse it with the closed-form deuce solution
    # so the recursion always terminates (deuce can otherwise go forever).
    if a >= 3 and b >= 3:
        d = p * p / (p * p + (1 - p) * (1 - p))   # P(win from deuce)
        if a == b:                     # deuce (40-40)
            return d
        elif a > b:                    # advantage server (AD-in)
            return p + (1 - p) * d
        else:                          # advantage returner (AD-out)
            return p * d
    # Otherwise, play the next point.
    return p * prob_win_game(p, a + 1, b) + (1 - p) * prob_win_game(p, a, b + 1)


# ---------------------------------------------------------------------------
# 2. TIEBREAK  -- first to 7 points, win by 2, 1-2-2-2 serve rotation
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def prob_win_tiebreak(pA, pB, first, a=0, b=0):
    """P(A wins the tiebreak). `first` ('A' or 'B') = who serves the first
    point of the tiebreak. `a`, `b` = A's and B's tiebreak points."""
    if a >= 7 and a - b >= 2:
        return 1.0
    if b >= 7 and b - a >= 2:
        return 0.0
    # Deuce tail (>= 6 each): approximate the remainder with an effective
    # per-point win prob averaged over the serve rotation. Tiebreaks rarely
    # reach here, so the error is negligible. (Refine later if you want.)
    if a >= 6 and b >= 6:
        q = 0.5 * pA + 0.5 * (1 - pB)
        d = q * q / (q * q + (1 - q) * (1 - q))
        if a == b:
            return d
        elif a > b:
            return q + (1 - q) * d
        else:
            return q * d
    # Who serves point number n = a + b?  1-2-2-2... rotation:
    # ((n + 1) // 2) % 2 == 0  ->  the player who served first.
    n = a + b
    server_is_first = (((n + 1) // 2) % 2 == 0)
    current_server = first if server_is_first else ('B' if first == 'A' else 'A')
    p_point_A = pA if current_server == 'A' else (1 - pB)
    return (p_point_A * prob_win_tiebreak(pA, pB, first, a + 1, b)
            + (1 - p_point_A) * prob_win_tiebreak(pA, pB, first, a, b + 1))


# ---------------------------------------------------------------------------
# 3. SET  -- first to 6 games, win by 2; 6-6 -> tiebreak; 7-5 ends it
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def prob_win_set(pA, pB, gA=0, gB=0, server='A'):
    """P(A wins the set). `server` = who serves the current game."""
    if gA == 6 and gB <= 4:
        return 1.0
    if gB == 6 and gA <= 4:
        return 0.0
    if gA == 7:                        # reached only as 7-5
        return 1.0
    if gB == 7:
        return 0.0
    if gA == 6 and gB == 6:            # tiebreak; the due server serves first
        return prob_win_tiebreak(pA, pB, server)
    # Probability A wins THIS game:
    if server == 'A':
        pg = prob_win_game(pA)         # A holds serve
        nxt = 'B'
    else:
        pg = 1 - prob_win_game(pB)     # A breaks B's serve
        nxt = 'A'
    return (pg * prob_win_set(pA, pB, gA + 1, gB, nxt)
            + (1 - pg) * prob_win_set(pA, pB, gA, gB + 1, nxt))


# ---------------------------------------------------------------------------
# 4. MATCH  -- best-of-3 or best-of-5 sets
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def prob_win_match(pA, pB, best_of=3, sA=0, sB=0, first_server='A'):
    """P(A wins the match). `best_of` is 3 or 5."""
    need = best_of // 2 + 1
    if sA == need:
        return 1.0
    if sB == need:
        return 0.0
    ps = prob_win_set(pA, pB, server=first_server)
    nxt = 'B' if first_server == 'A' else 'A'   # alternate first server each set
    return (ps * prob_win_match(pA, pB, best_of, sA + 1, sB, nxt)
            + (1 - ps) * prob_win_match(pA, pB, best_of, sA, sB + 1, nxt))


# ---------------------------------------------------------------------------
# 5. LIVE ENGINE  -- win probability from ANY mid-match state
#    Call this after every point with the updated score.  THIS is what your
#    real-time app calls on each incoming point event.
# ---------------------------------------------------------------------------
def live_match_win_prob(pA, pB, *, best_of=3,
                        sets_A=0, sets_B=0,
                        games_A=0, games_B=0,
                        points_A=0, points_B=0,
                        server='A', next_set_first='A'):
    """P(A wins the match) from a full mid-match state."""

    # If we're at 6-6, the "current game" is actually a tiebreak.
    if games_A == 6 and games_B == 6:
        p_set = prob_win_tiebreak(pA, pB, server, points_A, points_B)
    else:
        # Finish the current (normal) game from its current point score.
        if server == 'A':
            pg = prob_win_game(pA, points_A, points_B)       # A serving
            nxt = 'B'
        else:
            pg = 1 - prob_win_game(pB, points_B, points_A)   # B serving
            nxt = 'A'
        # P(A wins current SET), folding in the half-played game.
        p_set = (pg * prob_win_set(pA, pB, games_A + 1, games_B, nxt)
                 + (1 - pg) * prob_win_set(pA, pB, games_A, games_B + 1, nxt))

    # P(A wins MATCH), folding in the current-set result.
    p_match = (p_set * prob_win_match(pA, pB, best_of, sets_A + 1, sets_B, next_set_first)
               + (1 - p_set) * prob_win_match(pA, pB, best_of, sets_A, sets_B + 1, next_set_first))
    return p_match


# ---------------------------------------------------------------------------
# 6. TURNING SERVE STATS INTO pA / pB
# ---------------------------------------------------------------------------
def serve_probs(spw_A, spw_B, rpw_A=None, rpw_B=None, tour_avg_rpw=0.35):
    """Convert 'service points won' rates into the two model inputs.

    Simple baseline: use each player's own service-points-won rate.
    Opponent-adjusted (if you pass return stats): a player serves a bit worse
    against a better returner. This is the classic 'serve minus opponent's
    return edge' adjustment. Replace with a fitted model when you have data.
    """
    pA, pB = spw_A, spw_B
    if rpw_B is not None:
        pA = spw_A - (rpw_B - tour_avg_rpw)   # A's serve vs B's return
    if rpw_A is not None:
        pB = spw_B - (rpw_A - tour_avg_rpw)   # B's serve vs A's return
    # keep probabilities sane
    return min(max(pA, 0.01), 0.99), min(max(pB, 0.01), 0.99)


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Per-point serve-win probabilities. A strong server wins ~65% on serve.
    pA = 0.66   # A wins 66% of points on A's serve
    pB = 0.62   # B wins 62% of points on B's serve

    print("=== Pre-match ===")
    print(f"P(A wins best-of-3): {prob_win_match(pA, pB, best_of=3):.3f}")
    print(f"P(A wins best-of-5): {prob_win_match(pA, pB, best_of=5):.3f}")

    print("\n=== Live updates (best-of-3) ===")

    # A is up a break in set 1: leads 3-2, serving, 40-15.
    p = live_match_win_prob(pA, pB, best_of=3,
                            games_A=3, games_B=2,
                            points_A=3, points_B=1, server='A')
    print(f"A leads 3-2, serving 40-15:            {p:.3f}")

    # A dropped serve back: 3-3, B serving, 40-0 to B.
    p = live_match_win_prob(pA, pB, best_of=3,
                            games_A=3, games_B=3,
                            points_A=0, points_B=3, server='B')
    print(f"3-3, B serving 40-0:                   {p:.3f}")

    # A won set 1; set 2 tiebreak, A ahead 5-3, A served first in the tiebreak.
    p = live_match_win_prob(pA, pB, best_of=3,
                            sets_A=1, games_A=6, games_B=6,
                            points_A=5, points_B=3, server='A')
    print(f"A won set 1; set-2 tiebreak 5-3 (A):   {p:.3f}")

    # Match point for A: up a set and 5-4, serving, 40-30.
    p = live_match_win_prob(pA, pB, best_of=3,
                            sets_A=1, games_A=5, games_B=4,
                            points_A=3, points_B=2, server='A')
    print(f"A up a set, serving 5-4 40-30:         {p:.3f}")
