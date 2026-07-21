"""Cross-video object matching — ROADMAP Phase Ε.

Matches the same physical vehicle/person across DIFFERENT videos (multiple
cameras of the same event). Evidence, strongest first:

  1. License plates (vehicles): candidate-list comparison with edit-distance
     tolerance — plates survive viewpoint/lighting changes that break
     appearance features, and our per-char alternatives absorb single-char
     OCR disagreements between cameras.
  2. Appearance (VPE + HSV, same signatures as stitch.py) — cross-camera
     lighting/angle shifts degrade these, so appearance-ONLY matches use a
     stricter threshold and are explicitly labeled low-confidence for the
     human reviewer; the report's manual-pairing mode is the final arbiter.

Clock times are deliberately NOT used as a gate: cameras of one event are
rarely synchronized. (If two videos are the same camera, chain mode is the
right tool instead.)
"""

import itertools

import numpy as np

import stitch as stitch_mod

# Appearance-only threshold across cameras: stricter than the intra-video
# 0.88 because viewpoint/white-balance changes push same-object similarity
# down and different-object similarity up. Field-calibrate over time.
APPEARANCE_XCAM_THRESHOLD = 0.92
# With a plate agreement, appearance only needs to not-contradict.
APPEARANCE_WITH_PLATE_THRESHOLD = 0.80
PLATE_STRONG = 0.85


def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def plate_match_score(pa: dict | None, pb: dict | None) -> float | None:
    """Similarity of two auto-plate results in [0,1], None if either side
    has no plate. Compares the CANDIDATE LISTS (not just the headline): two
    cameras often disagree on one character, and the true plate tends to
    appear in both lists even when the headlines differ."""
    if not pa or not pb:
        return None
    cands_a = [c['plate'] for c in
               (pa.get('gr_candidates') or []) + (pa.get('candidates') or [])]
    cands_b = [c['plate'] for c in
               (pb.get('gr_candidates') or []) + (pb.get('candidates') or [])]
    if not cands_a or not cands_b:
        return None
    best = 0.0
    for ca, cb in itertools.product(cands_a[:4], cands_b[:4]):
        ca_c, cb_c = ca.replace('?', ''), cb.replace('?', '')
        if not ca_c or not cb_c:
            continue
        d = _edit_distance(ca, cb)
        sim = 1.0 - d / max(len(ca), len(cb))
        # Exact top-candidate agreement is the gold signal.
        if ca == cands_a[0] and cb == cands_b[0]:
            sim = min(1.0, sim + 0.1)
        best = max(best, sim)
    return best


def pair_score(ta: dict, tb: dict) -> tuple | None:
    """Score a candidate cross-video pair. Returns (score, evidence) or None
    when gates fail. evidence ∈ {'plate+appearance', 'plate', 'appearance'}."""
    if ta['class'] != tb['class']:
        return None
    app = stitch_mod.similarity(ta, tb)
    plate = plate_match_score(ta.get('plate'), tb.get('plate'))
    if plate is not None and plate >= PLATE_STRONG:
        if app >= APPEARANCE_WITH_PLATE_THRESHOLD:
            return (0.65 * plate + 0.35 * app, 'plate+appearance')
        return (0.6 * plate, 'plate')
    if app >= APPEARANCE_XCAM_THRESHOLD:
        # Low-confidence tier: flagged for the human / manual pairing mode.
        return (0.5 * app, 'appearance')
    return None


def find_reappearances(tracks: dict) -> list:
    """Same-video re-identification: pairs of tracks (same class) whose time
    intervals are DISJOINT — one left the scene, the other appeared later —
    but whose evidence says they may be the same physical object (a vehicle
    that leaves and returns, a person walking back into frame).

    Deliberately an ANNOTATION, never a merge: identity across a long gap is
    a human call (evidentiary caution), so the report links the two cards and
    the investigator judges. Evidence tiers mirror cross-video matching:
    plate-candidate agreement is strong, appearance alone is weak.

    Returns [{'a': earlier_tid, 'b': later_tid, 'gap': sec, 'score', 'evidence'}]
    sorted best-first.
    """
    pairs = []
    for (ia, ta), (ib, tb) in itertools.combinations(list(tracks.items()), 2):
        if ta['class'] != tb['class']:
            continue
        # Real absence gate: overlap means two different objects (the tracker
        # saw both at once); gaps under ~3s are ordinary tracking dropouts —
        # stitching's territory — not a genuine departure+return.
        gap = (max(ta['first_seen'], tb['first_seen'])
               - min(ta['last_seen'], tb['last_seen']))
        if gap < 3.0:
            continue
        # Two static tracks don't "reappear" — a parked car split in two is
        # stitching's job (same-spot merge), not a departure.
        if ta.get('static') and tb.get('static'):
            continue
        scored = pair_score(ta, tb)
        if scored is None:
            continue
        score, evidence = scored
        a, b = (ia, ib) if ta['first_seen'] <= tb['first_seen'] else (ib, ia)
        pairs.append({'a': a, 'b': b, 'gap': round(gap, 1),
                      'score': score, 'evidence': evidence})
    pairs.sort(key=lambda p: -p['score'])
    return pairs


def match_videos(per_video: dict) -> list:
    """per_video: {video_name: tracks_dict (pre-report, with _emb_* intact)}.

    Returns groups: [{'members': [(video, track_id)], 'score', 'evidence'}].
    Greedy best-pair-first with union-find; a group never contains two
    objects from the SAME video (they were already stitched there — two
    same-video tracks are two different physical objects by construction).
    """
    items = [(v, tid, t) for v, tracks in per_video.items()
             for tid, t in tracks.items()]
    idx = {(v, tid): i for i, (v, tid, _) in enumerate(items)}
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    videos_of_group: dict = {i: {items[i][0]} for i in range(len(items))}

    pairs = []
    for (va, ida, ta), (vb, idb, tb) in itertools.combinations(items, 2):
        if va == vb:
            continue
        scored = pair_score(ta, tb)
        if scored:
            pairs.append((scored[0], scored[1], idx[(va, ida)], idx[(vb, idb)]))
    pairs.sort(reverse=True, key=lambda p: p[0])

    evidence_of: dict[int, str] = {}
    score_of: dict[int, float] = {}
    for score, evidence, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # One object appears at most once per video.
        if videos_of_group[ri] & videos_of_group[rj]:
            continue
        parent[rj] = ri
        videos_of_group[ri] |= videos_of_group.pop(rj)
        merged_ev = {evidence_of.get(ri), evidence_of.get(rj), evidence}
        merged_ev.discard(None)
        evidence_of[ri] = ('plate+appearance'
                           if any('plate' in e for e in merged_ev)
                           and any('appearance' in e for e in merged_ev)
                           else merged_ev.pop())
        # pairs are sorted best-first, so the first score a root receives is
        # its strongest link — keep it as the group's headline score.
        score_of.setdefault(ri, score)
        score_of.setdefault(rj, score)

    groups: dict[int, list] = {}
    for i, (v, tid, _t) in enumerate(items):
        groups.setdefault(find(i), []).append((v, tid))
    out = []
    for r, members in groups.items():
        if len(members) < 2:
            continue
        out.append({
            'members': sorted(members),
            'evidence': evidence_of.get(r, 'appearance'),
            'score': round(score_of.get(r, 0.0), 3),
        })
    out.sort(key=lambda g: -g['score'])
    return out


def combined_plate(group_tracks: list, reader) -> dict | None:
    """Re-vote the plate over the UNION of all member snapshots — multiple
    viewpoints break the per-camera systematic blur that single-camera votes
    suffer from (the measured failure mode of confident-wrong reads)."""
    import cv2
    crops = []
    for t in group_tracks:
        for s in t.get('snapshots', [])[:4]:
            img = cv2.imdecode(np.frombuffer(s['jpeg'], np.uint8),
                               cv2.IMREAD_COLOR)
            if img is not None:
                crops.append(img)
    if not crops:
        return None
    return reader.read_from_crops(crops)
