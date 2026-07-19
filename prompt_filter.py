"""Stage-2 prompt filtering — the user-designed architecture, completed.

Detection ALWAYS runs closed-set (people + vehicles, yolo26l: fastest AND
most stable tracking). Free-text prompts are applied to the RESULTS:

  1. Structured path (deterministic, preferred): the prompt is parsed for a
     class hint and color words — "λευκό αυτοκίνητο" matches tracks with
     vehicle class + attrs.color == λευκό, computed by attributes.py.
  2. Semantic path (fallback for anything else — "αμάξι με σχάρα οροφής"):
     cosine of the track's visual-prompt embedding (already computed for
     stitching) against the text embedding of the prompt, using the same
     YOLOE-nano text encoder. RELATIVE ranking (within 90% of the best
     candidate) — the raw score scale is not calibrated, ranking is robust.

Matched tracks are ANNOTATED, never dropped: an investigative tool must not
hide objects — the report highlights matches and keeps everything visible.
"""

import numpy as np

# Color vocabulary → attributes.py color names (Greek + English prompts).
COLOR_WORDS = {
    'λευκό': 'λευκό', 'λευκη': 'λευκό', 'λευκός': 'λευκό', 'ασπρο': 'λευκό',
    'άσπρο': 'λευκό', 'white': 'λευκό',
    'μαύρο': 'μαύρο', 'μαυρη': 'μαύρο', 'μαύρος': 'μαύρο', 'black': 'μαύρο',
    'γκρι': 'γκρι/ασημί', 'ασημί': 'γκρι/ασημί', 'ασημι': 'γκρι/ασημί',
    'gray': 'γκρι/ασημί', 'grey': 'γκρι/ασημί', 'silver': 'γκρι/ασημί',
    'κόκκινο': 'κόκκινο', 'κοκκινο': 'κόκκινο', 'red': 'κόκκινο',
    'μπλε': 'μπλε', 'blue': 'μπλε', 'γαλάζιο': 'γαλάζιο',
    'πράσινο': 'πράσινο', 'πρασινο': 'πράσινο', 'green': 'πράσινο',
    'κίτρινο': 'κίτρινο', 'yellow': 'κίτρινο',
    'πορτοκαλί': 'πορτοκαλί', 'orange': 'πορτοκαλί',
    'καφέ': 'καφέ', 'brown': 'καφέ', 'μωβ': 'μωβ', 'purple': 'μωβ',
    'ροζ': 'ροζ', 'pink': 'ροζ',
}


def parse_prompt(prompt: str, vehicle_keywords, person_keywords) -> dict:
    low = prompt.lower()
    colors = sorted({v for w, v in COLOR_WORDS.items() if w in low})
    return {
        'prompt': prompt,
        'colors': colors,
        'wants_vehicle': any(k in low for k in vehicle_keywords),
        'wants_person': any(k in low for k in person_keywords),
    }


def structured_match(track: dict, parsed: dict,
                     vehicle_keywords, person_keywords) -> bool | None:
    """True/False when the prompt is decidable from class + color attributes;
    None when it needs the semantic path."""
    cls = track['class'].lower()
    is_vehicle = any(k in cls for k in vehicle_keywords)
    is_person = any(k in cls for k in person_keywords)
    if parsed['wants_vehicle'] and not is_vehicle:
        return False
    if parsed['wants_person'] and not is_person:
        return False
    if not parsed['colors']:
        # Class-only prompt ("αυτοκίνητο") → decided purely by class.
        if parsed['wants_vehicle'] or parsed['wants_person']:
            return True
        return None
    attrs = track.get('attrs') or {}
    track_colors = set()
    if attrs.get('color'):
        track_colors.add(attrs['color'])
    for info in (attrs.get('clothing') or {}).values():
        track_colors.add(info['color'])
    if not track_colors:
        return None  # no attribute evidence — let semantics decide
    return any(c in track_colors for c in parsed['colors'])


def _text_embedding(prompt: str, embed_model_path: str) -> np.ndarray | None:
    try:
        from ultralytics import YOLO
        yolo = YOLO(embed_model_path)
        tpe = yolo.get_text_pe([prompt])
        v = tpe.reshape(-1).float().cpu().numpy()
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    except Exception:  # noqa: BLE001 — semantic path is best-effort
        return None


def apply_prompts(tracks: dict, prompts: list, embed_model_path: str,
                  vehicle_keywords, person_keywords, log=None):
    """Annotate tracks in place: t['prompt_matches'] = [prompt, ...]."""
    for prompt in prompts:
        parsed = parse_prompt(prompt, vehicle_keywords, person_keywords)
        undecided = []
        for tid, t in tracks.items():
            verdict = structured_match(t, parsed, vehicle_keywords,
                                       person_keywords)
            if verdict is True:
                t.setdefault('prompt_matches', []).append(prompt)
            elif verdict is None:
                undecided.append(tid)

        if undecided:
            tpe = _text_embedding(prompt, embed_model_path)
            scored = []
            if tpe is not None:
                for tid in undecided:
                    vpe = tracks[tid].get('_emb_vpe')
                    if vpe is not None and vpe.shape == tpe.shape:
                        scored.append((float(np.dot(vpe, tpe)), tid))
            if scored:
                best = max(s for s, _ in scored)
                # Relative gate: the raw vpe·tpe scale is uncalibrated, but
                # ranking is stable — accept candidates within 90% of the
                # best (and require a sane positive best).
                for s, tid in scored:
                    if best > 0.05 and s >= 0.9 * best:
                        tracks[tid].setdefault('prompt_matches', []).append(prompt)
        if log:
            n = sum(1 for t in tracks.values()
                    if prompt in (t.get('prompt_matches') or []))
            log(f'Prompt "{prompt}": {n}/{len(tracks)} objects match')
