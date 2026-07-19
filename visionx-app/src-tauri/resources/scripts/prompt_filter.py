"""Structured result filters — user decision: free text is ABOLISHED.

The investigator picks from FIXED, reliable options — specific colors and
specific vehicle/person types — and the filters are applied to the results
deterministically (detector class + computed color attributes). No semantic
guessing, no uncalibrated similarity scores, nothing that can silently
mis-fire. Matches are ANNOTATED, never hidden.
"""

# Canonical color names = attributes.py vocabulary.
FILTER_COLORS = ['λευκό', 'μαύρο', 'γκρι/ασημί', 'κόκκινο', 'μπλε', 'γαλάζιο',
                 'πράσινο', 'κίτρινο', 'πορτοκαλί', 'καφέ', 'μωβ', 'ροζ']

# Selectable types → detector class names (closed-set COCO subset).
TYPE_MAP = {
    'car': 'car', 'αυτοκίνητο': 'car',
    'motorcycle': 'motorcycle', 'μοτοσικλέτα': 'motorcycle',
    'truck': 'truck', 'φορτηγό': 'truck',
    'bus': 'bus', 'λεωφορείο': 'bus',
    'bicycle': 'bicycle', 'ποδήλατο': 'bicycle',
    'person': 'person', 'άτομο': 'person',
}
TYPE_LABEL_EL = {'car': 'αυτοκίνητο', 'motorcycle': 'μοτοσικλέτα',
                 'truck': 'φορτηγό', 'bus': 'λεωφορείο',
                 'bicycle': 'ποδήλατο', 'person': 'άτομο'}


def _track_colors(track: dict) -> set:
    attrs = track.get('attrs') or {}
    colors = set()
    if attrs.get('color'):
        colors.add(attrs['color'])
    for info in (attrs.get('clothing') or {}).values():
        colors.add(info['color'])
    return colors


def apply_filters(tracks: dict, colors: list | None, types: list | None,
                  log=None) -> None:
    """Annotate tracks matching the active criteria (in place).

    Semantics: a track matches when its TYPE is among the selected types
    (or none selected) AND one of its COLORS is among the selected colors
    (or none selected). Matched tracks get 'prompt_matches' labels — the
    report renders them as 🔍 chips (field name kept for report compat)."""
    want_types = {TYPE_MAP[t.lower()] for t in (types or [])
                  if t.lower() in TYPE_MAP}
    want_colors = {c for c in (colors or []) if c in FILTER_COLORS}
    if not want_types and not want_colors:
        return
    matched = 0
    for t in tracks.values():
        cls = t['class'].lower()
        type_ok = not want_types or cls in want_types
        track_colors = _track_colors(t)
        color_ok = not want_colors or bool(track_colors & want_colors)
        if type_ok and color_ok:
            labels = []
            if want_colors:
                labels += sorted(track_colors & want_colors)
            if want_types and cls in want_types:
                labels.append(TYPE_LABEL_EL.get(cls, cls))
            t['prompt_matches'] = [' '.join(labels) or 'κριτήρια']
            matched += 1
    if log:
        log(f'Filters (colors={sorted(want_colors)}, '
            f'types={sorted(want_types)}): {matched}/{len(tracks)} match')
