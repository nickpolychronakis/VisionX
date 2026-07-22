"""VisionX HTML Report Generator"""

import html as html_mod
import json as json_mod
from pathlib import Path

try:
    from attributes import CSS_COLOR
except Exception:  # noqa: BLE001 — report must render even without the module
    CSS_COLOR = {}


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_report(tracks: dict, output_dir: str, video_name: str, video_path: str,
                    video_paths: dict | None = None) -> str:
    """Generate standalone HTML report with embedded thumbnails.

    video_paths: optional {interval filename -> absolute video path} so a
    COMBINED report (multiple source videos) can play each interval from the
    right file; defaults to the single video_path for per-video reports."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # file:// URIs for the in-report clip player (▶ per interval)
    def _uri(p):
        try:
            return Path(p).resolve().as_uri()
        except Exception:  # noqa: BLE001
            return ''
    uri_map = {}
    if video_paths:
        uri_map = {name: _uri(p) for name, p in video_paths.items()}
    default_uri = _uri(video_path) if video_path else ''

    # Playback overlay payload: per-track downsampled boxes (+ relative
    # plate box) so the clip player can draw the vehicle & plate during
    # replay — the live-preview experience, but for ONE object.
    playback_map = {}
    for track_id, track in tracks.items():
        pb = track.get('playback')
        if pb and pb.get('boxes'):
            entry = {'fps': pb['fps'], 'boxes': pb['boxes']}
            rel = (track.get('plate') or {}).get('rel_box')
            if rel:
                entry['plate'] = [round(float(v), 4) for v in rel]
            playback_map[str(track_id)] = entry

    # Generate detection cards HTML
    cards_html = []
    for track_id, track in sorted(tracks.items(), key=lambda x: x[1]['first_seen']):
        first_ts = format_timestamp(track['first_seen'])
        last_ts = format_timestamp(track['last_seen'])

        # Dwell time
        dwell = track.get('dwell_time', 0)
        if dwell >= 3600:
            dwell_str = f"{int(dwell // 3600)}h {int((dwell % 3600) // 60)}m"
        elif dwell >= 60:
            dwell_str = f"{int(dwell // 60)}m {int(dwell % 60)}s"
        else:
            dwell_str = f"{int(dwell)}s"

        # Direction arrow
        direction = track.get('direction', '●')

        # HTML-escape class names: they come from user --search prompts, so
        # they must never be injected raw into markup/JS (Stage 0 bug fix).
        cls_esc = html_mod.escape(track['class'], quote=True)
        title_esc = html_mod.escape(track['class'].upper(), quote=True)

        # Appearance intervals (post-stitching an object may have several:
        # occlusions / re-entries). Fallback to first/last for old callers.
        intervals = track.get('intervals') or [
            {'start': track['first_seen'], 'end': track['last_seen'],
             'file': track.get('first_seen_file')}]
        interval_rows = []
        for iv in intervals[:8]:
            s_ts, e_ts = format_timestamp(iv['start']), format_timestamp(iv['end'])
            file_prefix = f"{html_mod.escape(str(iv['file']))} @ " if iv.get('file') else ''
            play_uri = uri_map.get(iv.get('file'), default_uri)
            play_btn = ''
            if play_uri:
                # In-place clip replay (user request: research without
                # copy-pasting timestamps into a player). 2s lead-in.
                play_btn = (f'<span class="playbtn" '
                            f'onclick="openClip(\'{play_uri}\', '
                            f'{max(0.0, iv["start"] - 2):.1f}, {iv["end"] + 2:.1f}, '
                            f'\'{track_id}\')" '
                            f'title="Αναπαραγωγή αποσπάσματος με πλαίσια">&#9654;</span>')
            interval_rows.append(
                f'<span class="ts" onclick="copyTimestamp(\'{s_ts}\')" '
                f'title="Click to copy start timestamp">'
                f'{file_prefix}{s_ts} &ndash; {e_ts}</span>{play_btn}')
        if len(intervals) > 8:
            interval_rows.append(f'<span class="ts-more">+{len(intervals) - 8} more intervals</span>')

        # Badges: parked/static objects and stitched (merged) identities.
        badges = ''
        if track.get('host_vehicle'):
            badges += ('<span class="badge host" title="Το καπό του οχήματος '
                       'που φέρει την κάμερα (dashcam) — όχι ξεχωριστό '
                       'όχημα">ΟΧΗΜΑ ΛΗΨΗΣ</span>')
        if track.get('static'):
            badges += '<span class="badge parked" title="Stationary for its whole presence">PARKED</span>'
        merged_n = len(track.get('merged_from', []) or [])
        if merged_n > 1:
            badges += (f'<span class="badge merged" title="Automatically re-joined '
                       f'from {merged_n} track fragments">×{merged_n} stitched</span>')

        # Auto-read plate chip (vehicles): headline candidate + score; the
        # tooltip carries the alternative candidates, a click copies the
        # plate text, and the ⌕ chip copies a ready-made deep-analysis
        # command for the interactive plate.py tool aimed at this vehicle.
        plate_html = ''
        p = track.get('plate')
        if p:
            alts = ', '.join(f"{c['plate']} ({c['score']:.0%})"
                             for c in (p.get('candidates') or [])[:3])
            tip = html_mod.escape(
                f"Candidates (for DB search, not evidentiary): {alts}"
                f" — from {p.get('frames_used', 0)} frames, "
                f"plate ~{p.get('plate_px', '?')}px wide. Click to copy.",
                quote=True)
            plate_txt = html_mod.escape(p['plate'], quote=True)
            # The percentage is VOTE CONSENSUS, not correctness probability —
            # a tiny blurred plate can vote unanimously for a wrong read
            # (verified against a known ground-truth plate). Low-reliability
            # reads render in warning style with a ≈ prefix and alternatives
            # spelled out right below, not hidden in a tooltip.
            unreliable = p.get('low_conf')
            chip_cls = 'platechip uncertain' if unreliable else 'platechip'
            prefix = '&asymp; ' if unreliable else ''
            plate_html = (f'<div class="{chip_cls}" title="{tip}" '
                          f'onclick="copyTimestamp(\'{plate_txt}\')">'
                          f'\U0001F698 {prefix}{plate_txt} '
                          f'<span class="pscore">{p["score"]:.0%} consensus</span></div>')
            if unreliable and alts:
                plate_html += (f'<div class="plate-alts">uncertain read — '
                               f'alternatives: {html_mod.escape(alts)}</div>')
            if p.get('deep_cmd'):
                cmd = html_mod.escape(p['deep_cmd'], quote=True)
                plate_html += (f'<span class="deepchip" title="Copy the deep '
                               f'plate-analysis command (interactive tool) '
                               f'for this vehicle" '
                               f'onclick="copyTimestamp(\'{cmd}\')">&#8981;</span>')

        # Color attribute chips (vehicle body / person clothing) — computed
        # on the results, classical HSV voting (see attributes.py).
        attrs_html = ''
        a = track.get('attrs') or {}

        def _color_chip(label, name):
            dot = CSS_COLOR.get(name, '#888')
            return (f'<span class="attr"><i style="background:{dot}"></i>'
                    f'{html_mod.escape(label + name)}</span>')
        if a.get('color'):
            attrs_html += _color_chip('', a['color'])
        for part, label in (('upper', 'πάνω: '), ('lower', 'κάτω: ')):
            info = (a.get('clothing') or {}).get(part)
            if info:
                attrs_html += _color_chip(label, info['color'])
        for pm in track.get('prompt_matches') or []:
            attrs_html += (f'<span class="attr promptmatch">&#128269; '
                           f'{html_mod.escape(pm)}</span>')
        if attrs_html:
            attrs_html = f'<div class="attrs">{attrs_html}</div>'

        # Possible same-video re-appearances (vehicle/person left and came
        # back): link the two cards with the evidence tier spelled out.
        # Annotation only — merging over a long gap is the investigator's
        # call, never the algorithm's.
        reapp_html = ''
        _EVIDENCE_EL = {'plate': 'πινακίδα',
                        'plate+appearance': 'πινακίδα + εμφάνιση',
                        'appearance': 'μόνο εμφάνιση — χαμηλή βεβαιότητα'}
        for r in track.get('reappearance') or []:
            when = ('νωρίτερα' if r.get('when') == 'earlier' else 'αργότερα')
            ev = _EVIDENCE_EL.get(r.get('evidence'), r.get('evidence', ''))
            weak = r.get('evidence') == 'appearance'
            cls_extra = ' weak' if weak else ''
            reapp_html += (
                f'<div class="reapp{cls_extra}">&#128257; Πιθανή επανεμφάνιση: '
                f'ίδιο με #{r["other"]} ({when}, κενό {r.get("gap", 0):.0f}s) '
                f'&middot; απόδειξη: {html_mod.escape(ev)}</div>')

        # Best face shots (person tracks): extraction only, for human review.
        faces_html = ''
        for k, face in enumerate(track.get('faces') or []):
            faces_html += (f'<img src="data:image/jpeg;base64,{face["b64"]}" '
                           f'class="face" title="Best face shot — click to zoom" '
                           f'onclick="openLightbox(this.src, \'{title_esc} '
                           f'#{track_id} — face {k + 1}\')">')
        if faces_html:
            faces_html = f'<div class="faces">{faces_html}</div>'

        # Snapshot gallery: best-K crops, every one zoomable. Each snapshot
        # carries its full-scene context frame (red box on the object) in a
        # data attribute — the lightbox opens on the SCENE first so the
        # viewer sees where the crop was taken, then toggles to the close-up.
        snaps = track.get('snapshots_b64') or ([track['thumbnail']] if track.get('thumbnail') else [])
        snap_ts = track.get('snapshot_ts', [])
        snap_ctx = track.get('snapshots_ctx_b64', [])

        def _ctx_attr(k: int) -> str:
            if k < len(snap_ctx) and snap_ctx[k]:
                return f' data-ctx="data:image/jpeg;base64,{snap_ctx[k]}"'
            return ''

        thumb_src = f"data:image/jpeg;base64,{snaps[0]}" if snaps else ""
        minis = ''
        if len(snaps) > 1:
            for k, b64 in enumerate(snaps):
                ts_label = format_timestamp(snap_ts[k]) if k < len(snap_ts) else ''
                minis += (f'<img src="data:image/jpeg;base64,{b64}" class="mini"{_ctx_attr(k)} '
                          f'onclick="openLightbox(this.src, \'{title_esc} #{track_id} @ {ts_label}\', this.dataset.ctx)" '
                          f'title="{ts_label} — click to zoom">')

        card = f'''
        <div class="card" data-class="{cls_esc}">
            <div class="thumbcol">
                <img src="{thumb_src}" alt="{cls_esc} #{track_id}" class="thumbnail"{_ctx_attr(0)} onclick="openLightbox(this.src, '{title_esc} #{track_id}', this.dataset.ctx)" title="Click to zoom">
                <div class="gallery">{minis}</div>
            </div>
            <div class="info">
                <div class="title">{title_esc} #{track_id} {badges}</div>
                {attrs_html}{plate_html}{reapp_html}{faces_html}
                <div class="confidence">Confidence: {track['confidence']:.0%}</div>
                <div class="meta">
                    <span class="direction" title="Direction">{direction}</span>
                    <span class="dwell" title="Total visible duration">{dwell_str}</span>
                </div>
                <div class="timestamps">
                    {''.join(interval_rows)}
                </div>
            </div>
        </div>'''
        cards_html.append(card)

    # Get unique classes for filter buttons (escaped — user-supplied prompts)
    classes = sorted(set(t['class'] for t in tracks.values()))
    filter_buttons = '<button class="filter active" onclick="filterClass(\'all\')">All</button>\n'
    for cls in classes:
        count = sum(1 for t in tracks.values() if t['class'] == cls)
        c = html_mod.escape(cls, quote=True)
        filter_buttons += f'        <button class="filter" onclick="filterClass(\'{c}\')">{html_mod.escape(cls.title())} ({count})</button>\n'

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisionX Report - {video_name}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #fff;
        }}
        .video-info {{
            color: #888;
            margin-bottom: 20px;
        }}
        .instructions {{
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #0f3460;
        }}
        .instructions strong {{ color: #e94560; }}
        .filters {{
            position: sticky;
            top: 0;
            background: #1a1a2e;
            padding: 10px 0;
            margin-bottom: 20px;
            z-index: 100;
            border-bottom: 1px solid #333;
        }}
        .filter {{
            background: #16213e;
            color: #eee;
            border: 1px solid #0f3460;
            padding: 8px 16px;
            margin-right: 8px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .filter:hover {{ background: #0f3460; }}
        .filter.active {{
            background: #e94560;
            border-color: #e94560;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: #16213e;
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            transition: transform 0.2s;
        }}
        .card:hover {{ transform: translateY(-2px); }}
        .card.hidden {{ display: none; }}
        .thumbcol {{ display: flex; flex-direction: column; }}
        .thumbnail {{
            width: 120px;
            height: 120px;
            object-fit: cover;
            cursor: zoom-in;
            transition: opacity 0.2s;
        }}
        .thumbnail:hover {{ opacity: 0.8; }}
        .gallery {{ display: flex; flex-wrap: wrap; width: 120px; }}
        .mini {{
            width: 30px; height: 30px; object-fit: cover;
            cursor: zoom-in; opacity: 0.85; transition: opacity 0.15s;
        }}
        .mini:hover {{ opacity: 1; }}
        .badge {{
            font-size: 0.6em; vertical-align: middle; border-radius: 4px;
            padding: 2px 6px; margin-left: 6px; letter-spacing: 0.5px;
        }}
        .badge.parked {{ background: #0f3460; color: #7dd3fc; }}
        .badge.host {{ background: #4b5563; color: #e5e7eb; }}
        .badge.merged {{ background: #14532d; color: #86efac; }}
        .ts-more {{ color: #888; font-size: 0.85em; padding: 2px 10px; }}
        .platechip {{
            display: inline-block; font-family: monospace; font-weight: bold;
            letter-spacing: 2px; background: #f5f5f4; color: #1a1a2e;
            border: 2px solid #64748b; border-radius: 5px; padding: 2px 10px;
            margin: 4px 0; cursor: copy; font-size: 1.0em;
        }}
        .platechip:hover {{ background: #ffd75e; }}
        .platechip.uncertain {{
            background: #2a2118; color: #fbbf24;
            border: 2px dashed #b45309;
        }}
        .platechip.uncertain:hover {{ background: #3a2d1a; }}
        .plate-alts {{ color: #fbbf24; font-size: 0.78em; margin: 0 0 4px; }}
        .lightbox-nav {{
            position: fixed; top: 50%; transform: translateY(-50%);
            font-size: 42px; color: #fff; cursor: pointer; user-select: none;
            padding: 18px 14px; background: rgba(0,0,0,0.35); border-radius: 10px;
            z-index: 1002;
        }}
        .lightbox-nav:hover {{ background: rgba(0,0,0,0.6); }}
        .lightbox-nav.prev {{ left: 14px; }}
        .lightbox-nav.next {{ right: 14px; }}
        .lightbox-veh {{
            position: fixed; top: 14px; left: 50%; transform: translateX(-50%);
            display: flex; gap: 10px; z-index: 1002;
        }}
        .lightbox-veh span {{
            cursor: pointer; color: #fff; background: rgba(0,0,0,0.45);
            border: 1px solid rgba(255,255,255,0.25); border-radius: 8px;
            padding: 6px 14px; font-size: 14px; user-select: none;
        }}
        .lightbox-veh span:hover {{ background: rgba(0,0,0,0.7); }}
        .playbtn {{
            display: inline-block; cursor: pointer; color: #4ade80;
            background: rgba(74, 222, 128, 0.12); border: 1px solid rgba(74, 222, 128, 0.4);
            border-radius: 6px; padding: 1px 8px; margin-left: 6px; font-size: 0.85em;
        }}
        .playbtn:hover {{ background: rgba(74, 222, 128, 0.25); }}
        #clip-wrap {{ position: relative; display: inline-block; }}
        #clip-video {{ max-width: 92vw; max-height: 80vh; border-radius: 8px; display: block; }}
        #clip-canvas {{ position: absolute; left: 0; top: 0; pointer-events: none; }}
        .reapp {{
            display: inline-block; font-size: 0.8em; margin: 2px 0 4px;
            padding: 3px 8px; border-radius: 8px;
            border: 1px solid #38bdf8; color: #7dd3fc;
            background: rgba(56, 189, 248, 0.08);
        }}
        .reapp.weak {{ border-style: dashed; color: #94a3b8; border-color: #64748b; }}
        .pscore {{ font-size: 0.75em; color: #555; letter-spacing: 0; }}
        .deepchip {{
            cursor: copy; margin-left: 6px; color: #7dd3fc; font-size: 1.1em;
        }}
        .deepchip:hover {{ color: #e94560; }}
        .faces {{ display: flex; gap: 6px; margin: 4px 0; }}
        .face {{
            width: 52px; height: 52px; object-fit: cover; cursor: zoom-in;
            border-radius: 6px; border: 2px solid #7dd3fc;
        }}
        .face:hover {{ opacity: 0.85; }}
        .attrs {{ display: flex; gap: 8px; margin: 2px 0 4px; flex-wrap: wrap; }}
        .attr {{
            display: inline-flex; align-items: center; gap: 5px;
            font-size: 0.8em; color: #cbd5e1; background: #10192e;
            border-radius: 10px; padding: 2px 9px;
        }}
        .attr i {{
            width: 10px; height: 10px; border-radius: 50%;
            display: inline-block; border: 1px solid rgba(255,255,255,0.35);
        }}
        .attr.promptmatch {{ background: #143a2b; color: #7fd98a; }}
        /* For persons, show top (face) instead of center */
        .card[data-class="person"] .thumbnail {{
            object-position: top;
        }}
        .lightbox {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 2000;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }}
        .lightbox.active {{ display: flex; }}
        .lightbox img {{
            max-width: 90%;
            max-height: 80%;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }}
        .lightbox-title {{
            color: #fff;
            margin-top: 15px;
            font-size: 1.2em;
        }}
        .lightbox-close {{
            position: absolute;
            top: 20px;
            right: 30px;
            font-size: 40px;
            color: #fff;
            cursor: pointer;
            transition: color 0.2s;
        }}
        .lightbox-close:hover {{ color: #e94560; }}
        .info {{
            padding: 12px;
            flex: 1;
        }}
        .title {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 5px;
        }}
        .confidence {{
            color: #4ade80;
            margin-bottom: 5px;
        }}
        .meta {{
            display: flex;
            gap: 12px;
            margin-bottom: 8px;
            align-items: center;
        }}
        .direction {{
            font-size: 1.4em;
            line-height: 1;
        }}
        .dwell {{
            color: #fbbf24;
            font-size: 0.9em;
        }}
        .timestamps {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        .ts {{
            background: #0f3460;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-family: monospace;
            transition: background 0.2s;
        }}
        .ts:hover {{ background: #e94560; }}
        .toast {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #4ade80;
            color: #000;
            padding: 12px 24px;
            border-radius: 8px;
            opacity: 0;
            transition: opacity 0.3s;
            z-index: 1000;
        }}
        .toast.show {{ opacity: 1; }}
        .stats {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: #16213e;
            padding: 15px 20px;
            border-radius: 8px;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #e94560;
        }}
        .stat-label {{ color: #888; }}
    </style>
</head>
<body>
    <h1>VisionX Detection Report</h1>
    <div class="video-info">{Path(video_path).name}</div>

    <div class="instructions">
        <strong>How to use:</strong> Click on a timestamp to copy it.
        Then in VLC, press <strong>Ctrl+T</strong> (Windows/Linux) or <strong>Cmd+T</strong> (Mac)
        and paste the timestamp to jump to that moment.
    </div>

    <div class="stats">
        <div class="stat">
            <div class="stat-value">{len(tracks)}</div>
            <div class="stat-label">Total Detections</div>
        </div>
        {generate_class_stats(tracks)}
    </div>

    <div class="filters">
        {filter_buttons}
    </div>

    <div class="grid">
        {''.join(cards_html)}
    </div>

    <div class="toast" id="toast">Copied to clipboard!</div>

    <div class="lightbox" id="clipbox" onclick="closeClip()">
        <span class="lightbox-close">&times;</span>
        <div id="clip-wrap" onclick="event.stopPropagation()">
            <video id="clip-video" controls></video>
            <canvas id="clip-canvas"></canvas>
        </div>
        <div class="lightbox-title" id="clip-note"></div>
    </div>

    <div class="lightbox" id="lightbox" onclick="closeLightbox()">
        <span class="lightbox-close">&times;</span>
        <span class="lightbox-nav prev" onclick="lbStep(event, -1)" title="Προηγούμενο στιγμιότυπο ΙΔΙΟΥ αντικειμένου (←)">&#10094;</span>
        <img id="lightbox-img" src="" alt="" onclick="toggleLightboxView(event)">
        <span class="lightbox-nav next" onclick="lbStep(event, 1)" title="Επόμενο στιγμιότυπο ΙΔΙΟΥ αντικειμένου (→)">&#10095;</span>
        <div class="lightbox-veh">
            <span onclick="lbVehicle(event, -1)" title="Προηγούμενο αντικείμενο (↑)">&#9198; προηγ. αντικείμενο</span>
            <span onclick="lbVehicle(event, 1)" title="Επόμενο αντικείμενο (↓)">επόμ. αντικείμενο &#9197;</span>
        </div>
        <div class="lightbox-title" id="lightbox-title"></div>
    </div>

    <script>
        function copyTimestamp(ts) {{
            navigator.clipboard.writeText(ts).then(() => {{
                const toast = document.getElementById('toast');
                toast.classList.add('show');
                setTimeout(() => toast.classList.remove('show'), 2000);
            }});
        }}

        // Gallery navigation, two levels (user-designed):
        //   ← / →  (και τα πλαϊνά κουμπιά)  = snapshots of the SAME object
        //   ↑ / ↓  (και τα κουμπιά ⏮ ⏭)      = previous / next OBJECT
        // Title always shows position: "CAR #4 — στιγμιότυπο 2/4 · 3/9".
        const GROUPS = [];
        document.querySelectorAll('.card').forEach((card) => {{
            const imgs = Array.from(card.querySelectorAll('img.thumbnail, img.mini, img.face'));
            if (!imgs.length) return;
            const title = card.querySelector('.title')?.textContent?.trim() || '';
            GROUPS.push({{ title, imgs }});
            imgs.forEach((el, i) => {{
                const g = GROUPS.length - 1;
                el.addEventListener('click', (ev) => {{
                    ev.stopPropagation();
                    lbOpenAt(g, i);
                }});
            }});
        }});
        let lbG = -1, lbI = -1;

        function lbOpenAt(g, i) {{
            if (!GROUPS.length) return;
            lbG = ((g % GROUPS.length) + GROUPS.length) % GROUPS.length;
            const items = GROUPS[lbG].imgs;
            lbI = ((i % items.length) + items.length) % items.length;
            const el = items[lbI];
            openLightbox(el.src,
                `${{GROUPS[lbG].title}} — στιγμιότυπο ${{lbI + 1}}/${{items.length}} · ${{lbG + 1}}/${{GROUPS.length}}`,
                el.dataset.ctx);
        }}

        function lbStep(e, d) {{      // within the same object
            if (e) e.stopPropagation();
            if (lbG >= 0) lbOpenAt(lbG, lbI + d);
        }}

        function lbVehicle(e, d) {{   // jump objects, land on first snapshot
            if (e) e.stopPropagation();
            if (lbG >= 0) lbOpenAt(lbG + d, 0);
        }}

        document.addEventListener('keydown', (e) => {{
            if (!document.getElementById('lightbox').classList.contains('active')) return;
            if (e.key === 'ArrowLeft') {{ e.preventDefault(); lbStep(null, -1); }}
            if (e.key === 'ArrowRight') {{ e.preventDefault(); lbStep(null, 1); }}
            if (e.key === 'ArrowUp') {{ e.preventDefault(); lbVehicle(null, -1); }}
            if (e.key === 'ArrowDown') {{ e.preventDefault(); lbVehicle(null, 1); }}
        }});

        // Lightbox with scene context: when a snapshot has a context frame
        // (full scene, red box on the object), open on the SCENE first so
        // the viewer sees where the crop comes from; clicking the image
        // toggles scene <-> close-up. Snapshots without context (older
        // reports, faces) just zoom the crop as before.
        let lbCrop = '', lbCtx = '', lbTitle = '', lbShowingCtx = false;

        function lbRender() {{
            const img = document.getElementById('lightbox-img');
            img.src = lbShowingCtx ? lbCtx : lbCrop;
            const hint = lbCtx
                ? (lbShowingCtx ? ' — σκηνή (κλικ για κοντινό)'
                                : ' — κοντινό (κλικ για σκηνή)')
                : '';
            document.getElementById('lightbox-title').textContent = lbTitle + hint;
        }}

        function openLightbox(src, title, ctx) {{
            lbCrop = src;
            lbCtx = ctx || '';
            lbTitle = title;
            lbShowingCtx = !!lbCtx;
            lbRender();
            document.getElementById('lightbox').classList.add('active');
        }}

        function toggleLightboxView(e) {{
            if (!lbCtx) return;  // nothing to toggle — let the click close
            e.stopPropagation();
            lbShowingCtx = !lbShowingCtx;
            lbRender();
        }}

        const PLAYBACK = {json_mod.dumps(playback_map)};
        let clipEnd = 0, clipTrack = null, clipAnim = 0;

        function drawClipOverlay() {{
            const v = document.getElementById('clip-video');
            const c = document.getElementById('clip-canvas');
            clipAnim = requestAnimationFrame(drawClipOverlay);
            if (!clipTrack || !v.videoWidth || !v.clientWidth) return;
            if (c.width !== v.clientWidth) c.width = v.clientWidth;
            if (c.height !== v.clientHeight) c.height = v.clientHeight;
            const g = c.getContext('2d');
            g.clearRect(0, 0, c.width, c.height);
            const B = clipTrack.boxes;
            const f = v.currentTime * clipTrack.fps;
            if (f < B[0][0] - 8 || f > B[B.length - 1][0] + 8) return;
            let lo = 0, hi = B.length - 1;
            while (lo < hi) {{ const m = (lo + hi) >> 1; if (B[m][0] < f) lo = m + 1; else hi = m; }}
            const j = Math.max(1, lo);
            const b1 = B[j - 1], b2 = B[Math.min(j, B.length - 1)];
            const t = b2[0] > b1[0] ? Math.min(1, Math.max(0, (f - b1[0]) / (b2[0] - b1[0]))) : 0;
            const L = (a, b) => a + (b - a) * t;
            const sx = c.width / v.videoWidth, sy = c.height / v.videoHeight;
            const x1 = L(b1[1], b2[1]) * sx, y1 = L(b1[2], b2[2]) * sy;
            const x2 = L(b1[3], b2[3]) * sx, y2 = L(b1[4], b2[4]) * sy;
            g.lineWidth = 2;
            g.strokeStyle = '#4ade80';
            g.strokeRect(x1, y1, x2 - x1, y2 - y1);
            if (clipTrack.plate) {{
                const [rx, ry, rw, rh] = clipTrack.plate;
                g.strokeStyle = '#ffd200';
                g.strokeRect(x1 + rx * (x2 - x1), y1 + ry * (y2 - y1),
                             rw * (x2 - x1), rh * (y2 - y1));
            }}
        }}

        function openClip(uri, start, end, tid) {{
            const box = document.getElementById('clipbox');
            const v = document.getElementById('clip-video');
            const note = document.getElementById('clip-note');
            clipEnd = end;
            note.textContent = 'Απόσπασμα ' + start.toFixed(0) + 's – ' + end.toFixed(0) + 's';
            v.onerror = () => {{
                const dav = /\\.(dav|h264|h265)([#?]|$)/i.test(uri);
                note.textContent = dav
                    ? 'Μη υποστηριζόμενο format βίντεο (.dav) — αντιγράψτε το '
                      + 'timestamp και ανοίξτε το αρχείο σε VLC (Cmd/Ctrl+T).'
                    : 'Δεν ήταν δυνατή η αναπαραγωγή εδώ — πατήστε «Άνοιγμα '
                      + 'στον Browser» για πλήρη λειτουργία, ή χρησιμοποιήστε '
                      + 'το timestamp σε VLC.';
            }};
            clipTrack = (tid !== undefined && PLAYBACK[String(tid)]) || null;
            v.src = uri + '#t=' + start + ',' + end;
            box.classList.add('active');
            cancelAnimationFrame(clipAnim);
            clipAnim = requestAnimationFrame(drawClipOverlay);
            v.play().catch(() => {{}});
        }}
        function closeClip() {{
            const v = document.getElementById('clip-video');
            cancelAnimationFrame(clipAnim);
            clipTrack = null;
            v.pause();
            v.removeAttribute('src');
            v.load();
            document.getElementById('clipbox').classList.remove('active');
        }}
        // Stop at the interval's end (media fragments alone don't always)
        document.getElementById('clip-video').addEventListener('timeupdate', function () {{
            if (clipEnd && this.currentTime >= clipEnd) this.pause();
        }});

        function closeLightbox() {{
            document.getElementById('lightbox').classList.remove('active');
        }}

        // Close lightbox with Escape key
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeLightbox();
        }});

        function filterClass(cls) {{
            // Update active button
            document.querySelectorAll('.filter').forEach(btn => {{
                btn.classList.remove('active');
                if (btn.textContent.toLowerCase().startsWith(cls)) {{
                    btn.classList.add('active');
                }}
            }});
            if (cls === 'all') {{
                document.querySelector('.filter').classList.add('active');
            }}

            // Filter cards
            document.querySelectorAll('.card').forEach(card => {{
                if (cls === 'all' || card.dataset.class === cls) {{
                    card.classList.remove('hidden');
                }} else {{
                    card.classList.add('hidden');
                }}
            }});
        }}
    </script>
</body>
</html>'''

    # Save as video_report.html (same name as video).
    # encoding='utf-8' is REQUIRED: on Windows the default is cp1252, which
    # crashes on the direction arrows/Greek text (Stage 0 bug fix).
    report_file = output_path / f'{video_name}_report.html'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html)

    return str(report_file)


def generate_class_stats(tracks: dict) -> str:
    """Generate stat boxes for each class"""
    class_counts = {}
    for t in tracks.values():
        class_counts[t['class']] = class_counts.get(t['class'], 0) + 1

    html = ''
    for cls, count in sorted(class_counts.items()):
        html += f'''
        <div class="stat">
            <div class="stat-value">{count}</div>
            <div class="stat-label">{html_mod.escape(cls.title())}</div>
        </div>'''
    return html
