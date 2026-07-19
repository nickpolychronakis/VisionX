# -*- coding: utf-8 -*-
"""Cross-camera match report — ROADMAP Phase Ε.

Self-contained HTML: automatic match groups (evidence-labeled), every
object's card, and a MANUAL pairing mode — the human reviewer clicks two
cards to link objects the automatic pass missed (or unlink wrong ones).
Manual pairs live client-side and export as JSON, keeping the report a
plain portable file like every other VisionX artifact.
"""

import html as html_mod
import json
from pathlib import Path

from report import format_timestamp


def _card(video, tid, t):
    cls = html_mod.escape(t['class'], quote=True)
    thumb = (f"data:image/jpeg;base64,{t['thumbnail']}"
             if t.get('thumbnail') else '')
    plate = ''
    if t.get('plate'):
        plate = (f'<span class="mchip">{html_mod.escape(t["plate"]["plate"])}'
                 '</span>')
    iv = t.get('intervals') or []
    times = ', '.join(f"{format_timestamp(x['start'])}–{format_timestamp(x['end'])}"
                      for x in iv[:3])
    key = html_mod.escape(f'{video}|{tid}', quote=True)
    return (f'<div class="mcard" data-key="{key}" onclick="toggleSel(this)">'
            f'<img src="{thumb}">'
            f'<div class="mmeta"><b>{cls.upper()} #{tid}</b>{plate}'
            f'<span class="mvid">{html_mod.escape(str(video))}</span>'
            f'<span class="mtime">{times}</span></div></div>')


def generate_match_report(per_video: dict, groups: list, output_dir: str,
                          video_names: list) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ev_label = {
        'plate+appearance': ('ΠΙΝΑΚΙΔΑ + ΕΜΦΑΝΙΣΗ', 'ev-strong'),
        'plate': ('ΠΙΝΑΚΙΔΑ', 'ev-strong'),
        'appearance': ('ΜΟΝΟ ΕΜΦΑΝΙΣΗ — χαμηλή βεβαιότητα', 'ev-weak'),
    }

    grouped_keys = set()
    groups_html = ''
    for gi, g in enumerate(groups, 1):
        label, ev_cls = ev_label.get(g['evidence'], (g['evidence'], 'ev-weak'))
        combo = ''
        cp = g.get('combined_plate')
        if cp:
            combo = (f'<div class="combo">Συνδυαστική πινακίδα (όλες οι '
                     f'κάμερες): <span class="mchip big">'
                     f'{html_mod.escape(cp["plate"])}</span> '
                     f'{cp["score"]:.0%} συναίνεση από {cp["frames_used"]} καρέ'
                     '</div>')
        cards = ''
        for video, tid in g['members']:
            grouped_keys.add((video, tid))
            cards += _card(video, tid, per_video[video][tid])
        groups_html += (f'<div class="group"><div class="ghead">'
                        f'<b>Αντιστοίχιση {gi}</b>'
                        f'<span class="badge {ev_cls}">{label}</span>'
                        f'<span class="gscore">{g["score"]:.0%}</span></div>'
                        f'{combo}<div class="grow">{cards}</div></div>')
    if not groups_html:
        groups_html = ('<p class="note">Καμία αυτόματη αντιστοίχιση — '
                       'χρησιμοποιήστε τη χειροκίνητη σύνδεση παρακάτω.</p>')

    # Unmatched objects, grouped by video, all clickable for manual pairing.
    unmatched_html = ''
    for video in video_names:
        tracks = per_video.get(video) or {}
        cards = ''.join(_card(video, tid, t) for tid, t in sorted(tracks.items())
                        if (video, tid) not in grouped_keys)
        if cards:
            unmatched_html += (f'<h3>{html_mod.escape(str(video))}</h3>'
                               f'<div class="grow">{cards}</div>')

    html = f"""<!DOCTYPE html>
<html lang="el"><head><meta charset="utf-8">
<title>VisionX — Αντιστοίχιση καμερών</title>
<style>
 body {{ background:#14161a; color:#e8e8e8; margin:0 auto; max-width:1100px;
        padding:24px; font:15px/1.5 -apple-system,'Segoe UI',sans-serif; }}
 h1 {{ font-size:22px; }} h2 {{ color:#ffd75e; font-size:18px; margin-top:30px; }}
 h3 {{ color:#9aa3b0; font-size:14px; margin:14px 0 6px; }}
 .note {{ color:#9aa3b0; font-size:13.5px; }}
 .group {{ background:#181b20; border:1px solid #262b34; border-radius:10px;
        padding:12px 16px; margin:12px 0; }}
 .ghead {{ display:flex; align-items:center; gap:12px; margin-bottom:8px; }}
 .badge {{ font-size:11px; border-radius:4px; padding:2px 8px; }}
 .ev-strong {{ background:#1e4620; color:#7fd98a; }}
 .ev-weak {{ background:#3a2b12; color:#ffd75e; }}
 .gscore {{ color:#9aa3b0; margin-left:auto; }}
 .grow {{ display:flex; gap:10px; flex-wrap:wrap; }}
 .mcard {{ display:flex; gap:8px; background:#20242b; border-radius:8px;
        padding:8px; cursor:pointer; border:2px solid transparent;
        max-width:320px; }}
 .mcard img {{ width:84px; height:84px; object-fit:cover; border-radius:6px; }}
 .mcard.sel {{ border-color:#7dd3fc; }}
 .mcard.linked {{ border-color:#f472b6; }}
 .mmeta {{ display:flex; flex-direction:column; font-size:12.5px; gap:2px; }}
 .mvid {{ color:#7dd3fc; }} .mtime {{ color:#9aa3b0; font-family:monospace; }}
 .mchip {{ font-family:monospace; font-weight:bold; background:#f5f5f4;
        color:#1a1a2e; border-radius:4px; padding:0 6px; width:fit-content; }}
 .mchip.big {{ font-size:1.15em; letter-spacing:2px; padding:2px 10px; }}
 .combo {{ margin:6px 0 10px; color:#cbd5e1; }}
 .manual {{ background:#182028; border:1px solid #24303c; border-radius:10px;
        padding:12px 16px; margin:16px 0; }}
 #pairs li {{ margin:4px 0; }}
 button {{ background:#20242b; color:#e8e8e8; border:1px solid #3a4150;
        border-radius:6px; padding:6px 14px; cursor:pointer; }}
 button:hover {{ border-color:#7dd3fc; }}
 .disclaimer {{ background:#3a2b12; border:1px solid #7a5a1e; border-radius:8px;
        padding:10px 14px; margin:12px 0; color:#ffd75e; font-size:13.5px; }}
</style></head><body>
<h1>Αντιστοίχιση αντικειμένων μεταξύ καμερών</h1>
<p class="note">Βίντεο: {html_mod.escape(', '.join(str(v) for v in video_names))}
 — {len(groups)} αυτόματες αντιστοιχίσεις.</p>
<div class="disclaimer">⚠️ Οι αντιστοιχίσεις είναι ενδείξεις για διερευνητική
χρήση. Οι «ΜΟΝΟ ΕΜΦΑΝΙΣΗ» βασίζονται σε οπτική ομοιότητα μεταξύ καμερών —
επιβεβαιώστε με το μάτι πριν βασιστείτε σε αυτές.</div>

<h2>Αυτόματες αντιστοιχίσεις</h2>
{groups_html}

<h2>Χειροκίνητη σύνδεση</h2>
<div class="manual">
 <p class="note">Κάντε κλικ σε δύο κάρτες (από οποιοδήποτε σημείο της σελίδας)
 για να τις συνδέσετε χειροκίνητα. Οι συνδέσεις σας εξάγονται σε αρχείο JSON.</p>
 <ul id="pairs"></ul>
 <button onclick="exportPairs()">Εξαγωγή συνδέσεων (JSON)</button>
 <button onclick="clearPairs()">Καθαρισμός</button>
</div>

<h2>Μη αντιστοιχισμένα αντικείμενα</h2>
{unmatched_html or '<p class="note">Όλα τα αντικείμενα αντιστοιχίστηκαν.</p>'}

<script>
 let sel = null;
 const pairs = [];
 function toggleSel(card) {{
   if (sel === card) {{ card.classList.remove('sel'); sel = null; return; }}
   if (sel === null) {{ card.classList.add('sel'); sel = card; return; }}
   const a = sel.dataset.key, b = card.dataset.key;
   sel.classList.remove('sel');
   if (a !== b) {{
     pairs.push([a, b]);
     sel.classList.add('linked'); card.classList.add('linked');
     renderPairs();
   }}
   sel = null;
 }}
 function renderPairs() {{
   const ul = document.getElementById('pairs');
   ul.innerHTML = pairs.map((p, i) =>
     `<li>${{p[0]}} ⟷ ${{p[1]}} <button onclick="removePair(${{i}})">×</button></li>`
   ).join('');
 }}
 function removePair(i) {{ pairs.splice(i, 1); renderPairs(); }}
 function clearPairs() {{
   pairs.length = 0; renderPairs();
   document.querySelectorAll('.linked').forEach(c => c.classList.remove('linked'));
 }}
 function exportPairs() {{
   const blob = new Blob(
     [JSON.stringify({{manual_pairs: pairs, videos: {json.dumps([str(v) for v in video_names])}}}, null, 2)],
     {{type: 'application/json'}});
   const a = document.createElement('a');
   a.href = URL.createObjectURL(blob);
   a.download = 'visionx_manual_pairs.json';
   a.click();
 }}
</script>
</body></html>"""

    path = out / 'match_report.html'
    path.write_text(html, encoding='utf-8')
    return str(path)
