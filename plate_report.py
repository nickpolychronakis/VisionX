# -*- coding: utf-8 -*-
"""Self-contained HTML report for plate.py results (Greek UI).

Modeled on what professional forensic tools present (Amped FIVE-style):
ranked findings with per-character uncertainty, the fused/best images at
reading size, a multi-level enhancement panel per frame for HUMAN judgment
(examiners routinely review several contrast/gamma renderings side by side),
and a methodology section documenting every processing step + parameters for
reproducibility. All images are base64-embedded — one portable file, same
pattern as the existing report.py of VisionX.
"""
import base64
from pathlib import Path

import cv2


def _b64(img, q=90):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()


def _candidate_table(cands, title, note):
    if not cands:
        return ''
    rows = []
    for i, c in enumerate(cands, 1):
        flag = ' <span class="tag">GR</span>' if c.get('greek_pattern') else ''
        pct = f'{c["score"] * 100:.0f}%'
        bar = f'<div class="bar"><div style="width:{min(100, c["score"] * 100):.0f}%"></div></div>'
        rows.append(f'<tr><td class="rank">{i}</td>'
                    f'<td class="plate">{c["plate"]}{flag}</td>'
                    f'<td class="score">{pct}{bar}</td></tr>')
    return (f'<h2>{title}</h2><p class="note">{note}</p>'
            f'<table class="cands">{"".join(rows)}</table>')


def _per_char(cand):
    """Per-position cells: the chosen char with its probability, alternatives
    beneath — this is the row the user reads to build wildcard DB queries."""
    cells = []
    for pc in cand.get('per_char', []):
        ch = pc['char'] if pc['char'] != '_' else '·'
        if ch == '·' and not pc.get('alternatives'):
            continue
        alts = ' '.join(f'{a["char"] if a["char"] != "_" else "·"}&thinsp;{a["prob"] * 100:.0f}%'
                        for a in pc.get('alternatives', [])[:2])
        op = 0.25 + 0.75 * min(1.0, pc['prob'])
        cells.append(f'<div class="cc"><div class="ch" style="opacity:{op:.2f}">{ch}</div>'
                     f'<div class="cp">{pc["prob"] * 100:.0f}%</div>'
                     f'<div class="ca">{alts}</div></div>')
    return '<div class="ccrow">' + ''.join(cells) + '</div>'


def generate(out_dir, result, imgs, panels, meta):
    """Write report.html into out_dir. Returns its path."""
    out = Path(out_dir)
    top_free = result.get('candidates', [])
    top_gr = result.get('candidates_greek_projected', [])
    best = (top_gr or top_free)
    region = result.get('region_hint')

    img_html = ''
    if imgs.get('fused_large') is not None:
        img_html += (f'<figure><img src="{_b64(imgs["fused_large"])}">'
                     f'<figcaption>Συγχώνευση {result.get("fused_frames", 0)} καρέ '
                     '(fusion) — μέση τιμή πραγματικών pixel μετά από υποδιάστημα '
                     'ευθυγράμμισης· ΟΧΙ τεχνητή νοημοσύνη/γεννητικό μοντέλο.</figcaption></figure>')
    if imgs.get('best_frame_large') is not None:
        img_html += (f'<figure><img src="{_b64(imgs["best_frame_large"])}">'
                     '<figcaption>Καλύτερο μεμονωμένο καρέ (μέγιστη ευκρίνεια, '
                     'επιβεβαιωμένο από τον ανιχνευτή).</figcaption></figure>')
    if imgs.get('fused_deconv') is not None:
        img_html += (f'<figure><img src="{_b64(imgs["fused_deconv"])}">'
                     '<figcaption>Συγχώνευση + αποθόλωση Wiener — απόδοση ΜΟΝΟ '
                     'για οπτικό έλεγχο (δεν συμμετέχει στην ψηφοφορία OCR: '
                     'μετρήθηκε ότι δεν τη βελτιώνει).</figcaption></figure>')

    panel_html = ''
    for p in panels:
        cells = ''.join(f'<div class="pv"><img src="{_b64(im)}"><span>{lab}</span></div>'
                        for lab, im in p['variants'])
        read = f' — OCR: <b>{p["read"]}</b> ({p["conf"] * 100:.0f}%)' if p.get('read') else ''
        panel_html += (f'<div class="panel"><h4>Καρέ {p["frame"]}{read}</h4>'
                       f'<div class="pvrow">{cells}</div></div>')

    sheet_html = ''
    if imgs.get('sheet') is not None:
        sheet_html = (f'<h2>Όλα τα καρέ που ψήφισαν</h2>'
                      '<p class="note">Κάτω από κάθε καρέ: αριθμός καρέ, ευκρίνεια (sh), '
                      'ποιότητα ευθυγράμμισης (cc) και η ανάγνωση OCR του καρέ με τη '
                      'βεβαιότητά της (πράσινο = σίγουρη, πορτοκαλί = αμφίβολη).</p>'
                      f'<img class="sheet" src="{_b64(imgs["sheet"])}">')

    steps = '\n'.join(f'<li>{s}</li>' for s in meta.get('steps', []))
    models = ', '.join(meta.get('models', []))

    html = f"""<!DOCTYPE html>
<html lang="el"><head><meta charset="utf-8">
<title>Ανάλυση πινακίδας — {result.get('video', '')}</title>
<style>
 body {{ background:#14161a; color:#e8e8e8; font:15px/1.5 -apple-system, 'Segoe UI', sans-serif;
        max-width: 1060px; margin: 0 auto; padding: 24px; }}
 h1 {{ font-size: 22px; }} h2 {{ font-size: 18px; margin-top: 34px; color:#ffd75e; }}
 h4 {{ margin: 8px 0; }}
 .disclaimer {{ background:#3a2b12; border:1px solid #7a5a1e; border-radius:8px;
        padding:10px 14px; margin:14px 0; color:#ffd75e; }}
 .meta span {{ display:inline-block; background:#20242b; border-radius:6px;
        padding:3px 10px; margin:2px 6px 2px 0; }}
 table.cands {{ border-collapse: collapse; width: 100%; max-width: 560px; }}
 table.cands td {{ padding: 7px 12px; border-bottom: 1px solid #2a2f38; }}
 td.rank {{ color:#888; width:30px; }}
 td.plate {{ font: 700 22px/1 ui-monospace, monospace; letter-spacing: 3px; }}
 td.score {{ width: 180px; color:#aaa; }}
 .tag {{ font:11px/1 sans-serif; background:#1e4620; color:#7fd98a; border-radius:4px;
        padding:2px 6px; vertical-align: middle; letter-spacing:0; }}
 .bar {{ background:#242933; height:6px; border-radius:3px; margin-top:4px; }}
 .bar div {{ background:#ffd75e; height:6px; border-radius:3px; }}
 .note {{ color:#9aa3b0; font-size: 13.5px; }}
 .ccrow {{ display:flex; gap:6px; margin: 10px 0 22px; flex-wrap: wrap; }}
 .cc {{ background:#20242b; border-radius:8px; padding:8px 10px; text-align:center; min-width:52px; }}
 .ch {{ font: 700 26px/1.1 ui-monospace, monospace; }}
 .cp {{ color:#ffd75e; font-size:12px; margin-top:3px; }}
 .ca {{ color:#8b93a1; font-size:11px; margin-top:3px; }}
 figure {{ display:inline-block; margin: 8px 16px 8px 0; }}
 figure img {{ max-width: 480px; border-radius: 6px; }}
 figcaption {{ color:#9aa3b0; font-size: 12.5px; max-width: 480px; margin-top: 4px; }}
 .pvrow {{ display:flex; gap:8px; flex-wrap:wrap; }}
 .pv {{ text-align:center; }}
 .pv img {{ height: 84px; border-radius:4px; }}
 .pv span {{ display:block; color:#8b93a1; font-size:11.5px; margin-top:2px; }}
 .panel {{ background:#181b20; border:1px solid #262b34; border-radius:8px;
        padding:10px 14px; margin: 10px 0; }}
 img.sheet {{ max-width:100%; border-radius:6px; }}
 .how {{ background:#182028; border:1px solid #24303c; border-radius:8px;
        padding: 4px 16px 10px; margin: 14px 0; }}
 ol li, .how li {{ margin: 4px 0; }}
</style></head><body>

<h1>Ανάλυση πινακίδας — {Path(result.get('video', '')).name}</h1>
<div class="disclaimer">⚠️ Λίστα ΠΙΘΑΝΩΝ πινακίδων για διερευνητική αναζήτηση σε βάση
δεδομένων — ΔΕΝ αποτελεί αποδεικτικό/δικαστικό τεκμήριο. Τα σκορ είναι σχετική
κατάταξη, όχι βαθμονομημένες πιθανότητες.</div>

<div class="meta">
 <span>Καρέ έναρξης: {result.get('start_frame')} ({result.get('start_time_sec')}s)</span>
 <span>Καρέ που ακολουθήθηκαν: {result.get('frames_tracked')}</span>
 <span>Καρέ σε ψηφοφορία: {result.get('frames_ocred')}</span>
 <span>Fusion: {result.get('fused_frames', 0)} καρέ</span>
 {f'<span>Ένδειξη χώρας: {region}</span>' if region else ''}
</div>

<div class="how"><h4>Πώς διαβάζεται</h4><ul class="note">
 <li><b>Δύο λίστες:</b> η «ελεύθερη» καλύπτει και ξένες πινακίδες· η «ελληνική προβολή»
 περιορίζει κάθε θέση στα έγκυρα ελληνικά γράμματα (ΑΒΕΖΗΙΚΜΝΟΡΤΥΧ) και ψηφία —
 προτίμησέ τη αν το όχημα είναι ελληνικό.</li>
 <li><b>· (τελεία) ή χαμηλό ποσοστό σε θέση:</b> ο χαρακτήρας εκεί είναι αβέβαιος —
 χρησιμοποίησε wildcard στην αναζήτηση και δες τις εναλλακτικές της θέσης.</li>
 <li><b>Η ανάλυση ανά θέση</b> (κουτάκια) δείχνει το επικρατέστερο σύμβολο, το ποσοστό
 συναίνεσης και τις εναλλακτικές — αυτή χτίζει το ερώτημα προς τη βάση.</li>
 <li><b>Έλεγξε με το μάτι</b> τη συγχώνευση και το πάνελ επεξεργασιών: ο άνθρωπος
 συμπληρώνει σχήματα που το OCR χάνει σε θολές λήψεις.</li>
</ul></div>

{_candidate_table(top_gr, 'Υποψήφιες — ελληνικές πινακίδες',
                  'Η ψηφοφορία περιορισμένη στα έγκυρα γράμματα και ψηφία '
                  'των ελληνικών πινακίδων.')}
{_per_char(top_gr[0]) if top_gr else ''}
{_candidate_table(top_free, 'Υποψήφιες — ελεύθερη ανάγνωση',
                  'Χωρίς περιορισμό σχήματος — καλύπτει και ξένες πινακίδες.')}
{_per_char(top_free[0]) if top_free else ''}

<h2>Εικόνες ανάγνωσης</h2>
{img_html}

<h2>Πάνελ πολλαπλών επεξεργασιών (για ανθρώπινη κρίση)</h2>
<p class="note">Κάθε καρέ σε 5 αποδόσεις — αρχικό, τοπική αντίθεση (CLAHE), γάμμα
σκιών, γάμμα υπερέκθεσης, αρνητικό. Όλες είναι σημειακές/μονότονες πράξεις σε
πραγματικά pixel (καμία «εφεύρεση» χαρακτήρων)· διαφορετικές αποδόσεις αναδεικνύουν
διαφορετικές πινελιές — πρακτική αντίστοιχη των επαγγελματικών forensic εργαλείων.</p>
{panel_html}

{sheet_html}

<h2>Μεθοδολογία (για αναπαραγωγιμότητα)</h2>
<ol class="note">{steps}</ol>
<p class="note">Μοντέλα OCR (ensemble): {models}. Ανιχνευτής πινακίδας:
{meta.get('detector', '')}. Όλα τα μοντέλα εκτελούνται τοπικά (ONNX).</p>

</body></html>"""

    path = out / 'report.html'
    path.write_text(html, encoding='utf-8')
    return path
