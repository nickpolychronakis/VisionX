"""VisionX HTML Report Generator"""

from pathlib import Path


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_report(tracks: dict, output_dir: str, video_name: str, video_path: str) -> str:
    """Generate standalone HTML report with embedded thumbnails"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate detection cards HTML
    cards_html = []
    for track_id, track in sorted(tracks.items(), key=lambda x: x[1]['first_seen']):
        first_ts = format_timestamp(track['first_seen'])
        last_ts = format_timestamp(track['last_seen'])

        # Use embedded base64 thumbnail
        thumb_src = f"data:image/jpeg;base64,{track['thumbnail']}" if track.get('thumbnail') else ""

        card = f'''
        <div class="card" data-class="{track['class']}">
            <img src="{thumb_src}" alt="{track['class']} #{track_id}" class="thumbnail" onclick="openLightbox(this.src, '{track['class'].upper()} #{track_id}')" title="Click to zoom">
            <div class="info">
                <div class="title">{track['class'].upper()} #{track_id}</div>
                <div class="confidence">Confidence: {track['confidence']:.0%}</div>
                <div class="timestamps">
                    <span class="ts" onclick="copyTimestamp('{first_ts}')" title="Click to copy">
                        First: {first_ts}
                    </span>
                    <span class="ts" onclick="copyTimestamp('{last_ts}')" title="Click to copy">
                        Last: {last_ts}
                    </span>
                </div>
            </div>
        </div>'''
        cards_html.append(card)

    # Get unique classes for filter buttons
    classes = sorted(set(t['class'] for t in tracks.values()))
    filter_buttons = '<button class="filter active" onclick="filterClass(\'all\')">All</button>\n'
    for cls in classes:
        count = sum(1 for t in tracks.values() if t['class'] == cls)
        filter_buttons += f'        <button class="filter" onclick="filterClass(\'{cls}\')">{cls.title()} ({count})</button>\n'

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
        .thumbnail {{
            width: 120px;
            height: 120px;
            object-fit: cover;
            cursor: zoom-in;
            transition: opacity 0.2s;
        }}
        .thumbnail:hover {{ opacity: 0.8; }}
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
            margin-bottom: 10px;
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

    <div class="lightbox" id="lightbox" onclick="closeLightbox()">
        <span class="lightbox-close">&times;</span>
        <img id="lightbox-img" src="" alt="">
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

        function openLightbox(src, title) {{
            document.getElementById('lightbox-img').src = src;
            document.getElementById('lightbox-title').textContent = title;
            document.getElementById('lightbox').classList.add('active');
        }}

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

    # Save as video_report.html (same name as video)
    report_file = output_path / f'{video_name}_report.html'
    with open(report_file, 'w') as f:
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
            <div class="stat-label">{cls.title()}</div>
        </div>'''
    return html
