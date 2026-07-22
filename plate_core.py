"""Batch plate reading for the main vision pipeline — ROADMAP Phase B.

Thin adapter over plate.py (deliberately NOT a code-moving refactor: plate.py
is field-tested and ground-truth-calibrated, so this module REUSES its
ensemble/voting machinery instead of duplicating or relocating it). Given the
best snapshots of a vehicle track, it detects the plate in each crop, runs
the 3-model OCR ensemble, votes per character position and returns ranked
candidates — the same philosophy as the interactive tool, minus GUI, fusion
and tonal TTA (the auto pass must stay cheap: it runs on EVERY vehicle in a
video; the interactive tool remains the deep-analysis path).
"""

import numpy as np

import plate as P


class PlateReader:
    """Lazy-loaded detector + OCR ensemble for automatic per-track ALPR."""

    def __init__(self, detector_model: str = P.DEFAULT_DETECTOR_MODEL,
                 ocr_models: str = P.DEFAULT_OCR_MODELS):
        from fast_plate_ocr import LicensePlateRecognizer
        # Shared factory (providers/conf rationale lives in plate.py — DRY).
        self.detector = P.make_plate_detector(detector_model)
        self.recognizers = [
            {'name': m.strip(),
             'rec': LicensePlateRecognizer(m.strip(), device='cpu'),
             'gray': False}
            for m in ocr_models.split(',') if m.strip()
        ]

    def read_from_crops(self, crops: list, top: int = 3) -> dict | None:
        """Read a plate from a track's snapshot crops. Returns a result dict
        or None when no plate was detected in any crop (motorcycles seen from
        the front, distant vehicles etc. — absence is a normal outcome)."""
        plates = []  # (tight_crop, det_conf)
        rels = []    # plate box relative to its vehicle crop [rx, ry, rw, rh]
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            dets = self.detector.predict(crop)
            if not dets:
                continue
            d = max(dets, key=lambda x: float(x.confidence))
            bb = d.bounding_box
            # Shared geometry gate (plate.py) — timestamp/logo bars
            # are NOT plates; rationale lives with the constants.
            pw, ph = float(bb.x2 - bb.x1), float(bb.y2 - bb.y1)
            if not P.plate_geometry_ok(pw, ph, crop.shape[1]):
                continue
            ch2, cw2 = crop.shape[:2]
            rels.append([float(bb.x1) / cw2, float(bb.y1) / ch2,
                         pw / cw2, ph / ch2])
            tight = crop[max(0, int(bb.y1)):int(bb.y2),
                         max(0, int(bb.x1)):int(bb.x2)]
            # Below ~40px width there are no readable characters — feeding
            # such crops to OCR only adds confident-noise votes.
            if tight.size and tight.shape[1] >= 40 and tight.shape[0] >= 10:
                plates.append((tight, float(d.confidence)))
        if not plates:
            return None

        # Tilt-corrected renderings VOTE ALONGSIDE the originals (user
        # request: the auto pass must analyse the true plate outline like
        # the interactive tool). Same rules as plate.py: quad = orientation
        # only (never crop bounds), applied only on significant tilt, and
        # never replacing — per-position voting arbitrates.
        levelled = []
        for img, w in plates:
            try:
                quad, _conf = P.estimate_quad(img, None)
                if quad is None or P._quad_tilt_deg(quad) < P.RECTIFY_MIN_TILT:
                    continue
                lv = P.level_by_quad(img, quad)
                if lv is not None:
                    levelled.append((lv, w))
            except Exception:  # noqa: BLE001 — optional refinement
                continue
        plates.extend(levelled)

        # Per-position weighted vote across crops × ensemble models — the
        # exact scheme plate.py uses (weights = plate detector confidence).
        score = [dict() for _ in range(P.N_SLOTS)]
        for entry in self.recognizers:
            preds = P._run_ocr(entry, [img for img, _ in plates])
            for pred, (_, w) in zip(preds, plates):
                txt = pred.plate[:P.N_SLOTS].ljust(P.N_SLOTS, P.PAD)
                probs = list(pred.char_probs)[:P.N_SLOTS]
                probs += [0.0] * (P.N_SLOTS - len(probs))
                for pos, (c, p) in enumerate(zip(txt, probs)):
                    score[pos][c] = score[pos].get(c, 0.0) + w * float(p)
                    score[pos][P.PAD] = (score[pos].get(P.PAD, 0.0)
                                         + w * (1.0 - float(p)) * 0.25)
        dists = []
        for pos in range(P.N_SLOTS):
            t = sum(score[pos].values()) or 1.0
            dists.append({c: v / t for c, v in score[pos].items()})

        seen, free = set(), []
        for text, s, _ in P.beam_candidates(dists, top * 4):
            t = text.rstrip(P.PAD).replace(P.PAD, '?')
            if t and t not in seen:
                seen.add(t)
                free.append({'plate': t, 'score': round(min(1.0, s), 3)})
        gr = [{'plate': c.plate, 'score': round(c.score, 3)}
              for c in P.gr_projected_candidates(dists, top)]
        if not free and not gr:
            return None
        # Headline = the Greek-projected candidate when available: on the
        # ground-truth clip the projection scored markedly better than the
        # free vote (P(true char) 0.21→0.41), and this deployment's vehicles
        # are predominantly Greek. Foreign plates remain visible in
        # 'candidates' (free list) right below.
        best = gr[0] if gr else free[0]
        plate_px = max(img.shape[1] for img, _ in plates)
        return {
            'plate': best['plate'],
            'score': best['score'],
            'candidates': free[:top],
            'gr_candidates': gr,
            'det_conf': round(max(w for _, w in plates), 3),
            'frames_used': len(plates),
            'plate_px': plate_px,
            # Reliability flag for the UI. Ground-truth field cases: a 1-crop
            # dusk read voted 95% for a plate KNOWN to be wrong, and even a
            # 13-frame vote on a ~70px plate stayed confidently wrong on 3 of
            # 7 characters — when every crop carries the same blur, consensus
            # is mistaken with conviction. 90px threshold: below it the read
            # renders as uncertain with alternatives up front; genuinely
            # close-up plates (150px+) earn the confident style.
            'low_conf': plate_px < 90 or len(plates) < 3 or best['score'] < 0.5,
            # Median relative plate position inside the vehicle box — lets
            # the report draw the yellow plate rectangle during replay.
            'rel_box': ([float(np.median([r[i] for r in rels]))
                         for i in range(4)] if rels else None),
        }
