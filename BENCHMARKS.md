# VisionX Benchmarks

## Detector: closed-set YOLO26 vs open-vocabulary YOLOE-26 (2026-07-19)

Σκοπός: με ΣΤΑΘΕΡΟ πεδίο στόχων (person, bicycle, car, motorcycle, bus,
truck) αξίζει το βάρος του open-vocabulary YOLOE; Μηχάνημα: MacBook M3 Pro,
PyTorch MPS, 640px, conf 0.35, 25 καρέ πραγματικού dashcam κλιπ, median
ms/frame με mps.synchronize. (Χωρίς ground-truth labels — detections/frame
και confidence είναι ενδεικτικά· η ταχύτητα είναι το σκληρό νούμερο.)

| Μοντέλο | Τύπος | ms/frame | fps | dets/frame | mean conf |
|---|---|---:|---:|---:|---:|
| yolo26n | closed-set | 14.8 | 67 | 2.9 | 0.60 |
| **yolo26s** | closed-set | **11.8** | **85** | 3.8 | **0.74** |
| yolo26m | closed-set | 22.0 | 45 | **4.5** | 0.73 |
| yoloe-26n-seg | open-vocab | 17.3 | 58 | 3.0 | 0.55 |
| yoloe-26x-seg (τρέχον default) | open-vocab | 73.6 | 14 | 4.1 | 0.71 |

## Σταθερότητα tracking ανά μοντέλο (2026-07-19 — το μέτρο που μετράει)

Ένσταση χρήστη (σωστή): «περισσότερες ανιχνεύσεις ≠ σωστές ανιχνεύσεις· τα
μικρά μοντέλα έχαναν το tracking». Μετρήθηκε ο ΚΑΤΑΚΕΡΜΑΤΙΣΜΟΣ: πλήρες
BoT-SORT ανά μοντέλο στο ίδιο κλιπ (~6 πραγματικά οχήματα, 37 καρέ).

| Μοντέλο | ms/fr (με tracking) | Tracklets | Μήκη tracks |
|---|---:|---:|---|
| yolo26n | 81.8 | 4 | [36,35,31,11] — ΧΑΝΕΙ 2 οχήματα |
| yolo26s | 49.2 | 5 | [37,37,36,21,10] |
| yolo26m | 54.0 | 6 | [37,37,36,31,17,3] |
| **yolo26l** | 63.3 | **6** | **[37,37,37,37,31,1] — 4 τέλεια tracks** |
| yolo26x | 84.3 | 6 | [37,37,37,33,13,5] |
| yoloe-26x (τρέχον default) | 112.5 | **7 (χειρότερο)** | [37,37,36,30,8,1,1] |

### Συμπεράσματα (αναθεωρημένα, quality-first)

1. Η ανησυχία του χρήστη για τα μικρά μοντέλα ΕΠΙΒΕΒΑΙΩΝΕΤΑΙ: το nano χάνει
   ολόκληρα αντικείμενα. Απορρίπτεται ως default.
2. Το τρέχον default (yoloe-26x-seg) είναι ταυτόχρονα το ΠΙΟ αργό και το ΠΙΟ
   κατακερματισμένο — χειρότερο και στα δύο.
3. **Σύσταση: yolo26l ως default** για το σταθερό πεδίο (καλύτερη
   σταθερότητα, ~1.8× ταχύτερο από το σημερινό)· yolo26x ως επιλογή
   «μέγιστης ποιότητας»· YOLOE ΜΟΝΟ όταν δίνονται custom prompts.
4. Εκκρεμεί πριν την αλλαγή: CoreML/TensorRT export έλεγχος για yolo26,
   ενημέρωση setup downloads, επιβεβαίωση σε NVIDIA laptop
   (`tests/bench_detectors.py` + `bench_stability` αυτο-επιλέγουν CUDA), και
   επανάληψη σε μεγάλο υλικό σταθερής κάμερας.

Αναπαραγωγή: `venv/bin/python bench_detectors.py` (το script ζει στο ιστορικό
της συνεδρίας· αν χρειαστεί ξανά, ξαναγράφεται σε 5').
