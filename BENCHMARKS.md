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

### Συμπεράσματα

1. Για το σταθερό πεδίο ανθρώπων+οχημάτων, το **closed-set YOLO26 κυριαρχεί**:
   το yolo26m βρίσκει ΠΕΡΙΣΣΟΤΕΡΑ αντικείμενα από το τρέχον default
   (yoloe-26x) σε **3.3× την ταχύτητα**· το yolo26s σχεδόν εξίσου πολλά σε
   **5-6× την ταχύτητα** και με την υψηλότερη μέση βεβαιότητα.
2. Το YOLOE διατηρεί αξία ΜΟΝΟ όταν ο χρήστης δίνει ελεύθερα prompts
   («λευκό αυτοκίνητο», «σκύλος»).
3. Πρόταση αρχιτεκτονικής: **αυτόματη επιλογή μοντέλου** — χωρίς custom
   prompts ⇒ yolo26m (ή s για ταχύτητα), με custom prompts ⇒ YOLOE.
   Εκκρεμεί: CoreML/TensorRT export ελέγχος για yolo26, ενημέρωση setup
   downloads, και επαλήθευση σε NVIDIA laptop (η ιεραρχία ταχύτητας
   αναμένεται ίδια αλλά τα νούμερα θα μετρηθούν εκεί).

Αναπαραγωγή: `venv/bin/python bench_detectors.py` (το script ζει στο ιστορικό
της συνεδρίας· αν χρειαστεί ξανά, ξαναγράφεται σε 5').
