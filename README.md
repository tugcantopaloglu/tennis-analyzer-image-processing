# Tennis Match Analysis System

An OpenCV prototype for processing a recorded tennis video. It detects court lines, tracks motion-based player candidates with Kalman filters, and searches for ball-shaped contours. A second window maps tracked positions onto a court image.

## Setup and run

Use Python 3.11 or later and a desktop session that supports OpenCV windows. Local validation uses Python 3.13.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies and start the analyzer:

```bash
python -m pip install -r requirements.txt
python main.py
```

The repository includes `tennis.mp4` and `court.png`. Default paths resolve relative to `main.py`, so launching the script from another directory also finds these assets. To analyze another recording or use another court image, edit `VIDEO_DOSYA_YOL` or `KORT_RESIM_YOL` near the top of `main.py`. There is no command-line argument interface or live-camera input path.

An unreadable video ends processing with a warning. A missing or unreadable court image shows a placeholder mini-map while video processing continues. OpenCV codec support determines which recordings can be decoded.

## Displays and controls

The application opens `Tenis Mac Analizi` for the annotated video and `Kort` for the court mini-map. Set `IS_DEBUG = True` to also open `Hareket Maskesi`, which displays the motion mask when a court view is active.

| Key | Action |
| --- | --- |
| `q` | Quit during playback |
| `p` | Pause; press any key to resume |
| `+` | Increase playback speed, up to 4x |
| `-` | Decrease playback speed, down to 0.25x |
| `r` | Reset playback speed to 1x |

Player markers are black for the upper side and white for the lower side. The ball and its fading trail are yellow. Areas outside the detected court are blurred. `GAME` and `REPLAY` labels use court visibility and frame-count stability thresholds; they do not recognize broadcast replays semantically. Requested playback speed depends on processing time and the display event loop.

## Processing

| Task | Implementation |
| --- | --- |
| Court detection | Thresholding, Hough line segments, intersections, and geometric validation |
| Motion extraction | MOG2 background subtraction inside a padded court polygon |
| Player tracking | Contour size/shape filters, nearby box merging, and nearest-distance matching per court side |
| Position prediction | Kalman filters, with removal after too many unmatched frames |
| Ball detection | Contour area/shape filtering, player-box exclusion, and search near the last position |
| Mini-map | Homography, per-tracker smoothing, visible-bound clipping, and a fading ball trail |

Tuning constants retain their original names in `main.py`, including `HEDEF_GENISLIK` (video display width), `HEDEF_KROKI_GENISLIK` and `HEDEF_KROKI_YUKSEKLIK` (mini-map size), `MAKS_OYUNCU_ZIPLAMA_MESAFE` (player matching distance), `MAKS_OYUNCU_GORUNMEZ_KARE` (tracking timeout), and `DURUM_STABILITE_ESIGI` (status stability).

This is a heuristic analyzer for a suitable fixed court view. Camera cuts, moving cameras, occlusion, lighting changes, spectators, and small or blurred balls can cause missed detections or false positives. It does not use a trained model, annotated evaluation dataset, Hungarian assignment, or a dedicated white-line removal mask. No tracking accuracy, real-time throughput, or replay-classification metrics have been measured here.

## Validation

Run the small local regression suite:

```bash
python -m unittest discover -s tests -v
```

Tests generate two-frame videos and PNG images in temporary directories, decode them with OpenCV, and exercise court/ball detection and mini-map drawing on synthetic fixtures. Window display and keyboard input are mocked, so the suite can run without opening desktop windows. Capture wrappers also simulate invalid FPS metadata and unsupported seeking; one test adapts real Hough results to the older OpenCV line-array layout. The suite checks asset resolution, malformed inputs, cleanup, nonblocking playback delays, first-frame handling, Hough result layouts, and tracker-independent mini-map state. It does not establish accuracy on real matches or validate interactive desktop behavior.

The GitHub Actions workflow runs these tests and source/dependency security checks on Windows with Python 3.13. To run the security checks locally:

```bash
python -m pip install bandit pip-audit
python -m bandit main.py
python -m pip_audit -r requirements.txt
```

See [LICENSE](LICENSE) for the repository license.
