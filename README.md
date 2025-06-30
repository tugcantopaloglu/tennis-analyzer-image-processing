# Tennis Match Analysis System

## Project Description

This project is a real‑time analysis system for tennis matches that leverages **computer‑vision** techniques.  
From a live or recorded video stream it performs **court detection, player tracking, and ball tracking**.

---

## Features

### Core

| Feature                       | Description                                                   |
| ----------------------------- | ------------------------------------------------------------- |
| **Automatic Court Detection** | Detects court lines via Hough transform                       |
| **Player Tracking**           | Tracks two players independently with a Kalman filter         |
| **Ball Tracking**             | Detects and follows the tennis ball with a smart algorithm    |
| **Mini‑Map View**             | Real‑time 2‑D court sketch                                    |
| **State Analysis**            | Automatically switches between **GAME** and **REPLAY** states |

### Advanced

| Feature                 | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| **Adaptive Smoothing**  | Removes jitter in player positions                    |
| **Line Filtering**      | Prevents court lines from being classified as players |
| **Video Speed Control** | Adjustable playback speed 0.25× – 4×                  |
| **Debug Mode**          | Shows motion mask & filter visuals                    |
| **Blurred Background**  | Aesthetic blur for areas outside the court            |

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### File Layout

```
project_folder/
├── main_final.py        # Main entry point
├── tennis.mp4           # Video to analyse
├── court.png            # Court mini‑map image
├── requirements.txt     # Python deps
└── README.md
```

---

## 🚀 Usage

### Basic Run

```bash
python main_final.py
```

### Run in Debug Mode

Set `IS_DEBUG = True` in the code:

```python
IS_DEBUG = True  # Shows the motion‑mask window
```

---

## Controls

| Key | Action                  |
| --- | ----------------------- |
| `q` | Quit                    |
| `p` | Pause / Resume          |
| `+` | Increase playback speed |
| `-` | Decrease playback speed |
| `r` | Reset to normal speed   |

---

## Windows

The system opens **3 windows** (2 if DEBUG is off):

1. **Tennis Match Analysis** – Main video

   - Player boxes (Player 1: black, Player 2: white)
   - Ball detection & trail (yellow)
   - State indicator (GAME / REPLAY)
   - Blurred background effect

2. **Motion Mask** _(Debug only)_

   - Motion‑detection result
   - Colour heat‑map view

3. **Court Mini‑Map**
   - 2‑D court representation
   - Real‑time player positions
   - Ball trail (with fade)
   - State indicator

---

## Technical Details

### Algorithms

| Task                 | Method                                      |
| -------------------- | ------------------------------------------- |
| Court Detection      | Hough Line Transform + geometric validation |
| Player Tracking      | Kalman Filter + Hungarian assignment        |
| Ball Detection       | Contour analysis + temporal consistency     |
| Coordinate Transform | Homography matrix                           |

### Filters

- **Line Masking** – removes white court lines from motion detection
- **Morphological Ops** – dilate/erode for noise reduction
- **Adaptive Smoothing** – variable smoothing for small vs. large movements

### Performance Optimisations

- **ROI‑based processing**
- Smart search algorithms
- State‑stability filters

---

## Parameters

Main parameters are defined near the top of the code:

```python
# Video & display
TARGET_WIDTH = 800
TARGET_MAP_W, TARGET_MAP_H = 400, 650

# Player‑tracking
MAX_PLAYER_JUMP_DIST = 150
MAX_PLAYER_INVISIBLE_FRAMES = 35

# State stability
STATE_STABILITY_THRESH = 15
```

---

## Troubleshooting

### Video Not Found

- Ensure `tennis.mp4` is in the same folder as the code.
- Check that the video format is supported by OpenCV.

### Mini‑Map Not Displayed

- Make sure `court.png` exists.
- Confirm the image is PNG.

### Performance Issues

- Try reducing video resolution.
- Disable debug mode (`IS_DEBUG = False`).

Enjoy analysing your matches! 🎾
