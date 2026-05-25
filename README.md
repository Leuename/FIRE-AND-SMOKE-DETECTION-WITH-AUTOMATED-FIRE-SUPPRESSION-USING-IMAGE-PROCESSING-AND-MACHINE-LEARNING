# Fire and Smoke Detection with Automated Fire Suppression

<p align="center">
  <img width="284" height="207" alt="Fire suppression prototype image 1" src="https://github.com/user-attachments/assets/dc90c2d6-4075-48e1-b61f-c53a3a32698d" />
  <img width="233" height="299" alt="Fire suppression prototype image 2" src="https://github.com/user-attachments/assets/caf49a6b-ae2a-4da6-8e14-5b6aa6ddd13e" />
  <img width="299" height="247" alt="Fire suppression prototype image 3" src="https://github.com/user-attachments/assets/da64bb2b-98a9-44d3-b03b-54b3775ebdc7" />
  <img width="217" height="384" alt="Fire suppression prototype image 4" src="https://github.com/user-attachments/assets/1e90d8de-8241-4fa4-888c-7b74a4e6fcfa" />
</p>

Real-time **fire/smoke detection + automated suppression** prototype that combines:

- **ML vision detection** (YOLO on an RTSP/IP camera feed) for early fire recognition
- **Thermal tracking** (IR camera) to precisely localize the heat source
- **Hardware automation** (Arduino over serial) to steer a turret and activate a suppression servo

> This repository contains an academic/prototype implementation focused on practical end-to-end integration: camera input → detection → tracking → motor control → suppression.

## Project Overview
The system runs as a state machine:

1. **Monitoring (RGB/IP camera via RTSP):** a YOLO model detects fire in the scene.
2. **Tracking (thermal/IR camera):** once fire is detected, the system switches to IR tracking, searches for a valid thermal target, and locks on.
3. **Suppressing:** when the target is centered (deadzone), a servo output is enabled to trigger suppression.
4. **Auto-reset:** if no fire is detected for a period, the turret recenters and returns to monitoring.

Key implementation entrypoint: **`fire_detection.py`**

## Tech Stack
- **Python**
- **OpenCV** (video capture + overlays + image processing)
- **Ultralytics YOLO + PyTorch** (ML inference)
- **NumPy**
- **Arduino serial control** (motor + servo commands)

## Key Features
- Dual-camera workflow (**RTSP “watch” camera** + **thermal “track” camera**)
- State machine control (**MONITORING → TRACKING → SUPPRESSING**)
- Adaptive thermal thresholding and contour filtering (to reduce false positives)
- Turret control with **deadzone centering**, **axis limits**, and **search/sweep mode**
- Emergency stop and automatic return-to-monitoring behavior

## Notes / Requirements (High Level)
- Requires a trained YOLO weights file (configured in code, e.g. `MODEL_PATH`).
- Requires camera connectivity (RTSP credentials/IP for the monitoring camera, and a thermal camera index for IR tracking).
- Requires an Arduino connected via serial for motor/servo actuation.

## Media
Project images are included at the top of this README.
