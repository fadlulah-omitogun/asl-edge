# Preliminary Results — Pi Pipeline (Fadlulah)
Last updated: 2026-11-04

## Hardware & software
| | Laptop | Raspberry Pi |
|---|---|---|
| Device | HP laptop (Windows) | Raspberry Pi 4 Model B, 4GB, FLIRC passive case |
| OS | Windows | Raspberry Pi OS Trixie (Debian 13), 64-bit |
| Python | 3.12 (conda env `asl`) | 3.12.16 (uv venv `~/asl-env`; system Python is 3.13) |
| MediaPipe | 0.10.18 | 0.10.18 (newest with Pi aarch64 build + `mp.solutions`) |
| OpenCV / numpy | 4.11.0 / 1.26.4 | 4.11.0 / 1.26.4 |
| Camera | Aluratek AWC04F USB webcam | same (`/dev/video0`) |
| Resolution | 640×480 | 640×480 |

## Hand-landmark benchmark (`pi/hand_fps.py`, 200 frames, 1 hand)
| Run | Device | model_complexity | MediaPipe ms/frame | End-to-end FPS | Hand found | RAM | CPU temp |
|---|---|---|---|---|---|---|---|
| 1 | Laptop | 1 | 16.1 | 28.6 | 17/200* | — | — |
| 2 | Pi 4 | 1 | 126.9 | 7.7 | 88/200* | 155 MB | 50.6 °C |
| 3 | Pi 4 | 0 (lite) | 74.2 | 11.2 | 114/200* | 143 MB | 48.2 °C |

*Detection counts are low because the hand was poorly framed (too close / partly out of view), not a model issue. Rerun with whole hand + wrist in view at ~50 cm.

Pi `vcgencmd get_throttled` = `0x0` (no power or thermal throttling).

## Observations
- The Pi 4 is ~8× slower than the laptop for MediaPipe (127 vs 16 ms/frame at full complexity).
- The lite model (`model_complexity=0`) cut Pi inference time ~42% (127 → 74 ms) and raised FPS 7.7 → 11.2.
- End-to-end FPS is below MediaPipe-only FPS on the Pi, so camera capture and color conversion add overhead (~15 ms/frame).
- Losing the hand forces MediaPipe to rerun palm detection every frame, so good framing should improve speed as well as detection.
- At ~11 FPS, holding a letter for 5 frames ≈ 0.45 s — acceptable for fingerspelling.

## Setup issues & fixes (for the README)
- MediaPipe 1.x removed `mp.solutions`; pinned 0.10.x instead.
- MediaPipe 0.10.21 has no Linux aarch64 wheel; 0.10.18 is the newest that installs on the Pi.
- Imager wrote Trixie (Python 3.13, unsupported by MediaPipe); used `uv` to install Python 3.12 instead of reflashing.
- SSH was refused on first boot; fixed by creating an empty `ssh` file on the bootfs partition.
- Campus Wi-Fi blocks device-to-device connections; used a phone hotspot (Pi at 172.20.10.5).

## To do
- [ ] Rerun runs 2-3 with proper framing; record hand-found rate
- [ ] Optional: try 320×240 and `min_detection_confidence=0.3`
- [ ] Add headless mode to `demo_app.py`; lower `GAP_FRAMES` to ~6-8 for the Pi
- [ ] Get speaker/earbuds working on the 3.5mm jack; test `espeak-ng "hello"`
- [ ] Plug in the existing model once Nigel's is ready
- [ ] Later: compare classifier accuracy at model_complexity 0 vs 1