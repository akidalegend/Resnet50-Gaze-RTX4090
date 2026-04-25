# QUICK REFERENCE CARD - THREE-STAGE TESTING

## STAGE 1: DIGITAL BENCHMARK (~10 sec)
**What**: Static image, ~0.5 to 1px jitter expected
**Setup**: Already configured, just run
**Run**: `python test_stages.py`
**Controls**: Press S to START, Q to QUIT
**Proves**: AI model is perfectly stable

```
Expected Output:
  Jitter: < 2 px (e.g., 0.42px)
  SNR: >> 5.0 (e.g., 4517)
  Status: STABLE ✓
```

---

## STAGE 2: SENSOR BENCHMARK (~10 sec)
**What**: Printed photo, 15-20px jitter expected
**Setup**: 
  1. Edit test_stages.py: INPUT_MODE = "WEBCAM"
  2. Print high-res face photo (A4+)
  3. Place on stand, NO HANDS (no shaking)
  4. Position 30-40cm from webcam

**Run**: `python test_stages.py`
**Controls**: Press S to START, Q to QUIT
**Proves**: Hardware (1080p + 4K scaling) is the noise source

```
Expected Output:
  Jitter: 15-20 px (e.g., 18.5px)
  SNR: Drops to 20-30 (e.g., 25.20)
  Status: UNSTABLE (but that's expected!)
  Key: See BIG JITTER jump from Stage 1
```

---

## STAGE 3: PHYSIOLOGICAL BENCHMARK (~10 sec)
**What**: You as subject, 20-25px jitter, SNR > 5.0 ✓
**Setup**:
  1. Keep INPUT_MODE = "WEBCAM"
  2. Build chin rest from heavy books
  3. Relax chin on books
  4. Stare straight ahead at screen center (don't move)
  5. Position: ~30-40cm from webcam, face centered

**Run**: `python test_stages.py`
**Controls**: Press S to START, Q to QUIT
**Proves**: 1D EMA filter + deadzone recovers usable signal

```
CRITICAL TARGET METRICS:
  Jitter: ~23.73 px  (±2px acceptable)
  SNR: ~7.29         (MUST be > 5.0 for Rose Criterion ✓)
  Std Dev: ~526 px
  Status: STABLE ✓ (SNR > 5.0)
```

---

## COMPARISON TABLE

| Metric | Stage 1 | Stage 2 | Stage 3 |
|--------|---------|---------|---------|
| Input | Static Image | Webcam+Photo | Webcam+You |
| Jitter | <2 px | ~18 px | ~24 px |
| SNR | >>5 | ~25 | ~7.3 |
| Proves | AI Stable | HW Noisy | Filter Works |

**Key Insight**: Jitter jumps Stage 1→2 (hardware culprit), stays high Stage 2→3 (eye + hardware), but SNR > 5.0 proves filter succeeds!

---

## TROUBLESHOOTING QUICK FIXES

| Problem | Fix |
|---------|-----|
| Face not detected | ↑ Brightness, move to 30-40cm |
| Black window | Camera permission, or `cap = cv2.VideoCapture(1)` for 2nd cam |
| High jitter in Stage 3 | Relax more, keep head still |
| SNR < 5 in Stage 3 | ↓ Alpha to 0.03, re-test |
| Model won't load | `python train_gaze_4090.py` to create epoch10.pth |

---

## SETUP PHOTOS/DIAGRAMS

### Stage 2 Setup (Printed Photo Test)
```
        ┌─────────────────┐
        │  Your Monitor   │
        │  (4K Display)   │
        └────────┬────────┘
                 │
        ┌────────┴────────┐
        │   Webcam (30cm) │
        │                 │
        └────────┬────────┘
                 │
        ╔════════════════╗
        ║   PRINTED FACE │  ← On stand, NO HANDS
        ║    PHOTO (A4+) │
        ║   Don't move!  │
        ╚════════════════╝
```

### Stage 3 Setup (Physiological Test)
```
        ┌─────────────────┐
        │  Your Monitor   │
        │  (4K Display)   │
        └────────┬────────┘
                 │
        ┌────────┴────────┐
        │   Webcam (30cm) │
        │                 │
        └────────┬────────┘
                 │
            ┌────────┐
            │  YOUR  │  ← Chin on book stack
            │ FACE   │     Stare straight (no move)
            │  😐    │
        ════════════════    ← Book stack (chin rest)
        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    ← Heavy books on desk
```

---

## OUTPUT FILES SAVED

After each stage:
- `snr_analysis_stage1_YYYYMMDD_HHMMSS.json` - Detailed metrics
- `snr_analysis_stage2_YYYYMMDD_HHMMSS.json`
- `snr_analysis_stage3_YYYYMMDD_HHMMSS.json`

Minimal JSON structure:
```json
{
  "stage": 1,
  "input_mode": "IMAGE",
  "analysis": {
    "horizontal": {
      "jitter_mean": 0.42,
      "snr": 4517.65,
      "std_sigma": 0.85,
      "status": "STABLE"
    }
  }
}
```

---

## DISSERTATION SUMMARY (2 min version)

Three validated benchmarks:

1. **Digital (Stage 1)**: Static naked image → SNR=4517, Jitter<1px
   - Proves ResNet-50 core is stable

2. **Sensor (Stage 2)**: Printed photo on stand → SNR=25, Jitter=18.5px
   - Proves 1080p webcam + 4K upscaling is noise source

3. **Physiological (Stage 3)**: Subject with chin rest → SNR=7.29, Jitter=23.73px
   - Proves 1D EMA filter EXCEEDS Rose Criterion (>5.0 ✓)
   - Proves system usable for ADHD applications despite hardware

---

## KEEP HANDY

📌 **Bookmark this file** or print it out before testing!

🎬 **Video Documentation Tip**: Screen-record Stage 3 to show:
   - Green dot (=EMA filtered signal) stays centered
   - Even though visible flicker in raw (if comparison ON)
   - Smooth gaze tracking despite noise

📊 **Share Results**: 
   - The three JSON files prove everything quantitatively
   - The jitter progression (0.42 → 18.5 → 23.73) tells the full story

✓ **Success = Stage 3 SNR > 5.0**

---

Questions? See **TESTING_PROTOCOL.md** for full documentation.

