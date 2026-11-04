# Sign Language Translator — Real-Time ASL Alphabet & Word Recognition

> **Real-time sign language translator using MediaPipe Hands, PyTorch Static Residual MLP (v2), and Dynamic GRU Sequence Classifier (v1).**
> Immediate camera startup · Dual-mode static/dynamic toggle (`[m]`) · Temporal majority-vote smoothing · Automated benchmarks · Standalone Windows build.

> **Update:** Both AI models were originally trained mostly on synthetic (programmatically generated, non-photographic) data, which caused unreliable predictions on real hands. Both models have since been retrained on real data — see [Real-Data Retraining](#7-real-data-retraining-fix-for-synthetic-training-data) below.

---

## Directory Structure

```
Sign Language/
├── src/
│   ├── data/
│   │   ├── merge_datasets.py        # Processes public images to landmarks and merges
│   │   ├── audit.py                 # Audits static class distribution and imbalance
│   │   └── generate_dynamic_data.py # Synthesizes 30-frame dynamic word trajectory sequences
│   ├── models/
│   │   ├── mlp_static.py            # PyTorch MLP baseline static model definition
│   │   ├── static_model.py          # Track A — PyTorch StaticModelV2 (Residual MLP + LayerNorm)
│   │   └── dynamic_model.py         # Track B — PyTorch DynamicGRU sequence model & predictor
│   ├── inference/
│   │   ├── stability.py             # Temporal majority-vote prediction stabilizer
│   │   └── sequence_buffer.py       # Rolling 30-frame FIFO landmark buffer
│   ├── capture.py                   # Module 1 — Webcam capture with DirectShow fallback
│   ├── landmarks.py                 # Module 2 — MediaPipe landmark extraction (supports 2 hands)
│   ├── preprocessing.py             # Coordinate origin-centering, scaling, 3D augmentation & flex angles
│   ├── data_logger.py               # Module 3 — CSV & dynamic sequence logger
│   ├── data_utils.py                # Stratified train/val/test splits loader
│   ├── train.py                     # Track A — Static Model v2 training & hyperparameter tuning loop
│   ├── train_dynamic.py             # Track B — Dynamic GRU Model v1 training loop
│   └── main.py                      # Track C — Integrated dual-mode execution loop ([m] toggle)
├── scripts/
│   └── benchmark.py                 # Track C4 — Automated accuracy, F1, latency & FPS benchmark
├── data/
│   ├── landmarks.csv                # Merged static landmark coordinate dataset
│   └── dynamic/                     # Saved 30-frame sequence binary files (.npy)
├── docs/
│   ├── static_model_v2_card.md      # Model Card for Static Model v2
│   └── dynamic_model_v1_card.md     # Model Card for Dynamic GRU Model v1
├── results/
│   ├── mlp_v1_baseline.md           # Phase 1 baseline report (86.42% val accuracy)
│   ├── static_v2_metrics.txt        # Track A Static Model v2 test evaluation metrics
│   ├── dynamic_v1_metrics.txt       # Track B Dynamic GRU Model v1 test evaluation metrics
│   └── phase2_benchmark.md          # Track C4 complete automated benchmark report
├── models/
│   ├── hand_landmarker.task         # MediaPipe Hands task file
│   ├── mlp_v2.pt                    # Trained Static Model v2 PyTorch weights
│   ├── dynamic_v1.pt                # Trained Dynamic GRU Model v1 PyTorch weights
│   ├── class_labels.txt             # Static alphabet class labels list (24 letters)
│   └── dynamic_labels.txt           # Dynamic word vocabulary list (15 words)
└── requirements.txt                 # Python dependencies configuration
```

---

## Installation & Setup

1. Activate virtual environment:
```powershell
.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

---

## 1. Track A — Static Model v2 (Alphabet `A`–`Y`)

### Training Static Model v2
Train the upgraded `StaticModelV2` with 3D augmentation and residual blocks:
```powershell
python -u src/train.py --csv data/landmarks_real.csv --epochs 30 --augment --model-out models/mlp_v2.pt
```
*   **Current accuracy (real data):** **1.00 test accuracy** across all 24 static letters, evaluated on real photos + real self-recorded samples (see `results/static_v2_metrics.txt`). Previous figures (94.37% val / 94.52% test) were measured on a mostly-synthetic dataset and did not reflect real-world performance — see [section 7](#7-real-data-retraining-fix-for-synthetic-training-data).
*   Output artifacts saved to `models/mlp_v2.pt` and `results/static_v2_metrics.txt`.
*   To build `data/landmarks_real.csv` from a public image dataset instead of (or in addition to) manual recording, see `scripts/merge_from_zip.py`.

---

## 2. Track B — Dynamic Word Model v1 (15 Words)

### Generate / Record Dynamic Sequences

**Recommended: use real data.** The original synthetic generator below produces sequences with uniform sine-wave motion across all landmarks, which does not resemble real signing and was found to make the model unreliable in live testing. Prefer one of these real-data sources instead:
- `scripts/wlasl_to_sequences.py` — extracts real landmark sequences from the WLASL (Word-Level American Sign Language) video dataset for the target vocabulary, without needing to record yourself.
- `src/data_logger.py`, pressing `[d]` to start/stop recording your own real sequences.

Synthetic generation (fallback / supplementary only, not recommended as a sole data source):
```powershell
python -u src/data/generate_dynamic_data.py --output data/dynamic --samples 50
```

### Train Dynamic GRU Model v1
Train the 2-layer Bidirectional GRU sequence classifier on real sequences:
```powershell
python -u src/train_dynamic.py --data-dir data/dynamic_real --epochs 30
```
*   Output artifacts saved to `models/dynamic_v1.pt` and `results/dynamic_v1_metrics.txt`.
*   Note: real per-word sample counts from WLASL are typically much lower than the static model's per-letter counts (tens, not hundreds), so accuracy may vary more by word. Supplementing thin words with a few self-recorded samples via `data_logger.py` is recommended.

---

## 3. Track C — Integrated Live Dual-Mode Demo (`src/main.py`)

Launch the combined application supporting real-time webcam inference and mode toggling:
```powershell
python src/main.py
```

### Keypress Controls inside Application:
*   **`[m]`** : **Toggle Mode** between `Static (Alphabet)` and `Dynamic (Words)` without restarting.
*   **`[s]`** : Save current landmark frame to `data/landmarks.csv`.
*   **`[l]`** : Cycle active static target label.
*   **`[w]`** : Cycle active dynamic word target label.
*   **`[q]`** : Cleanly exit application window.

---

## 4. Track C4 — Automated Benchmarking (`scripts/benchmark.py`)

Run automated held-out test evaluation, inference latency benchmarking, and pipeline FPS measurement:
```powershell
python scripts/benchmark.py
```
*   Generates full benchmark report at `results/phase2_benchmark.md`.

---

## 5. Core AI & Inference Architecture (Author Contributions)

This repository features a production-ready AI inference engine with the following capabilities:
1. **TorchScript Export Pipeline:** Exported trained PyTorch static (`StaticModelV2`) and dynamic (`DynamicGRU`) models into optimized TorchScript (`.pt`) formats using `scripts/export_to_torchscript.py`.
2. **Unified Inference Wrapper:** Built the `SignPredictor` class (`src/inference/live_inference.py`) that seamlessly loads the compiled TorchScript models and label maps for fast, isolated inference.
3. **Motion-Segmented Sequence Buffer:** Integrated the `SequenceBuffer` with temporal interpolation to capture continuous 30-frame motion windows smoothly and trigger predictions upon gesture completion.
4. **End-to-End Integration Tests:** Designed and completed comprehensive integration tests (`scripts/test_integration.py`) verifying both static frame predictions and temporal sequence motion capture pipelines.

---

## 6. Phase 3 — Full Alpha Application (`src/ui_app.py`)

Turned the real-time pipeline into a usable application with a PyQt6 GUI, text accumulation, and text-to-speech.

### Launch the App
```powershell
python src/ui_app.py
```

### Features & Gesture Commands
- **Accumulated Sentence**: Build full sentences using the hold-to-confirm mechanism. Wait for the "Hold" progress to reach 100% to commit a prediction.
- **Gesture Commands** (When in DYNAMIC Mode):
  - **Space**: Sign the word `"more"`
  - **Delete/Backspace**: Sign the word `"no"`
  - **Clear Buffer**: Sign the word `"stop"`
- **Text-to-Speech**: Click "Speak Sentence" or toggle "Auto-Speak" to read text aloud using `pyttsx3`.
- **Robustness**: Application survives camera disconnects and handles 'no hand' frames gracefully.

**Known Limitations for Alpha**:
- J and Z are not supported — both require in-air motion to sign and are excluded from the static alphabet by design (not a bug). They could be added to the dynamic word model in the future.
- Webcams with poor lighting or high latency may drop frames or struggle with DYNAMIC sequences.
- Pyttsx3 TTS is synchronous internally and may cause a micro-stutter if auto-speak is enabled, although threaded.
- Fast signers might need to lower the "Hold Duration" slider in settings.
- Dynamic word accuracy varies more by word than static letter accuracy, due to lower real-sample counts per word (see section 7).

---

## 7. Real-Data Retraining (fix for synthetic training data)

**Problem found:** both models were originally trained mostly on synthetic, non-photographic data:
- Static model: 86% of training rows (7,200/8,400) came from `scripts/generate_asl_data.py`, a script that hand-codes approximate finger-curl values per letter — not real hand images.
- Dynamic model: 100% of training sequences came from `src/data/generate_dynamic_data.py`, which applies uniform sine-wave motion to all landmarks — not real signing motion.

This produced high reported accuracy (94-96%) because it was measured on held-out splits of the same synthetic distribution, but caused unreliable predictions on real hands in live use (e.g. static letters being misclassified).

**Fix applied:**
- `scripts/merge_from_zip.py` — extracts real landmarks directly from a zipped photo dataset (e.g. Kaggle ASL Alphabet) without extracting the archive to disk first.
- `scripts/wlasl_to_sequences.py` — extracts real landmark sequences from the WLASL video dataset for the dynamic word vocabulary.
- Both models retrained on real data only (plus real self-recorded samples). Static model now scores 1.00 test accuracy on real held-out data; dynamic model retrained on real signer videos instead of synthetic motion.
- Fixed several bugs surfaced during this work: MediaPipe VIDEO-mode timestamp monotonicity when reusing one detector across many videos, a frame-trim mismatch in WLASL's metadata vs. pre-trimmed clips, and a `train_dynamic.py` crash on classes too small to stratify-split.

**For anyone extending this project:** avoid relying on synthetic data generators (`generate_asl_data.py`, `generate_dynamic_data.py`) as a primary data source. They're useful for quick smoke-testing the pipeline, but do not generalize to real hands. Use `merge_from_zip.py`, `wlasl_to_sequences.py`, or `data_logger.py` for real training data instead.

---

## 8. Standalone Windows Build (.exe)

The app can be packaged into a standalone Windows executable that runs without Python installed, using PyInstaller.

### Prerequisites
```powershell
pip install pyinstaller
```

### Build
```powershell
pyinstaller --name SignLanguageTranslator --onedir --collect-all mediapipe --collect-all numpy --add-data "models;models" --add-data "docs;docs" --paths src src\ui_app.py
```

This produces `dist\SignLanguageTranslator\SignLanguageTranslator.exe`, along with an `_internal` folder that must stay alongside it (this is a `--onedir` build, not a single-file exe — this is intentional, as `--onefile` builds were less reliable with MediaPipe's bundled data).

### Notes for packaging
- `src/resource_path.py` resolves file paths correctly whether running from source or from inside the frozen executable. If you add new code that constructs paths relative to `__file__`, use `get_project_root()` from this module instead, or it will break in the packaged build.
- Some antivirus software (particularly enterprise/managed endpoint protection) may flag or lock files during the build process, since unsigned PyInstaller executables can trigger heuristic scans. If `Compress-Archive` fails with a file-in-use error when zipping the build for distribution, try 7-Zip instead, or use `robocopy` with retry flags (`/R:10 /W:2`) to copy the folder first.
- To distribute the build, zip the entire `dist\SignLanguageTranslator` folder (not just the `.exe`) and share via GitHub Releases.
#   p r o j e c t r e p o r t _ 2 3 0 5 1 7 7 5  
 #   s i g n _ l a n g u a g e -  
 