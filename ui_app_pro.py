"""Professional desktop interface for real-time sign language recognition.

Features:
- Dual-Engine recognition: Static (Alphabet A-Y) and Dynamic (Words)
- Advanced AR/HUD hand tracking: anti-aliased skeleton, glowing joints, corner brackets, handedness badges
- Dual-hand capture support for 2-handed words and single-hand signs with persistent spatial tracking
- Configurable capturing time duration & hold duration with live countdown/progress bars
- Modern dark obsidian & frosted glassmorphic UI with studio viewfinder camera frame
"""

import math
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtCore import QRectF
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtGui import (
    QColor, QFont, QImage, QPainter, QPen, QPixmap,
    QShortcut, QKeySequence, QTextDocument,
)
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QProgressBar, QPushButton, QScrollArea, QSlider,
    QTextEdit, QVBoxLayout, QFileDialog, QMessageBox,
    QWidget, QButtonGroup,
)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

COMMANDS = {"more": "space", "no": "delete", "stop": "clear"}
STATIC_METHODS = ("predict_static", "predict_frame", "predict_letter", "predict")
DYNAMIC_METHODS = ("predict_dynamic", "predict_sequence", "predict_word", "predict")

DYNAMIC_VOCABULARY = [
    "drink", "eat", "friend", "hello", "help", "love", "more", "name",
    "no", "please", "sorry", "stop", "thanks", "want", "yes"
]
DYNAMIC_VOCABULARY = sorted(set(DYNAMIC_VOCABULARY + [
    "bye", "good", "morning", "goodbye", "good morning", "good night",
    "thank you", "sorry", "please", "help", "water", "food", "bathroom",
    "book", "family", "school", "home", "work", "today", "tomorrow",
    "yesterday", "mother", "father", "teacher", "doctor", "understand",
    "again", "slow", "meet", "nice", "who", "what", "where", "when",
    "why", "how", "fine", "happy", "sad", "welcome",
]))

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),       # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),     # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky + palm
]

# Color palettes (BGR for OpenCV)
COLOR_RIGHT_BONE = (255, 215, 0)     # Electric Cyan (BGR)
COLOR_RIGHT_GLOW = (255, 160, 0)     # Cyan glow
COLOR_LEFT_BONE = (240, 80, 220)      # Neon Violet / Magenta (BGR)
COLOR_LEFT_GLOW = (200, 40, 180)      # Violet glow
COLOR_WHITE_CORE = (255, 255, 255)
COLOR_HUD_BG = (12, 16, 26)
COLOR_HUD_BORDER = (45, 60, 90)


def normalize_prediction(result):
    if result is None:
        return "", 0.0
    if isinstance(result, dict):
        return str(result.get("label", result.get("prediction", ""))), float(
            result.get("confidence", result.get("conf", 1.0))
        )
    if isinstance(result, (tuple, list)) and result:
        confidence = float(result[1]) if len(result) > 1 else 1.0
        return str(result[0]), confidence
    return str(result), 1.0


def resample_sequence(sequence: np.ndarray, target_length: int = 30) -> np.ndarray:
    """Linear temporal interpolation of sequence to target_length frames."""
    num_frames, num_features = sequence.shape
    if num_frames == target_length:
        return sequence.copy()
    if num_frames == 0:
        return np.zeros((target_length, num_features), dtype=np.float32)
    if num_frames == 1:
        return np.tile(sequence[0], (target_length, 1)).astype(np.float32)

    old_indices = np.linspace(0, num_frames - 1, num_frames)
    new_indices = np.linspace(0, num_frames - 1, target_length)
    resampled = np.zeros((target_length, num_features), dtype=np.float32)
    for feat_idx in range(num_features):
        resampled[:, feat_idx] = np.interp(new_indices, old_indices, sequence[:, feat_idx])
    return resampled


class RecognitionBackend:
    def __init__(self):
        from inference.live_inference import SignPredictor
        self.predictor = SignPredictor(str(ROOT / "models"))

    def predict(self, mode, data):
        methods = STATIC_METHODS if mode == "static" else DYNAMIC_METHODS
        failures = []
        for method_name in methods:
            method = getattr(self.predictor, method_name, None)
            if method is None:
                continue
            try:
                return normalize_prediction(method(data))
            except Exception as error:
                failures.append(f"{method_name}: {error}")
        available = [name for name in dir(self.predictor) if not name.startswith("_")]
        raise RuntimeError(
            f"Prediction failed ({'; '.join(failures) or 'no prediction method found'}). "
            f"Available methods: {', '.join(available)}"
        )


class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    prediction_ready = pyqtSignal(str, float, bool)
    status_changed = pyqtSignal(str, bool)
    capture_progress_changed = pyqtSignal(float, str)  # (progress 0.0-1.0, status_text)

    def __init__(self):
        super().__init__()
        self.mode = "static"              # "static" or "dynamic"
        self.running = True
        self.camera_enabled = False
        self.dynamic_window_frames = 18   # Fast 18-frame capture window
        self.prev_tracks = {}             # Track centers to stabilize hand slots: 0 -> (cx, cy), 1 -> (cx, cy)
        self.fps = 30.0

    def open_camera(self):
        try:
            from capture import open_camera as project_open_camera
            for _ in range(3):
                capture = project_open_camera(0, allow_mock=False)
                for _ in range(15):
                    success, frame = capture.read()
                    if success and frame is not None and frame.size > 0:
                        if float(frame.mean()) > 1.0:
                            return capture
                    time.sleep(0.05)
                capture.release()
                time.sleep(0.25)
        except Exception as error:
            self.status_changed.emit(f"Camera startup failed: {error}", True)
        return None

    def draw_hand_hud_professional(self, frame, hand_landmarks, hand_name: str, hand_idx: int):
        """Draw high-tech HUD skeleton, glowing joints, corner brackets, and handedness tag."""
        h, w = frame.shape[:2]
        is_right = ("right" in hand_name.lower()) or (hand_idx == 0 and "left" not in hand_name.lower())
        bone_color = COLOR_RIGHT_BONE if is_right else COLOR_LEFT_BONE
        glow_color = COLOR_RIGHT_GLOW if is_right else COLOR_LEFT_GLOW
        badge_title = "R • RIGHT HAND" if is_right else "L • LEFT HAND"

        points = []
        xs, ys = [], []
        for lm in hand_landmarks:
            px = int(np.clip(lm.x * w, 0, w - 1))
            py = int(np.clip(lm.y * h, 0, h - 1))
            points.append((px, py))
            xs.append(px)
            ys.append(py)

        if not points:
            return

        # 1. Bounding Box with Sci-Fi Corner Brackets
        pad = 22
        min_x, max_x = max(0, min(xs) - pad), min(w - 1, max(xs) + pad)
        min_y, max_y = max(0, min(ys) - pad), min(h - 1, max(ys) + pad)
        box_w, box_h = max_x - min_x, max_y - min_y
        corner_len = max(12, min(28, int(min(box_w, box_h) * 0.24)))

        # Subtle backdrop tint
        overlay = frame.copy()
        cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), (10, 15, 25), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # 4 Corner brackets (┌ ┐ └ ┘)
        cv2.line(frame, (min_x, min_y), (min_x + corner_len, min_y), bone_color, 2, cv2.LINE_AA)
        cv2.line(frame, (min_x, min_y), (min_x, min_y + corner_len), bone_color, 2, cv2.LINE_AA)
        cv2.line(frame, (max_x, min_y), (max_x - corner_len, min_y), bone_color, 2, cv2.LINE_AA)
        cv2.line(frame, (max_x, min_y), (max_x, min_y + corner_len), bone_color, 2, cv2.LINE_AA)
        cv2.line(frame, (min_x, max_y), (min_x + corner_len, max_y), bone_color, 2, cv2.LINE_AA)
        cv2.line(frame, (min_x, max_y), (min_x, min_y + corner_len), bone_color, 2, cv2.LINE_AA)
        cv2.line(frame, (max_x, max_y), (max_x - corner_len, max_y), bone_color, 2, cv2.LINE_AA)
        cv2.line(frame, (max_x, max_y), (max_x, max_y - corner_len), bone_color, 2, cv2.LINE_AA)

        # 2. Floating Handedness Pill Badge
        font = cv2.FONT_HERSHEY_SIMPLEX
        p_scale, p_thick = 0.44, 1
        (pw, ph), _ = cv2.getTextSize(badge_title, font, p_scale, p_thick)
        pill_x = max(10, min(w - pw - 18, min_x))
        pill_y = max(ph + 12, min_y - 8)

        pill_overlay = frame.copy()
        cv2.rectangle(pill_overlay, (pill_x - 6, pill_y - ph - 6), (pill_x + pw + 6, pill_y + 4), COLOR_HUD_BG, -1)
        cv2.addWeighted(pill_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (pill_x - 6, pill_y - ph - 6), (pill_x + pw + 6, pill_y + 4), glow_color, 1, cv2.LINE_AA)
        cv2.putText(frame, badge_title, (pill_x, pill_y), font, p_scale, bone_color, p_thick, cv2.LINE_AA)

        # 3. Palm Center Tracking Reticle
        if len(points) > 9:
            palm_cx = (points[0][0] + points[9][0]) // 2
            palm_cy = (points[0][1] + points[9][1]) // 2
            cv2.circle(frame, (palm_cx, palm_cy), 8, glow_color, 1, cv2.LINE_AA)
            cv2.line(frame, (palm_cx - 12, palm_cy), (palm_cx + 12, palm_cy), glow_color, 1, cv2.LINE_AA)
            cv2.line(frame, (palm_cx, palm_cy - 12), (palm_cx, palm_cy + 12), glow_color, 1, cv2.LINE_AA)

        # 4. Anti-Aliased Skeletal Connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                # Outer glow line
                cv2.line(frame, points[start_idx], points[end_idx], glow_color, 3, cv2.LINE_AA)
                # Inner crisp bone line
                cv2.line(frame, points[start_idx], points[end_idx], bone_color, 1, cv2.LINE_AA)

        # 5. Dual-Layer Glowing Landmark Nodes
        fingertip_indices = {4, 8, 12, 16, 20}
        for idx, (px, py) in enumerate(points):
            if idx in fingertip_indices:
                cv2.circle(frame, (px, py), 6, glow_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 6, COLOR_WHITE_CORE, 1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 2, COLOR_WHITE_CORE, -1, cv2.LINE_AA)
            elif idx == 0:
                cv2.circle(frame, (px, py), 7, glow_color, 2, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 3, COLOR_WHITE_CORE, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 4, glow_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 2, COLOR_WHITE_CORE, -1, cv2.LINE_AA)

    def draw_viewfinder_hud(self, frame, hand_count: int, motion: float, live_text: str = ""):
        """Render top viewfinder banner and status telemetry."""
        h, w = frame.shape[:2]

        # Top Bar Container
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (8, 12, 20), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.line(frame, (0, 42), (w, 42), (32, 44, 68), 1, cv2.LINE_AA)

        # Live Status Pulse
        pulse_color = (0, 230, 118) if hand_count > 0 else (120, 140, 160)
        cv2.circle(frame, (18, 21), 6, pulse_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (18, 21), 8, pulse_color, 1, cv2.LINE_AA)

        status_text = "AI ENGINE ACTIVE"
        if hand_count == 1:
            status_text = "TRACKING: 1 HAND DETECTED"
        elif hand_count >= 2:
            status_text = "TRACKING: 2 HANDS DETECTED (DUAL-HAND WORDS)"
        cv2.putText(frame, status_text, (34, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 230, 245), 1, cv2.LINE_AA)

        # Right Mode Badge
        mode_str = "MODE: STATIC (ALPHABET)" if self.mode == "static" else "MODE: DYNAMIC (WORDS)"
        mode_color = COLOR_RIGHT_BONE if self.mode == "static" else COLOR_LEFT_BONE
        (mw, _), _ = cv2.getTextSize(mode_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(frame, mode_str, (w - mw - 18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, mode_color, 1, cv2.LINE_AA)

        # Bottom Live Subtitle Bar if text exists
        if live_text:
            sub_y = h - 20
            (tw, th), _ = cv2.getTextSize(live_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            sub_x = (w - tw) // 2
            sub_overlay = frame.copy()
            cv2.rectangle(sub_overlay, (sub_x - 14, sub_y - th - 8), (sub_x + tw + 14, sub_y + 8), (10, 14, 24), -1)
            cv2.addWeighted(sub_overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (sub_x - 14, sub_y - th - 8), (sub_x + tw + 14, sub_y + 8), (60, 80, 120), 1, cv2.LINE_AA)
            cv2.putText(frame, live_text, (sub_x, sub_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    def run(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mediapipe_python
            from mediapipe.tasks.python import vision
            from preprocessing import normalize_flat_vector

            landmarker = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=mediapipe_python.BaseOptions(
                        model_asset_path=str(ROOT / "models" / "hand_landmarker.task")
                    ),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.30,
                    min_hand_presence_confidence=0.30,
                    min_tracking_confidence=0.30,
                )
            )
            backend = RecognitionBackend()
        except Exception as error:
            self.status_changed.emit(f"Initialization failed: {error}", True)
            return

        capture = None

        sequence = deque(maxlen=60)
        start_time = time.monotonic()
        frame_count = 0
        previous_hand_count = -1
        prev_features = None
        cooldown_until = 0.0

        if self.camera_enabled:
            self.status_changed.emit("Camera connected • Keep hands in view", False)
        else:
            self.status_changed.emit("Camera is fully off", False)

        while self.running:
            if not self.camera_enabled:
                if capture is not None:
                    capture.release()
                    capture = None
                self.status_changed.emit("Camera is fully off", False)
                time.sleep(0.15)
                continue

            if capture is None:
                capture = self.open_camera()
                if capture is None:
                    self.status_changed.emit("Camera unavailable. Check privacy and camera settings.", True)
                    time.sleep(0.5)
                    continue
                self.status_changed.emit("Camera connected • Keep hands in view", False)
                sequence.clear()
                prev_features = None

            success, frame = capture.read()
            if not success or frame is None:
                capture.release()
                capture = None
                if self.camera_enabled:
                    self.status_changed.emit("Camera connection interrupted. Waiting for Camera On...", True)
                time.sleep(0.25)
                continue
            if not self.camera_enabled:
                capture.release()
                capture = None
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = max(0, int((time.monotonic() - start_time) * 1000))

            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                timestamp_ms,
            )

            label, confidence, hand_present = "", 0.0, bool(result.hand_landmarks)
            hand_count = len(result.hand_landmarks)

            if hand_count != previous_hand_count:
                previous_hand_count = hand_count
                if hand_count == 1:
                    self.status_changed.emit("One hand detected • Ready for recognition", False)
                elif hand_count >= 2:
                    self.status_changed.emit("Two hands detected • Dual-hand word tracking active", False)

            motion = 0.0
            live_status_hint = ""

            if hand_present:
                height, width = frame.shape[:2]
                raw_features = np.zeros(126, dtype=np.float32)
                handed_predictions = getattr(result, "handedness", [])

                # Disambiguate and assign canonical slots (Slot 0 = Right, Slot 1 = Left)
                # To prevent slot-swapping in dynamic sequences, order stably:
                detected_hands = []
                for idx, lms in enumerate(result.hand_landmarks):
                    raw_pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
                    cx = float(np.mean(raw_pts[:, 0]))
                    cy = float(np.mean(raw_pts[:, 1]))

                    name = "Hand"
                    if idx < len(handed_predictions) and handed_predictions[idx]:
                        name = str(handed_predictions[idx][0].category_name).title()
                    detected_hands.append({"landmarks": lms, "raw": raw_pts, "name": name, "cx": cx, "cy": cy})

                if len(detected_hands) == 1:
                    # Single hand detected
                    h_info = detected_hands[0]
                    # In Static mode: slot 0 is the primary letter slot for either hand
                    # In Dynamic mode: slot 0 for single-hand words (hello, yes, no)
                    raw_features[0:63] = h_info["raw"].flatten()
                    self.draw_hand_hud_professional(frame, h_info["landmarks"], h_info["name"], 0)
                elif len(detected_hands) >= 2:
                    # Dual hands detected
                    # Disambiguate: In mirrored camera feed, user's physical right hand is on right half (higher cx)
                    # and left hand is on left half (lower cx)
                    h1, h2 = detected_hands[0], detected_hands[1]
                    if "right" in h1["name"].lower() and "left" in h2["name"].lower():
                        right_h, left_h = h1, h2
                    elif "left" in h1["name"].lower() and "right" in h2["name"].lower():
                        right_h, left_h = h2, h1
                    else:
                        # Fallback by horizontal coordinate in mirrored view
                        if h1["cx"] >= h2["cx"]:
                            right_h, left_h = h1, h2
                        else:
                            right_h, left_h = h2, h1

                    raw_features[0:63] = right_h["raw"].flatten()
                    raw_features[63:126] = left_h["raw"].flatten()

                    self.draw_hand_hud_professional(frame, right_h["landmarks"], "Right", 0)
                    self.draw_hand_hud_professional(frame, left_h["landmarks"], "Left", 1)

                features = np.asarray(normalize_flat_vector(raw_features.tolist()), dtype=np.float32)

                if prev_features is not None:
                    motion = float(np.mean(np.abs(features - prev_features)))
                prev_features = features.copy()

                try:
                    if self.mode == "static":
                        # Static Alphabet Mode: predict every stable frame
                        label, confidence = backend.predict("static", features)
                        sequence.clear()
                    else:
                        # Dynamic Word Mode: collect motion sequence across target capturing window
                        sequence.append(features)
                        frame_count += 1
                        target_f = max(12, min(30, self.dynamic_window_frames))
                        cur_f = min(target_f, len(sequence))
                        progress = cur_f / float(target_f)

                        now = time.monotonic()
                        if now < cooldown_until:
                            live_status_hint = "WORD CONFIRMED • COOLDOWN"
                            self.capture_progress_changed.emit(1.0, live_status_hint)
                        elif len(sequence) >= target_f and motion > 0.0006:
                            # Resample captured sequence of length target_f to exactly 30 frames for the model
                            seq_arr = np.array(list(sequence)[-target_f:], dtype=np.float32)
                            resampled = resample_sequence(seq_arr, target_length=30)
                            model_input = resampled[np.newaxis, ...]  # shape (1, 30, 126)

                            label, confidence = backend.predict("dynamic", model_input)
                            if label and confidence >= 0.40:
                                cooldown_until = now + 0.6  # 600ms cooldown to avoid repeat triggers
                                live_status_hint = f"WORD: {label.upper()} ({confidence*100:.0f}%)"
                            self.capture_progress_changed.emit(1.0, f"Captured: {label.upper()}")
                        else:
                            live_status_hint = f"CAPTURING MOTION: {cur_f}/{target_f}f ({progress*100:.0f}%)"
                            self.capture_progress_changed.emit(progress, live_status_hint)

                except Exception as error:
                    self.status_changed.emit(f"Recognition error: {error}", True)
            else:
                sequence.clear()
                prev_features = None
                frame_count = 0
                self.capture_progress_changed.emit(0.0, "Waiting for hands...")

            self.prediction_ready.emit(label, confidence, hand_present)

            # Draw top viewfinder HUD
            self.draw_viewfinder_hud(frame, hand_count, motion, live_text=live_status_hint)

            display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = display_rgb.shape
            image = QImage(
                display_rgb.data,
                width,
                height,
                channels * width,
                QImage.Format.Format_RGB888,
            )
            self.frame_ready.emit(image.copy())

        if capture is not None:
            capture.release()

    def stop(self):
        self.running = False
        self.wait(3000)

    def set_camera_enabled(self, enabled: bool):
        self.camera_enabled = enabled


QSS_THEME = """
QMainWindow, QWidget {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #07111f, stop:0.55 #0b1424, stop:1 #111827);
    color: #eaf0f7;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Arial, sans-serif;
    font-size: 13px;
}

/* Glassmorphic Surfaces */
QFrame#surface {
    background: rgba(18, 28, 48, 235);
    border: 1px solid #273650;
    border-radius: 14px;
}
QFrame#surface:hover {
    border-color: #2e3d5b;
}

/* Viewfinder Camera Surface Frame */
QFrame#videoSurface {
    background: #030712;
    border: 1px solid #2a4568;
    border-radius: 16px;
}

/* Typography Hierarchy */
QLabel#title {
    font-size: 22px;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.3px;
}
QLabel#subtitle {
    color: #94a3b8;
    font-size: 12px;
}
QLabel#section {
    color: #64748b;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}
QLabel#prediction {
    font-size: 42px;
    font-weight: 800;
    color: #00e5ff;
    padding: 2px;
}
QLabel#predictionStatus {
    font-size: 12px;
    font-weight: 600;
    color: #cbd5e1;
}

QLabel#statusOk {
    color: #10b981;
    font-weight: 600;
}
QLabel#statusError {
    color: #f87171;
    font-weight: 600;
}

QLabel#metric {
    background: #111b2d;
    border: 1px solid #263650;
    border-radius: 7px;
    padding: 6px 10px;
    color: #a8b7ca;
    font-size: 11px;
    font-weight: 600;
}
QLabel#signGuide {
    background: #0b1628;
    border: 1px solid #27415f;
    border-radius: 8px;
    padding: 8px 10px;
    color: #cbd5e1;
    font-size: 13px;
    line-height: 150%;
}

/* Buttons */
QPushButton {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 9px;
    padding: 7px 14px;
    font-weight: 600;
    color: #f1f5f9;
}
QPushButton:hover {
    border-color: #00e5ff;
    background: #25334a;
}
QPushButton:pressed {
    background: #131c2e;
}
QPushButton:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #00e5ff);
    border-color: #38bdf8;
    color: #050811;
    font-weight: 700;
}

QPushButton#modeStatic:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0369a1, stop:1 #00e5ff);
    border-color: #38bdf8;
    color: #04101c;
    font-weight: 800;
}
QPushButton#modeDynamic:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7e22ce, stop:1 #c084fc);
    border-color: #d8b4fe;
    color: #10041d;
    font-weight: 800;
}

QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981);
    border: 1px solid #34d399;
    color: #ffffff;
    font-weight: 700;
}
QPushButton#primary:hover {
    background: #10b981;
}

QPushButton#presetBtn {
    background: #131d2e;
    border: 1px solid #233148;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 11px;
    color: #94a3b8;
}
QPushButton#presetBtn:hover {
    color: #38bdf8;
    border-color: #38bdf8;
}

/* Reference Chips */
QPushButton#chip {
    background: #111a2c;
    border: 1px solid #24324c;
    border-radius: 6px;
    padding: 5px 6px;
    font-size: 11px;
    font-weight: 600;
    color: #cbd5e1;
}
QPushButton#chip:hover {
    background: #1a273f;
    border-color: #38bdf8;
    color: #38bdf8;
}

/* Text Editors */
QTextEdit {
    background: #090e1a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 10px;
    font-size: 18px;
    color: #f8fafc;
    line-height: 1.4;
}

/* Progress Bars */
QProgressBar {
    background: #131d2f;
    border: 1px solid #1e2a3f;
    border-radius: 5px;
    height: 10px;
    text-align: center;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981);
    border-radius: 4px;
}
QProgressBar#holding::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7e22ce, stop:1 #c084fc);
    border-radius: 4px;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 6px;
    background: #192438;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #00e5ff);
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #00e5ff;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QCheckBox {
    spacing: 8px;
    color: #94a3b8;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #334155;
    border-radius: 4px;
    background: #131c2e;
}
QCheckBox::indicator:checked {
    background: #0284c7;
    border-color: #38bdf8;
}
"""


class TranslatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Translator | Professional Dual-Engine Studio")
        self.resize(1366, 768)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(QSS_THEME)

        self.votes = deque(maxlen=9)
        self.current_candidate = None
        self.candidate_started = 0.0
        self.committed = False
        self.hold_seconds = 0.8
        self.dynamic_window_frames = 18
        self.confirmed_signs = 0
        self.session_started = time.monotonic()
        self.recognition_history = []
        self.tutorial_active = False
        self.light_theme = False
        self.sound_enabled = True
        self.tutorial_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.tutorial_index = 0
        self.frame_times = deque(maxlen=30)
        self.live_resolution = "No camera"

        self.setup_ui()

        # Keyboard shortcuts
        QShortcut(QKeySequence("M"), self, activated=self.toggle_mode)
        QShortcut(QKeySequence("Backspace"), self, activated=self.delete_character)
        QShortcut(QKeySequence("Escape"), self, activated=self.clear_sentence)
        QShortcut(QKeySequence("Space"), self, activated=lambda: self.add_text(" "))

        self.worker = CameraWorker()
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.prediction_ready.connect(self.process_prediction)
        self.worker.status_changed.connect(self.show_status)
        self.worker.capture_progress_changed.connect(self.update_capture_progress)
        self.worker.set_camera_enabled(False)
        self.worker.start()

        self.session_timer = QTimer(self)
        self.session_timer.setInterval(1000)
        self.session_timer.timeout.connect(self.update_session_metrics)
        self.session_timer.start()

    def setup_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 12, 14, 10)
        root_layout.setSpacing(8)
        self.page_scroll = None

        # ── Header ───────────────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(14)

        heading = QVBoxLayout()
        heading.setSpacing(2)
        heading.addWidget(self.make_label("Sign Language Translator", "title"))
        heading.addWidget(self.make_label("AI-powered accessibility workspace  •  Alphabet and word recognition", "subtitle"))
        header.addLayout(heading, 1)

        # Mode Selection Segmented Control
        mode_box = QHBoxLayout()
        mode_box.setSpacing(6)
        self.static_button = QPushButton("Static (Alphabet)")
        self.static_button.setObjectName("modeStatic")
        self.dynamic_button = QPushButton("Dynamic (Words)")
        self.dynamic_button.setObjectName("modeDynamic")

        for button, mode in ((self.static_button, "static"), (self.dynamic_button, "dynamic")):
            button.setCheckable(True)
            button.setMinimumHeight(38)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, value=mode: self.set_mode(value))

        self.static_button.setChecked(True)
        mode_box.addWidget(self.static_button)
        mode_box.addWidget(self.dynamic_button)
        header.addLayout(mode_box)

        self.camera_button = QPushButton("Camera On")
        self.camera_button.setCheckable(True)
        self.camera_button.setMinimumHeight(38)
        self.camera_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.camera_button.setChecked(False)
        self.camera_button.clicked.connect(self.toggle_camera)
        header.addWidget(self.camera_button)

        self.theme_button = QPushButton("Light Theme")
        self.theme_button.setCheckable(True)
        self.theme_button.setMinimumHeight(38)
        self.theme_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_button.clicked.connect(self.toggle_theme)
        header.addWidget(self.theme_button)

        root_layout.addLayout(header)

        # ── Main Content Split ───────────────────────────────────────────────
        content = QHBoxLayout()
        content.setSpacing(10)

        # Left Column: Viewfinder Frame & Status
        left = QVBoxLayout()
        left.setSpacing(10)

        video_frame = QFrame()
        video_frame.setObjectName("videoSurface")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(6, 6, 6, 6)

        self.video = QLabel("Initializing camera...")
        self.video.setMinimumSize(520, 390)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet("background: #05070d; border-radius: 12px; color: #64748b;")
        self.video.setText("Camera is Off")
        video_layout.addWidget(self.video)
        left.addWidget(video_frame, 1)

        # Status Bar
        status_bar = QHBoxLayout()
        self.status_icon = QLabel("●")
        self.status_icon.setStyleSheet("color: #10b981; font-size: 14px;")
        status_bar.addWidget(self.status_icon)
        self.status = QLabel("Initializing camera and recognition engines...")
        self.status.setObjectName("statusOk")
        self.status.setWordWrap(True)
        status_bar.addWidget(self.status, 1)
        left.addLayout(status_bar)

        content.addLayout(left, 3)

        # Right Column: Controls, Capturing Duration & Text Output
        right = QVBoxLayout()
        right.setSpacing(12)

        # 1. Recognition Hero Card
        recognition = self.surface()
        recognition_layout = QVBoxLayout(recognition)
        recognition_layout.setContentsMargins(10, 8, 10, 8)
        recognition_layout.setSpacing(4)

        rec_header = QHBoxLayout()
        rec_header.addWidget(self.make_label("CURRENT RECOGNITION", "section"))
        self.mode_tag = QLabel("[ STATIC: ALPHABET ]")
        self.mode_tag.setStyleSheet("color: #00e5ff; font-weight: 700; font-size: 11px;")
        rec_header.addWidget(self.mode_tag, 0, Qt.AlignmentFlag.AlignRight)
        recognition_layout.addLayout(rec_header)

        self.prediction = QLabel("—")
        self.prediction.setObjectName("prediction")
        self.prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recognition_layout.addWidget(self.prediction)

        self.prediction_status = QLabel("Waiting for hand in view")
        self.prediction_status.setObjectName("predictionStatus")
        self.prediction_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recognition_layout.addWidget(self.prediction_status)

        # Confidence Bar
        conf_row = QHBoxLayout()
        conf_row.addWidget(self.make_label("AI Confidence", "subtitle"))
        self.confidence_value = QLabel("0%")
        self.confidence_value.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        conf_row.addWidget(self.confidence_value, 0, Qt.AlignmentFlag.AlignRight)
        recognition_layout.addLayout(conf_row)

        self.confidence = QProgressBar()
        self.confidence.setRange(0, 100)
        self.confidence.setValue(0)
        recognition_layout.addWidget(self.confidence)

        # Hold & Capture Duration Bar
        hold_label_row = QHBoxLayout()
        self.capture_bar_title = self.make_label("Recognition Progress", "subtitle")
        hold_label_row.addWidget(self.capture_bar_title)
        self.capture_value = QLabel("0%")
        self.capture_value.setStyleSheet("color: #c084fc; font-size: 11px; font-weight: 600;")
        hold_label_row.addWidget(self.capture_value, 0, Qt.AlignmentFlag.AlignRight)
        recognition_layout.addLayout(hold_label_row)

        self.hold_progress = QProgressBar()
        self.hold_progress.setObjectName("holding")
        self.hold_progress.setRange(0, 100)
        self.hold_progress.setValue(0)
        recognition_layout.addWidget(self.hold_progress)

        right.addWidget(recognition)

        # 2. Translated Sentence Card
        sentence = self.surface()
        sentence_layout = QVBoxLayout(sentence)
        sentence_layout.setContentsMargins(14, 12, 14, 12)
        sentence_layout.setSpacing(8)

        sentence_layout.addWidget(self.make_label("TRANSLATED SENTENCE", "section"))
        self.sentence = QTextEdit()
        self.sentence.setReadOnly(True)
        self.sentence.setPlaceholderText("Recognized signs will accumulate here...")
        self.sentence.setMinimumHeight(55)
        sentence_layout.addWidget(self.sentence)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        speak = QPushButton("Speak Sentence")
        speak.setObjectName("primary")
        speak.setCursor(Qt.CursorShape.PointingHandCursor)
        speak.clicked.connect(self.speak_sentence)

        space = QPushButton("Space")
        space.setCursor(Qt.CursorShape.PointingHandCursor)
        space.clicked.connect(lambda: self.add_text(" "))

        delete = QPushButton("Delete")
        delete.setCursor(Qt.CursorShape.PointingHandCursor)
        delete.clicked.connect(self.delete_character)

        clear = QPushButton("Clear")
        clear.setCursor(Qt.CursorShape.PointingHandCursor)
        clear.clicked.connect(self.clear_sentence)

        actions.addWidget(speak, 2)
        actions.addWidget(space, 1)
        actions.addWidget(delete, 1)
        actions.addWidget(clear, 1)
        sentence_layout.addLayout(actions)

        copy = QPushButton("Copy Text")
        copy.setCursor(Qt.CursorShape.PointingHandCursor)
        copy.clicked.connect(self.copy_sentence)
        sentence_actions = QHBoxLayout()
        sentence_actions.addWidget(copy)
        sentence_layout.addLayout(sentence_actions)

        self.auto_speak = QCheckBox("Auto-speak confirmed dynamic words")
        self.auto_speak.setChecked(True)
        sentence_layout.addWidget(self.auto_speak)

        right.addWidget(sentence)

        # 3. Settings & Duration Configuration Card
        settings = self.surface()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(10, 8, 10, 8)
        settings_layout.setSpacing(5)

        # Capture Duration Pacing Controls
        dur_header = QHBoxLayout()
        self.duration_section_label = self.make_label("CAPTURING DURATION PACING", "section")
        dur_header.addWidget(self.duration_section_label)
        self.hold_value = QLabel("0.8 s (Static Hold)")
        self.hold_value.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 11px;")
        dur_header.addWidget(self.hold_value, 0, Qt.AlignmentFlag.AlignRight)
        settings_layout.addLayout(dur_header)

        # Sliders: Static Hold & Dynamic Sequence Window
        self.duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.duration_slider.setRange(5, 20)  # 0.5s to 2.0s
        self.duration_slider.setValue(8)
        self.duration_slider.valueChanged.connect(self.update_capture_duration)
        settings_layout.addWidget(self.duration_slider)

        # Presets
        presets = QHBoxLayout()
        presets.setSpacing(6)
        for label, val in [("Fast 0.5s", 5), ("Recommended 0.8s", 8), ("Stable 1.2s", 12)]:
            btn = QPushButton(label)
            btn.setObjectName("presetBtn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _c=False, v=val: self.duration_slider.setValue(v))
            presets.addWidget(btn)
        settings_layout.addLayout(presets)

        # Quick Reference
        settings_layout.addWidget(self.make_label("REFERENCE SIGNS", "section"))
        reference_scroll = QScrollArea()
        reference_scroll.setWidgetResizable(True)
        reference_scroll.setFrameShape(QFrame.Shape.NoFrame)
        reference_scroll.setMinimumHeight(190)
        reference_content = QWidget()
        reference_content.setObjectName("referenceContent")
        self.ref_stack = QVBoxLayout(reference_content)
        self.ref_stack.setContentsMargins(0, 0, 8, 0)
        self.ref_stack.setSpacing(6)
        self.alphabet_grid = self.create_alphabet_grid()
        self.sign_preview = QLabel()
        self.sign_preview.setObjectName("signPreview")
        self.sign_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sign_preview.setMinimumHeight(220)
        self.sign_preview.setPixmap(self.create_alphabet_preview(""))
        self.sign_guide = QLabel("Select a letter to see how to perform its sign.")
        self.sign_guide.setObjectName("signGuide")
        self.sign_guide.setWordWrap(True)
        self.sign_guide.setMinimumHeight(64)
        self.sign_guide.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.ref_stack.addWidget(self.sign_preview)
        self.ref_stack.addWidget(self.sign_guide)
        self.word_chips = self.create_word_buttons()
        self.ref_stack.addLayout(self.alphabet_grid)
        self.ref_stack.addLayout(self.word_chips)
        reference_scroll.setWidget(reference_content)
        settings_layout.addWidget(reference_scroll)

        right.addWidget(settings)
        content.addLayout(right, 2)
        root_layout.addLayout(content, 1)

        metrics = QHBoxLayout()
        metrics.setSpacing(8)
        self.session_signs = QLabel("Recognized signs: 0")
        self.session_duration = QLabel("Session: 00:00")
        self.processing_status = QLabel("Engine: Ready")
        for label in (self.session_signs, self.session_duration, self.processing_status):
            label.setObjectName("metric")
            metrics.addWidget(label)
        metrics.addStretch(1)
        root_layout.addLayout(metrics)

        session_panel = self.surface()
        session_layout = QHBoxLayout(session_panel)
        session_layout.setContentsMargins(10, 8, 10, 8)
        session_layout.setSpacing(8)
        session_layout.addWidget(self.make_label("SESSION TOOLS", "section"))

        utility_buttons = [
            ("Tutorial Mode", self.toggle_tutorial),
            ("Export PDF", self.export_pdf),
            ("Recognition History", self.show_history),
            ("Shortcuts", self.show_shortcuts),
            ("About", self.show_about),
            ("Privacy", self.show_privacy),
            ("Sound On", self.toggle_sound),
        ]
        for text, handler in utility_buttons:
            button = QPushButton(text)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(handler)
            session_layout.addWidget(button)
        session_layout.addStretch(1)
        root_layout.addWidget(session_panel)

        history_panel = self.surface()
        history_layout = QVBoxLayout(history_panel)
        history_layout.setContentsMargins(10, 8, 10, 8)
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setMaximumHeight(90)
        self.history_text.setPlaceholderText("Confirmed recognitions will appear here.")
        history_layout.addWidget(self.make_label("RECENT RECOGNITION HISTORY", "section"))
        history_layout.addWidget(self.history_text)
        root_layout.addWidget(history_panel)

        self.page_scroll = QScrollArea()
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.page_scroll.setWidget(root)
        self.setCentralWidget(self.page_scroll)

    def surface(self):
        frame = QFrame()
        frame.setObjectName("surface")
        return frame

    def make_label(self, text, object_name=""):
        label = QLabel(text)
        if object_name:
            label.setObjectName(object_name)
        return label

    def create_alphabet_grid(self):
        grid = QGridLayout()
        grid.setSpacing(4)
        for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            button = QPushButton(f"{letter}  Hand Sign")
            button.setObjectName("chip")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(
                lambda _checked=False, value=letter: self.show_sign_guide(value)
            )
            grid.addWidget(button, index // 13, index % 13)
        return grid

    def show_sign_guide(self, letter):
        instructions = {
            "A": "Closed fist with the thumb resting beside the index finger.",
            "B": "Straight fingers together; thumb folded across the palm.",
            "C": "Curve the fingers and thumb to form the letter C.",
            "D": "Point index finger upward; touch other fingertips to the thumb.",
            "E": "Curl all fingers downward and press fingertips against the thumb.",
            "F": "Touch index fingertip and thumb; keep the other fingers extended.",
            "G": "Point thumb and index finger horizontally toward the side.",
            "H": "Extend index and middle fingers together horizontally.",
            "I": "Make a fist and extend the little finger upward.",
            "J": "Draw a small J shape in the air with the little finger.",
            "K": "Middle finger and index finger point upward with thumb between.",
            "L": "Raise thumb and index finger to form an L shape.",
            "M": "Fold the thumb under the first three fingers.",
            "N": "Fold the thumb under the index and middle fingers.",
            "O": "Curve all fingers to touch the thumb and form an O.",
            "P": "Point middle finger and index finger downward with thumb between.",
            "Q": "Point thumb and index finger downward.",
            "R": "Cross the index and middle fingers.",
            "S": "Closed fist with the thumb across the front of the fingers.",
            "T": "Place the thumb between the index and middle fingers.",
            "U": "Hold index and middle fingers together and upright.",
            "V": "Extend and separate the index and middle fingers.",
            "W": "Extend index, middle, and ring fingers.",
            "X": "Hook the index finger while keeping other fingers closed.",
            "Y": "Extend the thumb and little finger, closing the other fingers.",
            "Z": "Trace the letter Z in the air with the index finger.",
        }
        self.sign_preview.setPixmap(self.create_alphabet_preview(letter))
        self.sign_guide.setText(
            f"LETTER {letter}\n"
            "HOW TO MAKE THE SIGN\n"
            f"{instructions[letter]}\n"
            "Keep the hand still for recognition."
        )
        self.prediction.setText(letter)
        self.prediction_status.setText("Follow the hand-shape guide below the alphabet.")
        if self.tutorial_active:
            if letter in self.tutorial_letters and letter != self.tutorial_letters[self.tutorial_index]:
                self.tutorial_index = self.tutorial_letters.index(letter)
            position = self.tutorial_index + 1
            self.sign_guide.setText(
                f"TUTORIAL {position}/{len(self.tutorial_letters)}  •  LETTER {letter}\n"
                "HOW TO MAKE THE SIGN\n"
                f"{instructions[letter]}\n"
                "Keep the hand still, then continue."
            )
            self.tutorial_index = min(self.tutorial_index + 1, len(self.tutorial_letters) - 1)

    def create_alphabet_preview(self, letter):
        image = QImage(520, 280, QImage.Format.Format_RGB888)
        image.fill(QColor("#07111f"))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(QPen(QColor("#23466c"), 2))
        painter.setBrush(QColor("#0b1b31"))
        painter.drawRoundedRect(16, 16, 488, 248, 16, 16)

        painter.setPen(QColor("#67e8f9"))
        painter.setFont(QFont("Segoe UI", 34, QFont.Weight.Bold))
        painter.drawText(36, 62, letter if letter else "?")
        painter.setPen(QColor("#8da4c1"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        painter.drawText(38, 88, "ASL HAND-SHAPE GUIDE" if letter else "SELECT A LETTER")

        if letter:
            asset = Path(__file__).resolve().parent / "assets" / "asl_alphabet" / f"{letter}.svg"
            renderer = QSvgRenderer(str(asset))
            if renderer.isValid():
                renderer.render(painter, QRectF(230, 48, 270, 210))
        painter.end()
        return QPixmap.fromImage(image)

    def create_word_buttons(self):
        layout = QGridLayout()
        layout.setSpacing(4)
        for index, word in enumerate(DYNAMIC_VOCABULARY):
            button = QPushButton(word.title())
            button.setObjectName("chip")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(
                lambda _checked=False, value=word: self.show_word_guide(value)
            )
            layout.addWidget(button, index // 4, index % 4)
        return layout

    def show_word_guide(self, word):
        word = word.lower()
        common_tips = {
            "hello": "Open hand near the forehead; move it outward as a greeting.",
            "bye": "Wave the open hand from the chin/side, usually repeated.",
            "goodbye": "Wave the open hand slowly away from the body.",
            "thank you": "Touch fingertips near the chin, then move the hand forward.",
            "please": "Rub a flat hand in a circle over the center of the chest.",
            "help": "Place one closed hand over an open palm; lift both upward.",
            "yes": "Make an S-hand and nod it up and down.",
            "no": "Extend index and middle fingers; close them toward the thumb.",
            "sorry": "Make a fist and circle it over the center of the chest.",
            "water": "Tap the W-handshape gently against the chin.",
            "bathroom": "Hold the T-handshape and shake it side to side.",
            "eat": "Tap flattened fingertips toward the mouth repeatedly.",
            "drink": "Move a C-hand shape toward the mouth like holding a cup.",
        }
        tip = common_tips.get(word, f"Perform the trained ASL sign for '{word.title()}'. Keep facial expression natural and perform a clear, controlled movement.")
        self.sign_guide.setText(f"{word.upper()}  •  {tip}")
        self.prediction.setText(word.title())

    def active_mode(self):
        return self.worker.mode

    def set_mode(self, mode):
        self.worker.mode = mode
        self.static_button.setChecked(mode == "static")
        self.dynamic_button.setChecked(mode == "dynamic")
        self.reset_prediction()

        if mode == "static":
            self.mode_tag.setText("[ STATIC: ALPHABET ]")
            self.mode_tag.setStyleSheet("color: #00e5ff; font-weight: 700; font-size: 11px;")
            self.prediction_status.setText("Static Mode • Hold hand steadily to confirm letter")
            self.capture_bar_title.setText("Hold Duration Progress")
            self.hold_value.setText(f"{self.hold_seconds:.1f} s (Static Hold)")
            self.prediction.setStyleSheet("font-size: 54px; font-weight: 800; color: #00e5ff;")
        else:
            self.mode_tag.setText("[ DYNAMIC: WORDS ]")
            self.mode_tag.setStyleSheet("color: #c084fc; font-weight: 700; font-size: 11px;")
            self.prediction_status.setText("Dynamic Mode • Perform motion gestures for words")
            self.capture_bar_title.setText("Dynamic Capture Window")
            self.hold_value.setText(f"{self.worker.dynamic_window_frames} Frames (Dynamic Window)")
            self.prediction.setStyleSheet("font-size: 54px; font-weight: 800; color: #c084fc;")

    def toggle_mode(self):
        self.set_mode("dynamic" if self.active_mode() == "static" else "static")

    def toggle_camera(self, checked=False):
        camera_on = bool(checked)
        self.worker.set_camera_enabled(camera_on)
        self.camera_button.setText("Camera On" if not camera_on else "Camera Off")
        self.camera_button.setChecked(camera_on)
        if camera_on:
            self.show_status("Camera resumed • Position your hands in the frame", False)
        else:
            self.video.clear()
            self.video.setText("Camera is Off")
            self.show_status("Camera is off", False)
            self.live_resolution = "No camera"
            self.processing_status.setText("Engine: Running  |  Camera: Fully Off")

    def update_capture_duration(self, value):
        self.hold_seconds = value / 10.0
        # Keep dynamic capture short: 12-30 frames (about 0.4-1.0 seconds).
        self.dynamic_window_frames = int(np.interp(value, [5, 20], [12, 30]))
        self.worker.dynamic_window_frames = self.dynamic_window_frames

        if self.active_mode() == "static":
            self.hold_value.setText(f"{self.hold_seconds:.1f} s (Static Hold)")
        else:
            self.hold_value.setText(f"{self.dynamic_window_frames} Frames (~{self.dynamic_window_frames/30.0:.1f}s)")

    def update_capture_progress(self, progress: float, status_text: str):
        if self.active_mode() == "dynamic":
            pct = int(progress * 100)
            self.hold_progress.setValue(pct)
            self.capture_value.setText(f"{pct}%")
            if status_text and not self.prediction.text().strip("—"):
                self.prediction_status.setText(status_text)

    def show_status(self, message, is_error):
        self.status.setText(message)
        self.status.setObjectName("statusError" if is_error else "statusOk")
        self.status.setStyle(self.status.style())
        self.status_icon.setStyleSheet("color: #f87171;" if is_error else "color: #10b981;")

    def show_frame(self, image):
        now = time.monotonic()
        self.frame_times.append(now)
        if len(self.frame_times) >= 2:
            fps = (len(self.frame_times) - 1) / max(0.001, self.frame_times[-1] - self.frame_times[0])
            self.processing_status.setText(
                f"Engine: Running  |  {fps:.0f} FPS  |  {image.width()}x{image.height()}"
            )
        self.video.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.video.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def reset_prediction(self):
        self.votes.clear()
        self.current_candidate = None
        self.candidate_started = 0.0
        self.committed = False
        self.prediction.setText("—")
        self.confidence.setValue(0)
        self.confidence_value.setText("0%")
        self.hold_progress.setValue(0)
        self.capture_value.setText("0%")

    def process_prediction(self, label, confidence, hand_present):
        if not hand_present:
            self.prediction_status.setText("Waiting for hands in view")
            self.reset_prediction()
            return

        if not label:
            if self.active_mode() == "static":
                self.prediction_status.setText("Align hand clearly and hold pose steady")
            return

        confidence = max(0.0, min(1.0, confidence))

        if self.active_mode() == "static":
            if confidence < 0.60:
                return

            self.votes.append(label.lower())
            candidate, count = Counter(self.votes).most_common(1)[0]
            if count < 5:
                return

            if candidate != self.current_candidate:
                self.current_candidate = candidate
                self.candidate_started = time.monotonic()
                self.committed = False

            elapsed = time.monotonic() - self.candidate_started
            progress = min(1.0, elapsed / self.hold_seconds)
            pct = int(progress * 100)

            self.prediction.setText(candidate.upper())
            self.prediction_status.setText(f"Holding '{candidate.upper()}' • {elapsed:.1f}s / {self.hold_seconds:.1f}s")
            self.confidence.setValue(int(confidence * 100))
            self.confidence_value.setText(f"{int(confidence * 100)}%")
            self.hold_progress.setValue(pct)
            self.capture_value.setText(f"{pct}%")

            if progress >= 1.0 and not self.committed:
                self.committed = True
                self.commit(candidate)
        else:
            # Dynamic Word Mode
            if confidence < 0.40:
                return

            self.prediction.setText(label.upper())
            self.prediction_status.setText(f"Word Captured: '{label.upper()}'")
            self.confidence.setValue(int(confidence * 100))
            self.confidence_value.setText(f"{int(confidence * 100)}%")
            self.commit(label.lower())

    def commit(self, token):
        self.confirmed_signs += 1
        self.session_signs.setText(f"Recognized signs: {self.confirmed_signs}")
        self.recognition_history.insert(0, f"{time.strftime('%H:%M:%S')}  •  {token.upper()}")
        self.recognition_history = self.recognition_history[:12]
        self.history_text.setPlainText("\n".join(self.recognition_history))
        if self.sound_enabled:
            threading.Thread(target=self.play_confirmation_sound, daemon=True).start()
        if self.active_mode() == "static":
            self.add_text(token.upper())
            return

        if token in COMMANDS:
            command = COMMANDS[token]
            if command == "space":
                self.add_text(" ")
            elif command == "delete":
                self.delete_character()
            else:
                self.clear_sentence()
            return

        self.add_text(token.title() + " ")
        if self.auto_speak.isChecked():
            self.speak_text(token.title())

    def add_text(self, text):
        self.sentence.setPlainText(self.sentence.toPlainText() + text)
        cursor = self.sentence.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.sentence.setTextCursor(cursor)

    def delete_character(self):
        text = self.sentence.toPlainText()
        if text:
            self.sentence.setPlainText(text[:-1])

    def clear_sentence(self):
        self.sentence.clear()

    def copy_sentence(self):
        text = self.sentence.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.show_status("Sentence copied to clipboard", False)

    def play_confirmation_sound(self):
        try:
            import winsound
            winsound.Beep(880, 90)
        except Exception:
            pass

    def export_pdf(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Recognition Report", "sign_language_session.pdf", "PDF Files (*.pdf)"
        )
        if not path:
            return
        document = QTextDocument()
        document.setPlainText(
            "SIGN LANGUAGE TRANSLATOR - SESSION REPORT\n\n"
            f"Recognized text: {self.sentence.toPlainText()}\n\n"
            f"Recognized signs: {self.confirmed_signs}\n"
            f"Recent history:\n{chr(10).join(self.recognition_history) or 'No signs recorded'}\n\n"
            "Privacy: Video processing is performed locally on this device."
        )
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        document.print(printer)
        self.show_status(f"Report exported to {path}", False)

    def show_history(self):
        content = "\n".join(self.recognition_history) or "No recognitions recorded in this session."
        QMessageBox.information(self, "Recognition History", content)

    def show_shortcuts(self):
        QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            "M  - Switch recognition mode\n"
            "Space  - Add a space\n"
            "Backspace  - Delete character\n"
            "Escape  - Clear sentence\n"
            "Camera  - Use the Camera On/Off button\n",
        )

    def show_about(self):
        QMessageBox.information(
            self,
            "About Sign Language Translator",
            "Real-time ASL accessibility workspace\n\n"
            "MediaPipe hand tracking\n"
            "PyTorch static and dynamic recognition models\n"
            "Dual-hand landmark processing\n"
            "Professional desktop interface built with PyQt6",
        )

    def show_privacy(self):
        QMessageBox.information(
            self,
            "Privacy",
            "Camera frames are processed locally on this device. "
            "The application does not upload your video to an external server.",
        )

    def toggle_tutorial(self):
        self.tutorial_active = not self.tutorial_active
        if self.tutorial_active:
            self.set_mode("static")
            self.tutorial_index = 0
            letter = self.tutorial_letters[self.tutorial_index]
            self.show_sign_guide(letter)
            self.sign_guide.setText(f"TUTORIAL 1/{len(self.tutorial_letters)}  •  LETTER {letter}\n{self.sign_guide.text()}")
        else:
            self.sign_guide.setText("Tutorial closed. Select any alphabet or word guide.")

    def toggle_theme(self):
        self.light_theme = not self.light_theme
        self.theme_button.setChecked(self.light_theme)
        self.theme_button.setText("Dark Theme" if self.light_theme else "Light Theme")
        if self.light_theme:
            self.setStyleSheet(QSS_THEME + """
QMainWindow, QWidget { background: #edf4f8; color: #1e293b; }
QFrame#surface { background: #ffffff; border: 1px solid #cbd5e1; }
QFrame#videoSurface { background: #dbe7ef; border: 1px solid #94a3b8; }
QTextEdit { background: #ffffff; color: #1e293b; border-color: #cbd5e1; }
""")
        else:
            self.setStyleSheet(QSS_THEME)

    def toggle_sound(self):
        self.sound_enabled = not self.sound_enabled
        self.processing_status.setText(
            f"Engine: Running  |  Sound {'On' if self.sound_enabled else 'Off'}"
        )

    def update_session_metrics(self):
        elapsed = int(time.monotonic() - self.session_started)
        minutes, seconds = divmod(elapsed, 60)
        self.session_duration.setText(f"Session: {minutes:02d}:{seconds:02d}")
        if self.worker.isRunning():
            self.processing_status.setText("Engine: Running")

    def speak_sentence(self):
        self.speak_text(self.sentence.toPlainText())

    def speak_text(self, text):
        clean = text.strip()
        if not clean:
            return

        def speak():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(clean)
                engine.runAndWait()
            except Exception:
                pass

        threading.Thread(target=speak, daemon=True).start()

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Sign Language Translator")
    window = TranslatorWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
