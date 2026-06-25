import csv
import io
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from math import hypot
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
import uvicorn

import config

# FINGER GESTURE INTEGRATION: Import MediaPipe library for finger landmark detection
import mediapipe as mp

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


class CameraStatus:
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"


class VideoCaptureThread(threading.Thread):
    def __init__(self, source: str) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.capture: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_timestamp: float = 0.0
        self.status: str = CameraStatus.DISCONNECTED
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.video_writer: Optional[cv2.VideoWriter] = None

    def run(self) -> None:
        logging.info("Starting video capture thread for source: %s", self.source)
        while not self.stop_event.is_set():
            if self.capture is None or not self.capture.isOpened():
                self._reconnect()
                continue

            ret, frame = self.capture.read()
            if not ret or frame is None:
                logging.warning("Frame read failed, reconnecting camera")
                self._reconnect()
                continue

            # Record raw frame directly in the capture thread to match the camera's natural speed
            if config.ENABLE_RECORDING and config.RECORD_RAW_VIDEO:
                if self.video_writer is None:
                    self._init_video_writer(frame.shape)
                if self.video_writer is not None:
                    self.video_writer.write(frame)

            with self.lock:
                self.latest_frame = frame
                self.latest_timestamp = time.time()
                self.status = CameraStatus.CONNECTED

            time.sleep(config.CAPTURE_WAIT)

        logging.info("Stopping video capture thread")
        self._release()

    def _reconnect(self) -> None:
        self.status = CameraStatus.RECONNECTING
        logging.info("Attempting reconnect to camera source: %s", self.source)
        self._release()
        time.sleep(config.RECONNECT_DELAY)

        self.capture = cv2.VideoCapture(self.source)
        if self.capture.isOpened():
            logging.info("Camera reconnected successfully")
            self.status = CameraStatus.CONNECTED
        else:
            logging.error("Failed to open camera source: %s", self.source)
            self.status = CameraStatus.DISCONNECTED
            self._release()

    def _init_video_writer(self, frame_shape: Tuple[int, int, int]) -> None:
        if self.video_writer is not None:
            return
        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        output_path = config.RECORDING_DIR / f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        config.RECORDING_DIR.mkdir(parents=True, exist_ok=True)
        self.video_writer = cv2.VideoWriter(str(output_path), fourcc, config.VIDEO_FPS, (width, height))
        if self.video_writer.isOpened():
            logging.info("Recording raw video in capture thread, writing to %s", output_path)
        else:
            logging.error("Failed to initialize video writer in capture thread")
            self.video_writer = None

    def _release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def get_frame(self) -> Tuple[Optional[np.ndarray], float, str]:
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None, self.latest_timestamp, self.status

    def stop(self) -> None:
        self.stop_event.set()


class YoloProcessor(threading.Thread):
    def __init__(self, model_path: str, confidence: float, iou: float) -> None:
        super().__init__(daemon=True)
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.model = self._load_model()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_timestamp: float = 0.0
        self.annotated_frame: Optional[np.ndarray] = None
        self.results: Optional[List[Dict[str, Any]]] = None
        self.inference_time: float = 0.0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def _load_model(self) -> YOLO:
        target_path = Path(self.model_path)
        if target_path.exists():
            logging.info("Loading YOLO11 model from %s", target_path)
            return YOLO(str(target_path))

        logging.warning("Custom model %s not found, falling back to %s", self.model_path, config.FALLBACK_MODEL)
        try:
            return YOLO(config.FALLBACK_MODEL)
        except Exception as exc:
            logging.exception("Failed to load fallback model: %s", exc)
            raise RuntimeError("No YOLO model available") from exc

    def run(self) -> None:
        logging.info("Starting inference thread")
        while not self.stop_event.is_set():
            frame, timestamp, _ = self._read_source_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            if timestamp <= self.latest_timestamp:
                time.sleep(0.005)
                continue

            self.latest_timestamp = timestamp
            self._process_frame(frame)

        logging.info("Stopping inference thread")

    def _read_source_frame(self) -> Tuple[Optional[np.ndarray], float, str]:
        return self.source_frame_fetcher()  # type: ignore[attr-defined]

    def _process_frame(self, frame: np.ndarray) -> None:
        start_time = time.perf_counter()
        try:
            results = self.model(frame, conf=self.confidence, iou=self.iou)[0]
        except Exception as exc:
            logging.exception("YOLO inference failed: %s", exc)
            return

        elapsed = time.perf_counter() - start_time
        annotated = frame.copy()
        detections = []

        # Check if the YOLO model provides body pose keypoints (wrists, shoulders, etc.)
        # so we can use them downstream to detect hand-raise gestures.
        has_keypoints = results.keypoints is not None
        keypoints_xy = None
        keypoints_conf = None
        if has_keypoints:
            try:
                keypoints_xy = results.keypoints.xy.cpu().numpy()
                keypoints_conf = results.keypoints.conf.cpu().numpy()
            except Exception as e:
                logging.warning("Failed to extract keypoints: %s", e)
                has_keypoints = False

        if results.boxes is not None and len(results.boxes) > 0:
            for idx, (box, cls_id, score) in enumerate(zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf)):
                x1, y1, x2, y2 = map(int, box.tolist())
                class_name = self.model.names[int(cls_id)] if int(cls_id) in self.model.names else str(int(cls_id))
                label = f"{class_name}: {float(score):.2f}"
                color = self._get_color(int(cls_id))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.FONT_SCALE,
                    color,
                    config.FONT_THICKNESS,
                )
                det = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "class_name": class_name,
                    "confidence": float(score),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
                # Inject keypoints coordinates and confidence scores for this person detection
                # to allow the FloorPositionDetector to verify hand-raise gestures.
                if has_keypoints and keypoints_xy is not None and idx < len(keypoints_xy):
                    det["keypoints"] = keypoints_xy[idx].tolist()
                if has_keypoints and keypoints_conf is not None and idx < len(keypoints_conf):
                    det["keypoints_conf"] = keypoints_conf[idx].tolist()
                detections.append(det)

        with self.lock:
            self.annotated_frame = annotated
            self.results = detections
            self.inference_time = elapsed

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        np.random.seed(class_id)
        return tuple(int(c) for c in np.random.randint(50, 255, size=3))

    def stop(self) -> None:
        self.stop_event.set()

    def set_source_fetcher(self, fetcher: Any) -> None:
        self.source_frame_fetcher = fetcher

    def get_results(self) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, Any]]], float, float]:
        with self.lock:
            annotated = self.annotated_frame.copy() if self.annotated_frame is not None else None
            results = list(self.results) if self.results else []
            return annotated, results, self.inference_time, self.latest_timestamp


@dataclass
class FloorPosition:
    number: int
    center: Tuple[int, int]
    radius: int

    def contains(self, point: Tuple[float, float]) -> bool:
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        return dx * dx + dy * dy <= self.radius * self.radius


@dataclass
class TrackedPerson:
    track_id: int
    last_position: Tuple[float, float]
    last_seen: float


class PersonTracker:
    def __init__(self, max_distance: float, timeout: float) -> None:
        self.max_distance = max_distance
        self.timeout = timeout
        self.next_id = 1
        self.tracks: Dict[int, TrackedPerson] = {}

    def assign_ids(self, person_points: List[Tuple[float, float]], timestamp: float) -> List[int]:
        assigned_ids: List[int] = []
        used_tracks: Set[int] = set()

        for point in person_points:
            best_id: Optional[int] = None
            best_distance = self.max_distance

            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                distance = hypot(point[0] - track.last_position[0], point[1] - track.last_position[1])
                if distance < best_distance:
                    best_distance = distance
                    best_id = track_id

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = TrackedPerson(track_id=best_id, last_position=point, last_seen=timestamp)
            else:
                self.tracks[best_id].last_position = point
                self.tracks[best_id].last_seen = timestamp

            used_tracks.add(best_id)
            assigned_ids.append(best_id)

        self._remove_stale_tracks(timestamp)
        return assigned_ids

    def _remove_stale_tracks(self, timestamp: float) -> None:
        stale_ids = [track_id for track_id, track in self.tracks.items() if timestamp - track.last_seen > self.timeout]
        for track_id in stale_ids:
            del self.tracks[track_id]


class FloorPositionDetector:
    def __init__(self) -> None:
        self.positions = [FloorPosition(**position) for position in config.FLOOR_POSITIONS]
        # GESTURE DETECTOR INTEGRATION: Removed regional gesture rectangle occupancy tracking.
        # Hand-raise gestures are now monitored screen-wide globally.
        self.tracker = PersonTracker(max_distance=config.FLOOR_TRACKER_MAX_DISTANCE, timeout=config.FLOOR_TRACKER_TIMEOUT)
        self.occupied_positions: List[int] = []
        self.gesture_state: int = 0
        self.hand_previously_raised: bool = False

        # FINGER GESTURE INTEGRATION: Initialize MediaPipe Hands tracking configuration for counting fingers.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def update(self, detections: List[Dict[str, Any]], timestamp: float, frame: Optional[np.ndarray] = None) -> List[int]:
        person_detections = [d for d in detections if d.get("class_name", "").lower() == "person"]
        person_points: List[Tuple[float, float]] = []

        for detection in person_detections:
            x1 = float(detection.get("x1", 0))
            y1 = float(detection.get("y1", 0))
            x2 = float(detection.get("x2", 0))
            y2 = float(detection.get("y2", 0))
            foot_x = (x1 + x2) / 2.0
            foot_y = y2
            person_points.append((foot_x, foot_y))

        ids = self.tracker.assign_ids(person_points, timestamp)
        occupied: Set[int] = set()

        for detection, track_id, point in zip(person_detections, ids, person_points):
            detection["track_id"] = track_id
            detection["foot_point"] = point

            # Check year floor circles
            for position in self.positions:
                if position.contains(point):
                    occupied.add(position.number)

        # FINGER GESTURE INTEGRATION: Process frame with MediaPipe using YOLO Pose crop or whole-screen fallback.
        if frame is not None:
            try:
                hand_detected_in_crop = False
                # Iterate through detections to check wrist keypoints for cropping hand region
                for detection in person_detections:
                    kpts = detection.get("keypoints")
                    kpts_conf = detection.get("keypoints_conf")
                    x1 = float(detection.get("x1", 0))
                    y1 = float(detection.get("y1", 0))
                    x2 = float(detection.get("x2", 0))
                    y2 = float(detection.get("y2", 0))
                    person_h = y2 - y1

                    if kpts is not None and kpts_conf is not None:
                        # Wrist indices: Left wrist is 9, Right wrist is 10
                        for wrist_idx in [9, 10]:
                            if wrist_idx < len(kpts) and wrist_idx < len(kpts_conf):
                                wrist_conf = kpts_conf[wrist_idx]
                                if wrist_conf > 0.4:
                                    wrist_x, wrist_y = kpts[wrist_idx][0], kpts[wrist_idx][1]
                                    # Calculate crop size relative to person height (around 35% of height)
                                    crop_size = max(64, min(512, int(0.35 * person_h)))
                                    
                                    # Calculate crop boundary coordinates
                                    x_start = max(0, int(wrist_x - crop_size // 2))
                                    y_start = max(0, int(wrist_y - crop_size // 2))
                                    x_end = min(frame.shape[1], int(wrist_x + crop_size // 2))
                                    y_end = min(frame.shape[0], int(wrist_y + crop_size // 2))
                                    
                                    if (x_end - x_start) >= 16 and (y_end - y_start) >= 16:
                                        crop = frame[y_start:y_end, x_start:x_end]
                                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                        results = self.hands.process(crop_rgb)
                                        
                                        if results.multi_hand_landmarks:
                                            hand_detected_in_crop = True
                                            for hand_landmarks in results.multi_hand_landmarks:
                                                landmarks = hand_landmarks.landmark
                                                index_up = landmarks[8].y < landmarks[6].y
                                                middle_up = landmarks[12].y < landmarks[10].y
                                                ring_up = landmarks[16].y < landmarks[14].y
                                                pinky_up = landmarks[20].y < landmarks[18].y
                                                
                                                up_count = sum([index_up, middle_up, ring_up, pinky_up])
                                                
                                                # Explicit ON/OFF trigger: 2 fingers sets state to 1, 1 finger resets to 0.
                                                if up_count == 2:
                                                    if self.gesture_state != 1:
                                                        self.gesture_state = 1
                                                        logging.info("Gesture state set to: 1 (2 fingers detected in wrist crop)")
                                                    break
                                                elif up_count == 1:
                                                    if self.gesture_state != 0:
                                                        self.gesture_state = 0
                                                        logging.info("Gesture state reset to: 0 (1 finger detected in wrist crop)")
                                            if hand_detected_in_crop:
                                                break
                        if hand_detected_in_crop:
                            break

                # Fallback to whole screen if no hands detected in crops
                if not hand_detected_in_crop:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(frame_rgb)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            landmarks = hand_landmarks.landmark
                            index_up = landmarks[8].y < landmarks[6].y
                            middle_up = landmarks[12].y < landmarks[10].y
                            ring_up = landmarks[16].y < landmarks[14].y
                            pinky_up = landmarks[20].y < landmarks[18].y
                            
                            up_count = sum([index_up, middle_up, ring_up, pinky_up])
                            
                            # Explicit ON/OFF trigger: 2 fingers sets state to 1, 1 finger resets to 0.
                            if up_count == 2:
                                if self.gesture_state != 1:
                                    self.gesture_state = 1
                                    logging.info("Gesture state set to: 1 (2 fingers detected on whole screen)")
                                break
                            elif up_count == 1:
                                if self.gesture_state != 0:
                                    self.gesture_state = 0
                                    logging.info("Gesture state reset to: 0 (1 finger detected on whole screen)")
            except Exception as exc:
                logging.warning("Error checking finger gesture: %s", exc)

        self.occupied_positions = sorted(occupied)
        return self.occupied_positions

    def draw_floor_positions(self, frame: np.ndarray) -> None:
        # Draw year floor positions
        for position in self.positions:
            occupied = position.number in self.occupied_positions
            color = config.FLOOR_OCCUPIED_COLOR if occupied else config.FLOOR_POSITION_COLOR
            cv2.circle(frame, position.center, position.radius, color, config.FLOOR_POSITION_LINE_THICKNESS)
            label = f"{position.number}"
            text_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FLOOR_POSITION_FONT_SCALE,
                config.FLOOR_POSITION_FONT_THICKNESS,
            )
            text_x = position.center[0] - text_size[0] // 2
            text_y = position.center[1] + text_size[1] // 2
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FLOOR_POSITION_FONT_SCALE,
                config.FLOOR_POSITION_TEXT_COLOR,
                config.FLOOR_POSITION_FONT_THICKNESS,
                cv2.LINE_AA,
            )

    def draw_person_ids(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> None:
        for detection in detections:
            if detection.get("class_name", "").lower() != "person":
                continue
            track_id = detection.get("track_id")
            if track_id is None:
                continue

            x1 = int(detection.get("x1", 0))
            y2 = int(detection.get("y2", 0))
            label = f"ID:{track_id}"
            cv2.putText(
                frame,
                label,
                (x1, min(y2 + 24, frame.shape[0] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                config.FLOOR_OCCUPIED_COLOR,
                config.FONT_THICKNESS,
                cv2.LINE_AA,
            )


class DetectionApp:
    def __init__(self) -> None:
        self.config = config
        self._ensure_directories()
        self.capture_thread = VideoCaptureThread(self.config.CAMERA_SOURCE)
        self.processor = YoloProcessor(
            model_path=self.config.MODEL_PATH,
            confidence=self.config.CONFIDENCE,
            iou=self.config.IOU_THRESHOLD,
        )
        self.processor.set_source_fetcher(self.capture_thread.get_frame)
        self.floor_position_detector = FloorPositionDetector()
        self.last_saved_frame: Optional[np.ndarray] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.frame_count: int = 0
        self.last_display_time: float = time.time()
        self.last_logged_timestamp: float = 0.0
        self.fps: float = 0.0
        self.latest_stream_frame: Optional[np.ndarray] = None
        self.latest_frame_lock = threading.Lock()
        self.web_app: Optional[FastAPI] = None
        self.web_server: Optional[uvicorn.Server] = None
        self.web_thread: Optional[threading.Thread] = None
        self._prepare_csv_log()

    def _ensure_directories(self) -> None:
        for path in [self.config.SCREENSHOT_DIR, self.config.RECORDING_DIR, self.config.LOG_DIR, Path("models")]:
            path.mkdir(parents=True, exist_ok=True)

    def _build_web_app(self) -> FastAPI:
        app = FastAPI(title="YOLO11 Live Detection")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # testing
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            html = f"""
            <!DOCTYPE html>
            <html lang="en">
                <head>
                    <meta charset="UTF-8" />
                    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                    <title>YOLO11 Live Detection</title>
                    <style>
                        :root {{
                            color-scheme: dark;
                            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                            background: #0d1117;
                            color: #f8fafc;
                        }}
                        * {{ box-sizing: border-box; }}
                        body {{ margin: 0; padding: 0; min-height: 100vh; background: radial-gradient(circle at top right, rgba(96,165,250,0.18), transparent 30%), #070b13; }}
                        header {{ padding: 24px 32px; border-bottom: 1px solid rgba(255,255,255,0.08); }}
                        .brand {{ display: flex; align-items: center; gap: 14px; margin-bottom: 8px; }}
                        .brand h1 {{ margin: 0; font-size: 1.8rem; letter-spacing: -0.04em; }}
                        .brand span {{ font-size: 0.95rem; color: #94a3b8; }}
                        .container {{ width: min(1200px, calc(100% - 32px)); margin: 28px auto; }}
                        .panel-grid {{ display: grid; grid-template-columns: 1.5fr 0.85fr; gap: 24px; align-items: start; }}
                        .card {{ background: rgba(15, 23, 42, 0.92); border: 1px solid rgba(148,163,184,0.12); border-radius: 24px; padding: 22px; box-shadow: 0 24px 64px rgba(15,23,42,0.35); }}
                        .card h2 {{ margin-top: 0; margin-bottom: 16px; font-size: 1.2rem; color: #e2e8f0; }}
                        .status-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }}
                        .status-card {{ padding: 18px; border-radius: 20px; background: rgba(30, 41, 59, 0.95); border: 1px solid rgba(148,163,184,0.1); }}
                        .status-label {{ display: block; margin-bottom: 8px; font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }}
                        .status-value {{ font-size: 1.60rem; font-weight: 700; color: #fff; }}
                        .status-value span {{ font-size: 1rem; color: #94a3b8; margin-left: 6px; }}
                        .info-list {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 10px; }}
                        .info-list li {{ display: flex; justify-content: space-between; gap: 16px; padding: 12px 14px; border-radius: 16px; background: rgba(71, 85, 105, 0.16); }}
                        .info-label {{ color: #cbd5e1; }}
                        .info-value {{ color: #fff; font-weight: 600; }}
                        .live-stream {{ width: 100%; border-radius: 22px; overflow: hidden; background: #020617; border: 1px solid rgba(148,163,184,0.12); }}
                        .live-stream img {{ width: 100%; height: auto; display: block; }}
                        .footer {{ margin-top: 26px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.08); color: #94a3b8; font-size: 0.92rem; }}
                        .hint {{ margin-top: 14px; font-size: 0.95rem; color: #cbd5e1; }}
                    </style>
                </head>
                <body>
                    <header>
                        <div class="brand">
                            <div>
                                <h1>YOLO11 Live Detection</h1>
                                <span>FastAPI web dashboard for DroidCam + YOLO11</span>
                            </div>
                        </div>
                    </header>
                    <main class="container">
                        <div class="panel-grid">
                            <section class="card">
                                <h2>Live Stream</h2>
                                <div class="live-stream">
                                    <img src="/stream" alt="Live YOLO stream" />
                                </div>
                                <p class="hint">If the stream does not appear, confirm the mobile camera URL is reachable and the service is running.</p>
                            </section>
                            <aside class="card">
                                <h2>Live Metrics</h2>
                                <div class="status-grid">
                                    <div class="status-card">
                                        <span class="status-label">Camera status</span>
                                        <span class="status-value" id="status">Connecting...</span>
                                    </div>
                                    <div class="status-card">
                                        <span class="status-label">FPS</span>
                                        <span class="status-value" id="fps">0</span>
                                    </div>
                                    <div class="status-card">
                                        <span class="status-label">Detections</span>
                                        <span class="status-value" id="detections">0</span>
                                    </div>
                                    <div class="status-card">
                                        <span class="status-label">Inference time</span>
                                        <span class="status-value" id="inference">0 ms</span>
                                    </div>
                                    <div class="status-card" style="grid-column: span 2;">
                                        <span class="status-label">Occupied positions</span>
                                        <span class="status-value" id="occupied">[]</span>
                                    </div>
                                    <!-- GESTURE DETECTOR INTEGRATION: Live metrics dashboard panel to display current gesture state -->
                                    <div class="status-card" style="grid-column: span 2;">
                                        <span class="status-label">Gesture State</span>
                                        <span class="status-value" id="gesture">0</span>
                                    </div>
                                </div>
                                <h2 style="margin-top: 24px;">Model Info</h2>
                                <ul class="info-list">
                                    <li><span class="info-label">Model path</span><strong class="info-value">{self.processor.model_path}</strong></li>
                                    <li><span class="info-label">Camera URL</span><strong class="info-value">{self.config.CAMERA_SOURCE}</strong></li>
                                </ul>
                            </aside>
                        </div>
                        <div class="footer">
                            <p>Use the keyboard controls in the application window if enabled: Q = Quit, S = Screenshot, R = Reconnect.</p>
                        </div>
                    </main>
                    <script>
                        async function updateStatus() {{
                            try {{
                                const response = await fetch('/status');
                                const data = await response.json();
                                document.getElementById('status').textContent = data.camera_status;
                                document.getElementById('detections').textContent = data.detections;
                                document.getElementById('fps').textContent = data.fps;
                                document.getElementById('inference').textContent = data.inference_time_ms + ' ms';
                                document.getElementById('occupied').textContent = JSON.stringify(data.occupied_positions);
                                // GESTURE DETECTOR INTEGRATION: Read current gesture toggle value and update the UI element
                                document.getElementById('gesture').textContent = data.gesture;
                            }} catch (error) {{
                                document.getElementById('status').textContent = 'Disconnected';
                                document.getElementById('occupied').textContent = '[]';
                                // GESTURE DETECTOR INTEGRATION: Fallback default value on network failure
                                document.getElementById('gesture').textContent = '0';
                            }}
                        }}
                        setInterval(updateStatus, 1500);
                        updateStatus();
                    </script>
                </body>
            </html>
            """
            return HTMLResponse(content=html)

        @app.get("/stream")
        async def stream() -> StreamingResponse:
            return StreamingResponse(self._frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

        @app.get("/status")
        async def status() -> Dict[str, Any]:
            _, _, camera_status = self.capture_thread.get_frame()
            _, detections, inference_time, _ = self.processor.get_results()
            return {
                "camera_status": camera_status,
                "detections": len(detections),
                "fps": round(self.fps, 1),
                "inference_time_ms": round(inference_time * 1000, 1),
                "occupied_positions": self.floor_position_detector.occupied_positions,
                # GESTURE DETECTOR INTEGRATION: Return the current hand-raise gesture toggle value (0 or 1)
                "gesture": self.floor_position_detector.gesture_state,
                "model": self.processor.model_path,
            }

        return app

    def _frame_generator(self) -> Generator[bytes, None, None]:
        boundary = b"--frame"
        while not self.capture_thread.stop_event.is_set():
            frame = self._get_latest_stream_frame()
            if frame is None:
                frame, _, _ = self.capture_thread.get_frame()

            if frame is None:
                time.sleep(0.05)
                continue

            success, jpeg = cv2.imencode('.jpg', frame)
            if not success:
                continue

            frame_bytes = jpeg.tobytes()
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(frame_bytes)).encode('utf-8')
                + b"\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            time.sleep(0.02)

    def _start_web_server(self) -> None:
        if not self.config.ENABLE_WEB_UI:
            return
        self.web_app = self._build_web_app()
        server_config = uvicorn.Config(
            self.web_app,
            host=self.config.WEBSERVER_HOST,
            port=self.config.WEBSERVER_PORT,
            log_level="info",
            loop="asyncio",
        )
        self.web_server = uvicorn.Server(server_config)
        self.web_thread = threading.Thread(target=self.web_server.run, daemon=True)
        self.web_thread.start()
        logging.info("Started FastAPI web server on http://%s:%d", self.config.WEBSERVER_HOST, self.config.WEBSERVER_PORT)

    def _stop_web_server(self) -> None:
        if self.web_server is not None:
            self.web_server.should_exit = True
        if self.web_thread is not None:
            self.web_thread.join(timeout=2.0)

    def _get_latest_stream_frame(self) -> Optional[np.ndarray]:
        with self.latest_frame_lock:
            return self.latest_stream_frame.copy() if self.latest_stream_frame is not None else None

    def _prepare_csv_log(self) -> None:
        if not self.config.DETECTIONS_CSV.exists():
            with open(self.config.DETECTIONS_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.config.CSV_HEADERS)
                writer.writeheader()
                logging.info("Created detection log file: %s", self.config.DETECTIONS_CSV)

    def _append_detections(self, detections: List[Dict[str, Any]]) -> None:
        if not detections:
            return
        with open(self.config.DETECTIONS_CSV, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.config.CSV_HEADERS)
            for detection in detections:
                filtered_detection = {key: detection.get(key) for key in self.config.CSV_HEADERS}
                writer.writerow(filtered_detection)

    def _init_video_writer(self, frame_shape: Tuple[int, int, int]) -> None:
        if self.video_writer is not None:
            return
        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.config.VIDEO_CODEC)
        output_path = self.config.RECORDING_DIR / f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        self.video_writer = cv2.VideoWriter(str(output_path), fourcc, self.config.VIDEO_FPS, (width, height))
        if self.video_writer.isOpened():
            logging.info("Recording enabled, writing to %s", output_path)
        else:
            logging.error("Failed to initialize video writer")
            self.video_writer = None

    def _draw_overlay(
        self,
        frame: np.ndarray,
        status: str,
        detection_count: int,
        class_counts: Counter,
        model_name: str,
        inference_time: float,
        occupied_positions: List[int],
    ) -> np.ndarray:
        overlay = frame.copy()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        texts = [
            f"Model: {model_name}",
            f"Status: {status}",
            f"FPS: {self.fps:.1f}",
            f"Inference: {inference_time * 1000:.0f} ms",
            f"Detections: {detection_count}",
            f"Occupied positions: {occupied_positions}",
            # GESTURE DETECTOR INTEGRATION: Render the gesture state onto the video frame overlay
            f"Gesture: {self.floor_position_detector.gesture_state}",
            f"Time: {timestamp}",
        ]

        for idx, text in enumerate(texts):
            cv2.putText(
                overlay,
                text,
                (10, 30 + idx * self.config.FONT_LINE_GAP),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.FONT_SCALE,
                self.config.LINE_COLOR,
                self.config.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        if class_counts:
            class_summary = ", ".join([f"{name}:{count}" for name, count in class_counts.items()])
            cv2.putText(
                overlay,
                class_summary,
                (10, 30 + len(texts) * self.config.FONT_LINE_GAP),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.FONT_SCALE,
                self.config.LINE_COLOR,
                self.config.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return overlay

    def _save_screenshot(self, frame: np.ndarray, annotated: Optional[np.ndarray]) -> None:
        if not self.config.SAVE_SCREENSHOTS:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_path = self.config.SCREENSHOT_DIR / f"screenshot_{timestamp}.png"
        annotated_path = self.config.SCREENSHOT_DIR / f"screenshot_{timestamp}_annotated.png"
        cv2.imwrite(str(original_path), frame)
        logging.info("Saved screenshot: %s", original_path)
        if annotated is not None:
            cv2.imwrite(str(annotated_path), annotated)
            logging.info("Saved annotated screenshot: %s", annotated_path)

    def _shutdown(self) -> None:
        logging.info("Shutting down application")
        self.capture_thread.stop()
        self.processor.stop()
        self.capture_thread.join(timeout=2.0)
        self.processor.join(timeout=2.0)
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

    def run(self) -> None:
        logging.info("Starting detection application")
        self.capture_thread.start()
        self.processor.start()
        self._start_web_server()

        if self.config.ENABLE_OPENCV_WINDOW:
            cv2.namedWindow(self.config.WINDOW_NAME, cv2.WINDOW_NORMAL)

        last_frame_timestamp = 0.0

        try:
            while True:
                frame, frame_timestamp, status = self.capture_thread.get_frame()
                if frame is None:
                    time.sleep(0.02)
                    continue

                annotated_frame, detections, inference_time, result_timestamp = self.processor.get_results()
                display_frame = annotated_frame if annotated_frame is not None else frame
                # FINGER GESTURE INTEGRATION: Pass raw frame to update to detect finger landmarks
                occupied_positions = self.floor_position_detector.update(detections, time.time(), frame)
                self.floor_position_detector.draw_floor_positions(display_frame)
                self.floor_position_detector.draw_person_ids(display_frame, detections)
                class_counts = Counter([item["class_name"] for item in detections])
                detection_count = len(detections)
                self.fps = self._calculate_fps()
                display_frame = self._draw_overlay(
                    display_frame,
                    status,
                    detection_count,
                    class_counts,
                    self.processor.model_path,
                    inference_time,
                    occupied_positions,
                )

                with self.latest_frame_lock:
                    self.latest_stream_frame = display_frame.copy()

                if self.config.ENABLE_OPENCV_WINDOW:
                    cv2.imshow(self.config.WINDOW_NAME, display_frame)

                # Annotated video recording (if enabled and not recording raw frames)
                if self.video_writer is None and self.config.ENABLE_RECORDING and not self.config.RECORD_RAW_VIDEO:
                    self._init_video_writer(display_frame.shape)

                if self.video_writer is not None:
                    self.video_writer.write(display_frame)

                if annotated_frame is not None and result_timestamp != self.last_logged_timestamp:
                    self._append_detections(detections)
                    self.last_logged_timestamp = result_timestamp

                key = -1
                if self.config.ENABLE_OPENCV_WINDOW:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logging.info("Quit command received")
                        break
                    if key == ord("s"):
                        self._save_screenshot(frame, annotated_frame)
                    if key == ord("r"):
                        logging.info("Reconnect command received")
                        self.capture_thread._reconnect()
                else:
                    time.sleep(0.01)

                if frame_timestamp == last_frame_timestamp:
                    continue
                last_frame_timestamp = frame_timestamp

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received")
        finally:
            self._shutdown()

    def _calculate_fps(self) -> float:
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_display_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_display_time = now
        return self.fps


def configure_logging() -> None:
    log_path = config.LOG_DIR / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_path), mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    configure_logging()
    logging.info("Application initializing")
    app = DetectionApp()
    app.run()


if __name__ == "__main__":
    main()
