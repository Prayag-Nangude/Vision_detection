import csv
import io
import json
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import hypot
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn

import config

# FINGER GESTURE INTEGRATION: Import MediaPipe library for finger landmark detection
import mediapipe as mp



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
    # POSE-BASED OCCUPANCY: History list of right wrist coordinates (x, y, timestamp) for swipe detection
    right_wrist_history: List[Tuple[float, float, float]] = field(default_factory=list)


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
        self.gesture_zone_rect = list(config.GESTURE_ZONE_RECT)
        # GESTURE DETECTOR INTEGRATION: Removed regional gesture rectangle occupancy tracking.
        # Hand-raise gestures are now monitored screen-wide globally.
        self.tracker = PersonTracker(max_distance=config.FLOOR_TRACKER_MAX_DISTANCE, timeout=config.FLOOR_TRACKER_TIMEOUT)
        self.occupied_positions: List[int] = []
        self.gesture_state: int = 0
        # POSE-BASED OCCUPANCY: Array representing occupied years where a chest-level hand-raise gesture is active (video stopped)
        self.video_years: List[int] = []
        self.hand_previously_raised: bool = False
        # ANTIGRAVITY ADDITION: Track whether a person and/or hand is inside the gesture zone
        self.person_in_gesture_zone: bool = False
        self.hand_in_gesture_zone: bool = False
        # ANTIGRAVITY ADDITION: State variables for sequence-based gesture password trigger (3->4 for ON, 4->3 for OFF)
        self.gesture_sequence: List[int] = []              # History of stable finger counts (e.g. [3, 4] or [4, 3])
        self.last_stable_finger_count: Optional[int] = None # Current candidate finger count being debounced
        self.stable_count_frames: int = 0                  # Frame count for debouncing
        self.last_hand_seen_time: float = 0.0              # Timestamp of last frame with a hand
        self.last_sequence_time: float = 0.0               # Timestamp of the start of the current sequence
 
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
 
    def update_positions(self, new_positions: List[Dict[str, Any]], gesture_zone: Optional[List[int]] = None) -> None:
        updated = []
        for pos in new_positions:
            center = tuple(pos["center"]) if isinstance(pos["center"], list) else pos["center"]
            updated.append(FloorPosition(
                number=int(pos["number"]),
                center=center,
                radius=int(pos["radius"])
            ))
        self.positions = updated
        if gesture_zone is not None:
            self.gesture_zone_rect = list(gesture_zone)

    def update(self, detections: List[Dict[str, Any]], timestamp: float, frame: Optional[np.ndarray] = None) -> List[int]:
        # ANTIGRAVITY ADDITION: Reset the zone status flags for this frame
        self.person_in_gesture_zone = False
        self.hand_in_gesture_zone = False

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

            # POSE-BASED OCCUPANCY: Try using left/right ankles from YOLO Pose before falling back
            keypoints = detection.get("keypoints")
            keypoints_conf = detection.get("keypoints_conf")
            used_pose = False

            if keypoints is not None and keypoints_conf is not None:
                # POSE-BASED OCCUPANCY: Check if keypoints arrays have enough elements (Left Ankle = 15, Right Ankle = 16)
                if len(keypoints) > 16 and len(keypoints_conf) > 16:
                    left_ankle, right_ankle = keypoints[15], keypoints[16]
                    left_conf, right_conf = keypoints_conf[15], keypoints_conf[16]
                    conf_threshold = 0.5 # Min confidence to trust the keypoint detection

                    # POSE-BASED OCCUPANCY: Validate keypoint confidence and ensure coordinate validity
                    valid_left = left_conf > conf_threshold and left_ankle[0] > 0 and left_ankle[1] > 0
                    valid_right = right_conf > conf_threshold and right_ankle[0] > 0 and right_ankle[1] > 0

                    if valid_left or valid_right:
                        used_pose = True # Set flag to skip standard bounding box fallback
                        standing_years = [] # List to track which years this specific person occupies
                        # POSE-BASED OCCUPANCY: Check each year circle against left and right ankles independently
                        for position in self.positions:
                            is_inside = False
                            if valid_left:
                                dx_l = left_ankle[0] - position.center[0]
                                dy_l = left_ankle[1] - position.center[1]
                                if dx_l * dx_l + dy_l * dy_l <= position.radius * position.radius:
                                    is_inside = True
                            if not is_inside and valid_right:
                                dx_r = right_ankle[0] - position.center[0]
                                dy_r = right_ankle[1] - position.center[1]
                                if dx_r * dx_r + dy_r * dy_r <= position.radius * position.radius:
                                    is_inside = True
                            if is_inside:
                                occupied.add(position.number)
                                standing_years.append(position.number) # Record year occupied

                        # POSE-BASED OCCUPANCY: Process chest-level hand-raise and right hand swipe gestures if person is tracked
                        track = self.tracker.tracks.get(track_id)
                        
                        # POSE-BASED OCCUPANCY: Extract shoulder, hip, and wrist coordinates on left/right sides
                        left_shoulder, right_shoulder = keypoints[5], keypoints[6]
                        left_hip, right_hip = keypoints[11], keypoints[12]
                        left_wrist, right_wrist = keypoints[9], keypoints[10]

                        left_shoulder_conf = keypoints_conf[5]
                        right_shoulder_conf = keypoints_conf[6]
                        left_hip_conf = keypoints_conf[11]
                        right_hip_conf = keypoints_conf[12]
                        left_wrist_conf = keypoints_conf[9]
                        right_wrist_conf = keypoints_conf[10]

                        # POSE-BASED OCCUPANCY: Check left hand raised to chest/mid-body level
                        left_hand_raised = False
                        if left_shoulder_conf > 0.5 and left_hip_conf > 0.5 and left_wrist_conf > 0.5:
                            chest_y_l = (left_shoulder[1] + left_hip[1]) / 2.0
                            if left_wrist[1] < chest_y_l and left_wrist[0] > 0 and left_wrist[1] > 0:
                                left_hand_raised = True

                        # POSE-BASED OCCUPANCY: Check right hand raised to chest/mid-body level
                        right_hand_raised = False
                        if right_shoulder_conf > 0.5 and right_hip_conf > 0.5 and right_wrist_conf > 0.5:
                            chest_y_r = (right_shoulder[1] + right_hip[1]) / 2.0
                            if right_wrist[1] < chest_y_r and right_wrist[0] > 0 and right_wrist[1] > 0:
                                right_hand_raised = True

                        # POSE-BASED OCCUPANCY: Add standing years to video array if hand raise is detected
                        if (left_hand_raised or right_hand_raised) and standing_years:
                            for y in standing_years:
                                if y not in self.video_years:
                                    self.video_years.append(y)
                                    logging.info(f"POSE-BASED OCCUPANCY: Year {y} added to video array (chest-level hand-raise)")

                        # POSE-BASED OCCUPANCY: Track right wrist coordinate history for swipe detection
                        if track is not None and right_wrist_conf > 0.5 and right_wrist[0] > 0 and right_wrist[1] > 0:
                            current_time = time.time()
                            track.right_wrist_history.append((right_wrist[0], right_wrist[1], current_time))
                            # Keep only history from the last 0.8 seconds to avoid stale calculations
                            track.right_wrist_history = [
                                item for item in track.right_wrist_history
                                if current_time - item[2] <= 0.8
                            ]

                            # POSE-BASED OCCUPANCY: Detect horizontal swipe (large X diff, small Y diff, short time window)
                            if len(track.right_wrist_history) >= 5:
                                oldest = track.right_wrist_history[0]
                                newest = track.right_wrist_history[-1]
                                dx_swipe = newest[0] - oldest[0]
                                dy_swipe = newest[1] - oldest[1]
                                dt_swipe = newest[2] - oldest[2]

                                if 0.05 < dt_swipe < 0.6:
                                    if abs(dx_swipe) > 120 and abs(dy_swipe) < 80:
                                        # POSE-BASED OCCUPANCY: Right swipe detected, clear history to prevent multiple triggers
                                        track.right_wrist_history = []
                                        for y in standing_years:
                                            if y in self.video_years:
                                                self.video_years.remove(y)
                                                logging.info(f"POSE-BASED OCCUPANCY: Year {y} removed from video array (right swipe)")

            # POSE-BASED OCCUPANCY: Standard fallback to bounding box bottom-center if pose is unavailable/low confidence
            if not used_pose:
                # Check year floor circles using default midpoint
                for position in self.positions:
                    if position.contains(point):
                        occupied.add(position.number)

        # ANTIGRAVITY ADDITION: Check if any detected person's bounding box intersects with the gesture zone rectangle
        rect_x_min, rect_y_min, rect_x_max, rect_y_max = self.gesture_zone_rect
        for detection in person_detections:
            x1 = float(detection.get("x1", 0))
            y1 = float(detection.get("y1", 0))
            x2 = float(detection.get("x2", 0))
            y2 = float(detection.get("y2", 0))
            # Overlap exists if they are not completely separated on either axis
            if not (x2 < rect_x_min or x1 > rect_x_max or y2 < rect_y_min or y1 > rect_y_max):
                self.person_in_gesture_zone = True
                break

        current_time = time.time()

        # ANTIGRAVITY ADDITION: Cleanup sequence history and debounce tracker if no hand has been seen for 5.0 seconds
        if current_time - self.last_hand_seen_time > 5.0:
            self.gesture_sequence = []
            self.last_stable_finger_count = None
            self.stable_count_frames = 0

        # ANTIGRAVITY ADDITION: Cleanup sequence if the sequence itself has timed out (longer than 5.0 seconds since starting)
        if self.gesture_sequence and (current_time - self.last_sequence_time > 5.0):
            self.gesture_sequence = []

        # FINGER GESTURE INTEGRATION: Process frame with MediaPipe to detect 1 (OFF) or 2 (ON) raised fingers.
        # ANTIGRAVITY ADDITION: Run sequence-based gesture trigger logic only if a person intersects with the gesture detection zone rectangle
        if frame is not None and self.person_in_gesture_zone:
            try:
                # Crop gesture zone from the frame
                h, w = frame.shape[:2]
                x_start = max(0, min(rect_x_min, w))
                y_start = max(0, min(rect_y_min, h))
                x_end = max(0, min(rect_x_max, w))
                y_end = max(0, min(rect_y_max, h))
                
                if (x_end - x_start) >= 16 and (y_end - y_start) >= 16:
                    crop = frame[y_start:y_end, x_start:x_end]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(crop_rgb)
                    if results.multi_hand_landmarks:
                        # ANTIGRAVITY ADDITION: Mark that a hand/gesture is actively detected inside the zone and update hand tracker timestamp
                        self.hand_in_gesture_zone = True
                        self.last_hand_seen_time = current_time
                        
                        # Process first detected hand
                        hand_landmarks = results.multi_hand_landmarks[0]
                        landmarks = hand_landmarks.landmark
                        # Check index, middle, ring, pinky fingers status (y tip < y pip)
                        index_up = landmarks[8].y < landmarks[6].y
                        middle_up = landmarks[12].y < landmarks[10].y
                        ring_up = landmarks[16].y < landmarks[14].y
                        pinky_up = landmarks[20].y < landmarks[18].y

                        up_count = sum([index_up, middle_up, ring_up, pinky_up])

                        # POSE-BASED OCCUPANCY: Enforce that a 3-finger count must strictly match Option B (Index folded, others extended)
                        if up_count == 3 and not (not index_up and middle_up and ring_up and pinky_up):
                            up_count = 0

                        # ANTIGRAVITY ADDITION: Debouncing (consecutive frame filtering)
                        if up_count == self.last_stable_finger_count:
                            self.stable_count_frames += 1
                        else:
                            self.last_stable_finger_count = up_count
                            self.stable_count_frames = 1

                        # Register count if stable for 2 consecutive frames
                        if self.stable_count_frames == 2:
                            stable_count = up_count
                            
                            # Only count 3 or 4 fingers as valid sequence inputs
                            if stable_count in [3, 4]:
                                # Append to sequence history only if it's different from the last recorded step
                                if not self.gesture_sequence or self.gesture_sequence[-1] != stable_count:
                                    if not self.gesture_sequence:
                                        self.last_sequence_time = current_time  # Start of sequence timer
                                    self.gesture_sequence.append(stable_count)
                                    logging.info(f"Registered sequence step: {self.gesture_sequence}")

                                    # Check for ON transition (3 fingers -> 4 fingers)
                                    if self.gesture_sequence[-2:] == [3, 4]:
                                        if self.gesture_state != 1:
                                            self.gesture_state = 1
                                            logging.info("Gesture state set to: 1 (ON Sequence [3, 4] completed)")
                                        self.gesture_sequence = []

                                    # Check for OFF transition (4 fingers -> 3 fingers)
                                    elif self.gesture_sequence[-2:] == [4, 3]:
                                        if self.gesture_state != 0:
                                            self.gesture_state = 0
                                            logging.info("Gesture state reset to: 0 (OFF Sequence [4, 3] completed)")
                                        self.gesture_sequence = []
            except Exception as exc:
                logging.warning("Error checking finger gesture: %s", exc)

        self.occupied_positions = sorted(occupied)
        return self.occupied_positions

    def draw_floor_positions(self, frame: np.ndarray) -> None:
        # ANTIGRAVITY ADDITION: Draw the gesture detection zone rectangle and its label on the frame (with Grey -> Blue -> Green state colors)
        rect_x_min, rect_y_min, rect_x_max, rect_y_max = self.gesture_zone_rect
        
        if self.hand_in_gesture_zone:
            rect_color = config.GESTURE_ZONE_ACTIVE_COLOR
        elif self.person_in_gesture_zone:
            rect_color = config.GESTURE_ZONE_PERSON_COLOR
        else:
            rect_color = config.GESTURE_ZONE_DEFAULT_COLOR

        cv2.rectangle(frame, (rect_x_min, rect_y_min), (rect_x_max, rect_y_max), rect_color, config.GESTURE_ZONE_THICKNESS)
        
        label_text = "GESTURE ZONE"
        cv2.putText(
            frame,
            label_text,
            (rect_x_min, max(rect_y_min - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            rect_color,
            2,
            cv2.LINE_AA,
        )

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
                        
                        .stream-container {{ position: relative; width: 100%; border-radius: 22px; overflow: hidden; background: #020617; border: 1px solid rgba(148,163,184,0.12); user-select: none; -webkit-user-select: none; }}
                        .stream-container img {{ width: 100%; height: auto; display: block; user-select: none; -webkit-user-drag: none; }}
                        
                        .sticky-stream {{
                            position: sticky;
                            top: 24px;
                        }}
                        
                        #editor-list {{
                            max-height: 400px;
                            overflow-y: auto;
                            padding-right: 6px;
                            display: grid;
                            gap: 16px;
                        }}
                        
                        #editor-list::-webkit-scrollbar {{
                            width: 6px;
                        }}
                        #editor-list::-webkit-scrollbar-track {{
                            background: rgba(255, 255, 255, 0.02);
                            border-radius: 8px;
                        }}
                        #editor-list::-webkit-scrollbar-thumb {{
                            background: rgba(255, 255, 255, 0.1);
                            border-radius: 8px;
                        }}
                        
                        .footer {{ margin-top: 26px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.08); color: #94a3b8; font-size: 0.92rem; }}
                        .hint {{ margin-top: 14px; font-size: 0.95rem; color: #cbd5e1; }}
                        
                        .btn-primary {{
                            background: linear-gradient(135deg, #2563eb, #1d4ed8);
                            color: #fff;
                            border: none;
                            border-radius: 12px;
                            padding: 8px 16px;
                            font-size: 0.9rem;
                            font-weight: 600;
                            cursor: pointer;
                            transition: all 0.2s ease;
                            box-shadow: 0 4px 12px rgba(37,99,235,0.2);
                            display: flex;
                            align-items: center;
                            gap: 6px;
                        }}
                        .btn-primary:hover {{
                            transform: translateY(-1px);
                            box-shadow: 0 6px 16px rgba(37,99,235,0.3);
                        }}
                        .btn-primary.active {{
                            background: linear-gradient(135deg, #10b981, #059669);
                            box-shadow: 0 4px 12px rgba(16,185,129,0.2);
                        }}
                        .editor-row {{
                            background: rgba(30, 41, 59, 0.4);
                            border: 1px solid rgba(148, 163, 184, 0.08);
                            border-radius: 16px;
                            padding: 12px;
                            transition: all 0.2s ease;
                            cursor: pointer;
                        }}
                        .editor-row.selected {{
                            border-color: #3b82f6;
                            background: rgba(59, 130, 246, 0.08);
                        }}
                        .editor-row-header {{
                            display: flex;
                            justify-content: space-between;
                            font-weight: 700;
                            margin-bottom: 8px;
                            font-size: 0.95rem;
                        }}
                        .editor-inputs {{
                            display: grid;
                            grid-template-columns: repeat(2, 1fr);
                            gap: 8px;
                            margin-bottom: 8px;
                        }}
                        .editor-input-group {{
                            display: flex;
                            align-items: center;
                            gap: 6px;
                            font-size: 0.85rem;
                            color: #94a3b8;
                        }}
                        .editor-input-group input {{
                            width: 100%;
                            background: #020617;
                            border: 1px solid rgba(148, 163, 184, 0.15);
                            border-radius: 8px;
                            padding: 4px 8px;
                            color: #fff;
                            font-size: 0.85rem;
                        }}
                        .editor-slider-group {{
                            display: flex;
                            align-items: center;
                            gap: 8px;
                            font-size: 0.85rem;
                            color: #94a3b8;
                        }}
                        .editor-slider-group input {{
                            flex: 1;
                        }}
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
                            <section class="card sticky-stream">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                                    <h2 style="margin: 0;">Live Stream</h2>
                                    <button id="toggle-edit-mode" class="btn-primary" onclick="toggleEditMode()">
                                        ⚙️ Edit Coordinates
                                    </button>
                                </div>
                                <div class="stream-container">
                                    <img src="/stream" alt="Live YOLO stream" />
                                    <svg id="interactive-overlay" viewBox="0 0 2560 1440" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></svg>
                                </div>
                                <p class="hint">If the stream does not appear, confirm the mobile camera URL is reachable and the service is running.</p>
                            </section>
                            
                            <div>
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
                                
                                <section id="editor-card" class="card" style="display: none; margin-top: 24px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                                        <h2 style="margin: 0;">Positions Editor</h2>
                                        <span id="save-status" style="font-size: 0.85rem; color: #10b981; opacity: 0; transition: opacity 0.5s;">Saved!</span>
                                    </div>
                                    <div id="editor-list" style="display: grid; gap: 16px;"></div>
                                    <div style="margin-top: 20px; display: flex; gap: 12px;">
                                        <button class="btn-primary active" style="flex: 1; justify-content: center;" onclick="saveToServer()">Save Configurations</button>
                                    </div>
                                </section>
                            </div>
                        </div>
                        <div class="footer">
                            <p>Use the keyboard controls in the application window if enabled: Q = Quit, S = Screenshot, R = Reconnect.</p>
                        </div>
                    </main>
                    <script>
                        let editMode = false;
                        let positions = [];
                        let gestureZone = [1750, 900, 2050, 1200];
                        let selectedIndex = -1; // -2 for gesture zone, -3 for gesture zone resize handle
                        let isDragging = false;
                        let dragOffset = {{ x: 0, y: 0 }};

                        async function updateStatus() {{
                            try {{
                                const response = await fetch('/status');
                                const data = await response.json();
                                document.getElementById('status').textContent = data.camera_status;
                                document.getElementById('detections').textContent = data.detections;
                                document.getElementById('fps').textContent = data.fps;
                                document.getElementById('inference').textContent = data.inference_time_ms + ' ms';
                                document.getElementById('occupied').textContent = JSON.stringify(data.occupied_positions);
                                document.getElementById('gesture').textContent = data.gesture;
                            }} catch (error) {{
                                document.getElementById('status').textContent = 'Disconnected';
                                document.getElementById('occupied').textContent = '[]';
                                document.getElementById('gesture').textContent = '0';
                            }}
                        }}

                        async function loadPositions() {{
                            try {{
                                const response = await fetch('/api/positions');
                                const data = await response.json();
                                positions = data.years;
                                gestureZone = data.gesture_zone;
                                if (editMode) {{
                                    renderSVG();
                                    renderEditorList();
                                }}
                            }} catch (error) {{
                                console.error('Failed to load positions', error);
                            }}
                        }}

                        function toggleEditMode() {{
                            editMode = !editMode;
                            const btn = document.getElementById('toggle-edit-mode');
                            const panel = document.getElementById('editor-card');
                            const overlay = document.getElementById('interactive-overlay');

                            if (editMode) {{
                                btn.textContent = '🔒 Lock Positions';
                                btn.classList.add('active');
                                panel.style.display = 'block';
                                overlay.style.pointerEvents = 'auto';
                                loadPositions();
                            }} else {{
                                btn.textContent = '⚙️ Edit Coordinates';
                                btn.classList.remove('active');
                                panel.style.display = 'none';
                                overlay.style.pointerEvents = 'none';
                                overlay.innerHTML = '';
                                saveToServer();
                            }}
                        }}

                        function getSVGCoords(e) {{
                            const svg = document.getElementById('interactive-overlay');
                            const rect = svg.getBoundingClientRect();
                            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
                            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
                            const x = (clientX - rect.left) * (2560 / rect.width);
                            const y = (clientY - rect.top) * (1440 / rect.height);
                            return {{ x: Math.round(x), y: Math.round(y) }};
                        }}

                        function renderSVG() {{
                            const overlay = document.getElementById('interactive-overlay');
                            overlay.innerHTML = '';
                            if (!editMode) return;

                            // 1. Render Gesture Zone Rectangle
                            const gzSelected = selectedIndex === -2 || selectedIndex === -3;
                            const x_min = gestureZone[0];
                            const y_min = gestureZone[1];
                            const x_max = gestureZone[2];
                            const y_max = gestureZone[3];
                            const width = x_max - x_min;
                            const height = y_max - y_min;

                            const gzGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                            
                            const gzRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                            gzRect.setAttribute('x', x_min);
                            gzRect.setAttribute('y', y_min);
                            gzRect.setAttribute('width', width);
                            gzRect.setAttribute('height', height);
                            gzRect.setAttribute('fill', gzSelected ? 'rgba(59, 130, 246, 0.15)' : 'rgba(255, 255, 255, 0.05)');
                            gzRect.setAttribute('stroke', gzSelected ? '#3b82f6' : 'rgba(255, 255, 255, 0.4)');
                            gzRect.setAttribute('stroke-width', gzSelected ? '4' : '2');
                            gzRect.setAttribute('stroke-dasharray', '8 4');
                            gzRect.style.cursor = 'move';

                            const startGZDrag = (e) => {{
                                e.preventDefault();
                                selectedIndex = -2;
                                isDragging = true;
                                const coords = getSVGCoords(e);
                                dragOffset.x = coords.x - gestureZone[0];
                                dragOffset.y = coords.y - gestureZone[1];
                                renderSVG();
                                renderEditorList();
                                highlightRow('gesture');
                            }};
                            gzRect.addEventListener('mousedown', startGZDrag);
                            gzRect.addEventListener('touchstart', startGZDrag);
                            gzGroup.appendChild(gzRect);

                            const gzText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                            gzText.setAttribute('x', x_min + width / 2);
                            gzText.setAttribute('y', y_min + height / 2 + 6);
                            gzText.setAttribute('fill', '#fff');
                            gzText.setAttribute('font-size', '22px');
                            gzText.setAttribute('font-weight', 'bold');
                            gzText.setAttribute('text-anchor', 'middle');
                            gzText.setAttribute('pointer-events', 'none');
                            gzText.textContent = 'GESTURE ZONE';
                            gzGroup.appendChild(gzText);

                            const handle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                            handle.setAttribute('cx', x_max);
                            handle.setAttribute('cy', y_max);
                            handle.setAttribute('r', '14');
                            handle.setAttribute('fill', '#3b82f6');
                            handle.setAttribute('stroke', '#fff');
                            handle.setAttribute('stroke-width', '2');
                            handle.style.cursor = 'se-resize';

                            const startGZResize = (e) => {{
                                e.preventDefault();
                                selectedIndex = -3;
                                isDragging = true;
                                const coords = getSVGCoords(e);
                                dragOffset.x = coords.x - gestureZone[2];
                                dragOffset.y = coords.y - gestureZone[3];
                                renderSVG();
                                renderEditorList();
                                highlightRow('gesture');
                            }};
                            handle.addEventListener('mousedown', startGZResize);
                            handle.addEventListener('touchstart', startGZResize);
                            gzGroup.appendChild(handle);

                            overlay.appendChild(gzGroup);

                            // 2. Render Year Circles
                            positions.forEach((pos, idx) => {{
                                const isSelected = idx === selectedIndex;
                                const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                                group.style.cursor = 'move';
                                
                                if (isSelected) {{
                                    const outer = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                                    outer.setAttribute('cx', pos.center[0]);
                                    outer.setAttribute('cy', pos.center[1]);
                                    outer.setAttribute('r', pos.radius + 8);
                                    outer.setAttribute('fill', 'none');
                                    outer.setAttribute('stroke', '#3b82f6');
                                    outer.setAttribute('stroke-width', '3');
                                    outer.setAttribute('stroke-dasharray', '5 3');
                                    group.appendChild(outer);
                                }}

                                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                                circle.setAttribute('cx', pos.center[0]);
                                circle.setAttribute('cy', pos.center[1]);
                                circle.setAttribute('r', pos.radius);
                                circle.setAttribute('fill', isSelected ? 'rgba(59, 130, 246, 0.25)' : 'rgba(255, 255, 255, 0.12)');
                                circle.setAttribute('stroke', isSelected ? '#3b82f6' : '#fff');
                                circle.setAttribute('stroke-width', isSelected ? '4' : '2');

                                const startDrag = (e) => {{
                                    e.preventDefault();
                                    selectedIndex = idx;
                                    isDragging = true;
                                    const coords = getSVGCoords(e);
                                    dragOffset.x = coords.x - pos.center[0];
                                    dragOffset.y = coords.y - pos.center[1];
                                    renderSVG();
                                    renderEditorList();
                                    highlightRow(idx);
                                }};

                                circle.addEventListener('mousedown', startDrag);
                                circle.addEventListener('touchstart', startDrag);
                                group.appendChild(circle);

                                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                                text.setAttribute('x', pos.center[0]);
                                text.setAttribute('y', pos.center[1] + 6);
                                text.setAttribute('fill', '#fff');
                                text.setAttribute('font-size', '20px');
                                text.setAttribute('font-weight', 'bold');
                                text.setAttribute('text-anchor', 'middle');
                                text.setAttribute('pointer-events', 'none');
                                text.textContent = pos.number;
                                group.appendChild(text);

                                overlay.appendChild(group);
                            }});
                        }}

                        function renderEditorList() {{
                            const list = document.getElementById('editor-list');
                            list.innerHTML = '';

                            // 1. Render Gesture Zone editor row
                            const gzSelected = selectedIndex === -2 || selectedIndex === -3;
                            const gzRow = document.createElement('div');
                            gzRow.className = 'editor-row' + (gzSelected ? ' selected' : '');
                            gzRow.id = 'editor-row-gesture';
                            gzRow.onclick = () => {{
                                selectedIndex = -2;
                                renderSVG();
                                renderEditorList();
                            }};

                            const gzWidth = gestureZone[2] - gestureZone[0];
                            const gzHeight = gestureZone[3] - gestureZone[1];

                            gzRow.innerHTML = `
                                <div class="editor-row-header">
                                    <span>Gesture Zone (Rectangle)</span>
                                </div>
                                <div class="editor-inputs" onclick="event.stopPropagation()">
                                    <div class="editor-input-group">
                                        <span>X Min:</span>
                                        <input type="number" value="${{gestureZone[0]}}" onchange="updateGZCoord(0, this.value)">
                                    </div>
                                    <div class="editor-input-group">
                                        <span>Y Min:</span>
                                        <input type="number" value="${{gestureZone[1]}}" onchange="updateGZCoord(1, this.value)">
                                    </div>
                                    <div class="editor-input-group">
                                        <span>X Max:</span>
                                        <input type="number" value="${{gestureZone[2]}}" onchange="updateGZCoord(2, this.value)">
                                    </div>
                                    <div class="editor-input-group">
                                        <span>Y Max:</span>
                                        <input type="number" value="${{gestureZone[3]}}" onchange="updateGZCoord(3, this.value)">
                                    </div>
                                </div>
                                <div class="editor-slider-group" onclick="event.stopPropagation()">
                                    <span>Width:</span>
                                    <input type="range" min="50" max="1000" value="${{gzWidth}}" oninput="updateGZDim('width', this.value)">
                                    <span>${{gzWidth}}px</span>
                                </div>
                                <div class="editor-slider-group" onclick="event.stopPropagation()">
                                    <span>Height:</span>
                                    <input type="range" min="50" max="1000" value="${{gzHeight}}" oninput="updateGZDim('height', this.value)">
                                    <span>${{gzHeight}}px</span>
                                </div>
                            `;
                            list.appendChild(gzRow);

                            // 2. Render Year Circles
                            positions.forEach((pos, idx) => {{
                                const isSelected = idx === selectedIndex;
                                const row = document.createElement('div');
                                row.className = 'editor-row' + (isSelected ? ' selected' : '');
                                row.id = `editor-row-${{idx}}`;
                                row.onclick = () => {{
                                    selectedIndex = idx;
                                    renderSVG();
                                    renderEditorList();
                                }};

                                row.innerHTML = `
                                    <div class="editor-row-header">
                                        <span>Year ${{pos.number}}</span>
                                    </div>
                                    <div class="editor-inputs" onclick="event.stopPropagation()">
                                        <div class="editor-input-group">
                                            <span>X:</span>
                                            <input type="number" value="${{pos.center[0]}}" onchange="updateCoord(${{idx}}, 0, this.value)">
                                        </div>
                                        <div class="editor-input-group">
                                            <span>Y:</span>
                                            <input type="number" value="${{pos.center[1]}}" onchange="updateCoord(${{idx}}, 1, this.value)">
                                        </div>
                                    </div>
                                    <div class="editor-slider-group" onclick="event.stopPropagation()">
                                        <span>Radius:</span>
                                        <input type="range" min="20" max="150" value="${{pos.radius}}" oninput="updateRadius(${{idx}}, this.value)">
                                        <span>${{pos.radius}}px</span>
                                    </div>
                                `;
                                list.appendChild(row);
                            }});
                        }}

                        function highlightRow(idx) {{
                            const container = document.getElementById('editor-list');
                            const row = document.getElementById(idx === 'gesture' ? 'editor-row-gesture' : `editor-row-${{idx}}`);
                            if (container && row) {{
                                const containerTop = container.scrollTop;
                                const containerBottom = containerTop + container.clientHeight;
                                const elemTop = row.offsetTop;
                                const elemBottom = elemTop + row.offsetHeight;
                                if (elemTop < containerTop) {{
                                    container.scrollTop = elemTop;
                                }} else if (elemBottom > containerBottom) {{
                                    container.scrollTop = elemBottom - container.clientHeight;
                                }}
                            }}
                        }}

                        function updateCoord(idx, coordIdx, val) {{
                            positions[idx].center[coordIdx] = parseInt(val) || 0;
                            renderSVG();
                            saveToServer();
                        }}

                        function updateRadius(idx, val) {{
                            positions[idx].radius = parseInt(val) || 50;
                            renderSVG();
                            renderEditorList();
                            saveToServer();
                        }}

                        function updateGZCoord(idx, val) {{
                            gestureZone[idx] = parseInt(val) || 0;
                            renderSVG();
                            saveToServer();
                        }}

                        function updateGZDim(dim, val) {{
                            const parsed = parseInt(val) || 50;
                            if (dim === 'width') {{
                                gestureZone[2] = gestureZone[0] + parsed;
                            }} else {{
                                gestureZone[3] = gestureZone[1] + parsed;
                            }}
                            renderSVG();
                            renderEditorList();
                            saveToServer();
                        }}

                        window.addEventListener('mousemove', (e) => {{
                            if (!isDragging || selectedIndex === -1) return;
                            e.preventDefault();
                            const coords = getSVGCoords(e);
                            
                            if (selectedIndex === -2) {{
                                const width = gestureZone[2] - gestureZone[0];
                                const height = gestureZone[3] - gestureZone[1];
                                const newX = Math.max(0, Math.min(2560 - width, coords.x - dragOffset.x));
                                const newY = Math.max(0, Math.min(1440 - height, coords.y - dragOffset.y));
                                gestureZone[0] = newX;
                                gestureZone[1] = newY;
                                gestureZone[2] = newX + width;
                                gestureZone[3] = newY + height;
                                renderSVG();
                                renderEditorList();
                            }} else if (selectedIndex === -3) {{
                                const newMaxX = Math.max(gestureZone[0] + 50, Math.min(2560, coords.x - dragOffset.x));
                                const newMaxY = Math.max(gestureZone[1] + 50, Math.min(1440, coords.y - dragOffset.y));
                                gestureZone[2] = newMaxX;
                                gestureZone[3] = newMaxY;
                                renderSVG();
                                renderEditorList();
                            }} else {{
                                positions[selectedIndex].center[0] = Math.max(0, Math.min(2560, coords.x - dragOffset.x));
                                positions[selectedIndex].center[1] = Math.max(0, Math.min(1440, coords.y - dragOffset.y));
                                renderSVG();
                                renderEditorList();
                            }}
                        }});

                        window.addEventListener('touchmove', (e) => {{
                            if (!isDragging || selectedIndex === -1) return;
                            e.preventDefault();
                            const coords = getSVGCoords(e);
                            
                            if (selectedIndex === -2) {{
                                const width = gestureZone[2] - gestureZone[0];
                                const height = gestureZone[3] - gestureZone[1];
                                const newX = Math.max(0, Math.min(2560 - width, coords.x - dragOffset.x));
                                const newY = Math.max(0, Math.min(1440 - height, coords.y - dragOffset.y));
                                gestureZone[0] = newX;
                                gestureZone[1] = newY;
                                gestureZone[2] = newX + width;
                                gestureZone[3] = newY + height;
                                renderSVG();
                                renderEditorList();
                            }} else if (selectedIndex === -3) {{
                                const newMaxX = Math.max(gestureZone[0] + 50, Math.min(2560, coords.x - dragOffset.x));
                                const newMaxY = Math.max(gestureZone[1] + 50, Math.min(1440, coords.y - dragOffset.y));
                                gestureZone[2] = newMaxX;
                                gestureZone[3] = newMaxY;
                                renderSVG();
                                renderEditorList();
                            }} else {{
                                positions[selectedIndex].center[0] = Math.max(0, Math.min(2560, coords.x - dragOffset.x));
                                positions[selectedIndex].center[1] = Math.max(0, Math.min(1440, coords.y - dragOffset.y));
                                renderSVG();
                                renderEditorList();
                            }}
                        }}, {{ passive: false }});

                        const stopDrag = () => {{
                            if (isDragging) {{
                                isDragging = false;
                                saveToServer();
                            }}
                        }};
                        window.addEventListener('mouseup', stopDrag);
                        window.addEventListener('touchend', stopDrag);

                        let saveTimeout;
                        function saveToServer() {{
                            clearTimeout(saveTimeout);
                            saveTimeout = setTimeout(async () => {{
                                try {{
                                    const payload = {{
                                        years: positions,
                                        gesture_zone: gestureZone
                                    }};
                                    const response = await fetch('/api/positions', {{
                                        method: 'POST',
                                        headers: {{ 'Content-Type': 'application/json' }},
                                        body: JSON.stringify(payload)
                                    }});
                                    const result = await response.json();
                                    if (result.status === 'success') {{
                                        const statusEl = document.getElementById('save-status');
                                        statusEl.style.opacity = '1';
                                        setTimeout(() => statusEl.style.opacity = '0', 2000);
                                    }}
                                }} catch (error) {{
                                    console.error('Failed to save positions to server', error);
                                }}
                            }}, 300);
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
                # POSE-BASED OCCUPANCY: Return the array of years where video has been stopped by hand gesture
                "video": self.floor_position_detector.video_years,
                "model": self.processor.model_path,
            }

        @app.get("/api/positions")
        async def get_positions() -> Dict[str, Any]:
            return {
                "years": [
                    {
                        "number": pos.number,
                        "center": list(pos.center),
                        "radius": pos.radius
                    }
                    for pos in self.floor_position_detector.positions
                ],
                "gesture_zone": list(self.floor_position_detector.gesture_zone_rect)
            }

        @app.post("/api/positions")
        async def save_positions(payload: Any = Body(...)) -> Dict[str, str]:
            try:
                new_positions = []
                gesture_zone = None
                
                if isinstance(payload, list):
                    new_positions = payload
                elif isinstance(payload, dict):
                    new_positions = payload.get("years", [])
                    gesture_zone = payload.get("gesture_zone", None)
                
                self.floor_position_detector.update_positions(new_positions, gesture_zone)
                
                serializable = {
                    "years": [
                        {
                            "number": int(pos["number"]),
                            "center": list(pos["center"]),
                            "radius": int(pos["radius"])
                        }
                        for pos in new_positions
                    ],
                    "gesture_zone": list(self.floor_position_detector.gesture_zone_rect)
                }
                with open(config.POSITIONS_FILE, "w") as f:
                    json.dump(serializable, f, indent=2)
                return {"status": "success", "message": "Positions saved successfully"}
            except Exception as e:
                logging.error(f"Failed to save positions: {e}")
                return {"status": "error", "message": str(e)}

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
        self._stop_web_server()
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
