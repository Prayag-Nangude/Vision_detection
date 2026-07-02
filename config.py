from pathlib import Path

# General camera settings
CAMERA_SOURCE: str = "rtsp://admin:Admin@123@192.168.4.148:554/Streaming/Channels/101"
# CAMERA_SOURCE: int = 0 

# Model settings
MODEL_PATH: str = "models/best.pt"
# FALLBACK_MODEL: Switched from 'yolo11n.pt' to 'yolo11n-pose.pt' to enable body keypoint extraction (wrists, shoulders) for hand-raise detection.
FALLBACK_MODEL: str = "yolo11n-pose.pt"
CONFIDENCE: float = 0.5
IOU_THRESHOLD: float = 0.45

# Screenshot and recording settings
SAVE_SCREENSHOTS: bool = True
# RECORDING SETTINGS: Disabled recording by default to prevent large files
ENABLE_RECORDING: bool = False
RECORD_RAW_VIDEO: bool = True  # If True, records raw (clean) camera frames without overlays/annotations.
SCREENSHOT_DIR: Path = Path("screenshots")
RECORDING_DIR: Path = Path("recordings")
LOG_DIR: Path = Path("logs")
DETECTIONS_CSV: Path = LOG_DIR / "detections.csv"

# Camera reconnect and performance tuning
RECONNECT_DELAY: float = 2.0
CAPTURE_WAIT: float = 0.02
MAX_FRAME_AGE_SECONDS: float = 1.0

# Video writer settings
VIDEO_CODEC: str = "mp4v"
VIDEO_FPS: int = 20
VIDEO_FRAME_SIZE: tuple[int, int] = (1280, 720)

# Display settings
WINDOW_NAME: str = "YOLO11 Live Detection"
FONT: int = 2
FONT_SCALE: float = 1.55
FONT_THICKNESS: int = 4
FONT_LINE_GAP: int = 80
LINE_COLOR: tuple[int, int, int] = (255, 255, 255)

# Misc
CSV_HEADERS: list[str] = ["timestamp", "class_name", "confidence", "x1", "y1", "x2", "y2"]

# Web UI settings
ENABLE_WEB_UI: bool = True
WEBSERVER_HOST: str = "0.0.0.0"
WEBSERVER_PORT: int = 8000
ENABLE_OPENCV_WINDOW: bool = True

import json

POSITIONS_FILE = Path("positions.json")

def load_positions_and_gesture() -> tuple[list[dict[str, object]], list[int]]:
    years = [
        {"number": 2011, "center": (1570, 1370), "radius": 50},
        {"number": 2012, "center": (1420, 1263), "radius": 50},
        {"number": 2013, "center": (1270, 1156), "radius": 50},
        {"number": 2014, "center": (1120, 1049), "radius": 50},
        {"number": 2019, "center": (640, 580), "radius": 50},
        {"number": 2020, "center": (791, 687), "radius": 50},
        {"number": 2021, "center": (943, 794), "radius": 50},
        {"number": 2022, "center": (1094, 901), "radius": 50},
    ]
    gesture_zone = [1750, 900, 2050, 1200]
    
    if POSITIONS_FILE.exists():
        try:
            with open(POSITIONS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if "years" in data:
                        loaded_years = data["years"]
                        for item in loaded_years:
                            item["center"] = tuple(item["center"])
                        years = loaded_years
                    if "gesture_zone" in data:
                        gesture_zone = list(data["gesture_zone"])
                elif isinstance(data, list):
                    for item in data:
                        item["center"] = tuple(item["center"])
                    years = data
        except Exception:
            pass
    return years, gesture_zone

FLOOR_POSITIONS, GESTURE_ZONE_RECT_LIST = load_positions_and_gesture()
FLOOR_POSITION_COLOR: tuple[int, int, int] = (200, 200, 200)
FLOOR_OCCUPIED_COLOR: tuple[int, int, int] = (34, 197, 94)
FLOOR_POSITION_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)
FLOOR_POSITION_FONT_SCALE: float = 0.85
FLOOR_POSITION_FONT_THICKNESS: int = 3
FLOOR_POSITION_LINE_THICKNESS: int = 3
FLOOR_TRACKER_MAX_DISTANCE: float = 120.0
FLOOR_TRACKER_TIMEOUT: float = 1.5

# ANTIGRAVITY ADDITION: Config settings for restricting gesture detection to a specific bounding box (Second Branch)
# Placed to the left of the 2011 circle (center: 2100, 1280) and shifted upwards
GESTURE_ZONE_RECT: tuple[int, int, int, int] = tuple(GESTURE_ZONE_RECT_LIST)  # (x_min, y_min, x_max, y_max)
GESTURE_ZONE_DEFAULT_COLOR: tuple[int, int, int] = (200, 200, 200)       # Grey (idle)
GESTURE_ZONE_PERSON_COLOR: tuple[int, int, int] = (255, 0, 0)            # Blue (person inside zone)
GESTURE_ZONE_ACTIVE_COLOR: tuple[int, int, int] = (34, 197, 94)          # Green (gesture detected)
GESTURE_ZONE_THICKNESS: int = 3