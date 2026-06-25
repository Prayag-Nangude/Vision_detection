# Guide for Claude: Python Environment & MediaPipe Gesture Logic

If you are assisting with this project on a new PC, please read this guide to understand the setup requirements and code design.

---

## 1. Environment & Python Version Requirements

* **Python Version:** **Python 3.12.x (64-bit)** is strictly required.
  * *Reason:* `mediapipe` does not have precompiled wheels for Windows on Python 3.13, causing build failures during installation.
* **Core Libraries:**
  * `mediapipe` (Standard MediaPipe Solutions Hands)
  * `opencv-python` (cv2)
  * `ultralytics` (YOLO Pose / YOLO11)
  * `fastapi` & `uvicorn` (Web Server)
  * `numpy`

### How to set up on Windows:
```powershell
# 1. Delete any existing virtual environment folder copied from another PC
Remove-Item -Recurse -Force .venv

# 2. Create a clean Python 3.12 virtual environment (using Python Launcher)
py -3.12 -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Start the application
python main.py
```

---

## 2. MediaPipe Integration Details

We use the standard MediaPipe Hands Solution. Here is how it is structured and used in `main.py`:

### Initialization:
```python
import mediapipe as mp

self.mp_hands = mp.solutions.hands
self.hands = self.mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

### Processing Frames:
```python
results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

### Landmark Access:
Each hand in `results.multi_hand_landmarks` contains 21 landmarks. They are accessed as:
```python
landmarks = hand_landmarks.landmark  # A list of 21 landmark objects
# Each landmark has: landmark.x, landmark.y, landmark.z (relative depth)
```

---

## 3. Rotation-Invariant Finger Checking

To prevent false positives when the hand is tilted or rotated, we do **not** use vertical `y`-coordinate comparisons. Instead, we use a **3D Euclidean distance ratio** check.

### Helper:
```python
def get_dist_3d(lm1, lm2):
    return ((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)**0.5
```

### Finger State Detection (Index, Middle, Ring, Pinky):
A finger is considered extended (UP) if the distance from the **Tip** to the **MCP joint (knuckle)** is greater than $1.3\times$ the distance from the **PIP joint** to the **MCP joint**:
$$\text{distance\_3d}(\text{Tip}, \text{MCP}) > 1.3 \times \text{distance\_3d}(\text{PIP}, \text{MCP})$$

* **Index:** MCP (5), PIP (6), Tip (8)
* **Middle:** MCP (9), PIP (10), Tip (12)
* **Ring:** MCP (13), PIP (14), Tip (16)
* **Pinky:** MCP (17), PIP (18), Tip (20)

```python
index_up = get_dist_3d(landmarks[8], landmarks[5]) > 1.3 * get_dist_3d(landmarks[6], landmarks[5])
middle_up = get_dist_3d(landmarks[12], landmarks[9]) > 1.3 * get_dist_3d(landmarks[10], landmarks[9])
ring_up = get_dist_3d(landmarks[16], landmarks[13]) > 1.3 * get_dist_3d(landmarks[14], landmarks[13])
pinky_up = get_dist_3d(landmarks[20], landmarks[17]) > 1.3 * get_dist_3d(landmarks[18], landmarks[17])
```

---

## 4. Sequence-Based Gesture Password

Instead of toggling instantly with 1 or 2 fingers, we use a sequence-based latch system to avoid casual hand movements accidentally toggling the state:

* **ON Sequence:** Show **3 fingers** then **4 fingers** within 5.0 seconds.
* **OFF Sequence:** Show **4 fingers** then **3 fingers** within 5.0 seconds.
* **Consecutive Frame Filtering:** The finger count must remain the same for at least 2 consecutive frames before registering as a stable state.
* **Timeout Cleanup:** If no hand is seen for 5.0 seconds, the sequence history and consecutive trackers are cleared.
