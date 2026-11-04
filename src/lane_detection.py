
import cv2
import json
import os
import csv
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# ==========================================
# 1. FILE PATHS
# ==========================================

VIDEO_PATH = "dataset/videos/traffic_video_03.mp4"
ROI_PATH = "config/lane_rois.json"

OUTPUT_DIR = "runs/traffiq/lane_detection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "lane_analytics.mp4")
CSV_PATH = os.path.join(OUTPUT_DIR, "lane_analytics.csv")

# ==========================================
# 2. SETTINGS
# ==========================================

VEHICLE_CLASSES = [2, 3, 5, 7]

# Movement measured across recent frames.
HISTORY_LENGTH = 10
MOVEMENT_THRESHOLD = 12  # pixels

# Traffic intensity thresholds (initial settings).
LOW_TRAFFIC_MAX = 3
MEDIUM_TRAFFIC_MAX = 7

# ==========================================
# 3. LOAD LANE REGIONS
# ==========================================

with open(ROI_PATH, "r") as file:
    roi_data = json.load(file)

lane_polygons = {
    lane: np.array(roi_data[lane], dtype=np.int32)
    for lane in ["lane_1", "lane_2", "lane_3", "lane_4"]
}

crossing_polygon = np.array(
    roi_data["crossing_zone"], dtype=np.int32
)

colors = {
    "lane_1": (0, 255, 0),
    "lane_2": (255, 0, 0),
    "lane_3": (0, 165, 255),
    "lane_4": (255, 0, 255)
}

crossing_color = (0, 255, 255)

# ==========================================
# 4. LOAD MODEL AND VIDEO
# ==========================================

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError("Cannot open traffic video.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 10

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not out.isOpened():
    raise RuntimeError("Could not create output video.")

# ==========================================
# 5. TRACKING STATE
# ==========================================

# Store recent positions and waiting information
track_states = {}

# Current frame lane information
lane_current_ids = {
    lane: set() for lane in lane_polygons
}

crossing_current_ids = set()

# CSV records
csv_records = []

# ==========================================
# 6. HELPER FUNCTIONS
# ==========================================

def point_inside_polygon(point, polygon):

    return cv2.pointPolygonTest(
        polygon,
        (float(point[0]), float(point[1])),
        False
    ) >= 0


def get_track_status(track_id, point, frame_number):

    if track_id not in track_states:

        track_states[track_id] = {
            "positions": deque(maxlen=HISTORY_LENGTH),
            "stationary_start": None,
            "last_seen": frame_number,
            "status": "Checking"
        }

    state = track_states[track_id]

    state["positions"].append(point)
    state["last_seen"] = frame_number

    positions = state["positions"]

    if len(positions) < HISTORY_LENGTH:

        state["status"] = "Checking"

        return "Checking", 0.0

    first_x, first_y = positions[0]
    last_x, last_y = positions[-1]

    displacement = np.sqrt(
        (last_x - first_x) ** 2 +
        (last_y - first_y) ** 2
    )

    if displacement < MOVEMENT_THRESHOLD:

        state["status"] = "Stationary"

        if state["stationary_start"] is None:
            state["stationary_start"] = frame_number

        waiting_time = (
            frame_number - state["stationary_start"]
        ) / fps

    else:

        state["status"] = "Moving"
        state["stationary_start"] = None
        waiting_time = 0.0

    return state["status"], waiting_time


def get_traffic_intensity(vehicle_count):

    if vehicle_count <= LOW_TRAFFIC_MAX:
        return "LOW"

    elif vehicle_count <= MEDIUM_TRAFFIC_MAX:
        return "MEDIUM"

    else:
        return "HIGH"


# ==========================================
# 7. PROCESS VIDEO
# ==========================================

frame_number = 0

print("Starting TRAFFIQ AI lane analytics...")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # Reset current frame counts
    lane_current_ids = {
        lane: set() for lane in lane_polygons
    }

    crossing_current_ids = set()

    # Store current frame vehicle details
    lane_stationary_ids = {
        lane: set() for lane in lane_polygons
    }

    lane_waiting_times = {
        lane: [] for lane in lane_polygons
    }

    # ======================================
    # 8. YOLO + BYTETRACK
    # ======================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=VEHICLE_CLASSES,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    # ======================================
    # 9. DRAW LANE REGIONS
    # ======================================

    for lane, polygon in lane_polygons.items():

        cv2.polylines(
            frame, [polygon], True, colors[lane], 2
        )

        x, y = polygon[0]

        cv2.putText(
            frame,
            lane.upper(),
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colors[lane],
            2
        )

    cv2.polylines(
        frame,
        [crossing_polygon],
        True,
        crossing_color,
        3
    )

    # ======================================
    # 10. IDENTIFY CURRENT VEHICLES
    # ======================================

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = (
            result.boxes.id.int().cpu().tolist()
        )

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = map(int, box)

            # Bottom-centre point
            point = (
                (x1 + x2) // 2,
                y2
            )

            # Check crossing zone first
            if point_inside_polygon(
                point, crossing_polygon
            ):

                crossing_current_ids.add(track_id)

                status, waiting_time = get_track_status(
                    track_id, point, frame_number
                )

                cv2.rectangle(
                    frame,
                    (x1, y1), (x2, y2),
                    crossing_color, 2
                )

                cv2.putText(
                    frame,
                    f"ID {track_id} CROSSING {status}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    crossing_color,
                    2
                )

                continue

            # Check four approach lanes
            assigned_lane = None

            for lane, polygon in lane_polygons.items():

                if point_inside_polygon(point, polygon):

                    assigned_lane = lane
                    break

            if assigned_lane is None:
                continue

            lane_current_ids[assigned_lane].add(track_id)

            status, waiting_time = get_track_status(
                track_id, point, frame_number
            )

            if status == "Stationary":

                lane_stationary_ids[assigned_lane].add(
                    track_id
                )

                lane_waiting_times[assigned_lane].append(
                    waiting_time
                )

            # Draw tracked vehicle
            cv2.rectangle(
                frame,
                (x1, y1), (x2, y2),
                colors[assigned_lane], 2
            )

            cv2.putText(
                frame,
                f"ID {track_id} {status}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colors[assigned_lane],
                2
            )

    # ======================================
    # 11. CALCULATE LANE ANALYTICS
    # ======================================

    y_position = 30

    for lane in lane_polygons:

        current_count = len(lane_current_ids[lane])

        queue_count = len(lane_stationary_ids[lane])

        waiting_values = lane_waiting_times[lane]

        average_waiting = (
            sum(waiting_values) / len(waiting_values)
            if waiting_values else 0.0
        )

        intensity = get_traffic_intensity(current_count)

        # Display lane analytics
        display_text = (
            f"{lane.upper()} | Vehicles: {current_count} | "
            f"Queue: {queue_count} | "
            f"Wait: {average_waiting:.1f}s | {intensity}"
        )

        cv2.putText(
            frame,
            display_text,
            (15, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colors[lane],
            2
        )

        y_position += 28

        # Save measurements
        csv_records.append({
            "frame": frame_number,
            "video_time_seconds": round(
                frame_number / fps, 2
            ),
            "lane": lane,
            "current_vehicles": current_count,
            "queue_length": queue_count,
            "average_waiting_seconds": round(
                average_waiting, 2
            ),
            "traffic_intensity": intensity
        })

    cv2.putText(
        frame,
        f"Crossing Vehicles: {len(crossing_current_ids)}",
        (15, y_position + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        crossing_color,
        2
    )

    # ======================================
    # 12. SAVE AND DISPLAY
    # ======================================

    out.write(frame)

    cv2.imshow(
        "TRAFFIQ AI - Queue Analytics",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Processing stopped by user.")
        break

# ==========================================
# 13. SAVE CSV
# ==========================================

cap.release()
out.release()
cv2.destroyAllWindows()

if csv_records:

    with open(CSV_PATH, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_records[0].keys()
        )

        writer.writeheader()
        writer.writerows(csv_records)

print("\n========== TRAFFIQ AI RESULTS ==========")

print("Processed frames:", frame_number)
print("Output video:", OUTPUT_PATH)
print("Analytics CSV:", CSV_PATH)

print("\nLane analytics completed.")
