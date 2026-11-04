
import cv2
import json
import os
import numpy as np

# ----------------------------------
# 1. FILE PATHS
# ----------------------------------

# Use the same original video used to create your lane coordinates
VIDEO_PATH = "dataset/videos/traffic_video_03.mp4"

OUTPUT_PATH = "config/lane_rois.json"

os.makedirs("config", exist_ok=True)

# ----------------------------------
# 2. OPEN VIDEO
# ----------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open video: {VIDEO_PATH}"
    )

# Read a frame at 30 seconds
cap.set(cv2.CAP_PROP_POS_MSEC, 30000)

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read a frame from the video.")

# ----------------------------------
# 3. DEFINE FIVE REGIONS
# ----------------------------------

lanes = {
    "1": [],
    "2": [],
    "3": [],
    "4": [],
    "5": []
}

current_lane = "1"

# Colours for each region (BGR format)
colors = {
    "1": (0, 255, 0),      # Green
    "2": (255, 0, 0),      # Blue
    "3": (0, 165, 255),    # Orange
    "4": (255, 0, 255),    # Pink
    "5": (0, 255, 255)     # Yellow - Crossing Zone
}

# ----------------------------------
# 4. DISPLAY LANE REGIONS
# ----------------------------------

def draw_display():

    display = frame.copy()

    for lane, points in lanes.items():

        color = colors[lane]

        # Draw connecting lines
        if len(points) >= 2:

            for i in range(len(points) - 1):

                cv2.line(
                    display,
                    points[i],
                    points[i + 1],
                    color,
                    2
                )

        # Fill polygon with transparent colour
        if len(points) >= 3:

            overlay = display.copy()

            polygon = np.array(points, dtype=np.int32)

            cv2.fillPoly(
                overlay,
                [polygon],
                color
            )

            display = cv2.addWeighted(
                overlay,
                0.18,
                display,
                0.82,
                0
            )

            cv2.polylines(
                display,
                [polygon],
                True,
                color,
                2
            )

            # Region label position
            center_x = sum(p[0] for p in points) // len(points)
            center_y = sum(p[1] for p in points) // len(points)

            label = (
                "CROSSING ZONE"
                if lane == "5"
                else f"LANE {lane}"
            )

            cv2.putText(
                display,
                label,
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # Draw individual points
        for point in points:

            cv2.circle(
                display,
                point,
                5,
                color,
                -1
            )

    # Current editing information
    current_label = (
        "CROSSING ZONE"
        if current_lane == "5"
        else f"LANE {current_lane}"
    )

    cv2.putText(
        display,
        f"Editing: {current_label}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        colors[current_lane],
        2
    )

    cv2.imshow(
        "TRAFFIQ AI - Lane ROI Marking",
        display
    )

# ----------------------------------
# 5. MOUSE CLICK FUNCTION
# ----------------------------------

def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        lanes[current_lane].append((x, y))

        draw_display()

# ----------------------------------
# 6. KEYBOARD CONTROLS
# ----------------------------------

cv2.namedWindow("TRAFFIQ AI - Lane ROI Marking")

cv2.setMouseCallback(
    "TRAFFIQ AI - Lane ROI Marking",
    mouse_callback
)

print("""
========================================
       TRAFFIQ AI - REGION MARKING
========================================

SELECT REGION:

1 : Lane 1
2 : Lane 2
3 : Lane 3
4 : Lane 4
5 : Crossing Zone

MOUSE:

LEFT CLICK : Add polygon point

KEYBOARD:

U : Undo last point
C : Clear selected region
S : Save all five regions
Q : Quit without saving

Draw at least 3 points for each region.

========================================
""")

draw_display()

# ----------------------------------
# 7. MAIN CONTROL LOOP
# ----------------------------------

while True:

    key = cv2.waitKey(0) & 0xFF

    # Select one of the five regions
    if key in [
        ord("1"),
        ord("2"),
        ord("3"),
        ord("4"),
        ord("5")
    ]:

        current_lane = chr(key)

        draw_display()

    # Undo last point
    elif key == ord("u"):

        if lanes[current_lane]:

            lanes[current_lane].pop()

            draw_display()

    # Clear selected region
    elif key == ord("c"):

        lanes[current_lane] = []

        draw_display()

        print(f"Cleared region {current_lane}")

    # Save all five regions
    elif key == ord("s"):

        if all(
            len(points) >= 3
            for points in lanes.values()
        ):

            data = {}

            for lane, points in lanes.items():

                if lane == "5":
                    region_name = "crossing_zone"
                else:
                    region_name = f"lane_{lane}"

                data[region_name] = [
                    list(point)
                    for point in points
                ]

            with open(OUTPUT_PATH, "w") as file:

                json.dump(
                    data,
                    file,
                    indent=4
                )

            print("\nAll five regions saved successfully!")

            print("Output:", OUTPUT_PATH)

            break

        else:

            print("\nSome regions are incomplete.")

            for lane, points in lanes.items():

                if len(points) < 3:

                    if lane == "5":
                        print("Crossing Zone needs more points.")
                    else:
                        print(f"Lane {lane} needs more points.")

    # Quit
    elif key == ord("q") or key == 27:

        print("Exited without saving.")

        break

# ----------------------------------
# 8. CLOSE WINDOW
# ----------------------------------

cv2.destroyAllWindows()
