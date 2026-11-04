
from ultralytics import YOLO
import os

# 1. Input traffic video
video_path = "dataset/videos/traffic_video_03.mp4"

# 2. Check whether the video exists
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")

# 3. Load pretrained lightweight YOLO model
model = YOLO("yolo11n.pt")

# 4. COCO vehicle class IDs
vehicle_classes = [2, 3, 5, 7]

print("Starting vehicle detection and tracking...")

# 5. Run YOLO + ByteTrack
results = model.track(
    source=video_path,
    tracker="bytetrack.yaml",
    classes=vehicle_classes,
    conf=0.25,
    imgsz=640,
    vid_stride=3,
    save=True,
    show=False,
    project="runs/traffiq",
    name="video_03_tracking",
    exist_ok=True
)

print("Tracking completed!")
print("Check the runs/traffiq folder for the output.")
