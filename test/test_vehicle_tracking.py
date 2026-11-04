
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Vehicle class IDs
vehicle_classes = [2, 3, 5, 7]

# Run tracking on sample video
results = model.track(
    source="YOUR_VIDEO_PATH.mp4",
    tracker="bytetrack.yaml",
    classes=vehicle_classes,
    conf=0.25,
    show=True,
    save=True
)

print("TRAFFIQ AI - VEHICLE TRACKING COMPLETED")
