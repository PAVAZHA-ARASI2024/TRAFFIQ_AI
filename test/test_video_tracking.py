
from ultralytics import YOLO

# Load the lightweight pretrained YOLO model
model = YOLO("yolo11n.pt")

# Vehicle class IDs in the COCO dataset
vehicle_classes = [2, 3, 5, 7]

# Process the first traffic video
results = model.track(
    source="dataset/videos/traffic_video_01.mp4",
    tracker="bytetrack.yaml",
    classes=vehicle_classes,
    conf=0.25,
    imgsz=640,
    vid_stride=3,
    save=True,
    show=False,
    project="runs/traffiq",
    name="video_01_tracking",
    exist_ok=True
)

print("Vehicle tracking completed!")
