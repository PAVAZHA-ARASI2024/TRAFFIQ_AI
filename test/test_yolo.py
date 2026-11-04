
from ultralytics import YOLO

# Load a lightweight pretrained YOLO model
model = YOLO("yolo11n.pt")

# Test detection using a sample image
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,
    conf=0.25
)

print("TRAFFIQ AI - YOLO TEST")
print("Vehicle detection test completed!")

for result in results:
    print("Detected objects:", len(result.boxes))
    print("Class IDs:", result.boxes.cls.tolist())
