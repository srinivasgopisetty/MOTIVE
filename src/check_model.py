from ultralytics import YOLO

model = YOLO("models/helmet.pt")

print("Number of classes:", len(model.names))
print("Classes:", model.names)