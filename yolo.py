import cv2
from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("yolo11n.pt")
model.train(data="coco8.yaml", epochs=100, imgsz=640)
model.export(format="imx", data="coco8.yaml")

for res in model("tcp://127.0.0.1:8888", stream=True):
    annotated_frame = res.plot()
    cv2.imshow("Camera", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
