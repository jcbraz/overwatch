# from picamera2.devices.imx500 import IMX500
from ultralytics import YOLO


# imx500 = IMX500(imx_model)

def main():
    model = YOLO("yolo11n.pt")
    model.export(format="imx")

    print("Hello from overwatch!")


if __name__ == "__main__":
    main()
