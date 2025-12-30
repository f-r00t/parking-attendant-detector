import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov11n_attendant.pt")

CLASS_NAMES = ["parking_attendant", "person"]
