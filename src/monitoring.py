import time
from collections import deque


class FPSMeter:
    def __init__(self, window: int = 30):
        self.window = window
        self.times = deque(maxlen=window)
        self.last_time = None

    def tick(self):
        now = time.time()
        if self.last_time is not None:
            dt = now - self.last_time
            self.times.append(dt)
        self.last_time = now

    def fps(self) -> float:
        if not self.times:
            return 0.0
        avg_dt = sum(self.times) / len(self.times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0


def summarize_detections(detections):
    total = len(detections)
    attendants = sum(1 for d in detections if d["cls"] == 1)
    others = total - attendants
    return {
        "total": total,
        "attendants": attendants,
        "others": others,
    }
