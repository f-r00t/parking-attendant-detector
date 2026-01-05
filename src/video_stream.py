import cv2
from detector import ParkingAttendantDetector
from monitoring import FPSMeter, summarize_detections


def run_video_stream(
    source=0,
    display_window_name="Parking Attendant Detector",
    save_output_path=None,
):
    """
    source: 0 for default webcam, or a video file path, or RTSP/HTTP URL.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open video source {source}")
        return

    # Optional video writer for recording
    writer = None
    if save_output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_output_path, fourcc, fps, (width, height))

    detector = ParkingAttendantDetector()
    fps_meter = FPSMeter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or cannot read frame.")
            break

        fps_meter.tick()

        detections = detector.predict(frame)
        summary = summarize_detections(detections)
        vis = detector.draw_detections(frame, detections)

        # Overlay debug info
        fps_val = fps_meter.fps()
        debug_text = f"FPS: {fps_val:5.1f} | total: {summary['total']} | attendants: {summary['attendants']}"
        cv2.putText(vis, debug_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(display_window_name, vis)

        # Optional save
        if writer is not None:
            writer.write(vis)

        # Press ESC to quit
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example: webcam
    run_video_stream(source=0)
    # Example: RTSP
    # run_video_stream(source="rtsp://user:pass@ip:port/stream", save_output_path="output.mp4")
