if __name__ == "__main__":
    import cv2
    import os
    from detector import ParkingAttendantDetector

    run_dir = os.getcwd()
    img_path = os.path.join(run_dir, "data", "sample.jpg")
    img = cv2.imread(img_path)

    detector = ParkingAttendantDetector()
    dets = detector.predict(img)
    print(dets)

    vis = detector.draw_detections(img, dets)
    cv2.imshow("detections", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
