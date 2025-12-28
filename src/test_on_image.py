if __name__ == "__main__":
    import cv2
    import os

    img_path = os.path.join(os.path.dirname(__file__), "..", "data", "images", "test", "sample.jpg")
    img = cv2.imread(img_path)

    detector = ParkingAttendantDetector()
    dets = detector.predict(img)
    print(dets)

    vis = detector.draw_detections(img, dets)
    cv2.imshow("detections", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
