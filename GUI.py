import sys, os
import cv2
import dlib
from scipy.spatial import distance
from logidrivepy import LogitechController
import time
from tensorflow.keras.models import load_model
import threading
import pygame  # مكتبة تشغيل الصوت

# تهيئة pygame لتشغيل الصوت
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")

# Load the drowsiness detection model
model = load_model('drowsiness_detection_model.h5')

# دالة حساب EAR
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

# دالة كشف EAR من الإطار
def detect_EAR(frame, face_detector, landmark):
    faces = face_detector(frame)
    if len(faces) == 0:
        return 0
    face = faces[0]
    face_landmarks = landmark(frame, face)
    leftEye = []
    rightEye = []

    for n in range(36, 42):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        leftEye.append((x, y))
        next_point = n + 1 if n != 41 else 36
        x2 = face_landmarks.part(next_point).x
        y2 = face_landmarks.part(next_point).y
        cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

    for n in range(42, 48):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        rightEye.append((x, y))
        next_point = n + 1 if n != 47 else 42
        x2 = face_landmarks.part(next_point).x
        y2 = face_landmarks.part(next_point).y
        cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

    left_ear = calculate_EAR(leftEye)
    right_ear = calculate_EAR(rightEye)
    EAR = round((left_ear + right_ear) / 2, 2)
    return EAR

# دالة التحكم في اهتزاز العجلة
def shake_wheel(drowsy_event, stop_event):
    controller = LogitechController()
    controller.steering_initialize()
    try:
        while not stop_event.is_set():
            if drowsy_event.is_set():
                controller.LogiPlayDirtRoadEffect(0, 50)
            else:
                controller.LogiPlayDirtRoadEffect(0, 0)
            controller.logi_update()
            time.sleep(0.1)
    finally:
        controller.steering_shutdown()

# البرنامج الرئيسي
def main():
    cap = cv2.VideoCapture(0)
    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    stop_event = threading.Event()
    drowsy_event = threading.Event()
    controller_thread = threading.Thread(target=shake_wheel, args=(drowsy_event, stop_event))
    controller_thread.start()

    window_closed = False
    alert_playing = False

    try:
        while True:
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            EAR = detect_EAR(gray, hog_face_detector, dlib_facelandmark)

            if EAR < 0.26:
                drowsy_event.set()
                if not alert_playing:
                    alert_sound.play(-1)  # تشغيل مستمر
                    alert_playing = True
                cv2.putText(frame, "DROWSY", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                print("Drowsy")
            else:
                drowsy_event.clear()
                if alert_playing:
                    alert_sound.stop()
                    alert_playing = False

            print(EAR)

            if not window_closed:
                cv2.imshow("Are you Sleepy", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break
            if not window_closed and cv2.getWindowProperty("Are you Sleepy", cv2.WND_PROP_VISIBLE) < 1:
                window_closed = True
                break
    finally:
        stop_event.set()
        controller_thread.join()
        cap.release()
        pygame.mixer.quit()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
