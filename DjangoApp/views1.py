import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
import time
import json
import sys
import os
from django.views.decorators import gzip
from django.shortcuts import render
from django.http import JsonResponse
from signRecognition.Handmodule import HandDetect
from signRecognition.collect import (
    mediapipe_detection,
    draw_styled_landmarks,
    extract_keypoints,
)
from signRecognition.train import actions
from tensorflow.keras.models import load_model
import mediapipe as mp
from django.http import StreamingHttpResponse
import threading


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.video.set(3, 640)  # Set width
        self.video.set(4, 480)  # Set heigh
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode(".jpg", image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        progress_bar_width = int(prob * 100)
        cv2.rectangle(
            output_frame,
            (0, 60 + num * 40),
            (progress_bar_width, 90 + num * 40),
            colors[num],
            -1,
        )
        cv2.putText(
            output_frame,
            f"{actions[num]}: {prob:.2f}",
            (0, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output_frame


# 1. New detection variables
global result_text
result_text = ""


def process_frame(cam):
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    colors = [
        (245, 117, 16),
        (117, 245, 16),
        (16, 117, 245),
        (60, 50, 200),
        (20, 200, 10),
        (200, 20, 10),
    ]
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "action.h5")

    model = load_model(model_path)
    detector = HandDetect()
    s = 0
    f = 0
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8
    mp_holistic = mp.solutions.holistic  # Holistic model

    boundary = "frame"
    context = {}

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        ptime = 0
        while True:
            image = cam.frame
            print("image:", image)
            image = cv2.flip(image, 1)
            hand = detector.findHands(image)
            hand = hand[0]

            if hand in (
                [0, 1, 0, 0, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
            ):
                static_gesture_detected = True
            else:
                static_gesture_detected = False
            print("hand:", hand)
            print("static_gesture_detected:", static_gesture_detected)

            if static_gesture_detected:
                if hand == [0, 1, 0, 0, 0]:
                    frame_data = cam.get_frame()
                    frame_np = np.frombuffer(frame_data, dtype=np.uint8)

                    if frame_np is not None and isinstance(frame_np, np.ndarray):
                        print("good")
                        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                        cv2.putText(
                            frame,
                            " good ",
                            (100, 450),
                            cv2.FONT_HERSHEY_COMPLEX,
                            2,
                            (255, 255, 255),
                            1,
                        )

                        _, encoded_frame = cv2.imencode(".jpg", frame)
                        frame_data = encoded_frame.tobytes()

                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + frame_data
                            + b"\r\n\r\n"
                        )
                    else:
                        # Gérez le cas où frame n'est pas valide (par exemple, en affichant un message d'erreur)
                        print("Erreur : frame n'est pas valide.")
                elif hand == [1, 0, 0, 0, 0]:
                    frame_data = cam.get_frame()
                    frame_np = np.frombuffer(frame_data, dtype=np.uint8)

                    if frame_np is not None and isinstance(frame_np, np.ndarray):
                        print("Help me")
                        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                        cv2.putText(
                            frame,
                            " Help me ",
                            (100, 450),
                            cv2.FONT_HERSHEY_COMPLEX,
                            2,
                            (255, 255, 255),
                            1,
                        )

                        _, encoded_frame = cv2.imencode(".jpg", frame)
                        frame_data = encoded_frame.tobytes()

                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + frame_data
                            + b"\r\n\r\n"
                        )
                    else:
                        # Gérez le cas où frame n'est pas valide (par exemple, en affichant un message d'erreur)
                        print("Erreur : frame n'est pas valide.")

                elif hand == [1, 1, 1, 1, 1]:
                    frame_data = cam.get_frame()
                    frame_np = np.frombuffer(frame_data, dtype=np.uint8)

                    if frame_np is not None and isinstance(frame_np, np.ndarray):
                        print("hello")
                        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                        cv2.putText(
                            frame,
                            " Hello ",
                            (100, 450),
                            cv2.FONT_HERSHEY_COMPLEX,
                            2,
                            (255, 255, 255),
                            1,
                        )

                        _, encoded_frame = cv2.imencode(".jpg", frame)
                        frame_data = encoded_frame.tobytes()

                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + frame_data
                            + b"\r\n\r\n"
                        )
                    else:
                        # Gérez le cas où frame n'est pas valide (par exemple, en affichant un message d'erreur)
                        print("Erreur : frame n'est pas valide.")

                elif hand == [0, 0, 1, 1, 1]:
                    frame_data = cam.get_frame()
                    frame_np = np.frombuffer(frame_data, dtype=np.uint8)

                    if frame_np is not None and isinstance(frame_np, np.ndarray):
                        print("OK")
                        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                        cv2.putText(
                            frame,
                            "OK",
                            (100, 450),
                            cv2.FONT_HERSHEY_COMPLEX,
                            2,
                            (255, 255, 255),
                            1,
                        )

                        _, encoded_frame = cv2.imencode(".jpg", frame)
                        frame_data = encoded_frame.tobytes()

                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + frame_data
                            + b"\r\n\r\n"
                        )
                    else:
                        # Gérez le cas où frame n'est pas valide (par exemple, en affichant un message d'erreur)
                        print("Erreur : frame n'est pas valide.")
                elif hand == [1, 1, 0, 0, 1]:
                    frame_data = cam.get_frame()
                    frame_np = np.frombuffer(frame_data, dtype=np.uint8)

                    if frame_np is not None and isinstance(frame_np, np.ndarray):
                        print("I love you")
                        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                        cv2.putText(
                            frame,
                            "I love you ",
                            (100, 450),
                            cv2.FONT_HERSHEY_COMPLEX,
                            2,
                            (255, 255, 255),
                            1,
                        )

                        _, encoded_frame = cv2.imencode(".jpg", frame)
                        frame_data = encoded_frame.tobytes()

                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + frame_data
                            + b"\r\n\r\n"
                        )
                    else:
                        # Gérez le cas où frame n'est pas valide (par exemple, en affichant un message d'erreur)
                        print("Erreur : frame n'est pas valide.")

            else:
                # Make detections
                image, results = mediapipe_detection(image, holistic)
                print("results", results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                keypoints_detected = keypoints is not None

                if keypoints_detected:
                    sequence.append(keypoints)
                    sequence = sequence[-50:]
                    print("Keypoints added to sequence:", keypoints)

                if len(sequence) == 50:
                    if keypoints_detected:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        print(actions[np.argmax(res)])
                        predictions.append(np.argmax(res))
                    else:
                        # Set probabilities to zero if no keypoints are detected
                        res = np.zeros(len(actions))

                    # 3. Viz logic
                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])
                    else:
                        # If the model doesn't detect a significant gesture, clear the sentence
                        sentence.clear()
                        predictions.clear()
                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = prob_viz(res, actions, image, colors)
                else:
                    # Set probabilities to zero if no keypoints are detected
                    res = np.zeros(len(actions))

                if not results.left_hand_landmarks and not results.right_hand_landmarks:
                    frame_data = cam.get_frame()
                    frame_np = np.frombuffer(frame_data, dtype=np.uint8)

                    if frame_np is not None and isinstance(frame_np, np.ndarray):
                        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                        cv2.putText(
                            frame,
                            "",
                            (100, 450),
                            cv2.FONT_HERSHEY_COMPLEX,
                            2,
                            (255, 255, 255),
                            1,
                        )

                        _, encoded_frame = cv2.imencode(".jpg", frame)
                        frame_data = encoded_frame.tobytes()

                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + frame_data
                            + b"\r\n\r\n"
                        )
                    else:
                        # Gérez le cas où frame n'est pas valide (par exemple, en affichant un message d'erreur)
                        print("Erreur : frame n'est pas valide.")

                else:
                    frame_data = cam.get_frame()
                    frame_np = np.frombuffer(frame_data, dtype=np.uint8)

                    if frame_np is not None and isinstance(frame_np, np.ndarray):
                        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                        cv2.putText(
                            frame,
                            " ".join(sentence),
                            (100, 450),
                            cv2.FONT_HERSHEY_COMPLEX,
                            2,
                            (255, 255, 255),
                            1,
                        )

                        _, encoded_frame = cv2.imencode(".jpg", frame)
                        frame_data = encoded_frame.tobytes()

                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + frame_data
                            + b"\r\n\r\n"
                        )
                    else:
                        # Gérez le cas où frame n'est pas valide (par exemple, en affichant un message d'erreur)
                        print("Erreur : frame n'est pas valide.")
                    print("sequence:", sequence)
                    print("sentence:", sentence)
                    print("predictions:", predictions)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


@gzip.gzip_page
def gesture_recognition_stream(request):
    try:
        cam = VideoCamera()
        print("Gesture recognition stream called")
        context = {"result_text": result_text}
        return StreamingHttpResponse(
            process_frame(cam), content_type="multipart/x-mixed-replace; boundary=frame"
        )
    except:
        pass
    return render(request, "gesture_translator.html", context)
