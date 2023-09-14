import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
from django.views.decorators import gzip
import time
from signRecognition.Handmodule import HandDetect
from signRecognition.collect import (
    mediapipe_detection,
    draw_styled_landmarks,
    extract_keypoints,
)
import json
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.views.decorators.http import require_POST
from signRecognition.train import actions
from tensorflow.keras.models import load_model
import mediapipe as mp
from django.http import JsonResponse
import os
import requests


# ptime = 0

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


def processCam(request):
    detector = HandDetect()
    s = 0
    f = 0

    # Create a rectangle to limit hand gestures
    rectangle_size = (200, 200)  # Width and height of the rectangle
    rectangle_center = (
        220,
        140,
    )  # Initial position of the rectangle at the center of the frame

    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8
    mp_holistic = mp.solutions.holistic  # Holistic model

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            # Read feed
            ret, image = cap.read()
            # print("image:", image)
            # print("ret", ret)
            # ctime = time.time()
            # fps = int(1 / (ctime - ptime))
            # ptime = ctime  # frame par seconde

            image = cv2.flip(image, 1)
            hand, hand1, lmls1, lmls2 = detector.findHands(image)
            # print("image after hand detection", image)

            if hand in (
                [0, 1, 0, 0, 0],
                [1, 1, 0, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
            ):
                static_gesture_detected = True
                global gesture
                gesture = ""
            else:
                static_gesture_detected = False
            # print("hand:", hand)
            print("static_gesture_detected:", static_gesture_detected)

            if static_gesture_detected:
                if hand == [0, 1, 0, 0, 0]:
                    gesture = "Good"
                    cv2.putText(
                        image,
                        " Good ",
                        (100, 450),
                        cv2.FONT_HERSHEY_COMPLEX,
                        2,
                        (255, 255, 255),
                        1,
                    )
                    f = f + 1
                    # time.sleep(2.7)
                    print(f)
                elif hand == [1, 1, 1, 1, 1]:
                    gesture = "Hello"
                    cv2.putText(
                        image,
                        " Hello ",
                        (100, 450),
                        cv2.FONT_HERSHEY_COMPLEX,
                        2,
                        (255, 255, 255),
                        1,
                    )
                    f = f + 1

                elif hand == [1, 0, 0, 0, 0]:
                    gesture = "Help me"
                    cv2.putText(
                        image,
                        " Help me ",
                        (100, 450),
                        cv2.FONT_HERSHEY_COMPLEX,
                        2,
                        (255, 255, 255),
                        1,
                    )
                    f = f + 1

                elif hand == [1, 1, 0, 0, 1]:
                    gesture = "I love you"
                    cv2.putText(
                        image,
                        "i love you ",
                        (150, 450),
                        cv2.FONT_HERSHEY_COMPLEX,
                        2,
                        (255, 255, 255),
                        1,
                    )
                    f = f + 1

                elif hand == [0, 0, 1, 1, 1]:
                    gesture = "OK"
                    cv2.putText(
                        image,
                        " OK ",
                        (150, 450),
                        cv2.FONT_HERSHEY_COMPLEX,
                        2,
                        (255, 255, 255),
                        1,
                    )
                    f = f + 1

                cv2.imshow("Sign Language recognition", image)

            else:
                # Make detections
                image, results = mediapipe_detection(image, holistic)
                # print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                keypoints_detected = keypoints is not None

                if keypoints_detected:
                    sequence.append(keypoints)
                    sequence = sequence[-50:]
                    # print("Keypoints added to sequence:", keypoints)

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
                    gesture = ""
                    cv2.putText(
                        image,
                        "",
                        (100, 450),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                else:
                    gesture = sentence
                    cv2.putText(
                        image,
                        " ".join(sentence),
                        (100, 450),
                        cv2.FONT_HERSHEY_COMPLEX,
                        2,
                        (255, 255, 255),
                        1,
                    )

                # print("sequence:", sequence)
                # print("sentence:", sentence)
                # print("predictions:", predictions)

                # Show to screen
                cv2.imshow("Sign Language recognition", image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord("q"):
                request.session["gesture_to_send"] = gesture
                request.session.save()
                print("gesture:", gesture)
                print("geeeeeesteee", request.session["gesture_to_send"])
                send_gesture_to_url(request)
                break

    cap.release()
    cv2.destroyAllWindows()
    return gesture


@csrf_exempt
def execute_processCam(request):
    processCam(request)
    # # request.session["gesture_to_send"] = gesture
    # # request.session.save()
    # print("gesture:", gesture)
    # print("geeeeeesteee", request.session["gesture_to_send"])
    # send_gesture_to_url(request)
    return JsonResponse(
        {"message": "La fonction processCam a été exécutée avec succès."}
    )

@csrf_exempt
def send_gesture_to_url(request):
    print("fonction déclenchée ")
    gesture = request.session.get("gesture_to_send", "")
    print("gesture renvoyée", gesture)
    if gesture:
        return JsonResponse({"gesture": gesture})  # Return the gesture in the response
    else:
        return JsonResponse({"message": "No gesture available"})
# @csrf_exempt
# def send_gesture_to_url(request):
#     print("fonction déclenchée ")
#     gesture = request.session.get("gesture_to_send", "")
#     print("gesture renvoyée", gesture)
#     if gesture:
#         url = "http://localhost:8000/send_gesture/"
#         data = {"gesture": gesture}
#         print("data", data)
#         headers = {"Content-Type": "application/json"}
#         response = requests.post(url, data=json.dumps(data), headers=headers)
#         print("data2", data)
#         if response.status_code == 200:
#             return JsonResponse({"message": "Gesture sent successfully"})

#     return JsonResponse({"message": "Invalid request method."})

