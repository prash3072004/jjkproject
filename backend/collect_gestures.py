"""
Gesture Data Collection Script
Run this standalone to collect training data via your webcam.
Usage: python collect_gestures.py
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

GESTURES = {
    "gojo": "GOJO (Infinity / Domain Expansion pose)",
    "sukuna": "SUKUNA (Malevolent Shrine / claw pose)"
}

COLLECT_SECONDS = 4
FPS_TARGET = 20


def extract_landmarks(hand_landmarks):
    """Flatten 21 hand landmarks (x, y, z) into a 63-element vector."""
    lm = hand_landmarks.landmark
    data = []
    # Normalize relative to wrist
    wrist = lm[0]
    for point in lm:
        data.extend([point.x - wrist.x, point.y - wrist.y, point.z - wrist.z])
    return data


def collect():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    all_data = {}

    # Load existing data if any
    data_file = os.path.join(DATA_DIR, "gestures.json")
    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            all_data = json.load(f)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        for gesture_name, gesture_desc in GESTURES.items():
            samples = []
            print(f"\n{'='*60}")
            print(f"Collecting gesture: {gesture_name.upper()}")
            print(f"Description: {gesture_desc}")
            print(f"Get ready! Collecting for {COLLECT_SECONDS} seconds...")
            print("Press SPACE to start collection, Q to quit")
            print("="*60)

            # Wait for user to press space
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hl in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

                cv2.putText(frame, f"Gesture: {gesture_name.upper()}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, "Press SPACE to start", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("JJK Gesture Collector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Collect samples
            start = time.time()
            frame_delay = 1.0 / FPS_TARGET

            while time.time() - start < COLLECT_SECONDS:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                elapsed = time.time() - start
                remaining = COLLECT_SECONDS - elapsed
                progress = int((elapsed / COLLECT_SECONDS) * 100)

                color = (0, 255, 0)  # green while collecting
                cv2.putText(frame, f"COLLECTING: {gesture_name.upper()}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Time left: {remaining:.1f}s | Samples: {len(samples)}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Progress bar
                bar_w = int((progress / 100) * 600)
                cv2.rectangle(frame, (10, 440), (10 + bar_w, 465), color, -1)
                cv2.rectangle(frame, (10, 440), (610, 465), (100, 100, 100), 2)

                if results.multi_hand_landmarks:
                    for hl in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                        lm_data = extract_landmarks(hl)
                        samples.append(lm_data)

                cv2.imshow("JJK Gesture Collector", frame)
                cv2.waitKey(1)
                time.sleep(frame_delay)

            print(f"Collected {len(samples)} samples for '{gesture_name}'")
            all_data[gesture_name] = samples

    cap.release()
    cv2.destroyAllWindows()

    # Save data
    with open(data_file, "w") as f:
        json.dump(all_data, f)

    print(f"\nData saved to {data_file}")
    print("Samples per gesture:")
    for k, v in all_data.items():
        print(f"  {k}: {len(v)} samples")

    return data_file


if __name__ == "__main__":
    collect()
    print("\nDone! Now run: python train_model.py")
