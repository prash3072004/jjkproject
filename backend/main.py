"""
FastAPI WebSocket backend for JJK gesture recognition.
Start with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import base64
import json
import os
import pickle
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="JJK Gesture Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "model" / "gesture_model.pkl"
LABELS_FILE = BASE_DIR / "model" / "labels.json"

# MediaPipe setup
mp_hands = mp.solutions.hands

# Load model + labels
clf = None
label_names = []

def load_model():
    global clf, label_names
    if MODEL_FILE.exists() and LABELS_FILE.exists():
        with open(MODEL_FILE, "rb") as f:
            clf = pickle.load(f)
        with open(LABELS_FILE, "r") as f:
            label_names = json.load(f)
        print(f"Model loaded. Labels: {label_names}")
        return True
    return False


load_model()


def extract_landmarks(hand_landmarks):
    """Normalize landmarks relative to wrist and flatten."""
    lm = hand_landmarks.landmark
    wrist = lm[0]
    data = []
    for point in lm:
        data.extend([point.x - wrist.x, point.y - wrist.y, point.z - wrist.z])
    return data


def decode_frame(b64_data: str):
    """Decode base64 JPEG frame from browser."""
    # Remove data URL prefix if present
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_data)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


@app.get("/")
async def root():
    return {"status": "ok", "model_loaded": clf is not None, "labels": label_names}


@app.get("/model-status")
async def model_status():
    return {
        "trained": clf is not None,
        "labels": label_names
    }


@app.websocket("/ws/detect")
async def detect_ws(websocket: WebSocket):
    """
    WebSocket endpoint for real-time gesture detection.
    Client sends: JSON { "frame": "<base64 jpeg>" }
    Server sends: JSON { "gesture": "gojo"|"sukuna"|"none", "confidence": 0.0-1.0, "landmarks": [...] }
    """
    await websocket.accept()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.5
    ) as hands:
        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                frame_b64 = msg.get("frame", "")

                if not frame_b64:
                    await websocket.send_text(json.dumps({"gesture": "none", "confidence": 0.0, "landmarks": []}))
                    continue

                frame = decode_frame(frame_b64)
                if frame is None:
                    await websocket.send_text(json.dumps({"gesture": "none", "confidence": 0.0, "landmarks": []}))
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                landmarks_out = []
                gesture = "none"
                confidence = 0.0

                if results.multi_hand_landmarks:
                    # Use first detected hand for classification
                    hl = results.multi_hand_landmarks[0]
                    lm_data = extract_landmarks(hl)

                    # Serialize landmark positions for drawing on frontend
                    for lm in hl.landmark:
                        landmarks_out.append({"x": lm.x, "y": lm.y})

                    if clf is not None:
                        vec = np.array(lm_data).reshape(1, -1)
                        proba = clf.predict_proba(vec)[0]
                        best_idx = int(np.argmax(proba))
                        confidence = float(proba[best_idx])

                        if confidence >= 0.75:
                            gesture = label_names[best_idx]

                await websocket.send_text(json.dumps({
                    "gesture": gesture,
                    "confidence": confidence,
                    "landmarks": landmarks_out
                }))

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"WS error: {e}")
            try:
                await websocket.close()
            except Exception:
                pass


@app.websocket("/ws/collect")
async def collect_ws(websocket: WebSocket):
    """
    WebSocket endpoint for in-browser gesture collection.
    Client sends: JSON { "frame": "<base64>", "gesture": "gojo"|"sukuna", "collecting": true|false }
    Server sends: JSON { "count": N, "landmarks": [...] }
    """
    await websocket.accept()

    import json as _json
    collected = {"gojo": [], "sukuna": []}

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:
        try:
            while True:
                raw = await websocket.receive_text()
                msg = _json.loads(raw)
                frame_b64 = msg.get("frame", "")
                gesture_label = msg.get("gesture", "")
                is_collecting = msg.get("collecting", False)
                action = msg.get("action", "")

                if action == "train":
                    # Train with collected data
                    # Save to data dir and train
                    data_dir = BASE_DIR / "data"
                    data_dir.mkdir(exist_ok=True)
                    with open(data_dir / "gestures.json", "w") as f:
                        _json.dump(collected, f)

                    # Train inline
                    try:
                        result = inline_train(collected)
                        load_model()
                        await websocket.send_text(_json.dumps({"status": "trained", "accuracy": result}))
                    except Exception as e:
                        await websocket.send_text(_json.dumps({"status": "error", "message": str(e)}))
                    continue

                if action == "reset":
                    collected = {"gojo": [], "sukuna": []}
                    await websocket.send_text(_json.dumps({"status": "reset"}))
                    continue

                if not frame_b64:
                    continue

                frame = decode_frame(frame_b64)
                if frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                landmarks_out = []
                if results.multi_hand_landmarks:
                    hl = results.multi_hand_landmarks[0]
                    lm_data = extract_landmarks(hl)
                    for lm in hl.landmark:
                        landmarks_out.append({"x": lm.x, "y": lm.y})

                    if is_collecting and gesture_label in collected:
                        collected[gesture_label].append(lm_data)

                counts = {k: len(v) for k, v in collected.items()}
                await websocket.send_text(_json.dumps({
                    "counts": counts,
                    "landmarks": landmarks_out,
                    "hand_detected": bool(results.multi_hand_landmarks)
                }))

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"Collect WS error: {e}")


def inline_train(data: dict):
    """Train model directly from in-memory gesture data."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X, y = [], []
    label_names_local = sorted(data.keys())

    for idx, name in enumerate(label_names_local):
        for sample in data[name]:
            X.append(sample)
            y.append(idx)

    X = np.array(X)
    y = np.array(y)

    if len(X) < 4:
        raise ValueError("Not enough samples to train. Collect more data.")

    # Simple split or just train on all if few samples
    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model_dir = BASE_DIR / "model"
    model_dir.mkdir(exist_ok=True)

    clf_local = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    clf_local.fit(X_train, y_train)

    acc = float((clf_local.predict(X_test) == y_test).mean())

    with open(model_dir / "gesture_model.pkl", "wb") as f:
        pickle.dump(clf_local, f)
    with open(model_dir / "labels.json", "w") as f:
        json.dump(label_names_local, f)

    return acc
