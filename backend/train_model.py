"""
Model Training Script
Run after collect_gestures.py to train the gesture classifier.
Usage: python train_model.py
"""

import json
import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "gestures.json")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_FILE = os.path.join(MODEL_DIR, "gesture_model.pkl")
LABELS_FILE = os.path.join(MODEL_DIR, "labels.json")

os.makedirs(MODEL_DIR, exist_ok=True)


def train():
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: No data found at {DATA_FILE}")
        print("Please run collect_gestures.py first!")
        return False

    with open(DATA_FILE, "r") as f:
        raw = json.load(f)

    X = []
    y = []
    label_names = sorted(raw.keys())

    for label_idx, label_name in enumerate(label_names):
        samples = raw[label_name]
        print(f"  {label_name}: {len(samples)} samples")
        for sample in samples:
            X.append(sample)
            y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    print(f"\nTotal samples: {len(X)}")
    print(f"Labels: {label_names}")

    if len(X) < 10:
        print("WARNING: Very few samples. Consider re-collecting data.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(X) > 10 else None
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    accuracy = (y_pred == y_test).mean()
    print(f"Test accuracy: {accuracy*100:.1f}%")

    # Save model and labels
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)

    with open(LABELS_FILE, "w") as f:
        json.dump(label_names, f)

    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Labels saved to {LABELS_FILE}")
    return True


if __name__ == "__main__":
    print("Training gesture model...")
    success = train()
    if success:
        print("\nDone! Now start the server: uvicorn main:app --reload")
