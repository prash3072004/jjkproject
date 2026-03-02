# 🌀 JJK Domain Expansion — Gesture Recognition

A real-time hand gesture recognition app themed around **Jujutsu Kaisen**. Show the webcam your custom domain expansion pose and watch the reveal unfold — full-screen character art and sound effect included.

---

## 📁 Project Structure

```
jjk/
├── backend/
│   ├── main.py              # FastAPI server (WebSocket detection + collection)
│   ├── collect_gestures.py  # Standalone CLI gesture collector (optional)
│   ├── train_model.py       # Standalone CLI model trainer (optional)
│   └── requirements.txt
├── frontend/
│   ├── index.html           # Full single-page web app (no build step needed)
│   ├── gojo.png
│   ├── sukuna.jpg
│   ├── gojo sound.mp3
│   └── sukuna sound.m4a
├── start_backend.bat        # ← Start the Python server
├── open_frontend.bat        # ← Open the app in browser
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Google Chrome or Microsoft Edge

### Step 1 — Start the backend
Double-click **`start_backend.bat`**  
*(On first run, this installs all Python dependencies automatically.)*

Or manually:
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 2 — Open the app
Double-click **`open_frontend.bat`**, or open `frontend/index.html` directly in Chrome/Edge.

> **Tip:** If the browser blocks camera access on a local file, run a mini server instead:
> ```bash
> cd frontend
> python -m http.server 3000
> ```
> Then visit `http://localhost:3000`

### Step 3 — Train your gestures
1. Click the **✋ Train Gestures** tab
2. Hold your **Gojo pose** (e.g. open palm / Infinity pose) → click **Record Gojo** → hold still for 3 seconds
3. Hold your **Sukuna pose** (e.g. claw / fingers curled) → click **Record Sukuna** → hold still for 3 seconds
4. Click **⚡ Train Model** — takes ~2 seconds

### Step 4 — Detect!
Switch to the **⚡ Detect** tab, strike your pose, and hold it steady for ~1 second.  
**Domain Expansion activates.** 🔥

---

## ⚙️ How It Works

```
Webcam frame (every 80ms)
  → base64 JPEG via WebSocket
    → MediaPipe extracts 21 hand landmarks
      → RandomForest classifier (200 trees)
        → Hold pose ≥1 sec at ≥75% confidence
          → Full-screen Domain Expansion reveal (image + sound)
```

| Component | Technology |
|---|---|
| Backend server | Python, FastAPI, uvicorn |
| Hand tracking | MediaPipe Hands |
| Gesture model | scikit-learn RandomForestClassifier |
| Frontend | HTML5, Vanilla JS, WebSockets |
| Camera | WebRTC `getUserMedia` |

---

## 🔄 Re-training Tips

- **More samples = better accuracy** — re-record any gesture and retrain anytime
- The trained model is saved to `backend/model/gesture_model.pkl` and persists across sessions (no re-training needed on next launch)
- Use **↺ Reset Data** in the Train tab to start collection from scratch

---

## 📦 Python Dependencies

```
fastapi
uvicorn[standard]
mediapipe
opencv-python
scikit-learn
numpy
python-multipart
websockets
Pillow
```
