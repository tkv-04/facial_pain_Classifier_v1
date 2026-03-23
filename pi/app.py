"""
Pain Detection Web Dashboard — Flask Backend
=============================================
Patient-based session management with JSON persistence.

Flow:
  1. User enters patient name on the dashboard
  2. Clicks "Start Session" → camera begins streaming + classifying
  3. Clicks "Stop Session"  → session summary saved to patients.json
  4. "Patient History" tab shows all past sessions

Usage:
    python app.py              # auto-detect camera
    python app.py --camera pi  # force PiCamera
    python app.py --camera sys # force system webcam
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request

# ---------------------------------------------------------------------------
# TFLite Runtime
# ---------------------------------------------------------------------------
try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_BACKEND = 'tflite_runtime'
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        TFLITE_BACKEND = 'tensorflow'
    except ImportError:
        print("[ERROR] Neither tflite-runtime nor tensorflow is installed.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# PiCamera2
# ---------------------------------------------------------------------------
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

# --- CONFIG -----------------------------------------------------------------
MODEL_PATH    = os.path.join(os.path.dirname(__file__), '..', 'models',
                             'pain_classifier_mobilenetv2.tflite')
IMG_SIZE      = (224, 224)
CLASS_NAMES   = ['mild', 'moderate', 'none', 'severe']
CASCADE_PATH  = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
CAPTURE_SIZE  = (640, 480)
FPS_TARGET    = 15
HISTORY_LEN   = 200
PATIENTS_FILE = os.path.join(os.path.dirname(__file__), 'patients.json')
# ----------------------------------------------------------------------------

app = Flask(__name__)

# --- Shared state ---
lock = threading.Lock()
sse_queues = []          # list of Queue objects for SSE listeners
output_frame = None
frame_lock = threading.Lock()

# --- Session state ---
session_active = False
session_data = {
    'patient_name': '',
    'session_id': '',
    'start_time': None,
    'predictions': [],    # list of {class, confidence, all_probs, timestamp}
}


# ===========================================================================
#  Patient JSON persistence
# ===========================================================================
def load_patients():
    if os.path.exists(PATIENTS_FILE):
        with open(PATIENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_patients(patients):
    with open(PATIENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(patients, f, indent=2, ensure_ascii=False)


def save_session():
    """Save the current session to patients.json."""
    global session_data
    if not session_data['patient_name']:
        return

    preds = session_data['predictions']
    # Compute summary stats
    class_counts = {}
    total_conf = 0
    for p in preds:
        c = p['class']
        class_counts[c] = class_counts.get(c, 0) + 1
        total_conf += p['confidence']

    dominant = max(class_counts, key=class_counts.get) if class_counts else 'none'
    avg_conf = round(total_conf / len(preds), 1) if preds else 0

    record = {
        'session_id': session_data['session_id'],
        'patient_name': session_data['patient_name'],
        'start_time': session_data['start_time'],
        'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_detections': len(preds),
        'class_counts': class_counts,
        'dominant_class': dominant,
        'avg_confidence': avg_conf,
        'predictions': preds,      # full detail for playback
    }

    patients = load_patients()
    patients.append(record)
    save_patients(patients)
    print(f"[INFO] Saved session for patient '{session_data['patient_name']}' "
          f"({len(preds)} detections)")


# ===========================================================================
#  Camera abstraction
# ===========================================================================
def detect_platform():
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return 'raspberry pi' in f.read().lower()
    except (FileNotFoundError, PermissionError):
        return False


def create_camera(mode: str):
    use_picam = False
    if mode == 'pi':
        if not PICAMERA_AVAILABLE:
            print("[ERROR] --camera pi specified but picamera2 is not installed.")
            sys.exit(1)
        use_picam = True
    elif mode == 'sys':
        use_picam = False
    else:
        use_picam = PICAMERA_AVAILABLE and detect_platform()

    if use_picam:
        print("[INFO] Using PiCamera2")
        picam = Picamera2()
        config = picam.create_preview_configuration(
            main={"size": CAPTURE_SIZE, "format": "RGB888"})
        picam.configure(config)
        picam.start()
        time.sleep(1)
        return picam, 'picamera2'
    else:
        print("[INFO] Using system webcam (OpenCV)")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
        if not cap.isOpened():
            print("[ERROR] Could not open system camera.")
            sys.exit(1)
        return cap, 'opencv'


def grab_frame(camera, backend):
    if backend == 'picamera2':
        return cv2.cvtColor(camera.capture_array(), cv2.COLOR_RGB2BGR)
    else:
        ret, frame = camera.read()
        return frame if ret else None


# ===========================================================================
#  TFLite inference
# ===========================================================================
def load_model():
    print(f"[INFO] Loading TFLite model ({TFLITE_BACKEND}): {MODEL_PATH}")
    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


def classify_face(interp, inp, out, face_img):
    img = cv2.resize(face_img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interp.set_tensor(inp[0]['index'], img)
    interp.invoke()
    preds = interp.get_tensor(out[0]['index'])[0]
    idx = int(np.argmax(preds))
    all_probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    return CLASS_NAMES[idx], float(preds[idx]), all_probs


# ===========================================================================
#  Background processing thread
# ===========================================================================
COLOR_MAP = {
    'none': (0, 200, 100), 'mild': (0, 220, 255),
    'moderate': (0, 140, 255), 'severe': (0, 50, 255),
}


def push_sse(data):
    dead = []
    for q in sse_queues:
        try:
            q.put_nowait(data)
        except queue.Full:
            pass
    # cleanup dead queues
    for q in dead:
        sse_queues.remove(q)


def processing_loop(camera, backend, interp, inp, out):
    global output_frame, session_data

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    frame_delay = 1.0 / FPS_TARGET

    while True:
        t0 = time.time()
        frame = grab_frame(camera, backend)
        if frame is None:
            time.sleep(0.05)
            continue

        active = session_active  # snapshot

        if active:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
                face_img = frame[y:y+h, x:x+w]
                cls, conf, all_probs = classify_face(interp, inp, out, face_img)

                now = datetime.now().strftime('%H:%M:%S')
                pred = {
                    'class': cls,
                    'confidence': round(conf * 100, 1),
                    'all_probs': {k: round(v * 100, 1) for k, v in all_probs.items()},
                    'timestamp': now,
                    'face_detected': True,
                    'session_active': True,
                }

                # Save to session
                with lock:
                    session_data['predictions'].append({
                        'class': cls,
                        'confidence': round(conf * 100, 1),
                        'all_probs': pred['all_probs'],
                        'timestamp': now,
                    })

                color = COLOR_MAP.get(cls, (0, 255, 0))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                label = f"{cls} ({conf*100:.1f}%)"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                push_sse(pred)
            else:
                now = datetime.now().strftime('%H:%M:%S')
                pred = {
                    'class': 'no_face', 'confidence': 0,
                    'all_probs': {}, 'timestamp': now,
                    'face_detected': False, 'session_active': True,
                }
                cv2.putText(frame, "No face detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                push_sse(pred)
        else:
            # Session not active — show idle message
            cv2.putText(frame, "Session not active", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            push_sse({'session_active': False, 'idle': True})

        # Encode frame for MJPEG
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            output_frame = jpeg.tobytes()

        elapsed = time.time() - t0
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)


# ===========================================================================
#  Flask routes
# ===========================================================================
def generate_mjpeg():
    while True:
        with frame_lock:
            fb = output_frame
        if fb is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + fb + b'\r\n')
        time.sleep(1.0 / FPS_TARGET)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/events')
def events():
    q = queue.Queue(maxsize=50)
    sse_queues.append(q)

    def stream():
        try:
            while True:
                try:
                    data = q.get(timeout=5)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'keepalive': True})}\n\n"
        finally:
            if q in sse_queues:
                sse_queues.remove(q)

    return Response(stream(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# --- Session control ---
@app.route('/api/session/start', methods=['POST'])
def start_session():
    global session_active, session_data
    body = request.get_json(force=True)
    name = body.get('patient_name', '').strip()
    if not name:
        return jsonify({'error': 'Patient name is required'}), 400

    with lock:
        session_active = True
        session_data = {
            'patient_name': name,
            'session_id': str(uuid.uuid4())[:8],
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predictions': [],
        }
    print(f"[INFO] Session started for patient: {name}")
    return jsonify({'status': 'started', 'patient_name': name,
                    'session_id': session_data['session_id']})


@app.route('/api/session/stop', methods=['POST'])
def stop_session():
    global session_active
    with lock:
        session_active = False
        save_session()
        sid = session_data['session_id']
        n = len(session_data['predictions'])
    print(f"[INFO] Session stopped. {n} detections saved.")
    return jsonify({'status': 'stopped', 'session_id': sid, 'detections_saved': n})


@app.route('/api/session/status')
def session_status():
    with lock:
        return jsonify({
            'active': session_active,
            'patient_name': session_data['patient_name'],
            'session_id': session_data['session_id'],
            'detections': len(session_data['predictions']),
        })


# --- Patient history ---
@app.route('/api/patients')
def get_patients():
    patients = load_patients()
    # Return summary (without full predictions array for listing)
    summaries = []
    for p in patients:
        summaries.append({
            'session_id': p['session_id'],
            'patient_name': p['patient_name'],
            'start_time': p['start_time'],
            'end_time': p['end_time'],
            'total_detections': p['total_detections'],
            'class_counts': p['class_counts'],
            'dominant_class': p['dominant_class'],
            'avg_confidence': p['avg_confidence'],
        })
    return jsonify(summaries)


@app.route('/api/patients/<session_id>')
def get_patient_detail(session_id):
    patients = load_patients()
    for p in patients:
        if p['session_id'] == session_id:
            return jsonify(p)
    return jsonify({'error': 'Session not found'}), 404


@app.route('/api/patients/<session_id>', methods=['DELETE'])
def delete_patient(session_id):
    patients = load_patients()
    patients = [p for p in patients if p['session_id'] != session_id]
    save_patients(patients)
    return jsonify({'status': 'deleted'})


# ===========================================================================
#  Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Pain Detection Web Dashboard")
    parser.add_argument('--camera', choices=['auto', 'pi', 'sys'], default='auto',
                        help="'auto' (detect platform), 'pi' (PiCamera), 'sys' (system webcam)")
    parser.add_argument('--port', type=int, default=5000, help="Web server port")
    args = parser.parse_args()

    interp, inp, out = load_model()
    camera, backend = create_camera(args.camera)
    print(f"[INFO] Camera backend: {backend}")

    t = threading.Thread(target=processing_loop,
                         args=(camera, backend, interp, inp, out),
                         daemon=True)
    t.start()

    print(f"[INFO] Dashboard at http://0.0.0.0:{args.port}")
    app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)


if __name__ == '__main__':
    main()
