"""
Live Pain Detection — Raspberry Pi + PiCamera2 Edition
=======================================================
Uses TFLite for lightweight inference and picamera2 for camera capture.
Runs on Raspberry Pi 3B+/4/5 with a PiCamera module (v2 or v3).

Usage:
    python live_pain_detection_pi.py [--headless]

    --headless : run without a display (prints predictions to terminal only)
"""

import argparse
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# PiCamera2 import — falls back to OpenCV capture if unavailable
# ---------------------------------------------------------------------------
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("[WARN] picamera2 not found — falling back to OpenCV VideoCapture.")

# ---------------------------------------------------------------------------
# TFLite Runtime — try the standalone package first, then tensorflow
# ---------------------------------------------------------------------------
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# --- CONFIG -----------------------------------------------------------------
MODEL_PATH   = 'models/pain_classifier_mobilenetv2.tflite'
IMG_SIZE      = (224, 224)
CLASS_NAMES   = ['mild', 'moderate', 'none', 'severe']  # must match training order
CASCADE_PATH  = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
CAPTURE_SIZE  = (640, 480)   # camera resolution
FPS_TARGET    = 15           # target frame rate on Pi
# ----------------------------------------------------------------------------


def load_tflite_model(model_path: str) -> tuple:
    """Load TFLite model and return (interpreter, input_details, output_details)."""
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def classify_face(interpreter, input_details, output_details, face_img: np.ndarray) -> tuple:
    """Run TFLite inference on a preprocessed face image.
    
    Returns (class_name, confidence).
    """
    face_input = cv2.resize(face_img, IMG_SIZE)
    face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
    face_input = face_input.astype(np.float32) / 255.0
    face_input = np.expand_dims(face_input, axis=0)

    interpreter.set_tensor(input_details[0]['index'], face_input)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    class_idx  = np.argmax(preds)
    class_name = CLASS_NAMES[class_idx]
    confidence = float(preds[class_idx])
    return class_name, confidence


def create_camera():
    """Create and start a camera capture source (PiCamera2 or OpenCV fallback)."""
    if PICAMERA_AVAILABLE:
        picam = Picamera2()
        config = picam.create_preview_configuration(
            main={"size": CAPTURE_SIZE, "format": "RGB888"}
        )
        picam.configure(config)
        picam.start()
        time.sleep(1)  # let the camera warm up
        return picam, 'picamera2'
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
        if not cap.isOpened():
            print("[ERROR] Could not open camera.")
            sys.exit(1)
        return cap, 'opencv'


def grab_frame(camera, backend: str) -> np.ndarray:
    """Grab a single BGR frame from the camera."""
    if backend == 'picamera2':
        # picamera2 returns RGB; convert to BGR for OpenCV drawing
        rgb_frame = camera.capture_array()
        return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = camera.read()
        if not ret:
            return None
        return frame


def release_camera(camera, backend: str):
    """Cleanly release the camera."""
    if backend == 'picamera2':
        camera.stop()
    else:
        camera.release()


def main():
    parser = argparse.ArgumentParser(description="Pain Detection on Raspberry Pi")
    parser.add_argument('--headless', action='store_true',
                        help='Run without display (terminal output only)')
    args = parser.parse_args()

    # --- Load model ---
    print(f"[INFO] Loading TFLite model: {MODEL_PATH}")
    interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)

    # --- Load face cascade ---
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print("[ERROR] Could not load Haar cascade for face detection.")
        sys.exit(1)

    # --- Start camera ---
    camera, backend = create_camera()
    print(f"[INFO] Camera backend: {backend}")
    print("[INFO] Running pain detection… Press 'q' to quit (or Ctrl+C in headless mode).")

    frame_delay = 1.0 / FPS_TARGET
    try:
        while True:
            t_start = time.time()

            frame = grab_frame(camera, backend)
            if frame is None:
                print("[WARN] Failed to grab frame — skipping.")
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            if len(faces) > 0:
                # Use the largest detected face
                (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
                face_img = frame[y:y+h, x:x+w]

                class_name, confidence = classify_face(
                    interpreter, input_details, output_details, face_img
                )

                label = f"{class_name} ({confidence*100:.1f}%)"
                print(f"  >> {label}")

                if not args.headless:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                if not args.headless:
                    cv2.putText(frame, "No face detected", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            if not args.headless:
                cv2.imshow('Pain Detection — Pi (press q to quit)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Throttle to target FPS
            elapsed = time.time() - t_start
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    # --- Cleanup ---
    release_camera(camera, backend)
    if not args.headless:
        cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == '__main__':
    main()
