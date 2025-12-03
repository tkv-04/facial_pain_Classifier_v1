# Pain Classification (Face-based)

This repository contains scripts and models for classifying facial images by pain severity (none / mild / moderate / severe). The main, user-facing script is `live_pain_detection.py` which runs a live webcam-based classifier using a saved Keras model.

**Quick overview**
- **Main script:** `live_pain_detection.py` — runs webcam capture, detects faces with OpenCV Haar cascade, and classifies the face using a TensorFlow Keras model.
- **Training scripts:** `train.py` and `train_pain_classifier.py` — for training and fine-tuning MobileNetV2-based classifiers.
- **Utilities:** `prepare_dataset.py`, `prepare_for_gemini.py`, `organize_by_gemini.py`, `count_images_per_class.py` — dataset preparation and classification helpers.
- **Pre-trained models included:** `pain_classifier_mobilenetv2.h5`, `pain_classifier_mobilenetv2.tflite`, `pain_model_imbalanced_corrected.h5`, `pain_model.tflite`, and fine-tuned variants.

**Requirements**
- See `requirements.txt` (pinned to versions found in the project's `myvenv`).

**Quickstart (Windows PowerShell)**
1. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r .\requirements.txt
```

3. Run the live classifier (webcam required):

```powershell
python .\live_pain_detection.py
```

Press `q` in the display window to quit.

**Files of interest**
- `live_pain_detection.py`: live webcam demo; loads `pain_classifier_mobilenetv2.h5` by default.
- `train_pain_classifier.py` and `train.py`: training pipelines that use `tensorflow.keras` and MobileNetV2 as backbone.
- `organize_by_gemini.py`: example script that calls a local model API (ollama-like) to label images in bulk.
- `prepare_for_gemini.py`: moves images into label folders based on a CSV produced by automated classification.

**Data layout**
- Expect dataset folders like `ollama_cleaned_dataset/` with subfolders for each class (`none`, `mild`, `moderate`, `severe`).

**Notes & troubleshooting**
- Webcam not found: ensure camera drivers are installed and no other app is using the camera. On some Windows systems you may need to allow camera access in Settings.
- OpenCV: the package `opencv-python` is used for capture and Haar cascade face detection. The cascade path used is `cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'`.
- TensorFlow GPU: `requirements.txt` pins `tensorflow==2.19.0` — if you need GPU acceleration, ensure your CUDA/cuDNN versions match TensorFlow's requirements.
- If you prefer to run inference with the TFLite model, I can add a small runner script showing how to do that.

**Reproducibility & env**
- The included `requirements.txt` contains pinned versions discovered in the repository's `myvenv` — install that to reproduce the same environment.

