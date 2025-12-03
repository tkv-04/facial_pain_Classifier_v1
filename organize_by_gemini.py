import os
import requests
import base64
import csv
import time

OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL = 'llava:13b'  # or 'llava:34b' or any vision model you pulled
IMAGE_DIR = 'to_filter/to_classify'
RESULT_CSV = 'CSVs/ollama_classification2.csv'
PROMPT = "Classify this face image as one of: mild, moderate, severe. Only return the class name."
RETRY_DELAY = 10  # seconds to wait before retrying connection


def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Load already processed images if resuming
processed = set()
if os.path.exists(RESULT_CSV):
    with open(RESULT_CSV, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[0]:
                processed.add(row[0])

all_images = [fname for fname in os.listdir(IMAGE_DIR) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
total_images = len(all_images)
completed = len(processed)

print(f"Total images: {total_images}. Already processed: {completed}.")

# Open CSV in append mode if resuming
with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    if completed == 0:
        writer.writerow(['filename', 'class'])
    for idx, fname in enumerate(all_images, 1):
        if fname in processed:
            continue
        img_path = os.path.join(IMAGE_DIR, fname)
        img_b64 = encode_image(img_path)
        data = {
            "model": MODEL,
            "prompt": PROMPT,
            "images": [img_b64],
            "stream": False
        }
        while True:
            try:
                response = requests.post(OLLAMA_URL, json=data)
                if response.status_code == 200:
                    result = response.json()
                    class_name = result.get('response', '').strip().lower()
                    writer.writerow([fname, class_name])
                    csvfile.flush()
                    completed += 1
                    print(f"[{completed}/{total_images}] {fname}: {class_name}")
                    break
                else:
                    print(f"Error for {fname}: {response.status_code} {response.text}")
                    # Retry on server connection errors
                    if response.status_code in [502, 503, 504, 500]:
                        print(f"Server error. Retrying in {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
                    else:
                        break
            except requests.exceptions.ConnectionError:
                print(f"Connection error. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)

print("Done! All images classified and results saved to", RESULT_CSV)