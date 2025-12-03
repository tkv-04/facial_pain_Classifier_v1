import os
import shutil
import csv

# --- CONFIG ---
SOURCE_DIR = 'to_filter'
OLLAMA_CSV = 'ollama_classification.csv'
DEST_DIR = 'ollama_cleaned_dataset'

os.makedirs(DEST_DIR, exist_ok=True)

with open(OLLAMA_CSV, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip header
    for row in reader:
        if len(row) < 2:
            continue
        filename, class_name = row[0].strip(), row[1].strip().lower()
        src_path = os.path.join(SOURCE_DIR, filename)
        dest_class_dir = os.path.join(DEST_DIR, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_class_dir)
        else:
            print(f"Warning: {src_path} not found.")

print("Done! Images have been organized by Ollama's classification.")