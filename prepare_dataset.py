import os
import shutil

# Define paths
SOURCE_ROOT = 'Pictures/Modified'
TO_FILTER_DIR = 'to_filter'
NONE_DIR = os.path.join(TO_FILTER_DIR, 'none')
TO_CLASSIFY_DIR = os.path.join(TO_FILTER_DIR, 'to_classify')

# Clear the to_filter directory if it exists
if os.path.exists(TO_FILTER_DIR):
    shutil.rmtree(TO_FILTER_DIR)

# Recreate necessary directories
os.makedirs(NONE_DIR, exist_ok=True)
os.makedirs(TO_CLASSIFY_DIR, exist_ok=True)

# Traverse the source directory and copy files
for root, dirs, files in os.walk(SOURCE_ROOT):
    for fname in files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            orig_path = os.path.join(root, fname)
            if 'Neutral' in root:
                # Copy to none directory
                dest_path = os.path.join(NONE_DIR, fname)
            else:
                # Copy to to_classify directory
                dest_path = os.path.join(TO_CLASSIFY_DIR, fname)
            shutil.copy2(orig_path, dest_path)

print("Dataset preparation complete. Images are organized in 'to_filter'.") 