import os

base = 'ollama_cleaned_dataset'
classes = ['none', 'mild', 'moderate', 'severe', 'severe pain']
counts = {}

for cls in classes:
    path = os.path.join(base, cls)
    if os.path.exists(path):
        counts[cls] = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    else:
        counts[cls] = 0

# Merge 'severe pain' into 'severe'
counts['severe'] += counts['severe pain']
del counts['severe pain']

print('Image counts per class:')
for cls in ['none', 'mild', 'moderate', 'severe']:
    print(f"{cls}: {counts[cls]}") 