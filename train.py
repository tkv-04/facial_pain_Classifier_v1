import os
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIG ---
ORIGINAL_DATASET_DIR = 'ollama_cleaned_dataset'
PROCESSED_DATASET_DIR = 'dataset_merged_severe'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
FINE_TUNE_EPOCHS = 5
MAX_CLASS_WEIGHT = 4.0   # Cap to prevent overcompensation

# --- 1. Merge 'severe pain' and 'severe' into 'severe' ---
os.makedirs(PROCESSED_DATASET_DIR, exist_ok=True)
for class_name in os.listdir(ORIGINAL_DATASET_DIR):
    src_folder = os.path.join(ORIGINAL_DATASET_DIR, class_name)
    if class_name.lower().strip() in ['severe', 'severe pain']:
        target_class = 'severe'
    else:
        target_class = class_name
    target_folder = os.path.join(PROCESSED_DATASET_DIR, target_class)
    os.makedirs(target_folder, exist_ok=True)
    for fname in os.listdir(src_folder):
        src_path = os.path.join(src_folder, fname)
        tgt_path = os.path.join(target_folder, fname)
        if not os.path.exists(tgt_path):
            shutil.copy2(src_path, tgt_path)

# --- 2. Load dataset with validation split ---
train_ds = image_dataset_from_directory(
    PROCESSED_DATASET_DIR,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True
)
val_ds = image_dataset_from_directory(
    PROCESSED_DATASET_DIR,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

# --- 3. Compute (and cap) class weights ---
all_labels = []
for _, labels in train_ds.unbatch():
    all_labels.append(tf.argmax(labels).numpy())
all_labels = np.array(all_labels)
raw_class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(class_names)),
    y=all_labels
)
# Cap maximum weight to avoid overcompensation
class_weights = {i: float(min(w, MAX_CLASS_WEIGHT)) for i, w in enumerate(raw_class_weights)}
print("Class weights (capped):", class_weights)

# --- 4. Targeted, GENTLE augmentation for minority classes only ---
minority_class_indices = [class_names.index(c) for c in ['moderate', 'severe']]

data_augmentation = models.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.08),      # slight
    layers.RandomZoom(0.08),
    layers.RandomBrightness(0.08),
])

def augment_minority(images, labels):
    class_indices = tf.argmax(labels, axis=1)
    minority_class_tensor = tf.constant(minority_class_indices, dtype=class_indices.dtype)
    mask = tf.reduce_any(tf.equal(tf.expand_dims(class_indices, axis=1), minority_class_tensor), axis=1)
    mask_expanded = tf.reshape(mask, (-1, 1, 1, 1))
    augmented = data_augmentation(images, training=True)
    images_aug = tf.where(mask_expanded, augmented, images)
    return images_aug, labels

train_ds = train_ds.map(augment_minority)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# --- 5. Model definition ---
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = models.Model(inputs, outputs)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 6. Train the model ---
earlystop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[earlystop]
)
model.save('pain_model_imbalanced_corrected.h5')
print("Saved Keras model.")

# --- 7. Fine-tune last 20 layers ---
print("Fine-tuning last 20 layers...")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=[earlystop]
)
model.save('pain_model_imbalanced_finetuned.h5')
print("Saved fine-tuned model.")

# --- 8. TensorFlow Lite conversion ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('pain_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Saved TensorFlow Lite model.")

# --- 9. Evaluate with confusion matrix ---
print("Evaluating on validation set...")
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

from collections import Counter
print("Validation label distribution:", Counter(y_true))
print("Classification report:")
print(classification_report(y_true, y_pred, target_names=class_names))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
