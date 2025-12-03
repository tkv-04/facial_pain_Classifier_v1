import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomBrightness

# --- CONFIG ---
DATASET_DIR = 'ollama_cleaned_dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
OUTPUT_MODEL = 'pain_classifier_mobilenetv2.h5'
OUTPUT_TFLITE = 'pain_classifier_mobilenetv2.tflite'

# --- LOAD DATASET ---
dataset = image_dataset_from_directory(
    DATASET_DIR,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset='both',
    seed=42
)
train_ds, val_ds = dataset

# Get class names before mapping/prefetching
class_names = train_ds.class_names

# --- DATA AUGMENTATION ---
data_augmentation = Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomBrightness(0.1)
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# --- PREFETCH ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# --- MODEL ---
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Fine-tune later if needed

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(train_ds.element_spec[1].shape[-1], activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- TRAIN ---
callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --- SAVE KERAS MODEL ---
model.save(OUTPUT_MODEL)
print(f"Saved Keras model to {OUTPUT_MODEL}")

# --- CONVERT TO TFLITE ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(OUTPUT_TFLITE, 'wb') as f:
    f.write(tflite_model)
print(f"Saved TensorFlow Lite model to {OUTPUT_TFLITE}")

# --- FINE-TUNING ---
print("\nStarting fine-tuning (unfreezing last 20 layers of MobileNetV2)...")
for layer in base_model.layers[-20:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])
finetune_epochs = 5
history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=finetune_epochs,
    callbacks=callbacks
)
model.save('pain_classifier_mobilenetv2_finetuned.h5')
print("Saved fine-tuned Keras model to pain_classifier_mobilenetv2_finetuned.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('pain_classifier_mobilenetv2_finetuned.tflite', 'wb') as f:
    f.write(tflite_model)
print("Saved fine-tuned TensorFlow Lite model to pain_classifier_mobilenetv2_finetuned.tflite")

# --- CLASS LABELS ---
print("Class indices:")
for idx, class_name in enumerate(class_names):
    print(f"{idx}: {class_name}") 