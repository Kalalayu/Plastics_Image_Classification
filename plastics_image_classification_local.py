
# Plastics Image Classification Script (Modified for local or deployment use)

import zipfile
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Set zip path and extraction directory
zip_path = 'Dataset_plastics_mhadlekar_UN-20250720T182641Z-1-001.zip'
extract_path = 'Dataset_plastics_mhadlekar_UN'

# Check if zip file exists
if not os.path.exists(zip_path):
    raise FileNotFoundError(f"Zip file not found at {zip_path}")

# Extract the dataset if not already extracted
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Set the base directory where images are stored
base_dir = os.path.join(extract_path, 'Dataset_plastics_mhadlekar_UN')

# Data Preprocessing
IMG_HEIGHT = 256
IMG_WIDTH = 256

image_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5,
    validation_split=0.25
)

train_data_gen = image_gen.flow_from_directory(
    directory=base_dir,
    batch_size=32,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical',
    subset='training',
    seed=123
)

val_data_gen = image_gen.flow_from_directory(
    directory=base_dir,
    batch_size=32,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical',
    subset='validation',
    seed=123
)

print(f"Number of training images: {train_data_gen.samples}")
print(f"Number of validation images: {val_data_gen.samples}")
print("\nClass indices:")
print(train_data_gen.class_indices)

# Build the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(train_data_gen.num_classes, activation='softmax')
])

model.summary()

# Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision()])

# Train the Model
epochs = 30
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // train_data_gen.batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // val_data_gen.batch_size
)

# Evaluate the Model
loss, accuracy, precision = model.evaluate(val_data_gen)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Precision: {precision:.4f}")

# Plot Accuracy and Precision
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
precision_train = history.history.get('precision')
val_precision = history.history.get('val_precision')
epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, precision_train, label='Training Precision')
plt.plot(epochs_range, val_precision, label='Validation Precision')
plt.legend(loc='lower right')
plt.title('Training and Validation Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.grid(True)

plt.tight_layout()
plt.show()

# Confusion Matrix
val_data_gen.reset()
y_true = val_data_gen.classes[val_data_gen.index_array]
y_pred_proba = model.predict(val_data_gen)
y_pred = np.argmax(y_pred_proba, axis=1)
class_names = list(val_data_gen.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
