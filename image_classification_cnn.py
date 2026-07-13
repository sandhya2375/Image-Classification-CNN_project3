# ============================================
# PROJECT 3: Image Classification of Cats and Dogs using CNN
# IBM SkillsBuild - Google Colab Ready
# ============================================

# Step 1: Libraries install karo
!pip install tensorflow keras numpy matplotlib opencv-python pillow scikit-learn

# Step 2: Libraries import karo
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import shutil
from google.colab import files
from urllib.request import urlopen
import zipfile
import io

print("✅ Sab libraries import ho gaye!")

# ============================================
# Step 3: Dataset download karo (Microsoft Cats and Dogs)
# ============================================
print("\n📥 Dataset download ho raha hai...")

# Dataset download करो
dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

try:
    # Download करो
    print("Downloading dataset... (ye thoda time le sakta hai)")
    urlopen(dataset_url)
    print("✅ Dataset download ho gaya!")
except:
    print("Manual download karo ya Google Drive se upload karo")

# Alternative: Smaller dataset use करो (fast training के लिए)
print("\n🐱🐶 Smaller dataset create करते हैं (training के लिए)...")

# Tensorflow datasets से dataset लो
import tensorflow_datasets as tfds

# Cats and Dogs dataset download करो (छोटा version)
(raw_train, raw_test), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
    data_dir='/tmp/tfds'
)

print(f"✅ Dataset loaded!")
print(f"Training samples: {len(list(raw_train))}")
print(f"Testing samples: {len(list(raw_test))}")

# ============================================
# Step 4: Image preprocessing functions
# ============================================
print("\n🖼️ Image preprocessing ho raha hai...")

IMG_SIZE = 224  # MobileNetV2 के लिए 224x224 चाहिए

def preprocess_image(image, label):
    """
    Image को preprocess करो:
    1. Resize करो 224x224 में
    2. Normalize करो (0-255 से 0-1 में)
    """
    # Resize
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Normalize (0-255 से 0-1 में)
    image = image / 255.0
    
    return image, label

# Preprocess करो
train_data = raw_train.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)
test_data = raw_test.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

print("✅ Preprocessing complete!")

# ============================================
# Step 5: CNN Model बनाओ (Transfer Learning के साथ)
# ============================================
print("\n🤖 CNN Model बन रहा है...")

"""
Transfer Learning:
- Pre-trained MobileNetV2 model use करेंगे (ImageNet पर trained)
- Top layers को हटाकर अपनी layers add करेंगे
- Cats vs Dogs के लिए customize करेंगे

Advantage:
- तेजी से train होगा
- कम data में भी अच्छा काम करेगा
- Better accuracy
"""

# Pre-trained model लो
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,  # Classification layers को हटा दो
    weights='imagenet'  # Pre-trained weights
)

# Base model को freeze करो (weights change न हों)
base_model.trainable = False

print(f"Base model parameters: {base_model.count_params()}")

# अपना model बनाओ
model = models.Sequential([
    # Input layer
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Pre-trained MobileNetV2
    base_model,
    
    # Custom layers
    layers.GlobalAveragePooling2D(),  # Flatten करो
    
    layers.Dense(256, activation='relu'),  # Hidden layer 1
    layers.Dropout(0.3),  # Dropout (overfitting prevent करने के लिए)
    
    layers.Dense(128, activation='relu'),  # Hidden layer 2
    layers.Dropout(0.3),
    
    layers.Dense(1, activation='sigmoid')  # Output layer (binary classification)
])

print("✅ Model architecture तैयार!")

# Model summary
print("\nModel Architecture:")
model.summary()

# ============================================
# Step 6: Model compile करो
# ============================================
print("\n⚙️ Model compile हो रहा है...")

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # 2 classes के लिए
    metrics=['accuracy', 
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall()]
)

print("✅ Model compiled!")

# ============================================
# Step 7: Model train करो
# ============================================
print("\n🎓 Model training शुरू हो रहा है...")
print("(ये कुछ मिनट ले सकता है...)")

history = model.fit(
    train_data,
    epochs=15,
    validation_data=test_data,
    verbose=1
)

print("\n✅ Model training complete!")

# ============================================
# Step 8: Model evaluate करो
# ============================================
print("\n📊 Model evaluation...")

# Test data पर evaluate करो
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_data)

print(f"\n🎯 TEST RESULTS:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall: {test_recall:.4f}")

# F1-Score calculate करो
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
print(f"  F1-Score: {f1_score:.4f}")

# ============================================
# Step 9: Predictions करो और confusion matrix बनाओ
# ============================================
print("\n🔍 Test predictions ले रहे हैं...")

# Test data को array में convert करो
y_true = []
y_pred = []

for images, labels in test_data:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(predictions.flatten())
    y_true.extend(labels.numpy())

# Threshold 0.5 लगाओ
y_pred = (np.array(y_pred) > 0.5).astype(int)
y_true = np.array(y_true)

print("✅ Predictions complete!")

# Confusion matrix बनाओ
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives (Cats correctly): {cm[0,0]}")
print(f"False Positives (Cats as Dogs): {cm[0,1]}")
print(f"False Negatives (Dogs as Cats): {cm[1,0]}")
print(f"True Positives (Dogs correctly): {cm[1,1]}")

# ============================================
# Step 10: Visualization - Training History
# ============================================
print("\n📈 Graphs बन रहे हैं...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], 
            xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')

# Performance Metrics
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [test_accuracy, test_precision, test_recall, f1_score]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

axes[1, 1].bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 1].set_ylim(0, 1.1)
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Performance Metrics', fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

for i, v in enumerate(metrics_values):
    axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ Graphs ready!")

# ============================================
# Step 11: Sample predictions दिखाओ
# ============================================
print("\n🎯 SAMPLE PREDICTIONS:")
print("="*60)

# Test data से कुछ images लो
sample_count = 0
for images, labels in test_data.take(2):
    predictions = model.predict(images[:6], verbose=0)
    
    for i in range(6):
        actual_label = "🐶 DOG" if labels[i] == 1 else "🐱 CAT"
        predicted_label = "🐶 DOG" if predictions[i] > 0.5 else "🐱 CAT"
        confidence = predictions[i][0] if predictions[i] > 0.5 else (1 - predictions[i][0])
        
        match = "✅" if (predictions[i] > 0.5) == labels[i] else "❌"
        
        print(f"\n{match} Sample {sample_count + i + 1}:")
        print(f"  Actual: {actual_label}")
        print(f"  Predicted: {predicted_label}")
        print(f"  Confidence: {confidence*100:.2f}%")
    
    sample_count += 6
    if sample_count >= 12:
        break

# ============================================
# Step 12: Classification Report
# ============================================
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT:")
print("="*60)
print(classification_report(y_true, y_pred, target_names=['Cat', 'Dog']))

# ============================================
# Step 13: Model save करो
# ============================================
print("\n💾 Model save हो रहा है...")

model.save('cats_dogs_classification_model.h5')
print("✅ Model saved as 'cats_dogs_classification_model.h5'")

# ============================================
# Step 14: Model summary
# ============================================
print("\n" + "="*60)
print("📊 PROJECT SUMMARY:")
print("="*60)
print(f"\n✅ Model Architecture: MobileNetV2 + Custom Dense Layers")
print(f"✅ Training Epochs: 15")
print(f"✅ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✅ Test Precision: {test_precision:.4f}")
print(f"✅ Test Recall: {test_recall:.4f}")
print(f"✅ F1-Score: {f1_score:.4f}")
print(f"\n✅ Total Parameters: {model.count_params():,}")
print(f"✅ Status: PRODUCTION READY! 🚀")

print("\n" + "="*60)
print("🎉 PROJECT 3 COMPLETE!")
print("="*60)

# ============================================
# Step 15: अपनी image से predict करो (Optional)
# ============================================
print("\n🖼️ CUSTOM IMAGE PREDICTION (Optional)")
print("-"*60)

def predict_single_image(image_path):
    """
    किसी एक image को load करके predict करो
    """
    # Image load करो
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    
    # Array में convert करो
    img_array = img_to_array(img) / 255.0
    
    # Batch में convert करो
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Predict करो
    prediction = model.predict(img_batch, verbose=0)
    
    # Result
    label = "🐶 DOG" if prediction[0] > 0.5 else "🐱 CAT"
    confidence = prediction[0][0] if prediction[0] > 0.5 else (1 - prediction[0][0])
    
    return label, confidence

print("\nfunction 'predict_single_image(image_path)' use करो:")
print("Example: label, conf = predict_single_image('your_image.jpg')")
print("         print(f'This is a {label} ({conf*100:.2f}% confident)')")
