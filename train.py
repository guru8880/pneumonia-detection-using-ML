import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info logs

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_cnn_model

# Paths
train_dir = '../data/train'
val_dir = '../data/val'

# Data generators
datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary')
val_data = datagen.flow_from_directory(val_dir, target_size=(150,150), batch_size=32, class_mode='binary')

# Build and train model
model = build_cnn_model()
history = model.fit(train_data, epochs=5, validation_data=val_data)

# Save model
os.makedirs('../results', exist_ok=True)
model.save('../results/cnn_pneumonia_model.h5')

print("Training completed and model saved successfully.")
