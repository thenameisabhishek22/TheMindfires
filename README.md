# TheMindfires
# Participated in Proxmed Hackathon

#CODE

import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Load and preprocess data
def load_data(case_id):
    ncct_path = os.path.join(case_id, f"{case_id}_NCCT.nii.gz")
    mask_path = os.path.join(case_id, f"{case_id}_ROI.nii.gz")
    
    ncct = nib.load(ncct_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    
    # Preprocess images and masks as needed (resize, normalize, etc.)
    # Example: normalize CT scan to values between 0 and 1
    ncct = (ncct - np.min(ncct)) / (np.max(ncct) - np.min(ncct))
    
    return ncct, mask

# Train the model
def train_model(train_data):
    model = unet_model(input_shape=train_data[0][0].shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint("model.h5", monitor='val_accuracy', save_best_only=True)

    history = model.fit(train_data, epochs=10, validation_split=0.2, callbacks=[checkpoint])

    return model, history

# Example usage
case_id = "/content/drive/MyDrive/HYPODENSITY_DATA/ProxmedImg001"
ncct, mask = load_data(case_id)
train_data = np.expand_dims(ncct, axis=-1), np.expand_dims(mask, axis=-1)
trained_model, training_history = train_model([train_data])

# Example: Plot training history
plt.plot(training_history.history['accuracy'], label='accuracy')
plt.plot(training_history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Example: Make predictions
predictions = trained_model.predict(np.expand_dims(ncct, axis=0))
predictions = (predictions > 0.5).astype(np.uint8)

# Example: Visualize predictions
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.squeeze(ncct), cmap='gray')
plt.title('Original NCCT Image')

plt.subplot(1, 3, 2)
plt.imshow(np.squeeze(mask), cmap='gray')
plt.title('Ground Truth Mask')

plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(predictions), cmap='gray')
plt.title('Predicted Mask')

plt.show()
