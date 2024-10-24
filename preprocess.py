import numpy as np
import pandas as pd

# Load the FER2013 dataset
data = pd.read_csv('data/fer2013')


# Extract pixels and emotions
pixels = data['pixels'].tolist()
emotions = pd.get_dummies(data['emotion']).values

# Preprocess pixel values and reshape to 48x48 grayscale images
X = np.array([np.fromstring(p, sep=' ') for p in pixels], dtype='float32')
X = X.reshape(X.shape[0], 48, 48, 1)

# Normalize pixel values
X /= 255.0

# Save preprocessed data for training
np.save('data/X.npy', X)
np.save('data/emotions.npy', emotions)

print("Preprocessing done. Data saved to 'data' folder.")
