import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt

# Load the model
loaded_model = load_model('digits.model')

# Test on your own files
# Here, you need to provide the path to your own image files
image_files = [r"C:\Users\conta\Downloads\seven.png"]

for image_file in image_files:
    img = plt.imread(image_file)
    img = img[:, :, 0]  # Convert RGB image to grayscale

    # Preprocess the image
    img = img / 255.0
    img = np.reshape(img, (1, 28, 28))  # Reshape to match input shape of the model

    # Make prediction
    prediction = loaded_model.predict(img)
    predicted_digit = np.argmax(prediction)

    plt.imshow(img[0], cmap='gray')
    plt.title(f'Predicted Digit: {predicted_digit}')
    plt.show()
    print(predicted_digit)
