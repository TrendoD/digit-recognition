import tensorflow as tf
import numpy as np
import cv2
from digit_recognition.model import load_model

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to 28x28 (MNIST size)
    img = cv2.resize(img, (28, 28))
    
    # Invert colors (MNIST has white digits on black background)
    img = 255 - img
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    # Add batch and channel dimensions
    img = np.expand_dims(img, axis=(0, -1))
    
    return img

def predict_digit(image_path):
    """Predict digit from image"""
    # Load trained model
    model = load_model('models/best_model.h5')
    
    # Preprocess image
    img = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img)
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions)
    
    return predicted_digit, confidence

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python -m digit_recognition.predict <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    digit, confidence = predict_digit(image_path)
    print(f"Predicted digit: {digit} (confidence: {confidence:.2%})")