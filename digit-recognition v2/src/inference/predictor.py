import numpy as np
import cv2
from src.preprocessing.image_processor import ImageProcessor

class Predictor:
    def __init__(self, model_path):
        self.processor = ImageProcessor()
        self.model = self.load_model(model_path)

    def load_model(self, path):
        """Load the trained model"""
        from src.model.digit_recognizer import DigitRecognizer
        recognizer = DigitRecognizer()
        recognizer.load_model(path)
        return recognizer.model

    def predict_digit(self, digit_image):
        """Predict a single digit"""
        # Normalize and reshape image
        img_array = digit_image.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        
        # Get prediction
        predictions = self.model.predict(img_array, verbose=0)
        digit = np.argmax(predictions[0])
        confidence = float(predictions[0][digit])
        
        return int(digit), confidence

    def predict(self, image_path):
        """Predict digits in an image"""
        try:
            # Process image
            print(f"Processing image: {image_path}")
            processed = self.processor.process(image_path)
            
            # Segment digits
            print("Segmenting digits...")
            digits = self.processor.segment_digits(processed['thresh'])
            
            if not digits:
                raise ValueError("No digits found in image")
            
            # Predict each digit
            predictions = []
            print(f"Making predictions for {len(digits)} digits...")
            for digit_image in digits:
                digit, confidence = self.predict_digit(digit_image)
                predictions.append({
                    'digit': digit,
                    'confidence': confidence
                })
            
            # Calculate overall confidence
            avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
            
            print(f"Successfully processed {len(digits)} digits")
            return {
                'digits': predictions,
                'confidence': avg_confidence,
                'original_image': processed['original'],
                'processed_image': processed['thresh'],
                'digit_images': digits
            }
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            raise