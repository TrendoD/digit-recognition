import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from src.model.digit_recognizer import DigitRecognizer

def evaluate_model():
    # Load trained model
    recognizer = DigitRecognizer()
    recognizer.load_model()

    # Load test data
    (_, _), (X_test, y_test) = mnist.load_data()
    X_test = X_test.astype('float32') / 255.0

    # Evaluate
    loss, acc = recognizer.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.2%}")

    # Generate predictions
    y_pred = np.argmax(recognizer.model.predict(X_test), axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Create results directory relative to project root
    results_dir = os.path.join(os.environ['PYTHONPATH'], 'results/evaluation')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save with absolute path
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))

if __name__ == '__main__':
    evaluate_model()
