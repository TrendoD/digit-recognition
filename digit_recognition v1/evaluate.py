import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

def evaluate_model(model, test_dataset):
    """Evaluate model performance"""
    # Evaluate on test data
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f'\nTest accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    # Generate predictions
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_pred = model.predict(test_dataset)
    y_pred = np.argmax(y_pred, axis=1)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('output/confusion_matrix.png')
    plt.close()

if __name__ == '__main__':
    from digit_recognition.model import load_model
    from digit_recognition.data_loader import load_data, create_datasets
    
    # Load trained model
    model = load_model('models/best_model.h5')
    
    # Load and prepare test data
    (_, _), (x_test, y_test) = load_data()
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    
    # Evaluate model and generate visualizations
    evaluate_model(model, test_dataset)