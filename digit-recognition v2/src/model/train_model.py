import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys
from pathlib import Path
import time
import os
from tqdm import tqdm

# Add project root to Python path
root_dir = str(Path(__file__).parent.parent.parent)
sys.path.append(root_dir)

from src.model.digit_recognizer import DigitRecognizer
from src.data.data_loader import DataLoader

def create_callbacks():
    # Ensure directory exists
    os.makedirs('models/saved_models', exist_ok=True)
    
    return [
        # Save best model
        ModelCheckpoint(
            'models/saved_models/digit_model.keras',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # Early stopping if no improvement
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateau
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

def train_model():
    try:
        # Initialize components
        data_loader = DataLoader()
        
        # Load and combine datasets
        (X_train, y_train), (X_test, y_test) = data_loader.combine_datasets()
        
        print("\nInitializing model...")
        recognizer = DigitRecognizer()
        
        # Create data generator with augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Configure training parameters
        BATCH_SIZE = 128  # Reduced batch size for CPU training
        
        # Train the model
        print("\nStarting training...")
        start_time = time.time()
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // BATCH_SIZE
        
        history = recognizer.model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            validation_data=(X_test, y_test),
            epochs=15,
            steps_per_epoch=steps_per_epoch,
            callbacks=create_callbacks(),
            verbose=1,
            workers=1,  # Set to 1 for more stable training
            use_multiprocessing=False  # Disable for stability
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Evaluate final model
        print("\nEvaluating model...")
        test_loss, test_accuracy = recognizer.model.evaluate(X_test, y_test, verbose=1)
        print(f"Test accuracy: {test_accuracy*100:.2f}%")
        
        return history, test_accuracy
        
    except Exception as e:
        import traceback
        print(f"\nERROR: Training failed!")
        print(str(e))
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    try:
        history, accuracy = train_model()
        print("\nTraining completed successfully!")
        print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    except Exception as e:
        sys.exit(1)