import tensorflow as tf
from .model import create_model
from .data_loader import load_data, create_datasets
import os

def train_model(save_dir='models', batch_size=32, epochs=20):
    """Train the digit recognition model"""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and prepare data
    (x_train, y_train), (x_test, y_test) = load_data()
    train_dataset, test_dataset = create_datasets(x_train, y_train, x_test, y_test, batch_size)
    
    # Create and compile model
    model = create_model()
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True
        )
    ]
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history, model

if __name__ == '__main__':
    train_model()