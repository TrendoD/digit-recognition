import tensorflow as tf
from tensorflow.keras import layers, models

class DigitRecognizer:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            # Input reshape layer
            layers.Reshape((28, 28, 1), input_shape=(28, 28)),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])

        # Compile with better optimizer settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_test, y_test, epochs=15):
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=128
        )

    def train_with_augmentation(self, datagen, X_train, y_train, X_test, y_test, 
                              epochs=20, batch_size=32):
        return self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_test, y_test),
            epochs=epochs,
            steps_per_epoch=len(X_train) // batch_size
        )

    def save_model(self, path='models/saved_models/digit_model.keras'):
        import os
        # Ensure proper file extension
        if not path.endswith('.keras'):
            path += '.keras'
        # Create directory structure if missing
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save model
        self.model.save(path)

    def load_model(self, path='models/saved_models/digit_model.keras'):
        self.model = tf.keras.models.load_model(path)