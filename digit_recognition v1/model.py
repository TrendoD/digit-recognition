from tensorflow.keras import layers, models

def create_model(input_shape=(28, 28, 1)):
    """Create CNN model for digit recognition"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def save_model(model, path):
    """Save model to specified path"""
    model.save(path)

def load_model(path):
    """Load model from specified path"""
    return models.load_model(path)