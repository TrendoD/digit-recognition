import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds
from pathlib import Path
from tqdm import tqdm

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
    def load_mnist(self):
        """Load MNIST dataset"""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Reshape to (samples, height, width, channels)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        print(f"MNIST shapes - Train: {x_train.shape}, Test: {x_test.shape}")
        return (x_train, y_train), (x_test, y_test)

    def load_emnist(self):
        """Load EMNIST dataset"""
        print("Loading EMNIST dataset...")
        emnist = tfds.load('emnist/digits', split=['train', 'test'], as_supervised=True)
        train_ds, test_ds = emnist[0], emnist[1]
        
        # Convert to numpy arrays
        x_train, y_train = [], []
        print("Processing EMNIST training data...")
        for image, label in tqdm(train_ds):
            # Process each image
            image = tf.transpose(image)  # Transpose for correct orientation
            image = image.numpy()  # Convert to numpy
            x_train.append(image)
            y_train.append(label.numpy())
        
        x_test, y_test = [], []
        print("Processing EMNIST test data...")
        for image, label in tqdm(test_ds):
            # Process each image
            image = tf.transpose(image)  # Transpose for correct orientation
            image = image.numpy()  # Convert to numpy
            x_test.append(image)
            y_test.append(label.numpy())
        
        # Convert lists to numpy arrays with correct shape
        x_train = np.array(x_train).reshape(-1, 28, 28, 1)
        x_test = np.array(x_test).reshape(-1, 28, 28, 1)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        print(f"EMNIST shapes - Train: {x_train.shape}, Test: {x_test.shape}")
        return (x_train, y_train), (x_test, y_test)

    def combine_datasets(self):
        """Combine MNIST and EMNIST datasets and normalize"""
        print("\nCombining datasets...")
        
        # Load datasets
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = self.load_mnist()
        (x_train_emnist, y_train_emnist), (x_test_emnist, y_test_emnist) = self.load_emnist()
        
        # Combine training data
        x_train = np.concatenate([x_train_mnist, x_train_emnist])
        y_train = np.concatenate([y_train_mnist, y_train_emnist])
        
        # Combine test data
        x_test = np.concatenate([x_test_mnist, x_test_emnist])
        y_test = np.concatenate([y_test_mnist, y_test_emnist])
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        print("\nFinal dataset shapes:")
        print(f"Training data: {x_train.shape}")
        print(f"Test data: {x_test.shape}")
        print(f"Total training samples: {len(x_train)}")
        print(f"Total test samples: {len(x_test)}")
        
        return (x_train, y_train), (x_test, y_test)