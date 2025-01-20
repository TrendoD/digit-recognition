import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

def load_data():
    """Load and preprocess MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Reshape data to add channel dimension
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    return (x_train, y_train), (x_test, y_test)

def create_datasets(x_train, y_train, x_test, y_test, batch_size=32):
    """Create TensorFlow datasets for training and testing"""
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, test_dataset