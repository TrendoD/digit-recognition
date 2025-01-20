# Digit Recognition Project

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits using the MNIST dataset.

## Project Structure

```
digit_recognition/
├── data_loader.py       # Data loading and preprocessing
├── model.py            # CNN model definition
├── train.py            # Training script
├── evaluate.py         # Evaluation and visualization
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the model:
   ```bash
   python -m digit_recognition.train
   ```
   
2. Evaluate the model:
   ```bash
   python -m digit_recognition.evaluate
   ```

## Results

The model achieves approximately 99% accuracy on the MNIST test set. Training progress and evaluation metrics are visualized using matplotlib.

## License

This project is licensed under the MIT License.