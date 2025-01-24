# Two-Digit Recognition System

## Features
- MNIST-based CNN model
- Real-time digit segmentation
- Confidence scoring
- Image preprocessing pipeline

## Setup
1. Create virtual environment:
```
python -m venv venv
venv\Scripts\activate
```
2. Install requirements:
```
pip install -r requirements.txt
```
3. Create required directories:
```
mkdir data\raw models\saved_models results
```

4. Force directory creation in code:
```python
import os
os.makedirs('models/saved_models', exist_ok=True)
```

## Complete Workflow Guide

### 1. System Setup
```bash
# Clone repository
git clone https://github.com/yourrepo/digit-recognition.git
cd digit-recognition

# Set up environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
mkdir data\raw models\saved_models results
```

### 2. Model Training
```bash
# Run training with default parameters
python src/model/train_model.py

# Optional arguments for advanced users:
# python src/model/train_model.py --epochs 20 --batch_size 64
```

### 3. Model Evaluation
```bash
# Run comprehensive evaluation
python src/model/evaluate_model.py

# Results will be saved in:
# - results/evaluation/confusion_matrix.png
# - Console output shows precision/recall metrics
```

### 4. Making Predictions
Create `demo.py`:
```python
from src.inference.predictor import Predictor
from src.utils.visualization import ResultVisualizer

# Initialize predictor
predictor = Predictor('models/saved_models/digit_model.keras')

# Predict and display
image_path = 'data/raw/test_image.png'  # Your image path
result = predictor.predict(image_path)

print(f"\nPrediction Result:")
print(f"Number: {result['number']:02d}")
print(f"Average Confidence: {result['confidence']:.2%}")

ResultVisualizer.display_results(
    result['original_image'],
    result['processed_image'],
    result
)
```

Run with:
```bash
python demo.py --image_path path/to/your_image.png
```

To use the default test image:
```bash
python demo.py
```

### 5. Generating Test Images (Optional)
```bash
python generate_test_image.py
```

## Troubleshooting
Q: Getting "No such file" errors?
A: Ensure:
1. Model file exists in models/saved_models/
2. Test images are in data/raw/
3. All directories are created

Q: Low confidence predictions?
A: Ensure input images:
- Have clear contrast
- Digits are centered
- Background is uniform
- Image size > 100x50 pixels
