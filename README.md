# Digit Recognition System

## Features
- MNIST-based CNN model
- Real-time digit segmentation
- Confidence scoring
- Image preprocessing pipeline

## Complete Workflow Guide

### 1. System Setup
```bash
# Clone repository
git clone https://github.com/TrendoD/digit-recognition
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
Only 1 Image Prediction:

Run with:
```bash
python digit_recognizer.py --image_path path/to/your_image.png
```

Direktori Full Predictions :

Run with :
```bash
python digit_recognizer.py --dir  path/to/your_folder
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
