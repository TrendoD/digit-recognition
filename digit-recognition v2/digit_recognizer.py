import argparse
import os
from tqdm import tqdm
from src.inference.predictor import Predictor
from src.utils.visualization import ResultVisualizer
import matplotlib.pyplot as plt

def process_single_image(predictor, image_path, show_plots=True):
    """Process a single image and show results"""
    try:
        # Make prediction
        result = predictor.predict(image_path)
        
        # Show results
        print(f"\nPrediction Results for {os.path.basename(image_path)}:")
        print(f"Detected Digits: {len(result['digits'])}")
        number = int(''.join(str(d['digit']) for d in result['digits']))
        print(f"Predicted Number: {number}")
        print(f"Overall Confidence: {result['confidence']:.2%}")
        
        print("\nDigit Breakdown:")
        for i, digit in enumerate(result['digits'], 1):
            print(f"Digit {i}: {digit['digit']} (Confidence: {digit['confidence']:.2%})")
        
        if show_plots:
            ResultVisualizer.display_results(
                result['original_image'],
                result['processed_image'],
                result
            )
            plt.show()
            
        return True, result
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False, None

def process_directory(predictor, directory, show_plots=False):
    """Process all images in a directory"""
    # Get list of image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    image_files = [f for f in os.listdir(directory) 
                  if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        print(f"No image files found in {directory}")
        return
    
    print(f"\nProcessing {len(image_files)} images in {directory}")
    
    # Initialize counters
    successful = 0
    failed = 0
    total_confidence = 0
    
    # Process each image with progress bar
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(directory, filename)
        print(f"Processing images: {i*20}%", end='\r')
        
        success, result = process_single_image(predictor, image_path, show_plots)
        if success:
            successful += 1
            total_confidence += result['confidence']
        else:
            failed += 1
    
    # Print summary
    print("\nSummary:")
    print(f"Total images processed: {len(image_files)}")
    print(f"Successful predictions: {successful}")
    print(f"Failed predictions: {failed}")
    if successful > 0:
        avg_confidence = total_confidence / successful
        print(f"Average confidence: {avg_confidence:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Digit Recognition Tool')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str, help='Path to single image')
    group.add_argument('--dir', type=str, help='Path to directory of images')
    parser.add_argument('--no_plot', action='store_true', help='Disable plot display')
    
    args = parser.parse_args()
    
    try:
        print("Loading model...")
        predictor = Predictor('models/saved_models/digit_model.keras')
        
        if args.image_path:
            process_single_image(predictor, args.image_path, not args.no_plot)
        elif args.dir:
            process_directory(predictor, args.dir, not args.no_plot)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()