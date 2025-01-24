import cv2
import numpy as np
from typing import Dict, List, Tuple, Union
import os

class ImageProcessor:
    def __init__(self, img_size: Tuple[int, int] = (28, 28)):
        self.img_size = img_size
        self.debug_mode = True
        self.debug_dir = "debug_output"
        os.makedirs(self.debug_dir, exist_ok=True)
        
    def save_debug(self, name: str, image: np.ndarray):
        if self.debug_mode:
            cv2.imwrite(f'{self.debug_dir}/debug_{name}.png', image)
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"Value range: [{image.min()}, {image.max()}]")
            print(f"Non-zero pixels: {np.count_nonzero(image)}")

    def process(self, image_path: str) -> Dict[str, np.ndarray]:
        """Enhanced image processing pipeline optimized for multiple digits"""
        # Read image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not read image: {image_path}")
        print(f"\nProcessing image: {image_path}")
        print(f"Original image shape: {original.shape}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        # Normalize size to ensure consistent processing
        scale = max(1, 400 / gray.shape[1])
        if scale != 1:
            new_width = int(gray.shape[1] * scale)
            new_height = int(gray.shape[0] * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        self.save_debug("1_resized", gray)

        # Enhanced bilateral filtering
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        self.save_debug("2_denoised", denoised)

        # Local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        self.save_debug("3_enhanced", enhanced)

        # Initial thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.save_debug("4_binary", binary)

        # Distance transform for better digit separation
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        dist = ((dist / dist.max()) * 255).astype(np.uint8)
        self.save_debug("5_distance", dist)

        # Adaptive thresholding with optimized parameters
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,  # Smaller block size for better digit separation
            5
        )
        self.save_debug("6_thresh", thresh)

        # Connect components that belong to the same digit
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        self.save_debug("7_morph", morph)

        # Clean small noise
        cleaned = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        self.save_debug("8_cleaned", cleaned)

        return {
            'original': original,
            'processed': cleaned,
            'gray': gray,
            'thresh': thresh,
            'dist': dist
        }

    def find_digit_contours(self, img: np.ndarray) -> List[Tuple]:
        """Improved digit contour detection with better separation"""
        print("\nFinding digit contours...")
        height, width = img.shape
        print(f"Image dimensions: {width}x{height}")

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        
        # Prepare debug image
        debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        
        # Initialize parameters
        min_area = (height * width) * 0.0005  # Lower threshold for small digits
        max_area = (height * width) * 0.9    # Higher threshold
        digit_contours = []

        # Process each component
        for i in range(1, num_labels):  # Skip background (label 0)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Calculate component properties
            aspect_ratio = w / float(h)
            component = (labels == i).astype(np.uint8) * 255
            
            # Check if component is too wide (might be multiple digits)
            if w > h * 1.5:  # If width is significantly larger than height
                # Try to split wide components
                hist = np.sum(component, axis=0)  # Vertical projection
                valleys = self.find_valleys(hist)
                
                if len(valleys) > 0:
                    # Split component at valleys
                    prev_x = x
                    for valley in valleys:
                        valley_x = x + valley
                        # Create sub-component
                        sub_w = valley_x - prev_x
                        if sub_w > 5:  # Minimum width threshold
                            digit_contours.append((prev_x, y, sub_w, h, area//len(valleys)))
                            cv2.rectangle(debug_img, (prev_x,y), (prev_x+sub_w,y+h), (0,255,0), 2)
                        prev_x = valley_x
                    
                    # Add last sub-component
                    final_w = (x + w) - prev_x
                    if final_w > 5:
                        digit_contours.append((prev_x, y, final_w, h, area//len(valleys)))
                        cv2.rectangle(debug_img, (prev_x,y), (prev_x+final_w,y+h), (0,255,0), 2)
                    continue

            # Regular component processing
            print(f"\nComponent {i}:")
            print(f"Position: ({x}, {y}), Size: {w}x{h}")
            print(f"Area: {area}, Aspect Ratio: {aspect_ratio:.2f}")

            # Enhanced filtering criteria
            if area < min_area:
                print("✗ Rejected - Too small")
                cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0,0,255), 1)
                continue

            if aspect_ratio < 0.2 or aspect_ratio > 2.0:
                print("✗ Rejected - Invalid aspect ratio")
                cv2.rectangle(debug_img, (x,y), (x+w,y+h), (255,0,0), 1)
                continue

            print("✓ Accepted as digit")
            cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0,255,255), 2)
            digit_contours.append((x, y, w, h, area))

        self.save_debug("contours", debug_img)

        # Sort contours left to right
        digit_contours.sort(key=lambda x: x[0])
        print(f"\nFound {len(digit_contours)} valid digits")
        return digit_contours

    def find_valleys(self, histogram, min_dist=10):
        """Find valleys in histogram for digit separation"""
        # Smooth histogram
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(histogram, kernel, mode='same')
        
        # Find local minima
        valleys = []
        for i in range(1, len(smoothed)-1):
            if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
                # Check if it's a significant valley
                left_max = max(smoothed[max(0, i-min_dist):i])
                right_max = max(smoothed[i+1:min(len(smoothed), i+min_dist)])
                if smoothed[i] < 0.5 * min(left_max, right_max):
                    valleys.append(i)
        
        return valleys

    def prepare_digit(self, img: np.ndarray, bbox: Tuple) -> np.ndarray:
        """Improved digit preparation"""
        x, y, w, h = bbox[:4]
        print(f"\nPreparing digit at position ({x}, {y}) with size {w}x{h}")
        
        # Extract digit with padding
        padding = int(min(w, h) * 0.15)  # Reduced padding
        x_start = max(0, x - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_start = max(0, y - padding)
        y_end = min(img.shape[0], y + h + padding)
        
        digit = img[y_start:y_end, x_start:x_end].copy()
        self.save_debug(f"digit_extract_{x}_{y}", digit)

        # Make square while preserving aspect ratio
        target_size = max(digit.shape) + padding * 2
        square = np.zeros((target_size, target_size), dtype=np.uint8)
        
        # Center the digit
        y_offset = (target_size - digit.shape[0]) // 2
        x_offset = (target_size - digit.shape[1]) // 2
        square[y_offset:y_offset+digit.shape[0], 
               x_offset:x_offset+digit.shape[1]] = digit
        self.save_debug(f"digit_square_{x}_{y}", square)

        # Resize with better quality
        resized = cv2.resize(square, self.img_size, interpolation=cv2.INTER_AREA)
        
        # Enhance contrast
        resized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        
        # Final thresholding with Otsu's method
        _, digit = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Ensure good stroke width
        kernel = np.ones((2,2), np.uint8)
        digit = cv2.dilate(digit, kernel, iterations=1)
        
        self.save_debug(f"digit_final_{x}_{y}", digit)
        return digit

    def segment_digits(self, processed_image: np.ndarray) -> List[np.ndarray]:
        """Segment and prepare digits"""
        print("\nSegmenting digits...")
        
        # Find digit contours with improved separation
        digit_contours = self.find_digit_contours(processed_image)
        
        if not digit_contours:
            raise ValueError("No digits found in image")

        # Prepare each digit
        digits = []
        for bbox in digit_contours:
            digit = self.prepare_digit(processed_image, bbox)
            digits.append(digit)

        return digits