import cv2
import matplotlib.pyplot as plt
import numpy as np

class ResultVisualizer:
    @staticmethod
    def display_results(original_image, processed_image, result):
        num_digits = len(result['digits'])
        
        # Calculate figure size and layout
        if num_digits <= 3:
            fig_width = 12
            cols = 2 + num_digits
        else:
            # For many digits, use two rows
            fig_width = max(12, 3 * num_digits)
            cols = num_digits
        
        plt.figure(figsize=(fig_width, 4 if num_digits <= 3 else 8))
        
        # Original Image
        plt.subplot(2 if num_digits > 3 else 1, cols, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Processed Image
        plt.subplot(2 if num_digits > 3 else 1, cols, 2)
        plt.imshow(processed_image, cmap='gray')
        plt.title('Processed Image')
        plt.axis('off')
        
        # Individual Digits
        for i, (digit_img, digit_info) in enumerate(zip(result['digit_images'], result['digits'])):
            plt_idx = 3 + i if num_digits <= 3 else num_digits + 1 + i
            plt.subplot(2 if num_digits > 3 else 1, cols, plt_idx)
            plt.imshow(digit_img, cmap='gray')
            plt.title(f"Digit {i+1}\nPred: {digit_info['digit']}\nConf: {digit_info['confidence']:.1%}")
            plt.axis('off')
        
        plt.tight_layout()
