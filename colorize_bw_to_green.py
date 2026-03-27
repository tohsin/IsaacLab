import argparse
from PIL import Image
import numpy as np

def colorize_green(input_path, output_path):
    try:
        # Load image and convert to grayscale ('L' mode)
        # This preserves all structural information and shapes
        img = Image.open(input_path).convert('L')
        img_data = np.array(img)
        
        # Create a new RGB image array initialized to black (0s)
        colored_data = np.zeros((*img_data.shape, 3), dtype=np.uint8)
        
        # Map the grayscale intensity directly to the Green channel (index 1)
        # Red (0) and Blue (2) channels remain 0.
        # This makes white (255) become bright green (0, 255, 0), and black stays black
        colored_data[:, :, 1] = img_data
        
        # Save the resulting image
        colored_img = Image.fromarray(colored_data, mode='RGB')
        colored_img.save(output_path)
        print(f"Successfully saved colorized image to: {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a black and white image to black and green.")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("output", help="Path to save the output image")
    args = parser.parse_args()
    
    colorize_green(args.input, args.output)
# python colorize_bw_to_green.py data/Flange_recorded/masks/frame_00232.png data/Flange_recorded/green_inspect.png
