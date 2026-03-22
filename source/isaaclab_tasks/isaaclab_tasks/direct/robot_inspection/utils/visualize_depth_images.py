import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Convert .npy depth maps to colored PNG images.")
    parser.add_argument("--input_dir", type=str, default="data/recorded_depth_data_eval/depth", help="Directory containing .npy files.")
    parser.add_argument("--output_dir", type=str, default="data/recorded_depth_data_eval/depth_images", help="Directory to save .png files.")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap (e.g. viridis, plasma, gray).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    npy_files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not npy_files:
        print(f"No .npy files found in {args.input_dir}")
        return

    print(f"Loading {len(npy_files)} files from {args.input_dir}...")
    
    # Get the colormap callable from matplotlib
    cmap = plt.get_cmap(args.cmap)

    for file_path in npy_files:
        try:
            depth = np.load(file_path)
            depth = np.squeeze(depth)  # Remove singleton dimensions (e.g., HxWx1 to HxW)
            
            # Identify valid depth pixels (exclude NaN, Infinity, and zero/negative depth)
            valid_mask = np.isfinite(depth) & (depth > 0)
            
            if np.any(valid_mask):
                vmin, vmax = np.min(depth[valid_mask]), np.max(depth[valid_mask])
                # Normalize valid pixels to [0, 1] range
                norm_depth = (depth - vmin) / (vmax - vmin + 1e-8)
            else:
                norm_depth = np.zeros_like(depth)

            # Clip guarantees floating point errors don't exceed boundaries
            norm_depth = np.clip(norm_depth, 0.0, 1.0)

            # Apply colormap mapping floats [0,1] -> RGBA float [0,1,0,1]
            rgba = cmap(norm_depth)

            # Mask invalid pixels as pure pitch black
            rgba[~valid_mask] = [0.0, 0.0, 0.0, 1.0]

            # Convert to 8-bit RGB
            rgb_8bit = (rgba[:, :, :3] * 255).astype(np.uint8)

            img = Image.fromarray(rgb_8bit)
            
            # Save the file
            base_name = os.path.basename(file_path).replace(".npy", ".png")
            out_path = os.path.join(args.output_dir, base_name)
            img.save(out_path)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"\nDone! Successfully saved {len(npy_files)} RGB depth images to {args.output_dir}/")

if __name__ == "__main__":
    main()
