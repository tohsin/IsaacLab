import os
import json
import imageio
from PIL import Image

def main():
    root = "/home/tosin/Documents/GitHub/IsaacLab/data/recorded_depth_data_eval"
    mp4_path = os.path.join(root, "best_episode.mp4")
    json_path = os.path.join(root, "transforms.json")
    rgb_dir = os.path.join(root, "rgb")
    
    if not os.path.exists(mp4_path): 
        print("No MP4 found!")
        return
    
    os.makedirs(rgb_dir, exist_ok=True)
    
    print("Extracting RGB frames from previous MP4...")
    reader = imageio.get_reader(mp4_path)
    count = 0
    for i, im in enumerate(reader):
        file_name = f"frame_{i:05d}.png"
        Image.fromarray(im).save(os.path.join(rgb_dir, file_name))
        count += 1
    
    print(f"Saved {count} RGB frames. Patching backward-compatible JSON...")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        for i, frame in enumerate(data.get("frames", [])):
            file_name = f"frame_{i:05d}.png"
            depth_file_name = f"frame_{i:05d}.npy"
            
            frame["file_path"] = f"rgb/{file_name}"
            frame["depth_file_path"] = f"depth/{depth_file_name}"
            
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    print("Backfill completed successfully.")

if __name__ == "__main__":
    main()
