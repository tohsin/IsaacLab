import pandas as pd
import argparse
import os

def summarize_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"[ERROR] Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)
    output_path = csv_path.replace(".csv", "_summary.txt")
    
    # Define the numerical columns we care about averaging
    numeric_cols = [
        'mean_faces', 'max_faces', 'mean_coverage_percent', 
        'mean_inspection_quality', 'success_rate_percent', 
        'timeout_rate_percent', 'crash_rate_percent', 
        'mean_episode_steps', 'crash_target_percent', 
        'crash_obstacle_percent', 'crash_wall_percent', 
        'crash_floor_percent', 'crash_tipover_percent'
    ]
    
    # Filter out any columns that might not exist in the CSV yet
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    with open(output_path, 'w') as f:
        def write_and_print(text):
            print(text)
            f.write(text + "\n")
            
        write_and_print(f"Evaluation Summary for: {os.path.basename(csv_path)}")
        write_and_print(f"Total Evaluation Runs (Rows): {len(df)}")
        write_and_print("\n" + "="*60)
        write_and_print("1. OVERALL SUMMARY (Mean across ALL objects and seeds)")
        write_and_print("="*60)
        
        overall = df[numeric_cols].mean()
        for col, val in overall.items():
            write_and_print(f"{col:<25}: {val:.2f}")
            
        write_and_print("\n" + "="*60)
        write_and_print("2. QUICK COMPARISON TABLE (Mean per object)")
        write_and_print("="*60)
        
        grouped = df.groupby('target_name')[numeric_cols].mean()
        
        # Select the most important columns to help decide which objects to drop
        important_cols = [c for c in [
            'mean_coverage_percent', 
            'success_rate_percent', 
            'crash_rate_percent', 
            'crash_target_percent',
            'crash_obstacle_percent'
        ] if c in grouped.columns]
        
        write_and_print(grouped[important_cols].to_string())
        
        write_and_print("\n" + "="*60)
        write_and_print("3. DETAILED METRICS PER OBJECT")
        write_and_print("="*60)
        
        for target, row in grouped.iterrows():
            write_and_print(f"\n--- Target: {target} ---")
            # Get the number of seeds completed for this target
            num_seeds = len(df[df['target_name'] == target])
            write_and_print(f"  Evaluations completed: {num_seeds}")
            
            for col in numeric_cols:
                write_and_print(f"  {col:<25}: {row[col]:.2f}")
                
        write_and_print("\n" + "="*60)
        write_and_print("4. MARKDOWN TABLE FORMAT")
        write_and_print("="*60)
        
        md_cols = ['target_name'] + important_cols
        header = "| " + " | ".join(md_cols) + " |"
        separator = "|" + "|".join(["---"] * len(md_cols)) + "|"
        write_and_print(header)
        write_and_print(separator)
        
        for target, row in grouped.iterrows():
            row_str = f"| {target} | "
            row_vals = [f"{row[c]:.2f}" for c in important_cols]
            row_str += " | ".join(row_vals) + " |"
            write_and_print(row_str)
                
    print(f"\n[INFO] A copy of this summary has been saved to: {output_path}")
base_dir = "/home/tosin/Documents/GitHub/IsaacLab/data/ablation/"
albaltion_file = "No_GRU"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a summary of the ablation evaluation CSV.")
    parser.add_argument("--csv", type=str, 
                        default=f"{base_dir}{albaltion_file}"+".csv",
                        help="Path to the ablation CSV file")
    args = parser.parse_args()
    
    summarize_csv(args.csv)

