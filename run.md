conda activate env_isaaclab
cd /home/tosin/Documents/GitHub/IsaacLab
scripts/reinforcement_learning/skrl/train_cont_gru_resnet_discrete_mu.py
Bug might arrise
conda install -c conda-forge gcc=12 -y



python source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/utils/generate_object_pointcloud.py --dataset_key rubiks_cube --output data/point_clouds/dataset/rubiks_cube.ply


#reconstruct point cloud from depth data
python source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/utils/convert_depth_to_ply.py \
    --data_path data/recorded_depth_data_eval/rubiks_cube \
    --output data/point_clouds/eval/rubiks_cube_recon.ply
# Genrate Ground truth  

cd /home/tosin/Documents/GitHub/IsaacLab

python3 \
source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/utils/generate_object_pointcloud.py \
--dataset_key ur10_mount \
--num_points 100000 \
--ray_backend open3d \
--num_views 128 \
--ray_oversample_factor 8 \
--ray_batch_size 1024 \
--output data/point_clouds/dataset/ur10_mount.ply \
--headless
# compare to ground truth
python source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/utils/compare_pointclouds.py \
    --source data/point_clouds/eval/rubiks_cube_recon.ply \
    --target data/point_clouds/dataset/rubiks_cube.ply

python3 \
source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/utils/view_pointcloud.py \
data/recorded_depth_data_eval/ur10_mount/run_2026-08-24_16-02-42/pointcloud_evaluation/episode_00001/ground_truth_coverage.ply \
data/recorded_depth_data_eval/ur10_mount/run_2026-08-24_16-02-42/pointcloud_evaluation/episode_00001/reconstruction_aligned.ply \
--backend open3d


# View point cloud
python3 source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/utils/view_pointcloud.py \
  path/to/reconstruction_aligned.ply \
  --backend open3d