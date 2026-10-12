def _subdivided_faces(base_faces: int, subdivisions: int) -> int:
    return base_faces * 4**subdivisions
a = _subdivided_faces(28, 3)
print(a)
print(a * 16 // 28)
# ================================================
# EVALUATION RESULTS (512 Episodes)
# ==================================================
# Mean Faces Discovered: 1520.56
# Std Deviation:         115.64
# Min Faces:             544
# Max Faces:             1664
# Mean Coverage:          50.69%
# Std Coverage:           3.85%
# Per-target results:
#   tessellated_t_block: 512 episodes | Faces: 1520.56 | Coverage: 50.69%
# Mean Crashes: 0.09
# Median Crashes: 0.00
# Episodes With Crash: 9.18%
# Total Terminated Due to Crash: 47 / 512
# Mean Faces Discovered (NO CRASHES): 1534.74
# Crash Sources (primary base_link contact):
#   inspection_target/tessellated_t_block: 46 (97.87% of crashes)
#   warehouse: 0 (0.00% of crashes)
#   unattributed: 1 (2.13% of crashes)