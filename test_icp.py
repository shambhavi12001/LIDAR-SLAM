import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
from icp import icp_with_yaw_grid_search


for model_name in ["drill", "liq_container"]:

    print("\nRunning ICP for:", model_name)

    target_pc = read_canonical_model(model_name)

    for i in range(4):

        source_pc = load_pc(model_name, i)

        pose, mse = icp_with_yaw_grid_search(
            source_pc,
            target_pc,
            yaw_bins=36,
            max_iter=50,
            max_corr_dist=0.20
        )

        print(f"{model_name} scan {i}: MSE = {mse:.6f}")

        visualize_icp_result(source_pc, target_pc, pose)
