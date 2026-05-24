import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import gtsam
from gtsam import (
    Pose2, NonlinearFactorGraph, Values,
    LevenbergMarquardtOptimizer,
    PriorFactorPose2, BetweenFactorPose2,
)

from scan_matching import (
    scan_match_dataset, load_hokuyo, make_se2_T4, inv_T4,
    scan_to_xy_lidar, apply_T_to_points_2d, wrap_angle,
)
from icp_warm_up.icp import icp_yaw_only_original_kabsch
from occupancy_mapping import build_occupancy_map, plot_occupancy
from color_map import build_color_map, plot_color_map

DATA_ROOT = "../data"
RGBD_ROOT = "../data/dataRGBD"

T_B_L = np.eye(4, dtype=np.float64)
T_B_L[:3, 3] = np.array([0.30183, 0.0, 0.51435])

FIXED_INTERVAL     = 10
PROXIMITY_RADIUS   = 2.0
MIN_POSE_SEP       = 20
ICP_FITNESS_THRESH = 0.01
MAX_CORR_DIST      = 0.3
MAX_ICP_ITER       = 50


def X(k):
    return gtsam.symbol("x", k)


ODOM_NOISE  = gtsam.noiseModel.Isotropic.Sigma(3, 0.05)
LOOP_NOISE  = gtsam.noiseModel.Isotropic.Sigma(3, 0.10)
PRIOR_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)



def sane_loop_closure(T_rel, max_dist=3.0, max_angle=1.0):
    dist = np.sqrt(T_rel[0,3]**2 + T_rel[1,3]**2)
    angle = abs(np.arctan2(T_rel[1,0], T_rel[0,0]))
    return dist < max_dist and angle < max_angle

def T4_to_Pose2(T):
    return Pose2(float(T[0, 3]), float(T[1, 3]),
                 float(np.arctan2(T[1, 0], T[0, 0])))


def scan_to_body(ranges_col, max_pts=800, r_cap=12.0):
    pts_l = scan_to_xy_lidar(ranges_col, max_pts=max_pts, r_cap=r_cap)
    pts_b = apply_T_to_points_2d(T_B_L, pts_l)
    return np.c_[pts_b, np.zeros(len(pts_b))].astype(np.float64)


def icp_between(scan_a, scan_b, init_T):
    T_rel, mse = icp_yaw_only_original_kabsch(
        scan_b, scan_a,
        init_T=init_T.astype(np.float64),
        max_iter=MAX_ICP_ITER,
        tol=1e-6,
        max_corr_dist=MAX_CORR_DIST,
        min_inliers=30,
    )
    return T_rel, mse


def build_pose_graph(ds):
    enc_path = f"{DATA_ROOT}/Encoders{ds}.npz"
    imu_path = f"{DATA_ROOT}/Imu{ds}.npz"
    hok_path = f"{DATA_ROOT}/Hokuyo{ds}.npz"

    _, icp_xyt = scan_match_dataset(
        enc_path, imu_path, hok_path,
        title=f"ICP {ds}",
        max_corr_dist=0.6,
        max_iter=50,
        submap_window=10,
    )
    icp_x, icp_y, icp_th = icp_xyt
    N = len(icp_x)

    _, ranges = load_hokuyo(hok_path)

    graph  = NonlinearFactorGraph()
    values = Values()

    prior_pose = Pose2(float(icp_x[0]), float(icp_y[0]), float(icp_th[0]))
    graph.add(PriorFactorPose2(X(0), prior_pose, PRIOR_NOISE))

    for k in range(N):
        values.insert(X(k), Pose2(float(icp_x[k]),
                                  float(icp_y[k]),
                                  float(icp_th[k])))

    print(f"  Adding {N-1} odometry factors ...")
    t0 = time.time()
    for k in range(1, N):
        T_prev = make_se2_T4(icp_x[k-1], icp_y[k-1], icp_th[k-1])
        T_curr = make_se2_T4(icp_x[k],   icp_y[k],   icp_th[k])
        T_rel  = inv_T4(T_prev) @ T_curr
        graph.add(BetweenFactorPose2(X(k-1), X(k),
                                     T4_to_Pose2(T_rel), ODOM_NOISE))
    print(f"  Odometry factors done in {time.time()-t0:.1f}s")

    scans = {}
    def get_scan(k):
        if k not in scans:
            scans[k] = scan_to_body(ranges[:, k])
        return scans[k]

    fixed_count = 0
    prox_count  = 0
    rejected    = 0

    total_fixed = N // FIXED_INTERVAL
    print(f"  Fixed-interval loop closure (every {FIXED_INTERVAL} poses, ~{total_fixed} pairs) ...")
    t0 = time.time()
    for i, k in enumerate(range(FIXED_INTERVAL, N, FIXED_INTERVAL)):
        j = k - FIXED_INTERVAL
        T_j    = make_se2_T4(icp_x[j], icp_y[j], icp_th[j])
        T_k    = make_se2_T4(icp_x[k], icp_y[k], icp_th[k])
        init_T = inv_T4(T_j) @ T_k
        T_rel, mse = icp_between(get_scan(j), get_scan(k), init_T)
        if mse < ICP_FITNESS_THRESH and sane_loop_closure(T_rel):
            graph.add(BetweenFactorPose2(X(j), X(k),
                                         T4_to_Pose2(T_rel), LOOP_NOISE))
            fixed_count += 1
        else:
            rejected += 1
        if (i+1) % 100 == 0:
            print(f"    {i+1}/{total_fixed} pairs, {fixed_count} accepted, {time.time()-t0:.0f}s elapsed")
    print(f"  Fixed-interval done: {fixed_count} factors in {time.time()-t0:.1f}s")

    print(f"  Proximity-based loop closure (radius={PROXIMITY_RADIUS}m) ...")
    xy      = np.stack([icp_x, icp_y], axis=1)
    tree    = cKDTree(xy)
    checked = set()
    # Only check every FIXED_INTERVAL-th pose to keep runtime manageable
    candidate_ks = range(0, N, FIXED_INTERVAL)
    for k in candidate_ks:
        for j in tree.query_ball_point(xy[k], PROXIMITY_RADIUS):
            if abs(k - j) < MIN_POSE_SEP:
                continue
            pair = (min(j, k), max(j, k))
            if pair in checked:
                continue
            checked.add(pair)
            T_j    = make_se2_T4(icp_x[j], icp_y[j], icp_th[j])
            T_k    = make_se2_T4(icp_x[k], icp_y[k], icp_th[k])
            init_T = inv_T4(T_j) @ T_k
            T_rel, mse = icp_between(get_scan(j), get_scan(k), init_T)
            if mse < ICP_FITNESS_THRESH and sane_loop_closure(T_rel):
                graph.add(BetweenFactorPose2(X(j), X(k),
                                             T4_to_Pose2(T_rel), LOOP_NOISE))
                prox_count += 1
            else:
                rejected += 1
        if k % 500 == 0:
            print(f"    proximity: {k}/{N} poses checked, {prox_count} found so far ...")

    print(f"  Factors: {N-1} odom + {fixed_count} fixed + "
          f"{prox_count} proximity ({rejected} rejected by MSE gate)")

    print(f"  Optimizing graph ({graph.size()} factors, {N} nodes) ...")
    t0 = time.time()
    result = LevenbergMarquardtOptimizer(graph, values).optimize()
    print(f"  Optimization done in {time.time()-t0:.1f}s")

    opt_x  = np.array([result.atPose2(X(k)).x()     for k in range(N)])
    opt_y  = np.array([result.atPose2(X(k)).y()     for k in range(N)])
    opt_th = np.array([result.atPose2(X(k)).theta() for k in range(N)])

    return icp_x, icp_y, icp_th, opt_x, opt_y, opt_th


def plot_trajectories(icp_x, icp_y, opt_x, opt_y, ds):
    plt.figure(figsize=(10, 10))
    plt.plot(icp_x, icp_y, 'b-', linewidth=0.8, label="Before PGO (ICP)")
    plt.plot(opt_x, opt_y, 'r-', linewidth=0.8, label="After PGO")
    plt.axis("equal"); plt.grid(True); plt.legend()
    plt.title(f"Dataset {ds}: Trajectory Before vs After PGO")
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.tight_layout()
    plt.savefig(f"trajectory_pgo_{ds}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    for ds in ["20", "21"]:
        print(f"\n{'='*60}\nDataset {ds}\n{'='*60}")

        icp_x, icp_y, icp_th, opt_x, opt_y, opt_th = build_pose_graph(ds)

        plot_trajectories(icp_x, icp_y, opt_x, opt_y, ds)

        hok_path = f"{DATA_ROOT}/Hokuyo{ds}.npz"
        kin_path = f"{DATA_ROOT}/Kinect{ds}.npz"

        print("  Building optimized occupancy map ...")
        occ_grid = build_occupancy_map(hok_path, opt_x, opt_y, opt_th)
        plot_occupancy(occ_grid, opt_x, opt_y,
                       title=f"Dataset {ds} Optimized Occupancy Map")

        print("  Building optimized color map ...")
        col_grid = build_color_map(
            ds          = ds,
            icp_x       = opt_x,
            icp_y       = opt_y,
            icp_th      = opt_th,
            kinect_npz  = kin_path,
            rgb_dir     = f"{RGBD_ROOT}/RGB{ds}",
            disp_dir    = f"{RGBD_ROOT}/Disparity{ds}",
            res         = 0.05,
            floor_z_tol = 0.08,
            step        = 3,
        )
        plot_color_map(col_grid, opt_x, opt_y,
                       title=f"Dataset {ds} Optimized Color Map")
