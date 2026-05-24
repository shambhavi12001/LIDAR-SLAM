import os
import numpy as np
import matplotlib.pyplot as plt
from main import integrate_enc_imu
from icp_warm_up.icp import icp_yaw_only_original_kabsch

icp_yaw_only = icp_yaw_only_original_kabsch

NUM_BEAMS = 1081
ANGLE_MIN = -135.0 * np.pi / 180.0
ANGLE_MAX  =  135.0 * np.pi / 180.0
ANGLES = np.linspace(ANGLE_MIN, ANGLE_MAX, NUM_BEAMS)
R_MIN, R_MAX = 0.15, 30.0

SNAPSHOT_STEPS = 5 

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def load_hokuyo(npz_path: str):
    hok = np.load(npz_path)
    t, ranges = hok["time_stamps"].astype(float), hok["ranges"].astype(float)
    if ranges.shape[0] != NUM_BEAMS and ranges.shape[1] == NUM_BEAMS:
        ranges = ranges.T
    return t, ranges

def interp_pose(t_ref, x_ref, y_ref, th_ref, t_query):
    return (np.interp(t_query, t_ref, x_ref),
            np.interp(t_query, t_ref, y_ref),
            np.interp(t_query, t_ref, th_ref))

def yaw_R(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

def make_se2_T4(dx: float, dy: float, dth: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = yaw_R(dth)
    T[:3, 3]  = np.array([dx, dy, 0.0], dtype=np.float64)
    return T

def scan_to_xy_lidar(r, max_pts=1200, r_cap=None):
    r = r.astype(float)
    rmax = min(R_MAX, float(r_cap)) if r_cap else R_MAX
    valid = np.isfinite(r) & (r > R_MIN) & (r < rmax)
    pts = np.stack([r[valid] * np.cos(ANGLES[valid]),
                    r[valid] * np.sin(ANGLES[valid])], axis=1)
    if pts.shape[0] > max_pts:
        pts = pts[np.random.choice(pts.shape[0], max_pts, replace=False)]
    return pts

def apply_T_to_points_2d(T4: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    pts3 = np.c_[pts2, np.zeros(len(pts2))].astype(np.float64)
    return ((T4[:3, :3] @ pts3.T).T + T4[:3, 3])[:, :2]

def inv_T4(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3]  = -(T[:3, :3].T @ T[:3, 3])
    return Ti

def _save_trajectory_snapshot(Xi, Yi, k, N, ds, out_dir):
    pct = int(round(100.0 * k / N))
    x = Xi[:k+1] - Xi[0]
    y = Yi[:k+1] - Yi[0]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x, y, 'b-', linewidth=0.8)
    ax.scatter(x[-1], y[-1], c='red', s=50, zorder=5, label='Current pose')
    ax.set_title(f"Dataset {ds} — ICP Trajectory ({pct}% complete)")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.axis("equal"); ax.grid(True); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"trajectory_ds{ds}_{pct:03d}pct.png"), dpi=100)
    plt.close()

def scan_match_dataset(enc_path, imu_path, hokuyo_path, title="",
                       max_corr_dist=0.30, max_iter=50, tol=1e-6,
                       max_pts=1200, r_cap=12.0, submap_window=10,
                       snapshot_dir="snapshots"):

    os.makedirs(snapshot_dir, exist_ok=True)
    ds = title.split()[-1] if title else "XX"

    T_b_l = np.eye(4, dtype=np.float64)
    T_b_l[:3, 3] = np.array([0.30183, 0.0, 0.51435], dtype=np.float64)

    t_enc, x_enc, y_enc, th_enc, _, _ = integrate_enc_imu(enc_path, imu_path)
    t_lid, ranges = load_hokuyo(hokuyo_path)
    N = ranges.shape[1]

    x_o, y_o, th_o = interp_pose(t_enc, x_enc, y_enc, th_enc, t_lid)
    Xo, Yo = x_o - x_o[0], y_o - y_o[0]

    T_world = make_se2_T4(x_o[0], y_o[0], th_o[0])
    Xi  = np.zeros(N)
    Yi  = np.zeros(N)
    Thi = np.zeros(N)
    Xi[0], Yi[0] = T_world[0, 3], T_world[1, 3]
    Thi[0]       = th_o[0]

    s0_world     = apply_T_to_points_2d(T_world @ T_b_l,
                       scan_to_xy_lidar(ranges[:, 0], max_pts=max_pts, r_cap=r_cap))
    submap_world = [s0_world]

    checkpoints = set(int(N * p / 100) for p in range(SNAPSHOT_STEPS, 101, SNAPSHOT_STEPS))

    print(f"Starting {title}...")
    for k in range(1, N):
        curr_l2    = scan_to_xy_lidar(ranges[:, k], max_pts=max_pts, r_cap=r_cap)
        curr_body3 = np.c_[apply_T_to_points_2d(T_b_l, curr_l2),
                           np.zeros(len(curr_l2))].astype(np.float64)

        T_prev  = make_se2_T4(x_o[k-1], y_o[k-1], th_o[k-1])
        T_curr  = make_se2_T4(x_o[k],   y_o[k],   th_o[k])
        init_T  = inv_T4(T_prev) @ T_curr

        submap_pts_world = np.vstack(submap_world)
        submap_prev3     = np.c_[
            apply_T_to_points_2d(inv_T4(T_world), submap_pts_world),
            np.zeros(len(submap_pts_world))
        ].astype(np.float64)

        T_rel, mse = icp_yaw_only(curr_body3, submap_prev3, init_T=init_T,
                                  max_iter=max_iter, tol=tol,
                                  max_corr_dist=max_corr_dist)

        dth_o = wrap_angle(th_o[k] - th_o[k-1])
        yaw_i = np.arctan2(T_rel[1, 0], T_rel[0, 0])
        T_use = init_T if (mse > 1e-2 or abs(wrap_angle(yaw_i - dth_o)) > 0.25) else T_rel

        T_world = T_world @ T_use
        Xi[k]   = T_world[0, 3]
        Yi[k]   = T_world[1, 3]
        Thi[k]  = np.arctan2(T_world[1, 0], T_world[0, 0])

        curr_world = apply_T_to_points_2d(T_world @ T_b_l, curr_l2)
        submap_world.append(curr_world)
        if len(submap_world) > submap_window:
            submap_world.pop(0)

        if k in checkpoints:
            _save_trajectory_snapshot(Xi, Yi, k, N, ds, snapshot_dir)

    return (Xo, Yo), (Xi, Yi, Thi)


def plot_compare(odom_xy, icp_xyt, title):
    plt.figure(figsize=(10, 6))
    plt.plot(odom_xy[0], odom_xy[1], 'r--', label="Odometry")
    plt.plot(icp_xyt[0] - icp_xyt[0][0],
             icp_xyt[1] - icp_xyt[1][0], 'b-', label="ICP-Refined")
    plt.axis("equal"); plt.grid(True); plt.legend(); plt.title(title)
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for i, ds in enumerate(["20", "21"]):
        odom, icp = scan_match_dataset(
            f"../data/Encoders{ds}.npz",
            f"../data/Imu{ds}.npz",
            f"../data/Hokuyo{ds}.npz",
            title=f"Dataset {ds}",
            max_corr_dist=0.60,
            max_iter=50,
            r_cap=15.0,
            submap_window=10,
            snapshot_dir=f"snapshots/ds{ds}",
        )
        ax[i].plot(odom[0], odom[1], 'r--', label="Odometry")
        ax[i].plot(icp[0] - icp[0][0], icp[1] - icp[1][0], 'b-', label="ICP-Refined")
        ax[i].axis("equal"); ax[i].grid(True)
        ax[i].set_title(f"Dataset {ds}")
        ax[i].set_xlabel("X [m]"); ax[i].set_ylabel("Y [m]")
        ax[i].legend()
    plt.tight_layout()
    plt.show()
