import os
import numpy as np
import matplotlib.pyplot as plt
from scan_matching import scan_match_dataset, load_hokuyo, make_se2_T4

NUM_BEAMS = 1081
ANGLE_MIN  = -135.0 * np.pi / 180.0
ANGLE_MAX  =  135.0 * np.pi / 180.0
ANGLES     = np.linspace(ANGLE_MIN, ANGLE_MAX, NUM_BEAMS)
R_MIN      = 0.15
R_MAX      = 30.0

LOG_ODD_OCC  =  0.85
LOG_ODD_FREE = -0.40
LOG_MAX      =  10.0
LOG_MIN      = -10.0

SNAPSHOT_STEPS = 5


def bresenham2d(x0, y0, x1, y1):
    cells = []
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx:  err += dx; y0 += sy
    return cells


class OccupancyGrid:
    def __init__(self, res=0.05, margin=2.0, x_min=-20, x_max=20, y_min=-20, y_max=20):
        self.res   = res
        self.x_min = x_min - margin
        self.y_min = y_min - margin
        self.x_max = x_max + margin
        self.y_max = y_max + margin
        self.W     = int(np.ceil((self.x_max - self.x_min) / res))
        self.H     = int(np.ceil((self.y_max - self.y_min) / res))
        self.log_odds = np.zeros((self.H, self.W), dtype=np.float32)

    def world_to_cell(self, x, y):
        c = int((x - self.x_min) / self.res)
        r = int((y - self.y_min) / self.res)
        return r, c

    def in_bounds(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W

    def update(self, sensor_xy, hit_xys):
        r0, c0 = self.world_to_cell(*sensor_xy)
        for hx, hy in hit_xys:
            r1, c1 = self.world_to_cell(hx, hy)
            ray = bresenham2d(c0, r0, c1, r1)
            for (c, r) in ray[:-1]:
                if self.in_bounds(r, c):
                    self.log_odds[r, c] = np.clip(
                        self.log_odds[r, c] + LOG_ODD_FREE, LOG_MIN, LOG_MAX)
            if self.in_bounds(r1, c1):
                self.log_odds[r1, c1] = np.clip(
                    self.log_odds[r1, c1] + LOG_ODD_OCC, LOG_MIN, LOG_MAX)

    def get_map(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))


def _save_occupancy_snapshot(grid, traj_x, traj_y, k, N, ds, out_dir):
    pct     = int(round(100.0 * k / N))
    prob    = grid.get_map()
    display = np.full(prob.shape, 0.5)
    display[prob > 0.6] = 0.0
    display[prob < 0.4] = 1.0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(display, cmap="gray", origin="lower",
              extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max])
    ax.plot(traj_x[:k+1], traj_y[:k+1], 'r-', linewidth=0.5, label="ICP Trajectory")
    ax.scatter(traj_x[k], traj_y[k], c='blue', s=40, zorder=5, label='Current pose')
    ax.set_title(f"Dataset {ds} — Occupancy Map ({pct}% complete)")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"occupancy_ds{ds}_{pct:03d}pct.png"), dpi=100)
    plt.close()
    print(f"  Saved occupancy snapshot: occupancy_ds{ds}_{pct:03d}pct.png")


def build_occupancy_map(hokuyo_path, icp_traj_x, icp_traj_y, icp_traj_th,
                        res=0.05, ds="XX", snapshot_dir="snapshots"):

    os.makedirs(snapshot_dir, exist_ok=True)

    T_b_l = np.eye(4)
    T_b_l[:3, 3] = np.array([0.30183, 0.0, 0.51435])

    _, ranges = load_hokuyo(hokuyo_path)
    N = ranges.shape[1]

    grid = OccupancyGrid(res=res,
                         x_min=icp_traj_x.min(), x_max=icp_traj_x.max(),
                         y_min=icp_traj_y.min(), y_max=icp_traj_y.max())

    # Checkpoints at every SNAPSHOT_STEPS % of scans
    checkpoints = set(
        int(N * p / 100)
        for p in range(SNAPSHOT_STEPS, 101, SNAPSHOT_STEPS)
    )

    for k in range(0, N, 2):
        T_wb = make_se2_T4(icp_traj_x[k], icp_traj_y[k], icp_traj_th[k])
        T_wl         = T_wb @ T_b_l
        lidar_origin = T_wl[:2, 3]

        r     = ranges[:, k]
        valid = np.isfinite(r) & (r > R_MIN) & (r < 8.0)
        aa    = ANGLES[valid]
        rr    = r[valid]

        pts_l = np.stack([rr * np.cos(aa),
                          rr * np.sin(aa),
                          np.zeros_like(rr),
                          np.ones_like(rr)], axis=1)
        pts_w = (T_wl @ pts_l.T).T[:, :2]

        grid.update(lidar_origin, pts_w)

        if k % 1000 == 0:
            print(f"  Processed {k}/{N} scans...")

        if k in checkpoints:
            _save_occupancy_snapshot(grid, icp_traj_x, icp_traj_y, k, N, ds, snapshot_dir)

    return grid


def plot_occupancy(grid, traj_x, traj_y, title="Final SLAM Map"):
    prob    = grid.get_map()
    display = np.full(prob.shape, 0.5)
    display[prob > 0.6] = 0.0
    display[prob < 0.4] = 1.0

    plt.figure(figsize=(12, 12))
    plt.imshow(display, cmap="gray", origin="lower",
               extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max])
    plt.plot(traj_x, traj_y, 'r-', linewidth=0.5, label="ICP Trajectory")
    plt.title(title)
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"occupancy_{title.replace(' ', '_')}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    for ds in ["20", "21"]:
        enc_path = f"../data/Encoders{ds}.npz"
        imu_path = f"../data/Imu{ds}.npz"
        hok_path = f"../data/Hokuyo{ds}.npz"

        print(f"\n--- Processing Dataset {ds} ---")

        odom_xy, icp_xyt = scan_match_dataset(
            enc_path, imu_path, hok_path,
            title=f"ICP {ds}",
            max_corr_dist=0.6,
            max_iter=30,
            submap_window=10,
            snapshot_dir=f"snapshots/ds{ds}",
        )

        icp_x, icp_y, icp_th = icp_xyt[0], icp_xyt[1], icp_xyt[2]

        grid = build_occupancy_map(
            hok_path, icp_x, icp_y, icp_th,
            ds=ds,
            snapshot_dir=f"snapshots/ds{ds}",
        )
        plot_occupancy(grid, icp_x, icp_y, title=f"Dataset {ds} Occupancy Map")
