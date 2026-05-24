import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scan_matching import scan_match_dataset, make_se2_T4

DATA_ROOT = "../data"
RGBD_ROOT = "../data/dataRGBD"

FX = FY = 585.05
CX = 242.94
CY = 315.84
IMG_H, IMG_W = 480, 640

CAM_T     = np.array([0.18, 0.005, 0.36])
CAM_PITCH = 0.36
CAM_YAW   = 0.021

COLOR_SNAPSHOT_STEPS = 5 


def _rot_y(a):
    return np.array([[ np.cos(a), 0, np.sin(a)],
                     [0,          1, 0         ],
                     [-np.sin(a), 0, np.cos(a)]])

def _rot_z(a):
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0,          0,         1]])

def build_T_body_cam():
    R_base = np.array([[0,  0, 1],
                       [-1, 0, 0],
                       [0, -1, 0]], dtype=float)
    R = _rot_z(CAM_YAW) @ _rot_y(CAM_PITCH) @ R_base
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = CAM_T
    return T

T_BODY_CAM = build_T_body_cam()
_ROWS, _COLS = np.mgrid[0:IMG_H, 0:IMG_W]


def load_disparity(path):
    img = np.array(Image.open(path))
    if img.ndim == 2:
        return img.astype(np.float32)
    return img[:, :, 0].astype(np.float32) * 256 + img[:, :, 1].astype(np.float32)

def disparity_to_depth(disp):
    dd    = -0.00304 * disp + 3.31
    valid = dd > 0
    depth = np.where(valid, 1.03 / dd, 0.0)
    return depth.astype(np.float32), dd.astype(np.float32)

def backproject_to_body(depth, dd):
    valid    = (depth > 0.5) & (depth < 4.0)
    Z        = depth[valid]
    X        = (_COLS[valid] - CX) * Z / FX
    Y        = (_ROWS[valid] - CY) * Z / FY
    ones     = np.ones(len(Z), dtype=np.float32)
    pts_cam  = np.stack([X, Y, Z, ones], axis=1)
    pts_body = (T_BODY_CAM @ pts_cam.T).T
    return pts_body[:, :3], _ROWS[valid], _COLS[valid], dd[valid]

def get_rgb_for_disparity_pixels(rgb_img, disp_rows, disp_cols, dd_vals):
    i     = disp_rows.astype(np.float32)
    j     = disp_cols.astype(np.float32)
    rgb_i = np.round((526.37 * i + 19276 - 7877.07 * dd_vals) / 585.051).astype(int)
    rgb_j = np.round((526.37 * j + 16662)                      / 585.051).astype(int)
    rgb_i = np.clip(rgb_i, 0, IMG_H - 1)
    rgb_j = np.clip(rgb_j, 0, IMG_W - 1)
    return rgb_img[rgb_i, rgb_j, :]


class ColorGrid:
    def __init__(self, res=0.05, margin=2.0,
                 x_min=-5, x_max=25, y_min=-10, y_max=25):
        self.res   = res
        self.x_min = x_min - margin
        self.y_min = y_min - margin
        self.x_max = x_max + margin
        self.y_max = y_max + margin
        self.W       = int(np.ceil((self.x_max - self.x_min) / res))
        self.H       = int(np.ceil((self.y_max - self.y_min) / res))
        self.rgb_sum = np.zeros((self.H, self.W, 3), dtype=np.float64)
        self.count   = np.zeros((self.H, self.W),    dtype=np.int32)

    def world_to_cell(self, wx, wy):
        c = ((wx - self.x_min) / self.res).astype(int)
        r = ((wy - self.y_min) / self.res).astype(int)
        return r, c

    def add_floor_points(self, world_xyz, colors_rgb, floor_z_tol=0.08):
        on_floor  = np.abs(world_xyz[:, 2]) < floor_z_tol
        wx        = world_xyz[on_floor, 0]
        wy        = world_xyz[on_floor, 1]
        rgb       = colors_rgb[on_floor].astype(np.float64)
        r, c      = self.world_to_cell(wx, wy)
        in_bounds = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W)
        r, c, rgb = r[in_bounds], c[in_bounds], rgb[in_bounds]
        np.add.at(self.rgb_sum, (r, c, 0), rgb[:, 0])
        np.add.at(self.rgb_sum, (r, c, 1), rgb[:, 1])
        np.add.at(self.rgb_sum, (r, c, 2), rgb[:, 2])
        np.add.at(self.count,   (r, c),    1)

    def get_image(self):
        img     = np.full((self.H, self.W, 3), 128, dtype=np.uint8)
        visited = self.count > 0
        img[visited] = (self.rgb_sum[visited] /
                        self.count[visited, np.newaxis]).astype(np.uint8)
        return img


def _save_color_snapshot(grid, icp_x, icp_y, k_disp, N_disp, ds, out_dir):
    pct = int(round(100.0 * k_disp / N_disp))
    img = grid.get_image()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, origin="lower",
              extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max])
    ax.plot(icp_x, icp_y, 'r-', linewidth=0.5, label="ICP Trajectory")
    ax.set_title(f"Dataset {ds} — Texture Map ({pct}% complete)")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"colormap_ds{ds}_{pct:03d}pct.png"), dpi=100)
    plt.close()


def build_color_map(ds, icp_x, icp_y, icp_th,
                    kinect_npz, rgb_dir, disp_dir,
                    res=0.05, floor_z_tol=0.08, step=3,
                    snapshot_dir="snapshots"):

    os.makedirs(snapshot_dir, exist_ok=True)

    kin    = np.load(kinect_npz)
    t_rgb  = kin["rgb_time_stamps"]
    t_disp = kin["disparity_time_stamps"]
    t_icp  = np.load(f"{DATA_ROOT}/Hokuyo{ds}.npz")["time_stamps"].astype(float)
    N_disp = len(t_disp)

    grid = ColorGrid(res=res,
                     x_min=icp_x.min(), x_max=icp_x.max(),
                     y_min=icp_y.min(), y_max=icp_y.max())

    checkpoints = set(
        int(N_disp * p / 100)
        for p in range(COLOR_SNAPSHOT_STEPS, 101, COLOR_SNAPSHOT_STEPS)
    )

    print(f"Building color map for dataset {ds} ({N_disp} disparity frames) ...")
    processed = 0

    for k_disp in range(0, N_disp, step):
        disp_path = os.path.join(disp_dir, f"disparity{ds}_{k_disp+1:04d}.png")
        if not os.path.exists(disp_path):
            disp_path = os.path.join(disp_dir, f"disparity{ds}_{k_disp+1}.png")
        if not os.path.exists(disp_path):
            continue

        depth, dd = disparity_to_depth(load_disparity(disp_path))

        t_d      = t_disp[k_disp]
        k_rgb    = int(np.argmin(np.abs(t_rgb - t_d)))
        rgb_path = os.path.join(rgb_dir, f"rgb{ds}_{k_rgb+1:04d}.png")
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(rgb_dir, f"rgb{ds}_{k_rgb+1}.png")
        if not os.path.exists(rgb_path):
            continue
        rgb_img = np.array(Image.open(rgb_path))

        pts_body, d_rows, d_cols, dd_vals = backproject_to_body(depth, dd)
        colors    = get_rgb_for_disparity_pixels(rgb_img, d_rows, d_cols, dd_vals)
        k_icp     = int(np.argmin(np.abs(t_icp - t_d)))
        T_wb      = make_se2_T4(icp_x[k_icp], icp_y[k_icp], icp_th[k_icp])
        ones      = np.ones((len(pts_body), 1), dtype=np.float32)
        pts_world = (T_wb @ np.hstack([pts_body, ones]).T).T[:, :3]

        grid.add_floor_points(pts_world, colors, floor_z_tol=floor_z_tol)
        processed += 1

        if processed % 100 == 0:
            print(f"  {k_disp}/{N_disp} frames processed ...")

        if k_disp in checkpoints:
            _save_color_snapshot(grid, icp_x, icp_y, k_disp, N_disp, ds, snapshot_dir)

    print(f"Done. {processed} frames used.")
    return grid


def plot_color_map(grid, icp_x, icp_y, title="Color Floor Map"):
    img = grid.get_image()
    plt.figure(figsize=(12, 12))
    plt.imshow(img, origin="lower",
               extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max])
    plt.plot(icp_x, icp_y, 'r-', linewidth=0.6, label="ICP Trajectory")
    plt.title(title)
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"color_map_{title.split()[-1]}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    for ds in ["20", "21"]:
        _, icp_xyt = scan_match_dataset(
            f"{DATA_ROOT}/Encoders{ds}.npz",
            f"{DATA_ROOT}/Imu{ds}.npz",
            f"{DATA_ROOT}/Hokuyo{ds}.npz",
            title=f"ICP {ds}",
            max_corr_dist=0.6,
            max_iter=30,
            submap_window=10,
            snapshot_dir=f"snapshots/ds{ds}",
        )
        icp_x, icp_y, icp_th = icp_xyt

        grid = build_color_map(
            ds          = ds,
            icp_x       = icp_x,
            icp_y       = icp_y,
            icp_th      = icp_th,
            kinect_npz  = f"{DATA_ROOT}/Kinect{ds}.npz",
            rgb_dir     = f"{RGBD_ROOT}/RGB{ds}",
            disp_dir    = f"{RGBD_ROOT}/Disparity{ds}",
            res         = 0.05,
            floor_z_tol = 0.08,
            step        = 3,
            snapshot_dir= f"snapshots/ds{ds}",
        )

        plot_color_map(grid, icp_x, icp_y, title=f"Dataset {ds}")
