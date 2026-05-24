import numpy as np
from scipy.spatial import cKDTree


def _yaw_R(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _best_fit_yaw(P: np.ndarray, Q: np.ndarray):
    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)

    P_centered = P - mu_P
    Q_centered = Q - mu_Q

    H = P_centered[:, :2].T @ Q_centered[:, :2]
    U, _, Vt = np.linalg.svd(H)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:
        Vt[1] *= -1
        R2 = Vt.T @ U.T

    theta = np.arctan2(R2[1, 0], R2[0, 0])
    R = _yaw_R(theta)
    t = mu_Q - R @ mu_P
    return R, t


def icp_yaw_only_original_kabsch(
    source: np.ndarray,
    target: np.ndarray,
    init_T: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
    max_corr_dist: float = 0.20,
    min_inliers: int = 30,
):
    P0 = source.astype(np.float64).copy()
    Q = target.astype(np.float64).copy()

    T = np.eye(4, dtype=np.float64)
    if init_T is not None:
        T = init_T.astype(np.float64).copy()

    tree = cKDTree(Q)
    prev_mse = np.inf

    for _ in range(max_iter):
        P = (T[:3, :3] @ P0.T).T + T[:3, 3]

        dists, idx = tree.query(P, k=1)
        valid = dists < max_corr_dist
        if int(valid.sum()) < min_inliers:
            break

        R, t = _best_fit_yaw(P0[valid], Q[idx[valid]])

        T_new = np.eye(4, dtype=np.float64)
        T_new[:3, :3] = R
        T_new[:3, 3] = t
        T = T_new

        P_new = (T[:3, :3] @ P0.T).T + T[:3, 3]
        d2, _ = tree.query(P_new, k=1)
        in2 = d2 < max_corr_dist
        mse = float(np.mean(d2[in2] ** 2)) if int(in2.sum()) >= min_inliers else float("inf")

        if abs(prev_mse - mse) < tol:
            prev_mse = mse
            break
        prev_mse = mse

    return T, prev_mse


def icp_with_yaw_grid_search(
    source: np.ndarray,
    target: np.ndarray,
    yaw_bins: int = 72,
    max_iter: int = 50,
    max_corr_dist: float = 0.20,
):
    S = source.astype(np.float64)
    Z = target.astype(np.float64)

    mu_S = S.mean(axis=0)
    mu_Z = Z.mean(axis=0)

    best_T = None
    best_mse = np.inf

    for k in range(int(yaw_bins)):
        yaw = 2.0 * np.pi * k / float(yaw_bins)
        R0 = _yaw_R(yaw)
        t0 = mu_Z - R0 @ mu_S

        init_T = np.eye(4, dtype=np.float64)
        init_T[:3, :3] = R0
        init_T[:3, 3] = t0

        T, mse = icp_yaw_only_original_kabsch(
            S, Z,
            init_T=init_T,
            max_iter=max_iter,
            tol=1e-6,
            max_corr_dist=max_corr_dist,
            min_inliers=30,
        )

        if mse < best_mse:
            best_mse = mse
            best_T = T

    return best_T, best_mse
