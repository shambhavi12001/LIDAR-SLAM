import numpy as np
import matplotlib.pyplot as plt

METERS_PER_TICK = 0.0022 

def integrate_enc_imu(enc_path: str, imu_path: str):
    enc = np.load(enc_path)
    imu = np.load(imu_path)

    t_enc = enc["time_stamps"].astype(float)
    counts = enc["counts"].astype(float)

    dist_right  = (counts[0] + counts[2]) * 0.5 * METERS_PER_TICK
    dist_left   = (counts[1] + counts[3]) * 0.5 * METERS_PER_TICK
    dist_center = (dist_right + dist_left) * 0.5

    t_imu = imu["time_stamps"].astype(float)
    
    raw_yaw_rate = imu["angular_velocity"][2]
    yaw_rate_sign = 1.0
    yaw_rate = np.interp(t_enc, t_imu, raw_yaw_rate) * yaw_rate_sign
    x  = np.zeros_like(t_enc)
    y  = np.zeros_like(t_enc)
    th = np.zeros_like(t_enc)

    EPS = 1e-6
    for k in range(1, len(t_enc)):
        dt = t_enc[k] - t_enc[k - 1]
        if dt <= 0:
            x[k], y[k], th[k] = x[k-1], y[k-1], th[k-1]
            continue

        v  = dist_center[k] / dt
        om = yaw_rate[k]
        th0 = th[k - 1]

        if abs(om) < EPS:
            x[k]  = x[k-1] + v * dt * np.cos(th0)
            y[k]  = y[k-1] + v * dt * np.sin(th0)
            th[k] = th0
        else:
            dth = om * dt
            # Movement along an arc
            x[k]  = x[k-1] + (v/om) * (np.sin(th0 + dth) - np.sin(th0))
            y[k]  = y[k-1] + (v/om) * (np.cos(th0) - np.cos(th0 + dth))
            th[k] = th0 + dth

    dt_arr = np.diff(t_enc)
    speed_arr = np.zeros_like(t_enc)
    good = dt_arr > 0
    speed_arr[1:][good] = dist_center[1:][good] / dt_arr[good]

    return t_enc, x, y, th, speed_arr, yaw_rate
if __name__ == "__main__":
    t1, x1, y1, th1, v1, w1 = integrate_enc_imu(
        "../data/Encoders20.npz",
        "../data/Imu20.npz"
    )

    t2, x2, y2, th2, v2, w2 = integrate_enc_imu(
        "../data/Encoders21.npz",
        "../data/Imu21.npz"
    )

    fig, ax = plt.subplots(1, 2, figsize=(12,6))

    ax[0].plot(x1, y1)
    ax[0].scatter(x1[0], y1[0])
    ax[0].scatter(x1[-1], y1[-1])
    ax[0].axis("equal")
    ax[0].set_title("Dataset 20")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].grid(True)

    ax[1].plot(x2, y2)
    ax[1].scatter(x2[0], y2[0])
    ax[1].scatter(x2[-1], y2[-1])
    ax[1].axis("equal")
    ax[1].set_title("Dataset 21")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].grid(True)

    plt.show()
