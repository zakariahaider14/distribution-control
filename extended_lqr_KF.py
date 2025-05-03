import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Define the system model with PV dynamics and load variations
A = np.array([
    [-0.5,  0.2,  0.1,  0  ],
    [ 0.2, -0.4,  0,   0.1 ],
    [ 0,    0,  -0.1,  0  ],
    [ 0,    0,   0,  -0.1 ]
])

B = np.array([
    [0.05,  0 ],
    [ 0,  0.05],
    [0.05, 0 ],
    [ 0, 0.05]
])

D = np.array([
    [0.02, 0],
    [0, 0.02],
    [0.01, 0],
    [0, 0.01]
])

# LQR tuning matrices
Q = np.diag([2, 2, 1, 1])  # Penalize voltage deviations
R = np.diag([0.25, 0.25])  # Penalize excessive control effort

# Solve Riccati equation for optimal LQR gain
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("LQR Gain K:\n", K)

# Simulation settings
dt = 0.1  # Time step
T = 80    # Total time
steps = int(T/dt)

# Initial state (small voltage deviations + PV generation)
x = np.array([[0.1], [-0.05], [0.5], [0.4]])
x_true = x.copy()

x_hist = []  # Store trajectory

# Load and solar variations (random fluctuations)
np.random.seed(42)
w = np.random.normal(0, 0.05, (2, steps))  # Load and PV disturbances

# Kalman Filter setup
H = np.eye(4)  # Measurement matrix (full state observed)
R_kf = np.diag([0.04, 0.04, 0.1, 0.1])  # Measurement noise covariance
P_kf = np.eye(4)  # Initial estimation error covariance
Q_kf = np.diag([0.001, 0.001, 0.002, 0.002])  # Process noise covariance
x_kf = np.zeros((4, 1))  # Initial estimate

for i in range(steps):
    # Apply LQR control
    u = -K @ x_kf  # Use Kalman estimate instead of true state

    # True system update (with disturbances)
    disturbance = D @ w[:, i].reshape(-1, 1)
    x_true = x_true + dt * (A @ x_true + B @ u + disturbance)

    # Measurement (with noise)
    z = H @ x_true + np.random.multivariate_normal(np.zeros(4), R_kf).reshape(-1, 1)

    # Kalman Filter update
    x_kf = A @ x_kf + B @ u  # Prediction step
    P_kf = A @ P_kf @ A.T + Q_kf

    # Measurement update
    K_kf = P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + R_kf)
    x_kf = x_kf + K_kf @ (z - H @ x_kf)
    P_kf = (np.eye(4) - K_kf @ H) @ P_kf

    x_hist.append(x_true.flatten())

x_hist = np.array(x_hist)

# Plot Results
time = np.arange(0, T, dt)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, x_hist[:, 0], label="Voltage Deviation at Bus 2")
plt.plot(time, x_hist[:, 1], label="Voltage Deviation at Bus 3")
plt.axhline(0, linestyle="--", color="black", alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Voltage Deviation (pu)")
plt.title("Voltage Control with PV, Load Variations, and Kalman Filter")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, x_hist[:, 2], label="PV Power at Bus 2")
plt.plot(time, x_hist[:, 3], label="PV Power at Bus 3")
plt.xlabel("Time (s)")
plt.ylabel("PV Power (pu)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
