import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
from scipy.optimize import minimize

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
Q = np.diag([2, 2, 1, 1])
R = np.diag([0.25, 0.25])

# Solve Riccati equation for LQR gain
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# Simulation settings
dt = 0.1  # Time step
T = 80     # Total time
steps = int(T/dt)

# Initial state (small voltage deviations + PV generation)
x_true = np.array([[0.1], [-0.05], [0.5], [0.4]])

# Load and solar variations
np.random.seed(42)
w = np.random.normal(0, 0.05, (2, steps))

# Noise covariances
R_kf = np.diag([0.04, 0.04, 0.1, 0.1])
Q_kf = np.diag([0.001, 0.001, 0.002, 0.002])

# Kalman Filter (KF)
P_kf = np.eye(4)
x_kf = np.zeros((4, 1))

# Extended Kalman Filter (EKF)
P_ekf = np.eye(4)
x_ekf = np.zeros((4, 1))

def f_ekf(x, u):
    return x + dt * (A @ x + B @ u)

def h_ekf(x):
    return x  # Full-state measurement


def fx(x, dt, u=None):
    """State transition function for UKF."""
    if u is None:
        u = np.zeros(2)  # Default value if no control input
    
    # Ensure x and u are properly shaped
    x = np.atleast_1d(x).reshape(-1)  # Make sure x is a 1D array
    u = np.atleast_1d(u).reshape(-1)  # Make sure u is a 1D array
    
    # Perform calculations
    x_dot = A @ x + B @ u
    x_next = x + dt * x_dot
    
    return x_next  # Return a 1D array


# Then update your UKF initialization
points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2, kappa=0)
ukf = UKF(dim_x=4, dim_z=4, dt=dt, fx=fx, hx=h_ekf, points=points)
ukf.x = np.zeros(4)  # Initial state
ukf.P = np.eye(4)    # Initial covariance
ukf.Q = Q_kf         # Process noise
ukf.R = R_kf         # Measurement noise



# Weighted Least Squares (WLS) State Estimation
def wls_estimate(z, H, R_wls):
    W = np.linalg.inv(R_wls)
    x_wls = np.linalg.inv(H.T @ W @ H) @ H.T @ W @ z
    return x_wls

x_wls = np.zeros((4, 1))
H_wls = np.eye(4)
R_wls = R_kf.copy()

# Store results
x_true_hist, x_kf_hist, x_ekf_hist, x_ukf_hist, x_wls_hist = [], [], [], [], []

for i in range(steps):
    u = -K @ x_kf  # Control applied to estimated state
    disturbance = D @ w[:, i].reshape(-1, 1)
    x_true = x_true + dt * (A @ x_true + B @ u + disturbance)
    
    # Measurements with noise
    z = x_true + np.random.multivariate_normal(np.zeros(4), R_kf).reshape(-1, 1)
    
    # KF Update
    x_kf = A @ x_kf + B @ u
    P_kf = A @ P_kf @ A.T + Q_kf
    K_kf = P_kf @ H_wls.T @ np.linalg.inv(H_wls @ P_kf @ H_wls.T + R_kf)
    x_kf = x_kf + K_kf @ (z - H_wls @ x_kf)
    P_kf = (np.eye(4) - K_kf @ H_wls) @ P_kf
    
    # EKF Update
    x_ekf = f_ekf(x_ekf, u)
    P_ekf = A @ P_ekf @ A.T + Q_kf
    K_ekf = P_ekf @ H_wls.T @ np.linalg.inv(H_wls @ P_ekf @ H_wls.T + R_kf)
    x_ekf = x_ekf + K_ekf @ (z - h_ekf(x_ekf))
    P_ekf = (np.eye(4) - K_ekf @ H_wls) @ P_ekf
    
    u_flat = u.flatten()  # Flatten to 1D array
    
    # UKF Update
    ukf.predict(u=u_flat)  # Pass the flattened control input
    ukf.update(z.flatten())  # Flatten the measurement
    x_ukf = ukf.x.reshape(-1, 1)  # Reshape back to column vector
    
    # WLS Update
    x_wls = wls_estimate(z, H_wls, R_wls).reshape(-1, 1)
    
    # Store results
    x_true_hist.append(x_true.flatten())
    x_kf_hist.append(x_kf.flatten())
    x_ekf_hist.append(x_ekf.flatten())
    x_ukf_hist.append(x_ukf.flatten())
    x_wls_hist.append(x_wls.flatten())

# Convert to arrays
x_true_hist = np.array(x_true_hist)
x_kf_hist = np.array(x_kf_hist)
x_ekf_hist = np.array(x_ekf_hist)
x_ukf_hist = np.array(x_ukf_hist)
x_wls_hist = np.array(x_wls_hist)

# Plot Results
time = np.arange(0, T, dt)
plt.figure(figsize=(10, 6))
plt.plot(time, x_true_hist[:, 0], 'k', label='True State')
plt.plot(time, x_kf_hist[:, 0], '--r', label='KF')
plt.plot(time, x_ekf_hist[:, 0], '--g', label='EKF')
plt.plot(time, x_ukf_hist[:, 0], '--b', label='UKF')
plt.plot(time, x_wls_hist[:, 0], '--m', label='WLS')
plt.xlabel('Time (s)')
plt.ylabel('Voltage Deviation')
plt.legend()
plt.grid()

plt.figure(figsize=(10, 6))
plt.plot(time, x_true_hist[:, 2], 'k', label='True PV')
plt.plot(time, x_kf_hist[:, 2], '--r', label='KF')
plt.plot(time, x_ekf_hist[:, 2], '--g', label='EKF')
plt.plot(time, x_ukf_hist[:, 2], '--b', label='UKF')
plt.plot(time, x_wls_hist[:, 2], '--m', label='WLS')
plt.xlabel('Time (s)')
plt.ylabel('PV Power')
plt.legend()
plt.grid()
plt.show()
