import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Define the extended state-space model
A = np.array([
    [-0.5,  0.2,  0.1,  0  ],
    [ 0.2, -0.4,  0,   0.1 ],
    [ 0,    0,  -0.1,  0  ],
    [ 0,    0,   0,  -0.1 ]
])

B = np.array([
    [0.15,  0 ],
    [ 0,  0.15],
    [0.5, 0 ],
    [ 0, 0.5]
])

D = np.array([
    [0],
    [0],
    [0.01],
    [0.01]
])

# Define LQR cost matrices
Q = np.diag([1, 1, 0.1, 0.1])  # Penalize voltage deviations more than PV fluctuations
R = np.diag([0.01, 0.01])  # Penalize excessive reactive power control

# Solve Riccati equation
P = scipy.linalg.solve_continuous_are(A, B, Q, R)

# Compute LQR gain
K = np.linalg.inv(R) @ B.T @ P

print("LQR Gain K:\n", K)

# Simulate system response
dt = 0.1  # Time step
T = 20     # Total simulation time
steps = int(T/dt)

# Initial states: Small voltage deviations and PV generation levels
x = np.array([[0.1], [-0.05], [0.5], [0.4]])

x_hist = []  # Store trajectory

# Simulate solar irradiance variation (Gaussian noise)
np.random.seed(42)
w = np.random.normal(0, 0.05, (steps, 1))  # Small fluctuations

for i in range(steps):
    u = -K @ x  # Compute LQR control
    x = x + dt * (A @ x + B @ u + D @ w[i])  # State update
    x_hist.append(x.flatten())

x_hist = np.array(x_hist)

# Plot Results
time = np.arange(0, T, dt)

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(time, x_hist[:, 0], label="Voltage Deviation at Bus 2")
plt.plot(time, x_hist[:, 1], label="Voltage Deviation at Bus 3")
plt.axhline(0, linestyle="--", color="black", alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Voltage Deviation (pu)")
plt.title("Voltage Control with PV Integration")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, x_hist[:, 2], label="PV Power Output at Bus 2")
plt.plot(time, x_hist[:, 3], label="PV Power Output at Bus 3")
plt.xlabel("Time (s)")
plt.ylabel("PV Power (pu)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
