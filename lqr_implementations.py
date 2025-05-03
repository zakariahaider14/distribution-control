import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Define state-space model (Voltage deviation dynamics)
A = np.array([[-0.5, 0.2],
              [0.2, -0.4]])

B = np.array([[0.1, 0],
              [0, 0.1]])

# LQR Cost Matrices
Q = np.array([[1, 0],
              [0, 1]])

R = np.array([[0.01, 0],
              [0, 0.01]])

# Solve Riccati equation to compute P
P = scipy.linalg.solve_continuous_are(A, B, Q, R)

# Compute optimal control gain K
K = np.linalg.inv(R) @ B.T @ P

print("LQR Gain K:\n", K)

# Simulate system response
dt = 0.1  # Time step
T = 10     # Total simulation time
steps = int(T/dt)

x = np.array([[0.1],  # Initial voltage deviation at Bus 2
              [-0.05]])  # Initial voltage deviation at Bus 3

x_hist = []  # Store trajectory

for _ in range(steps):
    u = -K @ x  # LQR control law
    x = x + dt * (A @ x + B @ u)  # State update
    x_hist.append(x.flatten())

x_hist = np.array(x_hist)

# Plot Results
time = np.arange(0, T, dt)
plt.plot(time, x_hist[:, 0], label="Voltage Deviation at Bus 2")
plt.plot(time, x_hist[:, 1], label="Voltage Deviation at Bus 3")
plt.axhline(0, linestyle="--", color="black", alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Voltage Deviation (pu)")
plt.title("Voltage Control Using LQR")
plt.legend()
plt.grid()
plt.show()
