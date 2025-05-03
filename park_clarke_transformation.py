import numpy as np
import matplotlib.pyplot as plt

# Clarke Transformation: abc → αβ
def clarke_transform(v_abc):
    T_clarke = np.array([
        [2/3, -1/3, -1/3],
        [0, np.sqrt(3)/3, -np.sqrt(3)/3]
    ])
    return T_clarke @ v_abc

# Inverse Clarke Transformation: αβ → abc
def inv_clarke_transform(v_alpha_beta):
    T_inv_clarke = np.array([
        [1, 0],
        [-1/2, np.sqrt(3)/2],
        [-1/2, -np.sqrt(3)/2]
    ])
    return T_inv_clarke @ v_alpha_beta

# Park Transformation: αβ → dq
def park_transform(v_alpha_beta, theta):
    T_park = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    return T_park @ v_alpha_beta

# Inverse Park Transformation: dq → αβ
def inv_park_transform(v_dq, theta):
    T_inv_park = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return T_inv_park @ v_dq

# Simulation: Convert 3-phase AC to DC using dq transformation
time = np.linspace(0, 2*np.pi, 100)  # One cycle (50Hz)
theta = 2 * np.pi * 50 * time  # Electrical angle

# Example 3-phase AC voltage
V_a = np.cos(theta)
V_b = np.cos(theta - 2*np.pi/3)
V_c = np.cos(theta + 2*np.pi/3)
V_abc = np.vstack((V_a, V_b, V_c))

# Apply Clarke Transformation (abc → αβ)
V_alpha_beta = np.array([clarke_transform(V_abc[:, i]) for i in range(len(time))]).T

# Apply Park Transformation (αβ → dq)
V_dq = np.array([park_transform(V_alpha_beta[:, i], theta[i]) for i in range(len(time))]).T

# Plot Results
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(time, V_a, label="Va")
plt.plot(time, V_b, label="Vb")
plt.plot(time, V_c, label="Vc")
plt.title("Three-Phase AC Voltage (Vabc)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, V_alpha_beta[0, :], label="Vα")
plt.plot(time, V_alpha_beta[1, :], label="Vβ")
plt.title("Two-Phase Stationary Voltage (Vαβ) - Clarke Transformation")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, V_dq[0, :], label="Vd (DC Component)")
plt.plot(time, V_dq[1, :], label="Vq (AC Component)")
plt.title("Two-Phase Rotating Voltage (Vdq) - Park Transformation")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
