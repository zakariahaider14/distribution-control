import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import pandapower as pp
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
from scipy.optimize import minimize

# Define the Pandapower network (IEEE 4-bus equivalent model)
net = pp.create_empty_network()

# Add Buses
b1 = pp.create_bus(net, vn_kv=1.0, name="Bus 1")  # Slack Bus
b2 = pp.create_bus(net, vn_kv=1.0, name="Bus 2")
b3 = pp.create_bus(net, vn_kv=1.0, name="Bus 3")
b4 = pp.create_bus(net, vn_kv=1.0, name="Bus 4")

# Add Slack Generator (Grid)
pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid")

# Add PV Generator
pp.create_sgen(net, bus=b3, p_mw=0.5, q_mvar=0.1, name="PV Plant")

# Add Loads (Dynamic Demand)
pp.create_load(net, bus=b2, p_mw=0.3, q_mvar=0.1, name="Load 1")
pp.create_load(net, bus=b4, p_mw=0.2, q_mvar=0.05, name="Load 2")

# Add Lines
pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, length_km=1,
                               r_ohm_per_km=0.01, x_ohm_per_km=0.05, c_nf_per_km=10,
                               max_i_ka=1)

pp.create_line_from_parameters(net, from_bus=b2, to_bus=b3, length_km=1,
                               r_ohm_per_km=0.01, x_ohm_per_km=0.04, c_nf_per_km=8,
                               max_i_ka=1)

pp.create_line_from_parameters(net, from_bus=b3, to_bus=b4, length_km=1,
                               r_ohm_per_km=0.01, x_ohm_per_km=0.03, c_nf_per_km=6,
                               max_i_ka=1)

# LQR Control System
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



x_true = np.array([[0.1], [-0.05], [0.5], [0.4]])

np.random.seed(42)
w = np.random.normal(0, 0.05, (2, steps))

R_kf = np.diag([0.04, 0.04, 0.1, 0.1])
Q_kf = np.diag([0.001, 0.001, 0.002, 0.002])

P_kf = np.eye(4)
x_kf = np.zeros((4, 1))

P_ekf = np.eye(4)
x_ekf = np.zeros((4, 1))

def fx(x, dt, u=None):
    """State transition function for UKF."""
    # Print the original shapes for debugging
    print(f"Original shapes - x: {x.shape}, type: {type(x)}")
    if u is not None:
        print(f"Original u shape: {u.shape}, type: {type(u)}")
    
    # For UKF, x comes in as a 1D array, but for EKF it might be a column vector
    if hasattr(x, 'shape') and len(x.shape) > 1:
        # It's a matrix/vector, flatten first
        x_vec = x.flatten()
    else:
        # It's already a 1D array or a single value
        x_vec = np.array(x, dtype=float)
    
    # Ensure x is exactly length 4
    if len(x_vec) != 4:
        raise ValueError(f"Expected x to have 4 elements, but got {len(x_vec)}")
    
    # Handle control input
    if u is None:
        u_vec = np.zeros(2)
    elif hasattr(u, 'shape') and len(u.shape) > 1:
        u_vec = u.flatten()
    else:
        u_vec = np.array(u, dtype=float)
    
    # Ensure u is exactly length 2
    if len(u_vec) != 2:
        raise ValueError(f"Expected u to have 2 elements, but got {len(u_vec)}")
    
    # Reshape for matrix multiplication
    x_col = x_vec.reshape(4, 1)
    u_col = u_vec.reshape(2, 1)
    
    # Calculate state transition
    x_dot = A @ x_col + B @ u_col
    x_next = x_col + dt * x_dot
    
    # For UKF, return a 1D array
    result = x_next.flatten()
    print(f"Result shape: {result.shape}")
    
    return result

points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2, kappa=0)
ukf = UKF(dim_x=4, dim_z=4, dt=dt, fx=fx, hx=lambda x: x, points=points)
ukf.x = np.zeros(4)
ukf.P = np.eye(4)
ukf.Q = Q_kf
ukf.R = R_kf

def wls_estimate(z, H, R_wls):
    W = np.linalg.inv(R_wls)
    x_wls = np.linalg.inv(H.T @ W @ H) @ H.T @ W @ z
    return x_wls

x_wls = np.zeros((4, 1))
H_wls = np.eye(4)
R_wls = R_kf.copy()

# Store results
x_true_hist, x_kf_hist, x_ekf_hist, x_ukf_hist, x_wls_hist, voltage_hist = [], [], [], [], [], []

# Initial power flow to establish baseline
pp.runpp(net)
base_voltage = net.res_bus.vm_pu.values.copy()
base_pv_p = 0.5  # Base active power of PV generator
base_pv_q = 0.1  # Base reactive power of PV generator

for i in range(steps):
    # Calculate control input based on Kalman Filter state estimate
    u = -K @ x_kf
    
    # Apply disturbance for true state (simulation only)
    disturbance = D @ w[:, i].reshape(-1, 1)
    x_true = x_true + dt * (A @ x_true + B @ u + disturbance)
    
    # Map control outputs to power system adjustments
    pv_p_adjustment = u[0, 0] * 0.1  # Scale control input to MW
    pv_q_adjustment = u[1, 0] * 0.1  # Scale control input to MVAr
    
    # Update the PV generator setpoint
    net.sgen.at[0, "p_mw"] = base_pv_p + x_true[2, 0]  # Base value + current PV state
    net.sgen.at[0, "q_mvar"] = base_pv_q + x_true[3, 0]  # Base value + current PV state
    
    # Apply control action
    net.sgen.at[0, "p_mw"] += pv_p_adjustment
    net.sgen.at[0, "q_mvar"] += pv_q_adjustment
    
    # Update load based on disturbance (optional)
    load_variation_p = disturbance[0, 0] * 0.1
    load_variation_q = disturbance[1, 0] * 0.05
    net.load.at[0, "p_mw"] = 0.3 + load_variation_p
    net.load.at[1, "p_mw"] = 0.2 + load_variation_p
    
    # Run power flow simulation
    pp.runpp(net)
    
    # Get measurements from power flow
    voltage_measurements = net.res_bus.vm_pu.values - base_voltage
    
    # Construct measurement vector from power flow results
    z = np.zeros((4, 1))
    z[0, 0] = voltage_measurements[1]  # Bus 2 voltage deviation
    z[1, 0] = voltage_measurements[3]  # Bus 4 voltage deviation
    z[2, 0] = net.res_sgen.p_mw[0] - base_pv_p  # PV active power deviation
    z[3, 0] = net.res_sgen.q_mvar[0] - base_pv_q  # PV reactive power deviation
    
    # Add measurement noise
    z += np.random.multivariate_normal(np.zeros(4), R_kf).reshape(-1, 1)
    
    # KF Update
    x_kf = A @ x_kf + B @ u
    P_kf = A @ P_kf @ A.T + Q_kf
    K_kf = P_kf @ H_wls.T @ np.linalg.inv(H_wls @ P_kf @ H_wls.T + R_kf)
    x_kf = x_kf + K_kf @ (z - H_wls @ x_kf)
    P_kf = (np.eye(4) - K_kf @ H_wls) @ P_kf
    
    # EKF Update
    x_ekf = fx(x_ekf, dt, u).reshape(-1, 1)  # Reshape back to column vector
    P_ekf = A @ P_ekf @ A.T + Q_kf
    K_ekf = P_ekf @ H_wls.T @ np.linalg.inv(H_wls @ P_ekf @ H_wls.T + R_kf)
    x_ekf = x_ekf + K_ekf @ (z - x_ekf)
    P_ekf = (np.eye(4) - K_ekf @ H_wls) @ P_ekf
    
    # UKF Update
    u_flat = u.flatten()
    ukf.predict(u=u_flat)
    ukf.update(z.flatten())
    x_ukf = ukf.x.reshape(-1, 1)
    
    # WLS Update
    x_wls = wls_estimate(z, H_wls, R_wls).reshape(-1, 1)
    
    # Store results
    x_true_hist.append(x_true.flatten())
    x_kf_hist.append(x_kf.flatten())
    x_ekf_hist.append(x_ekf.flatten())
    x_ukf_hist.append(x_ukf.flatten())
    x_wls_hist.append(x_wls.flatten())
    voltage_hist.append(net.res_bus.vm_pu.copy())

# Convert to arrays
x_true_hist = np.array(x_true_hist)
x_kf_hist = np.array(x_kf_hist)
x_ekf_hist = np.array(x_ekf_hist)
x_ukf_hist = np.array(x_ukf_hist)
x_wls_hist = np.array(x_wls_hist)
voltage_hist = np.array(voltage_hist)

# Create additional plots to show the integration
plt.figure(figsize=(15, 10))

# Plot 1: Voltage profiles
plt.subplot(2, 2, 1)
plt.plot(voltage_hist[:, 0], label="Bus 1 Voltage")
plt.plot(voltage_hist[:, 1], label="Bus 2 Voltage")
plt.plot(voltage_hist[:, 2], label="Bus 3 Voltage (PV)")
plt.plot(voltage_hist[:, 3], label="Bus 4 Voltage")
plt.xlabel("Time Step")
plt.ylabel("Voltage (p.u.)")
plt.title("Bus Voltage Profiles with LQR Control")
plt.legend()
plt.grid(True)

# Plot 2: True vs Estimated State (Bus 2 Voltage)
plt.subplot(2, 2, 2)
plt.plot(x_true_hist[:, 0], 'k', label="True State")
plt.plot(x_kf_hist[:, 0], '--r', label="KF")
plt.plot(x_ekf_hist[:, 0], '--g', label="EKF")
plt.plot(x_ukf_hist[:, 0], '--b', label="UKF")
plt.plot(x_wls_hist[:, 0], '--m', label="WLS")
plt.xlabel("Time Step")
plt.ylabel("Bus 2 Voltage Deviation")
plt.title("State Estimation Performance - Bus 2")
plt.legend()
plt.grid(True)

# Plot 3: PV Active Power
plt.subplot(2, 2, 3)
plt.plot(x_true_hist[:, 2] + base_pv_p, 'k', label="True PV Output")
plt.plot(x_kf_hist[:, 2] + base_pv_p, '--r', label="KF Estimate")
plt.plot(x_ekf_hist[:, 2] + base_pv_p, '--g', label="EKF Estimate")
plt.xlabel("Time Step")
plt.ylabel("PV Power (MW)")
plt.title("PV Power Output and Estimation")
plt.legend()
plt.grid(True)

# Plot 4: Control Inputs
plt.subplot(2, 2, 4)
control_inputs = [-K @ x_kf_hist[i].reshape(-1, 1) for i in range(steps)]
control_inputs = np.array(control_inputs).reshape(steps, 2)
plt.plot(control_inputs[:, 0], label="Control Input 1 (PV-P)")
plt.plot(control_inputs[:, 1], label="Control Input 2 (PV-Q)")
plt.xlabel("Time Step")
plt.ylabel("Control Input")
plt.title("LQR Control Inputs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()