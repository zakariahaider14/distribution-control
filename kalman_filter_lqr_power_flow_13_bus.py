import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import pandapower as pp
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
from scipy.optimize import minimize
import random

# Create IEEE 33-bus system
def create_ieee33_network_with_params():
    net = pp.create_empty_network()
    
    # Create buses (33 buses, indexed from 1 to 33)
    for i in range(1, 34):
        pp.create_bus(net, vn_kv=12.66, name=f"Bus {i}")
    
    # Line data with parameters: (from_bus, to_bus, r_ohm, x_ohm)
    line_data = [
        (1, 2, 0.0922, 0.0477), (2, 3, 0.493, 0.2511), (3, 4, 0.366, 0.1864), (4, 5, 0.3811, 0.1941),
        (5, 6, 0.819, 0.707), (6, 7, 0.1872, 0.6188), (7, 8, 1.7114, 1.2351), (8, 9, 1.03, 0.74),
        (9, 10, 1.04, 0.74), (10, 11, 0.1966, 0.065), (11, 12, 0.3744, 0.1238), (12, 13, 1.468, 1.155),
        (13, 14, 0.5416, 0.7129), (14, 15, 0.591, 0.526), (15, 16, 0.7463, 0.545), (16, 17, 1.289, 1.721),
        (17, 18, 0.732, 0.574), (2, 19, 0.164, 0.1565), (19, 20, 1.5042, 1.3554), (20, 21, 0.4095, 0.4784),
        (21, 22, 0.7089, 0.9373), (3, 23, 0.4512, 0.3083), (23, 24, 0.898, 0.7091), (24, 25, 0.896, 0.7011),
        (6, 26, 0.203, 0.1034), (26, 27, 0.2842, 0.1447), (27, 28, 1.059, 0.9337), (28, 29, 0.8042, 0.7006),
        (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.963), (31, 32, 0.31, 0.3619), (32, 33, 0.341, 0.5302)
    ]
    
    # Add lines with actual parameters instead of using standard types
    for from_bus, to_bus, r_ohm, x_ohm in line_data:
        # Adjust bus indices to be 0-indexed for pandapower
        pp_from_bus = from_bus - 1
        pp_to_bus = to_bus - 1
        
        # Calculate impedance and other parameters
        pp.create_line_from_parameters(
            net,
            from_bus=pp_from_bus,
            to_bus=pp_to_bus,
            length_km=1.0,  # Using unit length as R and X are already in ohms
            r_ohm_per_km=r_ohm,  
            x_ohm_per_km=x_ohm,
            c_nf_per_km=0.0,  
            max_i_ka=0.2,     
            name=f"Line {from_bus}-{to_bus}"
        )
    
    # Add external grid connection at bus 0
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, name="Grid Connection")
    
    return net

# Add IEEE 33-bus system loads
def add_ieee33_loads(net):
    # Standard IEEE 33-bus load data in kW and kVAr
    load_data = [
        (1, 100, 60), (2, 90, 40), (3, 120, 80), (4, 60, 30), (5, 60, 20),
        (6, 200, 100), (7, 200, 100), (8, 60, 20), (9, 60, 20), (10, 45, 30),
        (11, 60, 35), (12, 60, 35), (13, 120, 80), (14, 60, 10), (15, 60, 20),
        (16, 60, 20), (17, 90, 40), (18, 90, 40), (19, 90, 40), (20, 90, 40),
        (21, 90, 40), (22, 90, 40), (23, 90, 50), (24, 420, 200), (25, 420, 200),
        (26, 60, 25), (27, 60, 25), (28, 60, 20), (29, 120, 70), (30, 200, 600),
        (31, 150, 70), (32, 210, 100), (33, 60, 40)
    ]
    
    # Add loads
    for bus, p_kw, q_kvar in load_data:
        # Adjust to 0-indexed buses and convert to MW/MVAr
        pp.create_load(net, bus=bus-1, p_mw=p_kw/1000, q_mvar=q_kvar/1000, name=f"Load {bus}")
    
    return net

# Add 4 PV systems at strategic locations in the network
def add_pv_systems(net, pv_buses=None, pv_rating=0.5):
    if pv_buses is None:
        # Choose 4 random buses if not specified (avoiding bus 0, which is the slack bus)
        random.seed(42)  # For reproducibility
        pv_buses = random.sample(range(1, 33), 4)
    
    # Add PV generators at specified buses
    for i, bus in enumerate(pv_buses):
        pp.create_sgen(
            net, 
            bus=bus, 
            p_mw=pv_rating,
            q_mvar=0.0,  # Initially no reactive power
            name=f"PV_{i+1}_Bus_{bus+1}",
            type="PV"
        )
    
    return pv_buses

# Derive A and B matrices for state-space model through sensitivity analysis
def derive_control_matrices(net, pv_buses, monitored_buses=None):
    """
    Calculate A and B matrices for voltage control in the IEEE 33 bus system
    
    Parameters:
    -----------
    net : pandapower network
        The IEEE 33 bus network model
    pv_buses : list
        List of buses with PV installations (0-indexed)
    monitored_buses : list, optional
        List of buses to monitor for voltage control
        
    Returns:
    --------
    A : numpy.ndarray
        System dynamics matrix
    B : numpy.ndarray
        Control input matrix
    monitored_buses : list
        List of buses being monitored
    """
    # If no monitored buses are specified, use buses with PVs and some additional key buses
    if monitored_buses is None:
        # Use PV buses and some additional buses (e.g., at ends of feeders)
        monitored_buses = list(pv_buses) + [17, 22, 25, 32]  # Buses 18, 23, 26, 33 (1-indexed)
        monitored_buses = list(set(monitored_buses))  # Remove duplicates
    
    # Number of states and control inputs
    n_states = len(monitored_buses)
    n_controls = len(pv_buses) * 2  # P and Q for each PV
    
    # Run initial power flow
    pp.runpp(net)
    
    # Store original values
    original_voltages = {bus: net.res_bus.vm_pu[bus] for bus in monitored_buses}
    original_pv_p = {}
    original_pv_q = {}
    
    for bus in pv_buses:
        pv_idx = net.sgen[(net.sgen.bus == bus) & (net.sgen.type == "PV")].index
        if len(pv_idx) > 0:
            for idx in pv_idx:
                original_pv_p[idx] = net.sgen.p_mw[idx]
                original_pv_q[idx] = net.sgen.q_mvar[idx]
    
    # Calculate B matrix through sensitivity analysis
    B = np.zeros((n_states, n_controls))
    
    # Perturbation amount
    delta_p = 0.01  # MW
    delta_q = 0.01  # MVAr
    
    # For each PV, perturb P and Q and observe voltage changes
    for i, pv_bus in enumerate(pv_buses):
        pv_idx = net.sgen[(net.sgen.bus == pv_bus) & (net.sgen.type == "PV")].index
        
        if len(pv_idx) > 0:
            # Test P sensitivity
            for idx in pv_idx:
                net.sgen.at[idx, 'p_mw'] += delta_p
                
                # Run power flow with perturbation
                pp.runpp(net)
                
                # Measure voltage changes at monitored buses
                for j, bus in enumerate(monitored_buses):
                    v_with_p = net.res_bus.vm_pu[bus]
                    # dV/dP sensitivity
                    B[j, i*2] = (v_with_p - original_voltages[bus]) / delta_p
                
                # Restore original P
                net.sgen.at[idx, 'p_mw'] = original_pv_p[idx]
                
                # Test Q sensitivity
                net.sgen.at[idx, 'q_mvar'] += delta_q
                
                # Run power flow with perturbation
                pp.runpp(net)
                
                # Measure voltage changes at monitored buses
                for j, bus in enumerate(monitored_buses):
                    v_with_q = net.res_bus.vm_pu[bus]
                    # dV/dQ sensitivity
                    B[j, i*2 + 1] = (v_with_q - original_voltages[bus]) / delta_q
                
                # Restore original Q
                net.sgen.at[idx, 'q_mvar'] = original_pv_q[idx]
    
    # Build A matrix (system dynamics)
    A = np.zeros((n_states, n_states))
    
    # Diagonal elements represent voltage self-regulation (time constants)
    for i in range(n_states):
        A[i, i] = -0.5  # Conservative value for voltage dynamics
    
    # Add coupling between buses based on electrical distance
    for i, bus1 in enumerate(monitored_buses):
        for j, bus2 in enumerate(monitored_buses):
            if i != j:
                # Check if buses are directly connected
                connected = False
                for _, line in net.line.iterrows():
                    if (line.from_bus == bus1 and line.to_bus == bus2) or \
                       (line.from_bus == bus2 and line.to_bus == bus1):
                        connected = True
                        break
                
                if connected:
                    A[i, j] = 0.1  # Coupling for directly connected buses
                else:
                    # Weaker coupling for indirectly connected buses
                    A[i, j] = 0.01
    
    return A, B, monitored_buses

# Design LQR controller
def design_lqr_controller(A, B, voltage_weight=1.0, control_weight=0.1):
    """
    Design an LQR controller for voltage regulation
    """
    n_states = A.shape[0]
    n_controls = B.shape[1]
    
    # Q matrix - weights for state regulation
    Q = np.eye(n_states) * voltage_weight
    
    # R matrix - weights for control effort
    R = np.eye(n_controls) * control_weight
    
    # Solve Riccati equation
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    
    return K

# Setup Kalman Filter for state estimation
def setup_kalman_filter(A, B, process_noise=0.001, measurement_noise=0.005):
    """
    Initialize Kalman Filter for state estimation
    """
    n_states = A.shape[0]
    
    # Process noise covariance
    Q = np.eye(n_states) * process_noise
    
    # Measurement noise covariance
    R = np.eye(n_states) * measurement_noise
    
    # Initial state estimate and covariance
    x = np.zeros((n_states, 1))
    P = np.eye(n_states)
    
    # Measurement matrix (identity - we measure all states directly)
    H = np.eye(n_states)
    
    return {
        'A': A,
        'B': B,
        'Q': Q,
        'R': R,
        'H': H,
        'x': x,
        'P': P
    }

# Kalman Filter update
def update_kalman_filter(kf, u, z):
    """
    Update Kalman filter with new measurement
    
    Parameters:
    -----------
    kf : dict
        Kalman filter parameters
    u : numpy.ndarray
        Control input vector
    z : numpy.ndarray
        Measurement vector
        
    Returns:
    --------
    x_est : numpy.ndarray
        Updated state estimate
    """
    # Prediction step
    x_pred = kf['A'] @ kf['x'] + kf['B'] @ u
    P_pred = kf['A'] @ kf['P'] @ kf['A'].T + kf['Q']
    
    # Update step
    K = P_pred @ kf['H'].T @ np.linalg.inv(kf['H'] @ P_pred @ kf['H'].T + kf['R'])
    kf['x'] = x_pred + K @ (z - kf['H'] @ x_pred)
    kf['P'] = (np.eye(len(kf['x'])) - K @ kf['H']) @ P_pred
    
    return kf['x']

# Main simulation function
def run_voltage_control_simulation(net, pv_buses, monitored_buses=None, 
                                  sim_time=100, dt=0.1, load_disturbance=0.1):
    """
    Run a full simulation of voltage control with LQR controller and Kalman Filter
    
    Parameters:
    -----------
    net : pandapower.network
        The power system model
    pv_buses : list
        List of buses with PV installations
    monitored_buses : list, optional
        List of buses to monitor for voltage control
    sim_time : float
        Simulation time in seconds
    dt : float
        Time step in seconds
    load_disturbance : float
        Magnitude of random load disturbances
    
    Returns:
    --------
    results : dict
        Dictionary containing simulation results
    """
    # Derive control matrices
    A, B, monitored_buses = derive_control_matrices(net, pv_buses, monitored_buses)
    
    # Design LQR controller
    K = design_lqr_controller(A, B)
    
    # Setup Kalman Filter
    kf = setup_kalman_filter(A, B)
    
    # Run initial power flow
    pp.runpp(net)
    
    # Store reference voltages
    v_ref = {bus: net.res_bus.vm_pu[bus] for bus in monitored_buses}
    
    # Store original load values
    original_loads = {}
    for load_idx in net.load.index:
        original_loads[load_idx] = {
            'p_mw': net.load.p_mw[load_idx],
            'q_mvar': net.load.q_mvar[load_idx]
        }
    
    # Store original PV settings
    original_pv = {}
    for pv_bus in pv_buses:
        pv_idx = net.sgen[(net.sgen.bus == pv_bus) & (net.sgen.type == "PV")].index
        for idx in pv_idx:
            original_pv[idx] = {
                'p_mw': net.sgen.p_mw[idx],
                'q_mvar': net.sgen.q_mvar[idx]
            }
    
    # Setup simulation
    steps = int(sim_time / dt)
    
    # Initialize history arrays
    v_hist = np.zeros((steps, len(monitored_buses)))
    u_hist = np.zeros((steps, len(pv_buses) * 2))
    x_est_hist = np.zeros((steps, len(monitored_buses)))
    load_hist = np.zeros((steps, len(net.load)))
    
    # Create cloud-passing effect for PV outputs
    cloud_effect = np.zeros((steps, len(pv_buses)))
    for i, _ in enumerate(pv_buses):
        # Create random cloud effects at different times
        start = random.randint(20, 50)
        duration = random.randint(10, 30)
        cloud_effect[start:start+duration, i] = -0.5 * np.sin(np.linspace(0, np.pi, duration))
    
    # Main simulation loop
    for step in range(steps):
        # Apply load variations
        for i, load_idx in enumerate(net.load.index):
            # Random walk for loads
            if step > 0:
                prev_p = net.load.p_mw[load_idx]
                prev_q = net.load.q_mvar[load_idx]
                
                # Random walk with bounds
                p_change = load_disturbance * (random.random() - 0.5) * original_loads[load_idx]['p_mw']
                q_change = load_disturbance * (random.random() - 0.5) * original_loads[load_idx]['q_mvar']
                
                new_p = prev_p + p_change
                new_q = prev_q + q_change
                
                # Ensure loads remain within reasonable bounds
                new_p = max(0.5 * original_loads[load_idx]['p_mw'], 
                          min(new_p, 1.5 * original_loads[load_idx]['p_mw']))
                new_q = max(0.5 * original_loads[load_idx]['q_mvar'], 
                          min(new_q, 1.5 * original_loads[load_idx]['q_mvar']))
                
                net.load.at[load_idx, 'p_mw'] = new_p
                net.load.at[load_idx, 'q_mvar'] = new_q
                
                load_hist[step, i] = new_p
            else:
                load_hist[step, i] = net.load.p_mw[load_idx]
        
        # Apply cloud effects to PV generation
        for i, pv_bus in enumerate(pv_buses):
            pv_idx = net.sgen[(net.sgen.bus == pv_bus) & (net.sgen.type == "PV")].index
            for idx in pv_idx:
                # Apply cloud effect to base PV output
                cloud_factor = 1 + cloud_effect[step, i]
                base_p = original_pv[idx]['p_mw'] * cloud_factor
                net.sgen.at[idx, 'p_mw'] = max(0.1 * original_pv[idx]['p_mw'], base_p)
        
        # Run power flow
        try:
            pp.runpp(net)
            
            # Measure voltages
            v_measured = np.array([net.res_bus.vm_pu[bus] for bus in monitored_buses])
            
            # Calculate voltage deviations
            v_deviation = np.array([v_measured[i] - v_ref[bus] for i, bus in enumerate(monitored_buses)])
            
            # Add measurement noise
            v_measured_noisy = v_deviation + np.random.normal(0, 0.002, len(monitored_buses))
            
            # Update Kalman filter
            x_est = update_kalman_filter(kf, np.zeros((B.shape[1], 1)), v_measured_noisy.reshape(-1, 1))
            
            # Compute control input with LQR
            u = -K @ x_est
            
            # Apply control actions to PV systems
            for i, pv_bus in enumerate(pv_buses):
                pv_idx = net.sgen[(net.sgen.bus == pv_bus) & (net.sgen.type == "PV")].index
                
                if len(pv_idx) > 0:
                    for idx in pv_idx:
                        # Calculate new setpoints
                        p_adj = u[i*2, 0]
                        q_adj = u[i*2+1, 0]
                        
                        # Update PV outputs with constraints
                        base_p = net.sgen.p_mw[idx]  # Already includes cloud effects
                        base_q = original_pv[idx]['q_mvar']
                        
                        # Add control adjustments
                        new_p = base_p + p_adj
                        new_q = base_q + q_adj
                        
                        # Apply limits
                        max_p = original_pv[idx]['p_mw']
                        min_p = 0.0
                        max_q = 0.5 * max_p  # Typical PV inverter capability
                        min_q = -0.5 * max_p
                        
                        new_p = max(min_p, min(new_p, max_p))
                        new_q = max(min_q, min(new_q, max_q))
                        
                        # Update PV settings
                        net.sgen.at[idx, 'p_mw'] = new_p
                        net.sgen.at[idx, 'q_mvar'] = new_q
            
            # Store results
            v_hist[step, :] = v_measured
            u_hist[step, :] = u.flatten()
            x_est_hist[step, :] = x_est.flatten()
            
        except pp.powerflow.LoadflowNotConverged:
            print(f"Power flow did not converge at step {step}")
            # Use previous values if power flow doesn't converge
            if step > 0:
                v_hist[step, :] = v_hist[step-1, :]
                u_hist[step, :] = u_hist[step-1, :]
                x_est_hist[step, :] = x_est_hist[step-1, :]
    
    # Return results
    return {
        'voltage_history': v_hist,
        'control_history': u_hist,
        'estimate_history': x_est_hist,
        'load_history': load_hist,
        'monitored_buses': monitored_buses,
        'pv_buses': pv_buses,
        'A': A,
        'B': B,
        'K': K
    }

# Visualize results
def plot_simulation_results(net, results):
    """
    Create plots of simulation results
    """
    # Create time array
    steps = results['voltage_history'].shape[0]
    time = np.arange(0, steps) * 0.1  # Using dt=0.1s
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Voltages at monitored buses
    plt.subplot(2, 2, 1)
    for i, bus in enumerate(results['monitored_buses']):
        plt.plot(time, results['voltage_history'][:, i], label=f"Bus {bus+1}")
    plt.axhline(y=1.05, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (p.u.)")
    plt.title("Bus Voltages with LQR Control")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Control actions (PV adjustments)
    plt.subplot(2, 2, 2)
    for i, pv_bus in enumerate(results['pv_buses']):
        plt.plot(time, results['control_history'][:, i*2], label=f"P control - PV at Bus {pv_bus+1}")
        plt.plot(time, results['control_history'][:, i*2+1], '--', label=f"Q control - PV at Bus {pv_bus+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Control Signal")
    plt.title("LQR Control Actions")
    plt.legend()
    plt.grid(True)
    
    # Plot 3: True vs Estimated State (for the first monitored bus)
    plt.subplot(2, 2, 3)
    bus_idx = 0  # First monitored bus
    bus = results['monitored_buses'][bus_idx]
    v_ref = results['voltage_history'][0, bus_idx]  # Initial voltage as reference
    plt.plot(time, results['voltage_history'][:, bus_idx] - v_ref, label="True Deviation")
    plt.plot(time, results['estimate_history'][:, bus_idx], '--', label="KF Estimate")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage Deviation (p.u.)")
    plt.title(f"State Estimation for Bus {bus+1}")
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Load variations (first few loads)
    plt.subplot(2, 2, 4)
    max_loads = min(5, results['load_history'].shape[1])  # Plot up to 5 loads
    for i in range(max_loads):
        plt.plot(time, results['load_history'][:, i], label=f"Load {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Active Power (MW)")
    plt.title("Load Variations")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Create voltage histogram to show distribution
    plt.figure(figsize=(10, 6))
    for i, bus in enumerate(results['monitored_buses']):
        plt.hist(results['voltage_history'][:, i], bins=20, alpha=0.5, label=f"Bus {bus+1}")
    plt.axvline(x=1.05, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0.95, color='r', linestyle='--', alpha=0.5)
    plt.xlabel("Voltage (p.u.)")
    plt.ylabel("Frequency")
    plt.title("Voltage Distribution with LQR Control")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    # Create IEEE 33-bus system
    net = create_ieee33_network_with_params()
    net = add_ieee33_loads(net)
    
    # Select strategic buses for PV placement (you can change these)
    pv_buses = [7, 14, 24, 30]  # 0-indexed (buses 8, 15, 25, 31 in 1-indexed)
    pv_buses = add_pv_systems(net, pv_buses, pv_rating=0.5)
    
    # Optional: specify monitored buses (key buses for voltage control)
    monitored_buses = pv_buses + [17, 32]  # Include end of feeders
    
    # Run initial power flow
    try:
        pp.runpp(net)
        print("Initial power flow converged successfully.")
        
        # Print system summary
        print(f"System has {len(net.bus)} buses, {len(net.line)} lines, and {len(net.load)} loads.")
        print(f"PV systems installed at buses: {[b+1 for b in pv_buses]} (1-indexed)")
        
        # Check initial voltage profile
        print("\nInitial voltage profile:")
        low_v_buses = net.res_bus[net.res_bus.vm_pu < 0.95]
        if not low_v_buses.empty:
            print(f"Low voltage issues at buses: {low_v_buses.index.tolist()}")
            print(f"Minimum voltage: {net.res_bus.vm_pu.min():.4f} p.u. at bus {net.res_bus.vm_pu.idxmin()+1}")
        else:
            print("No voltage violations in initial state.")
        
        # Run simulation
        print("\nRunning simulation...")
        results = run_voltage_control_simulation(
            net, 
            pv_buses, 
            monitored_buses,
            sim_time=200,  # Longer simulation for better visualization
            dt=0.1,
            load_disturbance=0.05  # Smaller disturbance for stability
        )
        
        # Plot results
        plot_simulation_results(net, results)
        
        # Print performance metrics
        v_final = results['voltage_history'][-1, :]
        v_min = np.min(results['voltage_history'])
        v_max = np.max(results['voltage_history'])
        v_std = np.std(results['voltage_history'])
        
        print("\nPerformance metrics:")
        print(f"Final voltage range: {v_min:.4f} to {v_max:.4f} p.u.")
        print(f"Voltage standard deviation: {v_std:.4f} p.u.")
        
        # Calculate control effort
        control_energy = np.sum(np.square(results['control_history']))
        print(f"Total control effort: {control_energy:.4f}")
        
        return net, results
        
    except pp.powerflow.LoadflowNotConverged:
        print("Initial power flow did not converge. Check network parameters.")
        return None, None

if __name__ == "__main__":
    net, results = main()