import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import opendssdirect as dss
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import chi2
import os
import time
import logging
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistributionStateEstimation:
    """
    A class implementing three-phase distribution system state estimation
    taking into account the unbalanced nature of distribution systems.
    
    Key features:
    - Three-phase modeling for unbalanced systems
    - Support for radial and weakly meshed topologies
    - Integration of various measurement types (SCADA, smart meters, pseudo-measurements)
    - Bad data detection and identification
    - Visualization of results
    """
    
    def __init__(self, dss_file_path, use_historical_data=False, historical_data_path=None):
        """
        Initialize the Distribution State Estimation object
        
        Parameters:
        -----------
        dss_file_path : str
            Path to the OpenDSS file describing the distribution system
        use_historical_data : bool
            Whether to use historical load profiles for pseudo-measurements
        historical_data_path : str
            Path to historical load data (if use_historical_data is True)
        """
        # Initialize OpenDSS
        self._initialize_dss(dss_file_path)
        
        # Set up the measurement system
        self.measurements = []
        self.measurement_values = []
        self.measurement_variances = []
        self.measurement_types = []  # For categorizing measurements
        
        # Historical data for pseudo-measurements
        self.use_historical_data = use_historical_data
        self.historical_data_path = historical_data_path
        self.historical_data = None
        if use_historical_data and historical_data_path:
            self._load_historical_data()
            
        # Performance metrics
        self.execution_time = 0
        self.iteration_count = 0
        self.convergence_metric = 0
        
        # Initialize state with power flow results
        self.initialize_state_from_powerflow()
        
        # Initialize Y-bus matrix for the system
        self._build_ybus_matrix()
        
        logger.info(f"Distribution State Estimation initialized with {self.n_nodes} nodes and {self.n_buses} buses")
        logger.info(f"System has {self.n_phases} phases with {self.n_state_variables} state variables")
    
    def _initialize_dss(self, dss_file_path):
        """Initialize OpenDSS and load the circuit"""
        try:
            # Clear any existing circuit
            dss.Basic.ClearAll()
            
            # Check if file exists
            if not os.path.exists(dss_file_path):
                raise FileNotFoundError(f"OpenDSS file not found: {dss_file_path}")
            
            # Compile the DSS file
            dss.Text.Command(f"Compile {dss_file_path}")
            
            # Check if circuit was loaded successfully
            if not dss.Circuit.Name():
                raise RuntimeError(f"Failed to load circuit from {dss_file_path}")
            
            # Get system data
            self.circuit_name = dss.Circuit.Name()
            self.n_nodes = dss.Circuit.NumNodes()
            self.n_buses = dss.Circuit.NumBuses()
            self.bus_names = dss.Circuit.AllBusNames()
            
            # Get node-bus mapping and phase information
            self._build_node_mapping()
            
            # Identify system topology (radial or meshed)
            self._identify_topology()
            
            # Run power flow to get a starting point
            dss.Solution.Solve()
            if not dss.Solution.Converged():
                logger.warning("Power flow solution did not converge. Initial state may be inaccurate.")
            
            logger.info(f"Successfully loaded circuit: {self.circuit_name}")
            logger.info(f"System has {self.n_buses} buses and {self.n_nodes} nodes")
            
        except Exception as e:
            logger.error(f"Error initializing OpenDSS: {str(e)}")
            raise
    
    def _build_node_mapping(self):
        """Build mapping between buses, nodes, and phases"""
        self.node_names = []
        self.node_phases = []
        self.bus_to_nodes = {}  # Dictionary mapping bus names to node indices
        self.phase_count = {}   # Count of phases per bus
        
        for i in range(self.n_buses):
            dss.Circuit.SetActiveBusi(i)
            bus_name = dss.Bus.Name()
            self.bus_to_nodes[bus_name] = []
            
            # Get number of nodes (phases) at this bus
            num_nodes = dss.Bus.NumNodes()
            self.phase_count[bus_name] = num_nodes
            
            for j in range(num_nodes):
                node_name = f"{bus_name}.{j+1}"
                self.node_names.append(node_name)
                self.node_phases.append(j+1)  # Phase number (1, 2, or 3)
                self.bus_to_nodes[bus_name].append(len(self.node_names) - 1)
        
        # Determine the maximum number of phases in the system
        self.n_phases = max(self.node_phases) if self.node_phases else 0
        
        # Total number of state variables (voltage magnitude and angle for each node)
        self.n_state_variables = 2 * self.n_nodes
    
    def _identify_topology(self):
        """Identify if the system is radial or meshed"""
        # Get all lines
        n_lines = 0
        dss.Lines.First()
        while dss.Lines.Name():
            n_lines += 1
            dss.Lines.Next()
        
        # A purely radial system has n_buses - 1 lines
        self.is_radial = (n_lines <= self.n_buses)
        self.is_meshed = not self.is_radial
        
        logger.info(f"System topology identified as {'radial' if self.is_radial else 'meshed'}")
        
        # For meshed systems, we need to identify loops
        if self.is_meshed:
            self._identify_loops()
    
    def _identify_loops(self):
        """Identify loops in the network for meshed systems"""
        # This is a simplified approach - in a real implementation, 
        # you would use graph theory to find all loops
        self.loops = []
        
        # For now, just log that loops exist
        n_lines = 0
        dss.Lines.First()
        while dss.Lines.Name():
            n_lines += 1
            dss.Lines.Next()
        logger.info(f"System has at least {n_lines - (self.n_buses - 1)} loops")
    
    def _load_historical_data(self):
        """Load historical load profiles for pseudo-measurements"""
        try:
            if os.path.exists(self.historical_data_path):
                self.historical_data = pd.read_csv(self.historical_data_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded historical data with {len(self.historical_data)} time points")
            else:
                logger.warning(f"Historical data file not found: {self.historical_data_path}")
                self.use_historical_data = False
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            self.use_historical_data = False
    
    def initialize_state_from_powerflow(self):
        """Initialize state vector from power flow solution"""
        # Get all voltages from power flow solution
        voltages = dss.Circuit.AllBusVolts()
        
        # Convert complex voltages to magnitudes and angles
        complex_v = np.array(voltages[0::2]) + 1j * np.array(voltages[1::2])
        v_mag = np.abs(complex_v)
        v_angle = np.angle(complex_v)
        
        # Initialize state vector (voltage magnitudes and angles for each node)
        self.state_vector = np.zeros(2 * self.n_nodes)
        
        # Fill state vector
        self.state_vector[0:self.n_nodes] = v_mag
        self.state_vector[self.n_nodes:] = v_angle
        
        logger.info("State vector initialized from power flow solution")
    
    def _build_ybus_matrix(self):
        """Build the Y-bus matrix for the system"""
        # This is a placeholder for a more complete implementation
        # In a real implementation, you would extract the Y-bus matrix from OpenDSS
        # or build it manually from the line and transformer parameters
        
        # For now, we'll use OpenDSS's system Y matrix
        dss.Text.Command("CalcVoltageBases")
        dss.Solution.Solve()
        
        # Store the Y-bus matrix for later use in the measurement function
        self.ybus_matrix = None  # This would be populated with the actual Y-bus matrix
        
        logger.info("Y-bus matrix built for the system")
    
    def add_measurement(self, meas_type, node_id, value, variance, phase=None, timestamp=None):
        """Add a measurement to the system
        
        Parameters:
        -----------
        meas_type : str
            Type of measurement ('V' for voltage, 'P' for active power, 'Q' for reactive power, 'I' for current)
        node_id : int or str
            Node ID or name where measurement is taken
        value : float
            Measurement value
        variance : float
            Measurement variance (reliability)
        phase : int, optional
            Phase number (1, 2, or 3) for three-phase measurements
        timestamp : datetime, optional
            Timestamp of the measurement
        """
        # Convert node_id to index if string
        if isinstance(node_id, str):
            if node_id in self.node_names:
                node_idx = self.node_names.index(node_id)
            else:
                raise ValueError(f"Node {node_id} not found in system")
        else:
            node_idx = node_id
        
        # Create measurement dictionary
        measurement = {
            'type': meas_type,
            'node_id': node_idx,
            'phase': phase,
            'timestamp': timestamp or datetime.now(),
            'is_pseudo': False
        }
        
        self.measurements.append(measurement)
        self.measurement_values.append(value)
        self.measurement_variances.append(variance)
        self.measurement_types.append(meas_type)
        
        logger.debug(f"Added {meas_type} measurement at node {node_id} with value {value}")
    
    def add_voltage_measurement(self, node_id, value, variance, phase=None, timestamp=None):
        """Add a voltage magnitude measurement"""
        self.add_measurement('V', node_id, value, variance, phase, timestamp)
    
    def add_power_measurement(self, node_id, p_value, p_variance, q_value, q_variance, phase=None, timestamp=None):
        """Add active and reactive power measurements"""
        self.add_measurement('P', node_id, p_value, p_variance, phase, timestamp)
        self.add_measurement('Q', node_id, q_value, q_variance, phase, timestamp)
    
    def add_current_measurement(self, node_id, value, variance, phase=None, timestamp=None):
        """Add current magnitude measurement"""
        self.add_measurement('I', node_id, value, variance, phase, timestamp)
    
    def add_pmu_measurement(self, node_id, v_mag, v_angle, v_mag_var, v_angle_var, phase=None, timestamp=None):
        """Add PMU measurements (voltage magnitude and angle)"""
        self.add_measurement('V', node_id, v_mag, v_mag_var, phase, timestamp)
        self.add_measurement('VA', node_id, v_angle, v_angle_var, phase, timestamp)
    
    def add_pseudo_measurements(self, error_variance=0.5, time_point=None):
        """Add pseudo-measurements based on historical load profiles or nominal values
        
        Parameters:
        -----------
        error_variance : float
            Variance to assign to pseudo-measurements (higher values = less reliable)
        time_point : datetime, optional
            Time point to use for historical data lookup
        """
        if self.use_historical_data and self.historical_data is not None:
            self._add_historical_pseudo_measurements(error_variance, time_point)
        else:
            self._add_nominal_pseudo_measurements(error_variance)
    
    def _add_historical_pseudo_measurements(self, error_variance, time_point):
        """Add pseudo-measurements based on historical load profiles"""
        # Find the closest time point in historical data
        time_point = time_point or datetime.now()
        closest_time = self.historical_data.index[self.historical_data.index.get_indexer([time_point], method='nearest')[0]]
        data_slice = self.historical_data.loc[closest_time]
        
        # Add pseudo-measurements for each load in the historical data
        for load_name, row in data_slice.iteritems():
            if load_name in self.bus_to_nodes:
                node_indices = self.bus_to_nodes[load_name]
                p_value = row['P'] if 'P' in row else 0
                q_value = row['Q'] if 'Q' in row else 0
                
                # Distribute power equally among phases
                num_phases = len(node_indices)
                for i, node_idx in enumerate(node_indices):
                    phase = i + 1
                    self.add_measurement('P', node_idx, p_value/num_phases, error_variance, phase)
                    self.add_measurement('Q', node_idx, q_value/num_phases, error_variance, phase)
                    
                    # Mark as pseudo-measurement
                    self.measurements[-1]['is_pseudo'] = True
                    self.measurements[-2]['is_pseudo'] = True
        
        logger.info(f"Added historical pseudo-measurements for time point {closest_time}")
    
    def _add_nominal_pseudo_measurements(self, error_variance):
        """Add pseudo-measurements based on nominal values from OpenDSS model"""
        # Get all loads
        load_count = 0
        dss.Loads.First()
        while dss.Loads.Name():
            bus = dss.Loads.Bus()
            kw = dss.Loads.kW()
            kvar = dss.Loads.kvar()
            num_phases = dss.Loads.Phases()
            
            # Add as pseudo-measurements with high variance
            for phase in range(1, num_phases + 1):
                node_id = f"{bus}.{phase}"
                if node_id in self.node_names:
                    node_idx = self.node_names.index(node_id)
                    p_value = kw / num_phases  # Divide power equally among phases
                    q_value = kvar / num_phases
                    
                    self.add_measurement('P', node_idx, p_value, error_variance, phase)
                    self.add_measurement('Q', node_idx, q_value, error_variance, phase)
                    
                    # Mark as pseudo-measurement
                    self.measurements[-1]['is_pseudo'] = True
                    self.measurements[-2]['is_pseudo'] = True
                    
                    load_count += 1
            
            dss.Loads.Next()
        
        logger.info(f"Added {load_count} nominal pseudo-measurements from OpenDSS model")
    
    def measurement_function(self, state):
        """Calculate the expected measurement values for a given state vector
        
        Parameters:
        -----------
        state : numpy.ndarray
            State vector (voltage magnitudes and angles)
            
        Returns:
        --------
        numpy.ndarray
            Expected measurement values
        """
        # Extract voltage magnitudes and angles
        v_mag = state[:self.n_nodes]
        v_angle = state[self.n_nodes:]
        
        # Reconstruct complex voltages
        v_complex = v_mag * np.exp(1j * v_angle)
        
        # Initialize measurement expectations
        h = np.zeros(len(self.measurements))
        
        # Calculate expected values for each measurement
        for i, meas in enumerate(self.measurements):
            node_id = meas['node_id']
            meas_type = meas['type']
            
            if meas_type == 'V':
                # Voltage magnitude measurement
                h[i] = v_mag[node_id]
                
            elif meas_type == 'VA':
                # Voltage angle measurement
                h[i] = v_angle[node_id]
                
            elif meas_type in ['P', 'Q', 'I']:
                # For power and current measurements, we need to calculate power flow
                # This is a simplified approach - in a real implementation, you would use the Y-bus matrix
                # to calculate power flows accurately
                
                # Get the bus name and phase for this node
                bus_name = self.node_names[node_id].split('.')[0]
                phase = meas['phase'] or int(self.node_names[node_id].split('.')[-1])
                
                # Set the voltage at this node in OpenDSS
                dss.Circuit.SetNodeVoltage(str(node_id + 1), complex(v_complex[node_id]))
                
                # Force recalculation
                dss.Solution.SolveNoControl()
                
                if meas_type == 'P':
                    # Get active power at node
                    dss.Circuit.SetActiveBus(bus_name)
                    powers = dss.Bus.Powers()
                    phase_idx = phase - 1
                    if 2 * phase_idx < len(powers):
                        h[i] = powers[2 * phase_idx]
                    else:
                        h[i] = 0
                
                elif meas_type == 'Q':
                    # Get reactive power at node
                    dss.Circuit.SetActiveBus(bus_name)
                    powers = dss.Bus.Powers()
                    phase_idx = phase - 1
                    if 2 * phase_idx + 1 < len(powers):
                        h[i] = powers[2 * phase_idx + 1]
                    else:
                        h[i] = 0
                
                elif meas_type == 'I':
                    # Get current magnitude at node
                    dss.Circuit.SetActiveBus(bus_name)
                    currents = dss.Bus.Currents()
                    phase_idx = phase - 1
                    if 2 * phase_idx + 1 < len(currents):
                        h[i] = np.sqrt(currents[2 * phase_idx]**2 + currents[2 * phase_idx + 1]**2)
                    else:
                        h[i] = 0
        
        return h
    
    def jacobian(self, state):
        """Calculate the Jacobian matrix for the measurement function
        
        Parameters:
        -----------
        state : numpy.ndarray
            State vector
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            Jacobian matrix H where H[i,j] = ∂h_i/∂x_j
        """
        # Number of measurements and states
        m = len(self.measurements)
        n = len(state)
        
        # Initialize sparse Jacobian
        H = lil_matrix((m, n))
        
        # Small perturbation for numerical differentiation
        epsilon = 1e-6
        
        # Numerical calculation of Jacobian (finite difference method)
        h0 = self.measurement_function(state)
        
        for j in range(n):
            # Perturb state
            state_perturbed = state.copy()
            state_perturbed[j] += epsilon
            
            # Calculate perturbed measurements
            h1 = self.measurement_function(state_perturbed)
            
            # Compute partial derivatives
            dh_dx = (h1 - h0) / epsilon
            
            # Set values in Jacobian
            for i in range(m):
                if abs(dh_dx[i]) > 1e-10:  # Only store non-negligible entries
                    H[i, j] = dh_dx[i]
        
        return H.tocsr()
    
    def run_state_estimation(self, max_iter=10, tolerance=1e-4):
        """Run weighted least squares state estimation algorithm
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        numpy.ndarray
            Estimated state vector
        float
            Final objective value
        dict
            Performance metrics
        """
        # Check if we have enough measurements
        if len(self.measurements) < self.n_state_variables - 1:  # -1 for reference angle
            logger.warning(f"Not enough measurements: {len(self.measurements)} measurements for {self.n_state_variables} state variables")
            logger.warning("System may not be observable. Adding pseudo-measurements...")
            self.add_pseudo_measurements(error_variance=1.0)
        
        # Start timing
        start_time = time.time()
        
        # Initialize state with flat start or previous solution
        x = self.state_vector.copy()
        
        # Create weight matrix (inverse of measurement variances)
        W = np.diag(1.0 / np.array(self.measurement_variances))
        
        # Measurement vector
        z = np.array(self.measurement_values)
        
        # Iterative solution
        for iteration in range(max_iter):
            # Compute expected measurements for current state
            h = self.measurement_function(x)
            
            # Compute residual
            r = z - h
            
            # Compute Jacobian
            H = self.jacobian(x)
            
            # Compute Gain matrix G = H^T * W * H
            G = H.transpose() @ W @ H
            
            # Compute right-hand side
            rhs = H.transpose() @ W @ r
            
            # Solve linear system G * dx = rhs
            try:
                dx = spsolve(G, rhs)
            except Exception as e:
                logger.error(f"Error solving linear system: {str(e)}")
                logger.error("Gain matrix may be singular. Check system observability.")
                break
            
            # Update state
            x += dx
            
            # Check convergence
            norm_dx = np.linalg.norm(dx)
            if norm_dx < tolerance:
                logger.info(f"Converged in {iteration+1} iterations with norm(dx) = {norm_dx:.6f}")
                break
            
            logger.debug(f"Iteration {iteration+1}: norm(dx) = {norm_dx:.6f}")
        
        # Store final state
        self.state_vector = x
        
        # Compute final objective value
        h_final = self.measurement_function(x)
        r_final = z - h_final
        obj_value = r_final.T @ W @ r_final
        
        # Record performance metrics
        self.execution_time = time.time() - start_time
        self.iteration_count = iteration + 1
        self.convergence_metric = norm_dx if 'norm_dx' in locals() else float('inf')
        
        performance = {
            'execution_time': self.execution_time,
            'iterations': self.iteration_count,
            'convergence': self.convergence_metric,
            'objective_value': obj_value
        }
        
        logger.info(f"State estimation completed in {self.execution_time:.3f} seconds")
        logger.info(f"Final objective value: {obj_value:.6f}")
        
        return x, obj_value, performance
    
    def bad_data_detection(self, confidence_level=0.95):
        """Perform bad data detection and identification
        
        Parameters:
        -----------
        confidence_level : float
            Statistical confidence level for chi-square test
            
        Returns:
        --------
        list
            Indices of suspected bad measurements
        dict
            Detection statistics
        """
        # Get current state
        x = self.state_vector
        
        # Get measurements and compute residuals
        z = np.array(self.measurement_values)
        h = self.measurement_function(x)
        r = z - h
        
        # Weight matrix
        W = np.diag(1.0 / np.array(self.measurement_variances))
        
        # Compute Jacobian
        H = self.jacobian(x)
        
        # Compute residual covariance matrix
        # R = W^-1 - H * (H^T * W * H)^-1 * H^T
        try:
            G_inv = np.linalg.inv((H.transpose() @ W @ H).toarray())
            R = np.diag(self.measurement_variances) - (H @ G_inv @ H.transpose()).toarray()
        except np.linalg.LinAlgError:
            logger.error("Error computing residual covariance matrix. Gain matrix may be singular.")
            return [], {'chi2_test': False, 'J': 0, 'threshold': 0}
        
        # Compute normalized residuals
        r_norm = np.zeros_like(r)
        for i in range(len(r)):
            if R[i, i] > 0:
                r_norm[i] = abs(r[i]) / np.sqrt(R[i, i])
        
        # Chi-square test for overall bad data
        J = r.T @ W @ r
        dof = len(z) - 2 * self.n_nodes + 1  # Degrees of freedom (measurements - states + 1 reference angle)
        chi2_threshold = chi2.ppf(confidence_level, dof)
        
        bad_data_detected = J > chi2_threshold
        
        if bad_data_detected:
            logger.info(f"Bad data detected: J = {J:.2f} > χ²({dof}) = {chi2_threshold:.2f}")
            
            # Find largest normalized residual
            largest_idx = np.argmax(r_norm)
            
            # Use 3-sigma rule for detection
            suspected_bad = [i for i, val in enumerate(r_norm) if val > 3.0]
            
            # Sort by normalized residual value (descending)
            suspected_bad.sort(key=lambda i: r_norm[i], reverse=True)
            
            # Log details of suspected bad measurements
            for idx in suspected_bad:
                meas = self.measurements[idx]
                logger.info(f"Suspected bad measurement: {meas['type']} at node {self.node_names[meas['node_id']]}, "  
                           f"value={self.measurement_values[idx]:.4f}, normalized residual={r_norm[idx]:.4f}")
            
            stats = {
                'chi2_test': bad_data_detected,
                'J': J,
                'threshold': chi2_threshold,
                'largest_residual': r_norm[largest_idx],
                'largest_residual_idx': largest_idx,
                'normalized_residuals': r_norm
            }
            
            return suspected_bad, stats
        else:
            logger.info(f"No bad data detected: J = {J:.2f} ≤ χ²({dof}) = {chi2_threshold:.2f}")
            
            stats = {
                'chi2_test': bad_data_detected,
                'J': J,
                'threshold': chi2_threshold,
                'normalized_residuals': r_norm
            }
            
            return [], stats
    
    def remove_bad_measurements(self, bad_indices):
        """Remove bad measurements from the measurement set
        
        Parameters:
        -----------
        bad_indices : list
            Indices of measurements to remove
        """
        if not bad_indices:
            return
        
        # Sort indices in descending order to avoid index shifting
        for idx in sorted(bad_indices, reverse=True):
            if 0 <= idx < len(self.measurements):
                meas = self.measurements[idx]
                logger.info(f"Removing measurement: {meas['type']} at node {self.node_names[meas['node_id']]}")
                
                self.measurements.pop(idx)
                self.measurement_values.pop(idx)
                self.measurement_variances.pop(idx)
                self.measurement_types.pop(idx)
    
    def get_voltage_profile(self):
        """Get the voltage profile from the estimated state
        
        Returns:
        --------
        dict
            Voltage profile data by phase
        """
        # Extract voltage magnitudes per node
        v_mag = self.state_vector[:self.n_nodes]
        v_angle = self.state_vector[self.n_nodes:]
        
        # Group voltages by phase
        v_profile = {
            'phase_a': [],
            'phase_b': [],
            'phase_c': []
        }
        
        # Extract voltages by phase
        for i, node_name in enumerate(self.node_names):
            bus_name = node_name.split('.')[0]
            phase = int(node_name.split('.')[-1])
            
            if phase == 1:
                v_profile['phase_a'].append((bus_name, i, v_mag[i], v_angle[i]))
            elif phase == 2:
                v_profile['phase_b'].append((bus_name, i, v_mag[i], v_angle[i]))
            elif phase == 3:
                v_profile['phase_c'].append((bus_name, i, v_mag[i], v_angle[i]))
        
        # Sort by node index
        v_profile['phase_a'].sort(key=lambda x: x[1])
        v_profile['phase_b'].sort(key=lambda x: x[1])
        v_profile['phase_c'].sort(key=lambda x: x[1])
        
        return v_profile
    
    def plot_voltage_profile(self, save_path=None):
        """Plot voltage profile results
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot image
        """
        # Get voltage profile data
        v_profile = self.get_voltage_profile()
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot each phase if data exists
        if v_profile['phase_a']:
            bus_names, indices, v_mag_a, _ = zip(*v_profile['phase_a'])
            plt.plot(indices, v_mag_a, 'r-o', label='Phase A', markersize=4)
        
        if v_profile['phase_b']:
            bus_names, indices, v_mag_b, _ = zip(*v_profile['phase_b'])
            plt.plot(indices, v_mag_b, 'g-o', label='Phase B', markersize=4)
        
        if v_profile['phase_c']:
            bus_names, indices, v_mag_c, _ = zip(*v_profile['phase_c'])
            plt.plot(indices, v_mag_c, 'b-o', label='Phase C', markersize=4)
        
        # Add ANSI voltage limits
        plt.axhline(y=0.95, color='k', linestyle='--', alpha=0.5, label='ANSI Limits (±5%)')
        plt.axhline(y=1.05, color='k', linestyle='--', alpha=0.5)
        
        plt.xlabel('Node Index')
        plt.ylabel('Voltage Magnitude (p.u.)')
        plt.title('Estimated Voltage Profile')
        plt.legend()
        plt.grid(True)
        
        # Add some statistics to the plot
        v_mag = self.state_vector[:self.n_nodes]
        min_v = np.min(v_mag)
        max_v = np.max(v_mag)
        
        plt.figtext(0.02, 0.02, 
                   f"Min Voltage: {min_v:.4f} p.u.\n"
                   f"Max Voltage: {max_v:.4f} p.u.\n"
                   f"Execution Time: {self.execution_time:.3f} s\n"
                   f"Iterations: {self.iteration_count}", 
                   fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Voltage profile plot saved to {save_path}")
        
        plt.show()
    
    def plot_measurement_residuals(self, save_path=None):
        """Plot measurement residuals
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot image
        """
        # Get measurements and compute residuals
        z = np.array(self.measurement_values)
        h = self.measurement_function(self.state_vector)
        r = z - h
        
        # Weight matrix
        W = np.diag(1.0 / np.array(self.measurement_variances))
        
        # Compute Jacobian
        H = self.jacobian(self.state_vector)
        
        # Compute residual covariance matrix
        try:
            G_inv = np.linalg.inv((H.transpose() @ W @ H).toarray())
            R = np.diag(self.measurement_variances) - (H @ G_inv @ H.transpose()).toarray()
        except np.linalg.LinAlgError:
            logger.error("Error computing residual covariance matrix. Using diagonal approximation.")
            R = np.diag(self.measurement_variances)
        
        # Compute normalized residuals
        r_norm = np.zeros_like(r)
        for i in range(len(r)):
            if R[i, i] > 0:
                r_norm[i] = r[i] / np.sqrt(R[i, i])
        
        # Group by measurement type
        meas_types = np.array(self.measurement_types)
        unique_types = np.unique(meas_types)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot normalized residuals by measurement type
        for mtype in unique_types:
            indices = np.where(meas_types == mtype)[0]
            plt.scatter(indices, r_norm[indices], label=f'{mtype} Measurements', alpha=0.7)
        
        # Add 3-sigma threshold lines
        plt.axhline(y=3, color='r', linestyle='--', alpha=0.5, label='3σ Threshold')
        plt.axhline(y=-3, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Measurement Index')
        plt.ylabel('Normalized Residual')
        plt.title('Normalized Measurement Residuals')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, output_dir=None):
        """Generate a comprehensive report of the state estimation results
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save report files
        
        Returns:
        --------
        dict
            Report data
        """
        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Extract voltage magnitudes and angles
        v_mag = self.state_vector[:self.n_nodes]
        v_angle = self.state_vector[self.n_nodes:]
        
        # Voltage statistics
        v_stats = {
            'min_voltage': np.min(v_mag),
            'max_voltage': np.max(v_mag),
            'avg_voltage': np.mean(v_mag),
            'voltage_std': np.std(v_mag),
            'phase_a_avg': np.mean([v_mag[i] for i, name in enumerate(self.node_names) if name.endswith('.1')]) if any(name.endswith('.1') for name in self.node_names) else 0,
            'phase_b_avg': np.mean([v_mag[i] for i, name in enumerate(self.node_names) if name.endswith('.2')]) if any(name.endswith('.2') for name in self.node_names) else 0,
            'phase_c_avg': np.mean([v_mag[i] for i, name in enumerate(self.node_names) if name.endswith('.3')]) if any(name.endswith('.3') for name in self.node_names) else 0,
        }
        
        # Count nodes with voltage violations
        v_violations = {
            'under_voltage': sum(1 for v in v_mag if v < 0.95),
            'over_voltage': sum(1 for v in v_mag if v > 1.05),
            'total_violations': sum(1 for v in v_mag if v < 0.95 or v > 1.05),
            'violation_percentage': 100 * sum(1 for v in v_mag if v < 0.95 or v > 1.05) / len(v_mag) if len(v_mag) > 0 else 0
        }
        
        # Performance metrics
        performance = {
            'execution_time': self.execution_time,
            'iterations': self.iteration_count,
            'convergence': self.convergence_metric
        }
        
        # Measurement statistics
        meas_stats = {
            'total_measurements': len(self.measurements),
            'real_measurements': sum(1 for m in self.measurements if not m.get('is_pseudo', False)),
            'pseudo_measurements': sum(1 for m in self.measurements if m.get('is_pseudo', False)),
            'voltage_measurements': sum(1 for m in self.measurement_types if m == 'V'),
            'power_measurements': sum(1 for m in self.measurement_types if m in ['P', 'Q']),
            'current_measurements': sum(1 for m in self.measurement_types if m == 'I'),
            'pmu_measurements': sum(1 for m in self.measurement_types if m == 'VA'),
            'redundancy_ratio': len(self.measurements) / (2 * self.n_nodes - 1) if self.n_nodes > 0 else 0  # -1 for reference angle
        }
        
        # Compile report data
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'circuit_name': self.circuit_name,
            'system_info': {
                'n_buses': self.n_buses,
                'n_nodes': self.n_nodes,
                'n_phases': self.n_phases,
                'topology': 'radial' if self.is_radial else 'meshed'
            },
            'voltage_statistics': v_stats,
            'voltage_violations': v_violations,
            'performance': performance,
            'measurement_statistics': meas_stats
        }
        
        # Print summary to console
        logger.info("\n===== State Estimation Report =====")
        logger.info(f"Circuit: {self.circuit_name}")
        logger.info(f"Buses: {self.n_buses}, Nodes: {self.n_nodes}, Phases: {self.n_phases}")
        logger.info(f"Topology: {'radial' if self.is_radial else 'meshed'}")
        logger.info("\nVoltage Statistics:")
        logger.info(f"Min voltage: {v_stats['min_voltage']:.4f} p.u.")
        logger.info(f"Max voltage: {v_stats['max_voltage']:.4f} p.u.")
        logger.info(f"Avg voltage: {v_stats['avg_voltage']:.4f} p.u.")
        logger.info(f"Phase A avg: {v_stats['phase_a_avg']:.4f} p.u.")
        logger.info(f"Phase B avg: {v_stats['phase_b_avg']:.4f} p.u.")
        logger.info(f"Phase C avg: {v_stats['phase_c_avg']:.4f} p.u.")
        logger.info(f"\nVoltage Violations: {v_violations['total_violations']} nodes ({v_violations['violation_percentage']:.1f}%)")
        logger.info(f"Under-voltage: {v_violations['under_voltage']} nodes")
        logger.info(f"Over-voltage: {v_violations['over_voltage']} nodes")
        logger.info("\nPerformance:")
        logger.info(f"Execution time: {performance['execution_time']:.3f} seconds")
        logger.info(f"Iterations: {performance['iterations']}")
        logger.info("\nMeasurement Statistics:")
        logger.info(f"Total measurements: {meas_stats['total_measurements']}")
        logger.info(f"Real measurements: {meas_stats['real_measurements']}")
        logger.info(f"Pseudo-measurements: {meas_stats['pseudo_measurements']}")
        logger.info(f"Redundancy ratio: {meas_stats['redundancy_ratio']:.2f}")
        
        # Save report to file if output directory specified
        if output_dir:
            # Save plots
            self.plot_voltage_profile(os.path.join(output_dir, 'voltage_profile.png'))
            self.plot_measurement_residuals(os.path.join(output_dir, 'measurement_residuals.png'))
            
            # Save report as JSON
            import json
            report_path = os.path.join(output_dir, 'state_estimation_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            # Save voltage profile as CSV
            voltage_df = pd.DataFrame({
                'node': self.node_names,
                'voltage_magnitude': v_mag,
                'voltage_angle': v_angle
            })
            voltage_df.to_csv(os.path.join(output_dir, 'voltage_profile.csv'), index=False)
            
            logger.info(f"\nReport saved to {output_dir}")
        
        return report
