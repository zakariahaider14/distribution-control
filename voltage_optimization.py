"""
Voltage Optimization in Distribution System
- Maintains voltage within acceptable limits (0.95-1.05 p.u.)
- Minimizes system power losses
- Uses capacitor banks and voltage regulators as control variables
- Implements optimization using SciPy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import opendssdirect as dss

# === Constants ===
VOLTAGE_MIN = 0.95  # p.u.
VOLTAGE_MAX = 1.05  # p.u.
NOMINAL_VOLTAGE = 1.0  # p.u.

# === Helper Functions ===
def get_original_loads():
    """Get current load settings"""
    loads = {}
    for load_name in dss.Loads.AllNames():
        dss.Loads.Name(load_name)
        loads[load_name] = {
            'kW': dss.Loads.kW(),
            'kvar': dss.Loads.kvar(),
            'model': dss.Loads.Model(),
            'bus': dss.CktElement.BusNames()[0]  # Get the first bus name
        }
    return loads

def generate_load_profiles(original_loads, num_time_steps=24):
    """Generate load profiles for all loads"""
    profiles = {}
    
    # Create a basic load profile multiplier (24 hours)
    hourly_profile = np.array([
        0.6, 0.55, 0.5, 0.5, 0.55, 0.6,       # 12am-6am
        0.85, 1.0, 1.05, 0.95, 0.9, 0.9,      # 6am-12pm
        0.85, 0.85, 0.85, 0.9, 1.0, 1.1,      # 12pm-6pm
        1.05, 1.0, 0.9, 0.8, 0.7, 0.65        # 6pm-12am
    ])
    
    # Add some randomness to each load
    for load_name in original_loads:
        # Add random variation (+/- 10%)
        variation = np.random.uniform(0.9, 1.1, num_time_steps)
        profiles[load_name] = hourly_profile * variation
            
    return profiles

def set_loads_for_time(time_step, original_loads, load_profiles):
    """Set loads according to the profiles for a specific time step"""
    for load_name, profile in load_profiles.items():
        if load_name in original_loads:
            multiplier = profile[time_step]
            dss.Loads.Name(load_name)
            original_kw = original_loads[load_name]['kW']
            original_kvar = original_loads[load_name]['kvar']
            
            # Set new load values
            dss.Loads.kW(original_kw * multiplier)
            dss.Loads.kvar(original_kvar * multiplier)

def get_capacitor_names():
    """Get all capacitor names in the system"""
    return dss.Capacitors.AllNames()

def get_regulator_names():
    """Get all regulator control names in the system"""
    return dss.RegControls.AllNames()

def set_capacitor_state(cap_name, state):
    """Set capacitor state (1=on, 0=off)"""
    dss.Capacitors.Name(cap_name)
    if state == 1:
        dss.Text.Command(f"Capacitor.{cap_name}.States=[1,1,1]")
    else:
        dss.Text.Command(f"Capacitor.{cap_name}.States=[0,0,0]")

def set_regulator_tap(reg_name, tap_position):
    """Set regulator tap position (-16 to +16)"""
    dss.RegControls.Name(reg_name)
    # Convert tap position to tap value (typically 0.00625 p.u. per tap)
    tap_value = 1.0 + 0.00625 * tap_position
    
    # Extract transformer name from regcontrol
    xfmr_name = dss.RegControls.Transformer()
    
    # Disable automatic control
    dss.Text.Command(f"RegControl.{reg_name}.TapWinding=1")
    dss.Text.Command(f"RegControl.{reg_name}.Enabled=No")
    
    # Set tap directly on the transformer
    dss.Text.Command(f"Transformer.{xfmr_name}.Tap={tap_value:.6f}")
    
    # Store tap position for reporting
    dss.Text.Command(f"RegControl.{reg_name}.TapNum={tap_position}")

def get_voltage_violations():
    """Calculate voltage violations across all buses"""
    violations = 0
    max_deviation = 0
    total_deviation = 0
    total_nodes = 0
    violation_severity = 0  # Track how far outside limits voltages are
    
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        voltages = dss.Bus.puVmagAngle()[::2]  # Only magnitudes
        
        # Check each phase
        for v in voltages:
            if v > 0.1:  # Only consider valid voltages
                total_nodes += 1
                deviation = abs(v - NOMINAL_VOLTAGE)
                total_deviation += deviation
                max_deviation = max(max_deviation, deviation)
                
                # Check for violations
                if v < VOLTAGE_MIN:
                    violations += 1
                    # Penalize more for larger violations
                    violation_severity += (VOLTAGE_MIN - v) ** 2
                elif v > VOLTAGE_MAX:
                    violations += 1
                    # Penalize more for larger violations
                    violation_severity += (v - VOLTAGE_MAX) ** 2
    
    avg_deviation = total_deviation / total_nodes if total_nodes > 0 else 0
    
    return violations, max_deviation, avg_deviation, violation_severity

def get_system_losses():
    """Get total system losses in kW"""
    return dss.Circuit.Losses()[0] / 1000  # Convert W to kW

def get_all_bus_voltages():
    """Get voltages for all buses in the system"""
    voltages = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        v_pu = dss.Bus.puVmagAngle()[::2]  # Only magnitudes
        voltages[bus] = v_pu
    return voltages

# === Optimization Functions ===
class VoltageOptimizer:
    def __init__(self, dss_file_path):
        """Initialize the voltage optimizer"""
        self.dss_file_path = dss_file_path
        self.initialize_opendss()
        
        # Get system components
        self.capacitor_names = get_capacitor_names()
        self.regulator_names = get_regulator_names()
        
        # Store original load profiles
        self.original_loads = get_original_loads()
        self.load_profiles = generate_load_profiles(self.original_loads)
        
        # Optimization parameters
        self.voltage_violation_weight = 500  # Weight for voltage violations (reduced)
        self.loss_weight = 5  # Increased weight for losses
        self.deviation_weight = 100  # Weight for voltage deviation from nominal
        
        print(f"System loaded with {len(self.capacitor_names)} capacitors and {len(self.regulator_names)} regulators")
    
    def initialize_opendss(self):
        """Initialize OpenDSS with the specified circuit"""
        dss.Basic.ClearAll()
        
        # Verify file exists
        if not os.path.exists(self.dss_file_path):
            raise Exception(f"DSS file does not exist: {self.dss_file_path}")
        
        # Compile the file
        dss.Text.Command(f'Compile "{self.dss_file_path}"')
        
        # Disable automatic controls for optimization
        dss.Text.Command("Set ControlMode=OFF")
        
        # Solve the initial power flow
        dss.Solution.Solve()
    
    def apply_control_actions(self, control_vars):
        """Apply control actions from optimization variables"""
        # Extract control variables
        cap_idx = 0
        reg_idx = len(self.capacitor_names)
        
        # Set capacitor states
        for i, cap_name in enumerate(self.capacitor_names):
            cap_state = round(control_vars[cap_idx + i])  # Round to 0 or 1
            set_capacitor_state(cap_name, cap_state)
        
        # Set regulator taps
        for i, reg_name in enumerate(self.regulator_names):
            # Convert continuous value to discrete tap position (-16 to +16)
            tap_position = round(control_vars[reg_idx + i])
            # Ensure within limits
            tap_position = max(-16, min(16, tap_position))
            set_regulator_tap(reg_name, tap_position)
    
    def objective_function(self, control_vars, time_step):
        """
        Objective function for optimization:
        - Minimize power losses
        - Penalize voltage violations
        - Improve voltage profile
        """
        # Set loads for this time step
        set_loads_for_time(time_step, self.original_loads, self.load_profiles)
        
        # Apply control actions
        self.apply_control_actions(control_vars)
        
        # Solve power flow
        dss.Solution.Solve()
        
        # Check if solution converged
        if not dss.Solution.Converged():
            # Return a very high penalty for non-convergence
            return 1e6
        
        # Calculate voltage violations with improved metrics
        violations, max_deviation, avg_deviation, violation_severity = get_voltage_violations()
        
        # Get system losses
        losses = get_system_losses()
        
        # Count tap operations (to minimize unnecessary switching)
        tap_changes = sum(abs(round(control_vars[len(self.capacitor_names) + i])) for i in range(len(self.regulator_names)))
        
        # Count capacitor operations
        cap_operations = sum(round(control_vars[i]) for i in range(len(self.capacitor_names)))
        
        # Objective: minimize losses and voltage deviations with improved weighting
        objective = (
            self.loss_weight * losses + 
            self.voltage_violation_weight * violations +
            self.voltage_violation_weight * 10 * violation_severity +
            self.deviation_weight * avg_deviation +
            self.voltage_violation_weight * 2 * max_deviation +
            0.5 * tap_changes  # Small penalty for excessive tap changes
        )
        
        # Add a small reward for using capacitors when losses are high
        if losses > 150 and cap_operations > 0:
            objective -= 10 * cap_operations  # Incentivize capacitor use for high load periods
        
        return objective
    
    def optimize_for_time_step(self, time_step):
        """Run optimization for a specific time step"""
        # Initial control variables
        # For each capacitor: 0 (off) or 1 (on)
        # For each regulator: tap position (-16 to +16)
        initial_vars = np.zeros(len(self.capacitor_names) + len(self.regulator_names))
        
        # Use smarter initial values based on time of day
        # Morning and evening peaks often benefit from capacitors
        if 7 <= time_step <= 9 or 17 <= time_step <= 19:
            # During peak hours, start with capacitors ON
            for i in range(len(self.capacitor_names)):
                initial_vars[i] = 1.0
        
        # Set initial regulator taps based on time of day
        if len(self.regulator_names) > 0:
            # During high load periods, boost voltage slightly
            if 7 <= time_step <= 20:
                initial_vars[len(self.capacitor_names):] = 2.0
            else:
                initial_vars[len(self.capacitor_names):] = 0.0
        
        # Define bounds
        # Capacitors: 0 to 1
        # Regulators: -16 to 16
        bounds = []
        for _ in self.capacitor_names:
            bounds.append((0, 1))  # Capacitor bounds
        for _ in self.regulator_names:
            bounds.append((-16, 16))  # Regulator bounds
        
        # Run optimization with improved settings
        result = minimize(
            lambda x: self.objective_function(x, time_step),
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            options={
                'maxiter': 200,       # Increased max iterations
                'ftol': 1e-6,         # Tighter function tolerance
                'eps': 1e-3,          # Step size for finite difference
                'disp': True
            }
        )
        
        # If first optimization didn't find a good solution, try again with different starting point
        if result.fun > 10000:  # High objective value indicates poor solution
            print("Trying alternative starting point...")
            # Try with all capacitors on and regulators at different positions
            alt_initial_vars = np.zeros(len(self.capacitor_names) + len(self.regulator_names))
            
            # Set all capacitors on
            for i in range(len(self.capacitor_names)):
                alt_initial_vars[i] = 1.0
            
            # Set regulators to various positions
            if len(self.regulator_names) > 0:
                alt_initial_vars[len(self.capacitor_names):] = np.random.uniform(-8, 8, len(self.regulator_names))
            
            # Run optimization again
            alt_result = minimize(
                lambda x: self.objective_function(x, time_step),
                alt_initial_vars,
                method='SLSQP',
                bounds=bounds,
                options={
                    'maxiter': 200,
                    'ftol': 1e-6,
                    'eps': 1e-3,
                    'disp': True
                }
            )
            
            # Use the better result
            if alt_result.fun < result.fun:
                print("Alternative optimization found better solution")
                result = alt_result
        
        # Apply optimal control actions
        self.apply_control_actions(result.x)
        
        # Solve power flow with optimal settings
        dss.Solution.Solve()
        
        # Get results
        voltages = get_all_bus_voltages()
        losses = get_system_losses()
        violations, max_deviation, avg_deviation, violation_severity = get_voltage_violations()
        
        # Extract optimal control settings
        optimal_caps = {}
        optimal_regs = {}
        
        for i, cap_name in enumerate(self.capacitor_names):
            optimal_caps[cap_name] = round(result.x[i])
        
        for i, reg_name in enumerate(self.regulator_names):
            optimal_regs[reg_name] = round(result.x[len(self.capacitor_names) + i])
        
        return {
            'voltages': voltages,
            'losses': losses,
            'violations': violations,
            'max_deviation': max_deviation,
            'avg_deviation': avg_deviation,
            'violation_severity': violation_severity,
            'capacitors': optimal_caps,
            'regulators': optimal_regs,
            'success': result.success,
            'message': result.message
        }

def run_optimization(dss_file_path, output_dir="voltage_optimization_results", use_advanced=True):
    """Run the voltage optimization for all time steps"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize optimizer
    optimizer = VoltageOptimizer(dss_file_path)
    
    # Run optimization for each hour
    results = []
    optimized_losses = []
    optimized_violations = []
    optimized_max_deviations = []
    optimized_avg_deviations = []
    
    # Track best control settings for each hour
    best_capacitor_states = []
    best_regulator_taps = []
        
    for hour in range(24):
        print(f"\nOptimizing for hour {hour}...")
        result = optimizer.optimize_for_time_step(hour)
        results.append(result)
            
        # Store the results for plotting
        optimized_losses.append(result['losses'])
        optimized_violations.append(result['violations'])
            
        if 'max_deviation' in result:
            optimized_max_deviations.append(result['max_deviation'])
        if 'avg_deviation' in result:
            optimized_avg_deviations.append(result['avg_deviation'])
            
        # Store best control settings
        best_capacitor_states.append(result['capacitors'])
        best_regulator_taps.append(result['regulators'])
            
        # Print summary
        print(f"Hour {hour} optimization {'successful' if result['success'] else 'failed'}")
        print(f"Losses: {result['losses']:.2f} kW, Violations: {result['violations']}")
        print(f"Capacitor states: {result['capacitors']}")
        print(f"Regulator taps: {result['regulators']}")
            
    # Advanced feature: Post-process to smooth control actions
    if use_advanced:
        print("\nPost-processing to smooth control actions...")
        # This prevents excessive switching of capacitors and regulators
        smoothed_cap_states, smoothed_reg_taps = smooth_control_actions(
            best_capacitor_states, best_regulator_taps
        )
            
        # Re-evaluate with smoothed controls
        print("Re-evaluating with smoothed controls...")
        smoothed_results = evaluate_with_controls(
            dss_file_path, smoothed_cap_states, smoothed_reg_taps
        )
            
    no_control_losses = []
    no_control_violations = []
        
    # Run simulation for each hour with optimized controls
    for hour in range(24):
        # Set loads for this hour
        set_loads_for_time(hour, original_loads, load_profiles)
            
        # Apply capacitor settings
        for cap_name, state in capacitor_states[hour].items():
            set_capacitor_state(cap_name, state)
            
        # Apply regulator settings
        for reg_name, tap in regulator_taps[hour].items():
            set_regulator_tap(reg_name, tap)
            
        # Solve power flow
        dss.Solution.Solve()
            
        # Get results
        losses = get_system_losses()
        violations, _, _, _ = get_voltage_violations()
            
        optimized_losses.append(losses)
        optimized_violations.append(violations)
            
    # Run simulation for each hour without control
    dss.Basic.ClearAll()
    dss.Text.Command(f'Compile "{dss_file_path}"')
    dss.Text.Command("Set ControlMode=OFF")
        
    # Create default controls (all capacitors off, all regulators at neutral tap)
    default_cap_states = {}
    default_reg_taps = {}
        
    for cap_name in capacitor_states[0].keys():
        default_cap_states[cap_name] = 0  # All capacitors off
        
    for reg_name in regulator_taps[0].keys():
        default_reg_taps[reg_name] = 0  # All regulators at neutral tap
        
    for hour in range(24):
        # Set loads for this hour
        set_loads_for_time(hour, original_loads, load_profiles)
            
        # Apply default capacitor settings
        for cap_name, state in default_cap_states.items():
            set_capacitor_state(cap_name, state)
            
        # Apply default regulator settings
        for reg_name, tap in default_reg_taps.items():
            set_regulator_tap(reg_name, tap)
            
        # Solve power flow
        dss.Solution.Solve()
            
        # Get results
        losses = get_system_losses()
        violations, _, _, _ = get_voltage_violations()
            
        no_control_losses.append(losses)
        no_control_violations.append(violations)
                
    # === Plot results ===
    # Plot losses comparison
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(hours, optimized_losses, 'b-', marker='o', label='With Optimization')
    plt.plot(hours, no_control_losses, 'r--', marker='x', label='No Control')
        
    # Add voltage variation results if available
    if variation_losses is not None and len(variation_losses) >= 24:
        plt.plot(hours, variation_losses[:24], 'g-.', marker='^', label='With Load Variation')
        
    plt.xlabel('Hour')
    plt.ylabel('System Losses (kW)')
    plt.title('System Losses Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
        
    # Plot voltage violations comparison
    plt.subplot(3, 1, 2)
    plt.plot(hours, optimized_violations, 'b-', marker='o', label='With Optimization')
    plt.plot(hours, no_control_violations, 'r--', marker='x', label='No Control')
    plt.xlabel('Hour')
    plt.ylabel('Number of Voltage Violations')
    plt.title('Voltage Violations Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
        
    # Plot load multipliers if available
    if variation_load_multipliers is not None and len(variation_load_multipliers) >= 24:
        plt.subplot(3, 1, 3)
        plt.plot(hours, variation_load_multipliers[:24], 'g-', marker='^')
        plt.xlabel('Hour')
        plt.ylabel('Load Multiplier')
        plt.title('Load Variation Profile')
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimization_comparison.png'), dpi=300)
        
    # Save optimization results to CSV
    results_df = pd.DataFrame({
        'Hour': hours,
        'Optimized_Losses': optimized_losses,
        'No_Control_Losses': no_control_losses,
        'Optimized_Violations': optimized_violations,
        'No_Control_Violations': no_control_violations
    })
    results_df.to_csv(os.path.join(output_dir, 'optimization_results.csv'), index=False)
        
    # Create control action visualization
    cap_states = {}
    reg_taps = {}
        
    # Extract capacitor states and regulator taps for each hour
    for hour, result in enumerate(results):
        for cap, state in result['capacitors'].items():
            if cap not in cap_states:
                cap_states[cap] = []
            cap_states[cap].append(state)
            
        for reg, tap in result['regulators'].items():
            if reg not in reg_taps:
                reg_taps[reg] = []
            reg_taps[reg].append(tap)
            
    # Plot capacitor states
    if cap_states:
        plt.figure(figsize=(12, 4))
        for cap, states in cap_states.items():
            plt.step(hours, states, label=cap, where='post', linewidth=2)
        plt.xlabel('Hour')
        plt.ylabel('State (1=On, 0=Off)')
        plt.title('Optimal Capacitor States')
        plt.grid(True, alpha=0.3)
        plt.yticks([0, 1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'capacitor_states.png'), dpi=300)
            
    # Plot regulator taps
    if reg_taps:
        plt.figure(figsize=(12, 4))
        for reg, taps in reg_taps.items():
            plt.step(hours, taps, label=reg, where='post', linewidth=2)
        plt.xlabel('Hour')
        plt.ylabel('Tap Position')
        plt.title('Optimal Regulator Tap Positions')
        plt.grid(True, alpha=0.3)
        plt.yticks(range(-16, 17, 4))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'regulator_taps.png'), dpi=300)
            
    print(f"\nOptimization complete. Results saved to {output_dir}/")
    return results_df

def analyze_voltage_profiles(dss_file_path, capacitor_states, regulator_taps, variation_voltages=None, output_dir="optimization_results"):
    """Analyze voltage profiles from optimization and compare with voltage variation results"""
    # Initialize OpenDSS
    dss.Basic.ClearAll()
    dss.Text.Command(f'Compile "{dss_file_path}"')
    dss.Text.Command("Set ControlMode=OFF")
        
    # Get original loads
    original_loads = get_original_loads()
    load_profiles = generate_load_profiles(original_loads)
        
    # Prepare for results
    hours = list(range(24))
    optimized_voltages = {}
    no_control_voltages = {}
        
    # Select a few key buses to monitor
    monitor_buses = ['806', '810', '822', '848', '890']
        
    # Run simulation for each hour with optimized controls to get voltage profiles
    for hour in range(24):
        # Set loads for this hour
        set_loads_for_time(hour, original_loads, load_profiles)
            
        # Apply capacitor settings
        for cap_name, state in capacitor_states[hour].items():
            set_capacitor_state(cap_name, state)
            
        # Apply regulator settings
        for reg_name, tap in regulator_taps[hour].items():
            set_regulator_tap(reg_name, tap)
            
        # Solve power flow
        dss.Solution.Solve()
            
        # Get voltage profiles for monitored buses
        for bus in monitor_buses:
            dss.Circuit.SetActiveBus(bus)
            voltages = dss.Bus.puVmagAngle()
                
            # Extract just the voltage magnitudes (every other value)
            vmags = voltages[::2]
                
            if bus not in optimized_voltages:
                optimized_voltages[bus] = {}
                
            # Store voltages by phase
            for phase_idx, vmag in enumerate(vmags):
                phase_key = f"Phase {phase_idx+1}"
                if phase_key not in optimized_voltages[bus]:
                    optimized_voltages[bus][phase_key] = []
                optimized_voltages[bus][phase_key].append(vmag)
                
    # Run simulation for each hour without control to get voltage profiles
    dss.Basic.ClearAll()
    dss.Text.Command(f'Compile "{dss_file_path}"')
    dss.Text.Command("Set ControlMode=OFF")
        
    # Create default controls (all capacitors off, all regulators at neutral tap)
    default_cap_states = {}
    default_reg_taps = {}
        
    for cap_name in capacitor_states[0].keys():
        default_cap_states[cap_name] = 0  # All capacitors off
        
    for reg_name in regulator_taps[0].keys():
        default_reg_taps[reg_name] = 0  # All regulators at neutral tap
        
    for hour in range(24):
        # Set loads for this hour
        set_loads_for_time(hour, original_loads, load_profiles)
            
        # Apply default capacitor settings
        for cap_name, state in default_cap_states.items():
            set_capacitor_state(cap_name, state)
            
        # Apply default regulator settings
        for reg_name, tap in default_reg_taps.items():
            set_regulator_tap(reg_name, tap)
            
        # Solve power flow
        dss.Solution.Solve()
            
        # Get voltage profiles for monitored buses
        for bus in monitor_buses:
            dss.Circuit.SetActiveBus(bus)
            voltages = dss.Bus.puVmagAngle()
                
            # Extract just the voltage magnitudes (every other value)
            vmags = voltages[::2]
                
            if bus not in no_control_voltages:
                no_control_voltages[bus] = {}
                
            # Store voltages by phase
            for phase_idx, vmag in enumerate(vmags):
                phase_key = f"Phase {phase_idx+1}"
                if phase_key not in no_control_voltages[bus]:
                    no_control_voltages[bus][phase_key] = []
                no_control_voltages[bus][phase_key].append(vmag)
                
    # Plot voltage profiles for each monitored bus
    for bus in monitor_buses:
        plt.figure(figsize=(12, 8))
        plt.suptitle(f'Voltage Profile for Bus {bus}')
            
        for phase_idx, phase_key in enumerate(['Phase 1', 'Phase 2', 'Phase 3']):
            if phase_key in optimized_voltages[bus] and phase_key in no_control_voltages[bus]:
                plt.subplot(3, 1, phase_idx+1)
                plt.plot(hours, optimized_voltages[bus][phase_key], 'b-', marker='o', label='With Optimization')
                plt.plot(hours, no_control_voltages[bus][phase_key], 'r--', marker='x', label='No Control')
                    
                # Add voltage variation results if available for bus 806
                if variation_voltages is not None and bus == '806':
                    variation_phase_key = f'Phase {chr(65+phase_idx)}'  # Convert to 'Phase A', 'Phase B', etc.
                    if variation_phase_key in variation_voltages.columns:
                        plt.plot(hours, variation_voltages[variation_phase_key].values[:24], 'g-.', marker='^', label='With Load Variation')
                    
                plt.axhline(y=1.05, color='k', linestyle='--', alpha=0.5, label='Upper Limit')
                plt.axhline(y=0.95, color='k', linestyle='--', alpha=0.5, label='Lower Limit')
                plt.xlabel('Hour')
                plt.ylabel('Voltage (p.u.)')
                plt.title(f'{phase_key}')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig(os.path.join(output_dir, f'voltage_profile_bus_{bus}.png'), dpi=300)
            
    return optimized_voltages, no_control_voltages

def run_optimization(dss_file_path, output_dir="voltage_optimization_results", use_advanced=True):
    """Run voltage optimization for all time steps"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    # Initialize optimizer
    optimizer = VoltageOptimizer(dss_file_path)
        
    # Run optimization for each hour
    results = []
    best_capacitor_states = []
    best_regulator_taps = []
    optimized_losses = []
    optimized_violations = []
        
    for hour in range(24):
        print(f"\nOptimizing for hour {hour}...")
        result = optimizer.optimize_for_time_step(hour)
        results.append(result)
            
        # Store best control settings
        best_capacitor_states.append(result['capacitors'])
        best_regulator_taps.append(result['regulators'])
            
        # Print summary
        print(f"Hour {hour} optimization {'successful' if result['success'] else 'failed'}")
        print(f"Losses: {result['losses']:.2f} kW, Violations: {result['violations']}")
        print(f"Capacitor states: {result['capacitors']}")
        print(f"Regulator taps: {result['regulators']}")
            
        # Store optimization results
        optimized_losses.append(result['losses'])
        optimized_violations.append(result['violations'])
            
    # Advanced feature: Post-process to smooth control actions
    if use_advanced:
        print("\nPost-processing to smooth control actions...")
        # This prevents excessive switching of capacitors and regulators
        smoothed_cap_states, smoothed_reg_taps = smooth_control_actions(
            best_capacitor_states, best_regulator_taps
        )
            
        # Re-evaluate with smoothed controls
        print("Re-evaluating with smoothed controls...")
        smoothed_results = evaluate_with_controls(
            dss_file_path, smoothed_cap_states, smoothed_reg_taps
        )
            
        # Update results if smoothed controls are good enough
        if smoothed_results['avg_losses'] < 1.1 * np.mean(optimized_losses):
            print("Using smoothed control actions (reduced switching)")
            optimized_losses = smoothed_results['losses']
            optimized_violations = smoothed_results['violations']
            best_capacitor_states = smoothed_cap_states
            best_regulator_taps = smoothed_reg_taps
        
    # Define hours for x-axis
    hours = list(range(24))
        
    # Create comparison with no control
    # Reset all controls to default
    dss.Basic.ClearAll()
    dss.Text.Command(f'Compile "{dss_file_path}"')
    dss.Text.Command("Set ControlMode=OFF")
        
    # Get original loads
    original_loads = get_original_loads()
    load_profiles = generate_load_profiles(original_loads)
        
    # Run simulation without optimization
    no_control_losses = []
    no_control_violations = []
        
    for hour in range(24):
        # Set loads for this hour
        set_loads_for_time(hour, original_loads, load_profiles)
            
        # Turn off all capacitors
        for cap in get_capacitor_names():
            set_capacitor_state(cap, 0)
            
        # Set all regulators to neutral position
        for reg in get_regulator_names():
            set_regulator_tap(reg, 0)
            
        # Solve power flow
        dss.Solution.Solve()
            
        # Get results
        losses = get_system_losses()
        violations, _, _, _ = get_voltage_violations()
            
        no_control_losses.append(losses)
        no_control_violations.append(violations)
            
    # Load voltage variation results from voltage_profile_regulation.py
    variation_results = load_voltage_variation_results()
        
    # Extract data for comparison if available
    variation_losses = None
    variation_load_multipliers = None
    variation_voltages = None
        
    if variation_results:
        if 'losses' in variation_results:
            # Convert minute-by-minute losses to hourly average
            losses_df = variation_results['losses']
            # Extract hour from time string (format is 'hour:minute')
            losses_df['Hour'] = losses_df['Time'].apply(lambda x: int(x.split(':')[0]))
            # Group by hour and calculate average
            hourly_losses = losses_df.groupby('Hour')['Losses (kW)'].mean().reset_index()
            variation_losses = hourly_losses['Losses (kW)'].values
            
        if 'load_variation' in variation_results:
            # Convert minute-by-minute load multipliers to hourly average
            load_df = variation_results['load_variation']
            # Extract hour from time string
            load_df['Hour'] = load_df['Time'].apply(lambda x: int(x.split(':')[0]))
            # Group by hour and calculate average
            hourly_load = load_df.groupby('Hour')['Load Multiplier'].mean().reset_index()
            variation_load_multipliers = hourly_load['Load Multiplier'].values
            
        if 'bus_voltages' in variation_results:
            # Extract voltage data for main bus
            voltage_df = variation_results['bus_voltages']
            # Extract hour from time string
            voltage_df['Hour'] = voltage_df['Time'].apply(lambda x: int(x.split(':')[0]))
            # Group by hour and calculate average for each phase
            hourly_voltages = voltage_df.groupby('Hour')[['Phase A', 'Phase B', 'Phase C']].mean().reset_index()
            variation_voltages = hourly_voltages
    
    # === Plot results ===
    # Plot losses comparison
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(hours, optimized_losses, 'b-', marker='o', label='With Optimization')
    plt.plot(hours, no_control_losses, 'r--', marker='x', label='No Control')
    
    # Add voltage variation results if available
    if variation_losses is not None and len(variation_losses) >= 24:
        plt.plot(hours, variation_losses[:24], 'g-.', marker='^', label='With Load Variation')
    
    plt.xlabel('Hour')
    plt.ylabel('System Losses (kW)')
    plt.title('System Losses Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot voltage violations comparison
    plt.subplot(3, 1, 2)
    plt.plot(hours, optimized_violations, 'b-', marker='o', label='With Optimization')
    plt.plot(hours, no_control_violations, 'r--', marker='x', label='No Control')
    plt.xlabel('Hour')
    plt.ylabel('Number of Voltage Violations')
    plt.title('Voltage Violations Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot load multipliers if available
    if variation_load_multipliers is not None and len(variation_load_multipliers) >= 24:
        plt.subplot(3, 1, 3)
        plt.plot(hours, variation_load_multipliers[:24], 'g-', marker='^')
        plt.xlabel('Hour')
        plt.ylabel('Load Multiplier')
        plt.title('Load Variation Profile')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimization_comparison.png'), dpi=300)
    
    # Save optimization results to CSV
    results_df = pd.DataFrame({
        'Hour': hours,
        'Optimized_Losses': optimized_losses,
        'No_Control_Losses': no_control_losses,
        'Optimized_Violations': optimized_violations,
        'No_Control_Violations': no_control_violations
    })
    results_df.to_csv(os.path.join(output_dir, 'optimization_results.csv'), index=False)
    
    # Create control action visualization
    cap_states = {}
    reg_taps = {}
    
    # Extract capacitor states and regulator taps for each hour
    for hour, result in enumerate(results):
        for cap, state in result['capacitors'].items():
            if cap not in cap_states:
                cap_states[cap] = []
            cap_states[cap].append(state)
        
        for reg, tap in result['regulators'].items():
            if reg not in reg_taps:
                reg_taps[reg] = []
            reg_taps[reg].append(tap)
    
    # Plot capacitor states
    if cap_states:
        plt.figure(figsize=(12, 4))
        for cap, states in cap_states.items():
            plt.step(hours, states, label=cap, where='post', linewidth=2)
        plt.xlabel('Hour')
        plt.ylabel('State (1=On, 0=Off)')
        plt.title('Optimal Capacitor States')
        plt.grid(True, alpha=0.3)
        plt.yticks([0, 1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'capacitor_states.png'), dpi=300)
    
    # Plot regulator taps
    if reg_taps:
        plt.figure(figsize=(12, 4))
        for reg, taps in reg_taps.items():
            plt.step(hours, taps, label=reg, where='post', linewidth=2)
        plt.xlabel('Hour')
        plt.ylabel('Tap Position')
        plt.title('Optimal Regulator Tap Positions')
        plt.grid(True, alpha=0.3)
        plt.yticks(range(-16, 17, 4))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'regulator_taps.png'), dpi=300)
    
    print(f"\nOptimization complete. Results saved to {output_dir}/")
    return results_df

def smooth_control_actions(capacitor_states, regulator_taps):
    """Smooth control actions to reduce excessive switching"""
    smoothed_cap_states = []
    smoothed_reg_taps = []
    
    # Process capacitor states to reduce switching
    for hour in range(24):
        if hour == 0:
            smoothed_cap_states.append(capacitor_states[hour])
        else:
            # For each capacitor, check if it's worth switching
            new_cap_state = {}
            for cap_name in capacitor_states[hour].keys():
                # Get previous state
                prev_state = smoothed_cap_states[-1].get(cap_name, 0)
                current_state = capacitor_states[hour].get(cap_name, 0)
                
                # Only switch if state has been consistent for 2 hours
                if hour >= 2 and current_state == capacitor_states[hour-1].get(cap_name, 0):
                    new_cap_state[cap_name] = current_state
                else:
                    # Otherwise maintain previous state to avoid frequent switching
                    new_cap_state[cap_name] = prev_state
            
            smoothed_cap_states.append(new_cap_state)
    
    # Process regulator taps to reduce excessive movement
    for hour in range(24):
        if hour == 0:
            smoothed_reg_taps.append(regulator_taps[hour])
        else:
            # For each regulator, limit tap changes
            new_reg_taps = {}
            for reg_name in regulator_taps[hour].keys():
                # Get previous tap position
                prev_tap = smoothed_reg_taps[-1].get(reg_name, 0)
                current_tap = regulator_taps[hour].get(reg_name, 0)
                
                # Limit tap changes to at most 2 positions at a time
                if abs(current_tap - prev_tap) <= 2:
                    new_reg_taps[reg_name] = current_tap
                else:
                    # Move at most 2 taps in the direction of the desired position
                    if current_tap > prev_tap:
                        new_reg_taps[reg_name] = prev_tap + 2
                    else:
                        new_reg_taps[reg_name] = prev_tap - 2
            
            smoothed_reg_taps.append(new_reg_taps)
    
    return smoothed_cap_states, smoothed_reg_taps

def load_voltage_variation_results(variation_results_dir="voltage_variation_results"):
    """Load voltage variation results from the voltage profile regulation analysis"""
    results = {}
    
    try:
        # Load bus voltages
        bus_voltage_file = os.path.join(variation_results_dir, "bus_806_voltages.csv")
        if os.path.exists(bus_voltage_file):
            bus_voltages = pd.read_csv(bus_voltage_file)
            results['bus_voltages'] = bus_voltages
        
        # Load system losses
        losses_file = os.path.join(variation_results_dir, "system_losses.csv")
        if os.path.exists(losses_file):
            losses = pd.read_csv(losses_file)
            results['losses'] = losses
        
        # Load load variation data
        load_file = os.path.join(variation_results_dir, "load_variation.csv")
        if os.path.exists(load_file):
            load_variation = pd.read_csv(load_file)
            results['load_variation'] = load_variation
            
        print(f"Successfully loaded voltage variation results from {variation_results_dir}")
        return results
    except Exception as e:
        print(f"Error loading voltage variation results: {e}")
        return None

def evaluate_with_controls(dss_file_path, capacitor_states, regulator_taps):
    """Evaluate system performance with given control settings"""
    # Initialize OpenDSS
    dss.Basic.ClearAll()
    dss.Text.Command(f'Compile "{dss_file_path}"')
    dss.Text.Command("Set ControlMode=OFF")
    
    # Get original loads
    original_loads = get_original_loads()
    load_profiles = generate_load_profiles(original_loads)
    
    # Run simulation with specified controls
    losses = []
    violations = []
    
    for hour in range(24):
        # Set loads for this hour
        set_loads_for_time(hour, original_loads, load_profiles)
        
        # Apply capacitor settings
        for cap_name, state in capacitor_states[hour].items():
            set_capacitor_state(cap_name, state)
        
        # Apply regulator settings
        for reg_name, tap in regulator_taps[hour].items():
            set_regulator_tap(reg_name, tap)
        
        # Solve power flow
        dss.Solution.Solve()
        
        # Get results
        loss_kw = get_system_losses()
        violations_count, _, _, _ = get_voltage_violations()
        
        losses.append(loss_kw)
        violations.append(violations_count)
    
    return {
        'losses': losses,
        'violations': violations,
        'avg_losses': np.mean(losses),
        'avg_violations': np.mean(violations)
    }

if __name__ == "__main__":
    # Path to the DSS file
    dss_file_path = "ieee34Mod1.dss"
    
    # Run optimization
    results_df = run_optimization(dss_file_path)
    
    # Load voltage variation results
    variation_results = load_voltage_variation_results()
    variation_voltages = None
    
    if variation_results and 'bus_voltages' in variation_results:
        # Process voltage data for comparison
        voltage_df = variation_results['bus_voltages']
        voltage_df['Hour'] = voltage_df['Time'].apply(lambda x: int(x.split(':')[0]))
        hourly_voltages = voltage_df.groupby('Hour')[['Phase A', 'Phase B', 'Phase C']].mean().reset_index()
        variation_voltages = hourly_voltages
    
    # Get the capacitor states and regulator taps from the optimization
    optimizer = VoltageOptimizer(dss_file_path)
    results = [optimizer.optimize_for_time_step(hour) for hour in range(24)]
    
    # Extract capacitor states and regulator taps
    capacitor_states = [result['capacitors'] for result in results]
    regulator_taps = [result['regulators'] for result in results]
    
    # Analyze voltage profiles
    print("\nAnalyzing voltage profiles...")
    analyze_voltage_profiles(dss_file_path, capacitor_states, regulator_taps, 
                           variation_voltages, output_dir="voltage_optimization_results")
    
    print("\nVoltage profile analysis complete. Results saved to voltage_optimization_results/")
    print("\nIntegration of voltage variation results with optimization complete.")
    print("Check the output directory for comprehensive analysis results.")
    
    # Print summary of improvements
    print("\nSummary of Optimization Results:")
    avg_opt_losses = results_df['Optimized_Losses'].mean()
    avg_no_control_losses = results_df['No_Control_Losses'].mean()
    loss_reduction = ((avg_no_control_losses - avg_opt_losses) / avg_no_control_losses) * 100
    
    avg_opt_violations = results_df['Optimized_Violations'].mean()
    avg_no_control_violations = results_df['No_Control_Violations'].mean()
    violation_reduction = ((avg_no_control_violations - avg_opt_violations) / max(1, avg_no_control_violations)) * 100
    
    print(f"Average Loss Reduction: {loss_reduction:.2f}%")
    print(f"Average Voltage Violation Reduction: {violation_reduction:.2f}%")
    
    if variation_results and 'losses' in variation_results:
        # Compare with voltage variation approach
        losses_df = variation_results['losses']
        losses_df['Hour'] = losses_df['Time'].apply(lambda x: int(x.split(':')[0]))
        hourly_losses = losses_df.groupby('Hour')['Losses (kW)'].mean().reset_index()
        avg_var_losses = hourly_losses['Losses (kW)'].mean()
        
        var_loss_reduction = ((avg_no_control_losses - avg_var_losses) / avg_no_control_losses) * 100
        opt_vs_var = ((avg_var_losses - avg_opt_losses) / avg_var_losses) * 100
        
        print(f"\nComparison with Voltage Variation Approach:")
        print(f"Voltage Variation Loss Reduction: {var_loss_reduction:.2f}%")
        print(f"Optimization vs Variation Improvement: {opt_vs_var:.2f}%")
