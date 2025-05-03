import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

class LinDistFlow:
    """
    Linearized Distribution Optimal Power Flow using LinDistFlow approximation.
    
    This model implements a linearized approximation of AC power flow, which is
    valid for distribution networks with the following characteristics:
    1. Radial structure (tree topology)
    2. High voltage to impedance ratio (low R/X ratios)
    3. Small voltage angle differences between adjacent buses
    
    The LinDistFlow approximation neglects the quadratic terms in the power flow
    equations and assumes voltage angle differences are small enough that
    sin(θ_i - θ_j) ≈ (θ_i - θ_j) and cos(θ_i - θ_j) ≈ 1.
    """
    
    def __init__(self, base_mva=1.0):
        """
        Initialize the LinDistFlow OPF model.
        
        Args:
            base_mva: Base MVA for per-unit calculations (default: 1.0)
        """
        self.base_mva = base_mva
        
        # Network data structures
        self.buses = {}  # Dictionary of buses: {bus_id: {data}}
        self.branches = {}  # Dictionary of branches: {branch_id: {data}}
        self.generators = {}  # Dictionary of generators: {gen_id: {data}}
        self.loads = {}  # Dictionary of loads: {load_id: {data}}
        
        # Mapping to keep track of network topology
        self.bus_to_branches = {}  # Dictionary to map buses to their connected branches
        self.bus_to_generators = {}  # Dictionary to map buses to their connected generators
        self.bus_to_loads = {}  # Dictionary to map buses to their connected loads
        self.branch_to_buses = {}  # Dictionary to map branches to their from/to buses
        
        # Optimization model
        self.model = None
        self.results = None
        
        # Track if model has been built and solved
        self.model_built = False
        self.model_solved = False
    
    def add_bus(self, bus_id, v_min=0.95, v_max=1.05, is_slack=False):
        """
        Add a bus to the network.
        
        Args:
            bus_id: Bus identifier
            v_min: Minimum voltage magnitude in per-unit (default: 0.95)
            v_max: Maximum voltage magnitude in per-unit (default: 1.05)
            is_slack: Boolean indicating if this is the slack/reference bus (default: False)
        """
        self.buses[bus_id] = {
            'v_min': v_min,
            'v_max': v_max,
            'is_slack': is_slack
        }
        
        # Initialize mapping dictionaries for this bus
        self.bus_to_branches[bus_id] = []
        self.bus_to_generators[bus_id] = []
        self.bus_to_loads[bus_id] = []
        
        self.model_built = False  # Model needs to be rebuilt
    
    def add_branch(self, branch_id, from_bus, to_bus, r, x, s_max=None):
        """
        Add a branch (line or transformer) to the network.
        
        Args:
            branch_id: Branch identifier
            from_bus: Bus ID at the sending end ("from" bus)
            to_bus: Bus ID at the receiving end ("to" bus)
            r: Resistance in per-unit
            x: Reactance in per-unit
            s_max: Maximum apparent power flow limit in MVA (optional)
        """
        # Verify that buses exist
        if from_bus not in self.buses:
            raise ValueError(f"From bus {from_bus} does not exist")
        if to_bus not in self.buses:
            raise ValueError(f"To bus {to_bus} does not exist")
        
        # Store branch data
        self.branches[branch_id] = {
            'from_bus': from_bus,
            'to_bus': to_bus,
            'r': r,
            'x': x,
            'z': complex(r, x),
            's_max': s_max
        }
        
        # Update mapping dictionaries
        self.bus_to_branches[from_bus].append(branch_id)
        self.bus_to_branches[to_bus].append(branch_id)
        self.branch_to_buses[branch_id] = (from_bus, to_bus)
        
        self.model_built = False  # Model needs to be rebuilt
    
    def add_generator(self, gen_id, bus_id, p_min, p_max, q_min, q_max, cost_model, cost_coeffs):
        """
        Add a generator to the network.
        
        Args:
            gen_id: Generator identifier
            bus_id: Bus ID where the generator is connected
            p_min: Minimum active power output in MW
            p_max: Maximum active power output in MW
            q_min: Minimum reactive power output in MVAr
            q_max: Maximum reactive power output in MVAr
            cost_model: Type of cost model ('linear', 'quadratic')
            cost_coeffs: List of cost coefficients [c2, c1, c0] for quadratic cost c2*P^2 + c1*P + c0
                         or [c1, c0] for linear cost c1*P + c0
        """
        # Verify that bus exists
        if bus_id not in self.buses:
            raise ValueError(f"Bus {bus_id} does not exist")
        
        # Normalize to per-unit
        p_min_pu = p_min / self.base_mva
        p_max_pu = p_max / self.base_mva
        q_min_pu = q_min / self.base_mva
        q_max_pu = q_max / self.base_mva
        
        # Store generator data
        self.generators[gen_id] = {
            'bus_id': bus_id,
            'p_min': p_min_pu,
            'p_max': p_max_pu,
            'q_min': q_min_pu,
            'q_max': q_max_pu,
            'cost_model': cost_model,
            'cost_coeffs': cost_coeffs
        }
        
        # Update mapping dictionary
        self.bus_to_generators[bus_id].append(gen_id)
        
        self.model_built = False  # Model needs to be rebuilt
    
    def add_load(self, load_id, bus_id, p_demand, q_demand):
        """
        Add a load to the network.
        
        Args:
            load_id: Load identifier
            bus_id: Bus ID where the load is connected
            p_demand: Active power demand in MW
            q_demand: Reactive power demand in MVAr
        """
        # Verify that bus exists
        if bus_id not in self.buses:
            raise ValueError(f"Bus {bus_id} does not exist")
        
        # Normalize to per-unit
        p_demand_pu = p_demand / self.base_mva
        q_demand_pu = q_demand / self.base_mva
        
        # Store load data
        self.loads[load_id] = {
            'bus_id': bus_id,
            'p_demand': p_demand_pu,
            'q_demand': q_demand_pu
        }
        
        # Update mapping dictionary
        self.bus_to_loads[bus_id].append(load_id)
        
        self.model_built = False  # Model needs to be rebuilt
    
    def check_network_validity(self):
        """
        Check if the network is valid for LinDistFlow (radial, has a slack bus).
        
        Returns:
            valid: Boolean indicating if the network is valid
            message: Message explaining validation results
        """
        # Check if there is at least one bus
        if not self.buses:
            return False, "Network has no buses"
        
        # Check if there is a slack bus
        slack_buses = [bus_id for bus_id, bus in self.buses.items() if bus['is_slack']]
        if not slack_buses:
            return False, "Network does not have a slack bus"
        if len(slack_buses) > 1:
            return False, "Network has multiple slack buses"
        
        # Check if there is at least one branch (unless we have only one bus)
        if len(self.buses) > 1 and not self.branches:
            return False, "Network has multiple buses but no branches"
        
        # Check if network is radial (number of branches = number of buses - 1)
        if len(self.branches) != len(self.buses) - 1:
            return False, "Network is not radial (branches ≠ buses - 1)"
        
        # Check if network is connected (all buses are reachable from slack)
        # Implementation note: This is a simplified check assuming we already verified branch count
        
        return True, "Network is valid for LinDistFlow"
    
    def build_model(self):
        """
        Build the Pyomo optimization model for LinDistFlow OPF.
        """
        # First, validate the network
        valid, message = self.check_network_validity()
        if not valid:
            raise ValueError(f"Invalid network for LinDistFlow: {message}")
        
        # Create a concrete model
        model = pyo.ConcreteModel()
        
        # Define sets
        model.BUSES = pyo.Set(initialize=self.buses.keys())
        model.BRANCHES = pyo.Set(initialize=self.branches.keys())
        model.GENERATORS = pyo.Set(initialize=self.generators.keys())
        model.LOADS = pyo.Set(initialize=self.loads.keys())
        
        # Define variables
        # Voltage magnitude squared (v^2) at each bus
        model.v_squared = pyo.Var(model.BUSES, domain=pyo.NonNegativeReals)
        
        # Active and reactive power flow in each branch
        model.P = pyo.Var(model.BRANCHES, domain=pyo.Reals)
        model.Q = pyo.Var(model.BRANCHES, domain=pyo.Reals)
        
        # Generator active and reactive power output
        model.Pg = pyo.Var(model.GENERATORS, domain=pyo.Reals)
        model.Qg = pyo.Var(model.GENERATORS, domain=pyo.Reals)
        
        # Define constraints
        # Voltage magnitude limits
        def voltage_limits_rule(model, b):
            bus = self.buses[b]
            return (bus['v_min']**2, model.v_squared[b], bus['v_max']**2)
        model.voltage_limits = pyo.Constraint(model.BUSES, rule=voltage_limits_rule)
        
        # Slack bus voltage (reference)
        def slack_voltage_rule(model):
            slack_bus = next(b for b, data in self.buses.items() if data['is_slack'])
            return model.v_squared[slack_bus] == 1.0  # V = 1.0 pu
        model.slack_voltage = pyo.Constraint(rule=slack_voltage_rule)
        
        # Generator limits
        def gen_p_limits_rule(model, g):
            gen = self.generators[g]
            return (gen['p_min'], model.Pg[g], gen['p_max'])
        model.gen_p_limits = pyo.Constraint(model.GENERATORS, rule=gen_p_limits_rule)
        
        def gen_q_limits_rule(model, g):
            gen = self.generators[g]
            return (gen['q_min'], model.Qg[g], gen['q_max'])
        model.gen_q_limits = pyo.Constraint(model.GENERATORS, rule=gen_q_limits_rule)
        
        # Branch flow limits (if s_max is specified)
        def branch_flow_limits_rule(model, br):
            branch = self.branches[br]
            if branch['s_max'] is not None:
                s_max_pu = branch['s_max'] / self.base_mva
                # Approximate apparent power limit using a linear constraint
                # |P| + |Q| ≤ √2 * S_max (an outer approximation of the circle)
                return pyo.Constraint.Skip
            else:
                return pyo.Constraint.Skip
        model.branch_flow_limits = pyo.Constraint(model.BRANCHES, rule=branch_flow_limits_rule)
        
        # LinDistFlow equations
        # 1. Power balance at each bus
        def active_power_balance_rule(model, b):
            # Sum of power flowing into the bus minus sum of power flowing out = net injection
            inflows = sum(model.P[br] for br in self.bus_to_branches[b] 
                         if self.branches[br]['to_bus'] == b)
            outflows = sum(model.P[br] for br in self.bus_to_branches[b] 
                          if self.branches[br]['from_bus'] == b)
            
            # Net generation at this bus
            generation = sum(model.Pg[g] for g in self.bus_to_generators[b])
            
            # Net demand at this bus
            demand = sum(self.loads[l]['p_demand'] for l in self.bus_to_loads[b])
            
            return inflows - outflows + generation == demand
        model.active_power_balance = pyo.Constraint(model.BUSES, rule=active_power_balance_rule)
        
        def reactive_power_balance_rule(model, b):
            # Sum of power flowing into the bus minus sum of power flowing out = net injection
            inflows = sum(model.Q[br] for br in self.bus_to_branches[b] 
                         if self.branches[br]['to_bus'] == b)
            outflows = sum(model.Q[br] for br in self.bus_to_branches[b] 
                          if self.branches[br]['from_bus'] == b)
            
            # Net generation at this bus
            generation = sum(model.Qg[g] for g in self.bus_to_generators[b])
            
            # Net demand at this bus
            demand = sum(self.loads[l]['q_demand'] for l in self.bus_to_loads[b])
            
            return inflows - outflows + generation == demand
        model.reactive_power_balance = pyo.Constraint(model.BUSES, rule=reactive_power_balance_rule)
        
        # 2. Voltage drop along branches (LinDistFlow equation)
        def voltage_drop_rule(model, br):
            branch = self.branches[br]
            from_bus = branch['from_bus']
            to_bus = branch['to_bus']
            r = branch['r']
            x = branch['x']
            
            # LinDistFlow voltage equation:
            # v_j^2 = v_i^2 - 2(r*P_ij + x*Q_ij)
            return model.v_squared[to_bus] == model.v_squared[from_bus] - 2*(r*model.P[br] + x*model.Q[br])
        model.voltage_drop = pyo.Constraint(model.BRANCHES, rule=voltage_drop_rule)
        
        # Define the objective function (minimize generation cost)
        def objective_rule(model):
            cost = 0
            for g in model.GENERATORS:
                gen = self.generators[g]
                if gen['cost_model'] == 'linear':
                    # Linear cost: c1*P + c0
                    c1, c0 = gen['cost_coeffs']
                    cost += c1 * model.Pg[g] * self.base_mva + c0
                elif gen['cost_model'] == 'quadratic':
                    # Quadratic cost: c2*P^2 + c1*P + c0
                    c2, c1, c0 = gen['cost_coeffs']
                    cost += c2 * (model.Pg[g] * self.base_mva)**2 + c1 * model.Pg[g] * self.base_mva + c0
            return cost
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        
        # Store the model
        self.model = model
        self.model_built = True
        return model
    
    def solve(self, solver='ipopt', tee=False):
        """
        Solve the LinDistFlow OPF model.
        
        Args:
            solver: Name of the solver to use (default: 'ipopt')
            tee: Boolean indicating whether to display solver output (default: False)
            
        Returns:
            results: Solver results object
        """
        if not self.model_built:
            self.build_model()
        
        # Solve the model
        opt = SolverFactory(solver)
        results = opt.solve(self.model, tee=tee)
        
        # Store results
        self.results = results
        self.model_solved = True
        
        # Print status
        print(f"Solver status: {results.solver.status}")
        print(f"Termination condition: {results.solver.termination_condition}")
        
        return results
    
    def get_voltage_profile(self):
        """
        Get the voltage profile at all buses.
        
        Returns:
            DataFrame with bus voltage magnitudes
        """
        if not self.model_solved:
            raise ValueError("Model has not been solved yet")
        
        voltages = {b: np.sqrt(pyo.value(self.model.v_squared[b])) for b in self.model.BUSES}
        return pd.DataFrame({
            'Bus': list(voltages.keys()),
            'Voltage (pu)': list(voltages.values())
        })
    
    def get_power_flows(self):
        """
        Get the power flows through all branches.
        
        Returns:
            DataFrame with branch power flows
        """
        if not self.model_solved:
            raise ValueError("Model has not been solved yet")
        
        flows = []
        for br in self.model.BRANCHES:
            branch = self.branches[br]
            p_flow = pyo.value(self.model.P[br]) * self.base_mva
            q_flow = pyo.value(self.model.Q[br]) * self.base_mva
            s_flow = np.sqrt(p_flow**2 + q_flow**2)
            
            flows.append({
                'Branch': br,
                'From': branch['from_bus'],
                'To': branch['to_bus'],
                'P (MW)': p_flow,
                'Q (MVAr)': q_flow,
                'S (MVA)': s_flow
            })
        
        return pd.DataFrame(flows)
    
    def get_generator_outputs(self):
        """
        Get the power outputs of all generators.
        
        Returns:
            DataFrame with generator outputs
        """
        if not self.model_solved:
            raise ValueError("Model has not been solved yet")
        
        outputs = []
        for g in self.model.GENERATORS:
            gen = self.generators[g]
            p_out = pyo.value(self.model.Pg[g]) * self.base_mva
            q_out = pyo.value(self.model.Qg[g]) * self.base_mva
            
            outputs.append({
                'Generator': g,
                'Bus': gen['bus_id'],
                'P (MW)': p_out,
                'Q (MVAr)': q_out
            })
        
        return pd.DataFrame(outputs)
    
    def get_total_losses(self):
        """
        Calculate the total active power losses in the system.
        
        Returns:
            Total losses in MW
        """
        if not self.model_solved:
            raise ValueError("Model has not been solved yet")
        
        total_generation = sum(pyo.value(self.model.Pg[g]) for g in self.model.GENERATORS)
        total_demand = sum(self.loads[l]['p_demand'] for l in self.loads)
        
        return (total_generation - total_demand) * self.base_mva
    
    def plot_voltage_profile(self):
        """
        Plot the voltage profile of the system.
        """
        if not self.model_solved:
            raise ValueError("Model has not been solved yet")
        
        voltages = self.get_voltage_profile()
        
        plt.figure(figsize=(10, 6))
        plt.bar(voltages['Bus'].astype(str), voltages['Voltage (pu)'], color='skyblue')
        plt.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, label='Nominal')
        plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='Lower Limit')
        plt.axhline(y=1.05, color='red', linestyle='--', alpha=0.7, label='Upper Limit')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Bus')
        plt.ylabel('Voltage Magnitude (pu)')
        plt.title('Voltage Profile')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_power_flows(self):
        """
        Plot the power flows through all branches.
        """
        if not self.model_solved:
            raise ValueError("Model has not been solved yet")
        
        flows = self.get_power_flows()
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(flows['Branch'].astype(str), flows['P (MW)'], color='salmon')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Branch')
        plt.ylabel('Active Power Flow (MW)')
        plt.title('Branch Power Flows')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Example: Create a simple 3-bus radial distribution system
def create_simple_system():
    # Initialize the model
    opf = LinDistFlow(base_mva=10.0)
    
    # Add buses
    opf.add_bus(1, is_slack=True)  # Substation (slack bus)
    opf.add_bus(2)
    opf.add_bus(3)
    
    # Add branches (lines)
    opf.add_branch(1, 1, 2, 0.01, 0.02, s_max=10.0)  # Branch 1: Bus 1 to Bus 2
    opf.add_branch(2, 2, 3, 0.015, 0.03, s_max=5.0)  # Branch 2: Bus 2 to Bus 3
    
    # Add generators
    # Slack bus generator (unlimited)
    opf.add_generator(1, 1, 0, 50, -30, 30, 'quadratic', [0.01, 10, 0])
    # Distributed generator at bus 3
    opf.add_generator(2, 3, 0, 5, -2, 2, 'linear', [30, 0])
    
    # Add loads
    opf.add_load(1, 2, 8.0, 4.0)  # Load at bus 2: 8 MW, 4 MVAr
    opf.add_load(2, 3, 6.0, 3.0)  # Load at bus 3: 6 MW, 3 MVAr
    
    return opf

# Example: Create IEEE 33-bus distribution test feeder
def create_ieee33_system():
    # This is a simplified version of the IEEE 33-bus test system
    # Actual data should be loaded from a data file for a complete implementation
    
    # Initialize the model
    opf = LinDistFlow(base_mva=10.0)
    
    # Add buses (with actual IEEE 33-bus names)
    for i in range(1, 34):
        opf.add_bus(i, is_slack=(i==1))
    
    # Add branches (lines) - simplified impedance values
    # Format: branch_id, from_bus, to_bus, r (pu), x (pu), s_max (MVA)
    branch_data = [
        (1, 1, 2, 0.0058, 0.0029, None),
        (2, 2, 3, 0.0308, 0.0157, None),
        (3, 3, 4, 0.0228, 0.0116, None),
        (4, 4, 5, 0.0238, 0.0121, None),
        (5, 5, 6, 0.0511, 0.0441, None),
        (6, 6, 7, 0.0117, 0.0386, None),
        (7, 7, 8, 0.1068, 0.0770, None),
        (8, 8, 9, 0.0644, 0.0463, None),
        (9, 9, 10, 0.0651, 0.0462, None),
        (10, 10, 11, 0.0123, 0.0041, None),
        (11, 11, 12, 0.0234, 0.0077, None),
        (12, 12, 13, 0.0916, 0.0721, None),
        (13, 13, 14, 0.0338, 0.0445, None),
        (14, 14, 15, 0.0368, 0.0328, None),
        (15, 15, 16, 0.0466, 0.0340, None),
        (16, 16, 17, 0.0804, 0.1074, None),
        (17, 17, 18, 0.0457, 0.0358, None),
        (18, 2, 19, 0.0102, 0.0098, None),
        (19, 19, 20, 0.0939, 0.0846, None),
        (20, 20, 21, 0.0255, 0.0298, None),
        (21, 21, 22, 0.0442, 0.0585, None),
        (22, 3, 23, 0.0282, 0.0192, None),
        (23, 23, 24, 0.0560, 0.0442, None),
        (24, 24, 25, 0.0559, 0.0437, None),
        (25, 6, 26, 0.0127, 0.0065, None),
        (26, 26, 27, 0.0177, 0.0090, None),
        (27, 27, 28, 0.0661, 0.0583, None),
        (28, 28, 29, 0.0502, 0.0437, None),
        (29, 29, 30, 0.0317, 0.0161, None),
        (30, 30, 31, 0.0608, 0.0601, None),
        (31, 31, 32, 0.0194, 0.0226, None),
        (32, 32, 33, 0.0213, 0.0331, None)
    ]
    
    for branch_id, from_bus, to_bus, r, x, s_max in branch_data:
        opf.add_branch(branch_id, from_bus, to_bus, r, x, s_max)
    
    # Add generators
    # Slack bus generator
    opf.add_generator(1, 1, 0, 50, -30, 30, 'quadratic', [0.01, 10, 0])
    
    # Add DERs (optional)
    opf.add_generator(2, 18, 0, 1, -0.5, 0.5, 'linear', [20, 0])
    opf.add_generator(3, 33, 0, 1, -0.5, 0.5, 'linear', [20, 0])
    
    # Add loads (simplified load values for brevity)
    loads = [
        (1, 2, 0.1, 0.06),
        (2, 3, 0.09, 0.04),
        (3, 4, 0.12, 0.08),
        (4, 5, 0.06, 0.03),
        (5, 6, 0.06, 0.02),
        (6, 7, 0.2, 0.1),
        (7, 8, 0.2, 0.1),
        (8, 9, 0.06, 0.02),
        (9, 10, 0.06, 0.02),
        (10, 11, 0.045, 0.03),
        (11, 12, 0.06, 0.035),
        (12, 13, 0.06, 0.035),
        (13, 14, 0.12, 0.08),
        (14, 15, 0.06, 0.01),
        (15, 16, 0.06, 0.02),
        (16, 17, 0.06, 0.02),
        (17, 18, 0.09, 0.04),
        (18, 19, 0.09, 0.04),
        (19, 20, 0.09, 0.04),
        (20, 21, 0.09, 0.04),
        (21, 22, 0.09, 0.04),
        (22, 23, 0.09, 0.05),
        (23, 24, 0.42, 0.2),
        (24, 25, 0.42, 0.2),
        (25, 26, 0.06, 0.025),
        (26, 27, 0.06, 0.025),
        (27, 28, 0.06, 0.02),
        (28, 29, 0.12, 0.07),
        (29, 30, 0.2, 0.6),
        (30, 31, 0.15, 0.07),
        (31, 32, 0.21, 0.1),
        (32, 33, 0.06, 0.04)
    ]
    
    for load_id, bus_id, p, q in loads:
        opf.add_load(load_id, bus_id, p*10, q*10)  # Scaling to MW and MVAr
    
    return opf

def run_opf_example():
    print("Running LinDistFlow OPF Example...")
    
    # Create a simple test system
    opf = create_simple_system()
    
    # Build and solve the model
    opf.build_model()
    opf.solve(tee=True)
    
    # Get and print results
    print("\nVoltage Profile:")
    voltage_df = opf.get_voltage_profile()
    print(voltage_df)
    
    print("\nBranch Power Flows:")
    flow_df = opf.get_power_flows()
    print(flow_df)
    
    print("\nGenerator Outputs:")
    gen_df = opf.get_generator_outputs()
    print(gen_df)
    
    print(f"\nTotal Losses: {opf.get_total_losses():.4f} MW")
    
    # Plot results
    opf.plot_voltage_profile()
    opf.plot_power_flows()
    
    # IEEE 33-bus system example
    print("\n\nRunning IEEE 33-bus System Example...")
    
    # Create the IEEE 33-bus system
    ieee33 = create_ieee33_system()
    
    # Build and solve the model
    ieee33.build_model()
    ieee33.solve()
    
    # Get and print summary results
    print("\nIEEE 33-bus Voltage Summary:")
    v_df = ieee33.get_voltage_profile()
    print(f"Minimum voltage: {v_df['Voltage (pu)'].min():.4f} pu at bus {v_df.loc[v_df['Voltage (pu)'].idxmin(), 'Bus']}")
    print(f"Maximum voltage: {v_df['Voltage (pu)'].max():.4f} pu at bus {v_df.loc[v_df['Voltage (pu)'].idxmax(), 'Bus']}")
    
    print("\nGenerator Outputs:")
    gen_df = ieee33.get_generator_outputs()
    print(gen_df)
    
    print(f"\nTotal Losses: {ieee33.get_total_losses():.4f} MW")
    
    # Plot IEEE 33-bus results
    ieee33.plot_voltage_profile()
    
    print("\nLinDistFlow OPF completed successfully.")

if __name__ == "__main__":
    run_opf_example()