import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

class ThreePhaseLinDistFlow:
    """
    Three-Phase Unbalanced Linear Distribution Flow Optimal Power Flow model.
    
    This class implements a linearized three-phase power flow model for unbalanced
    distribution systems, extending the single-phase LinDistFlow approach to handle
    multiple phases with mutual coupling between conductors.
    
    Key features:
    - Models three-phase unbalanced systems (phases a, b, c)
    - Accounts for mutual impedance between phases
    - Handles single-phase, two-phase, and three-phase laterals
    - Includes distributed generation (DG) optimization
    - Considers voltage constraints per phase
    """
    
    def __init__(self, base_mva=1.0):
        """
        Initialize the Three-Phase LinDistFlow OPF model.
        
        Args:
            base_mva: Base MVA for per-unit calculations (default: 1.0)
        """
        self.base_mva = base_mva
        
        # Network data structures
        self.buses = {}  # Dictionary of buses: {bus_id: {data}}
        self.branches = {}  # Dictionary of branches: {branch_id: {data}}
        self.generators = {}  # Dictionary of generators: {gen_id: {data}}
        self.loads = {}  # Dictionary of loads: {load_id: {data}}
        
        # Mapping dictionaries
        self.bus_to_branches = {}  # {bus_id: [branch_ids]}
        self.bus_to_generators = {}  # {bus_id: [gen_ids]}
        self.bus_to_loads = {}  # {bus_id: [load_ids]}
        self.branch_to_buses = {}  # {branch_id: (from_bus, to_bus)}
        
        # Set of phases
        self.phases = ['a', 'b', 'c']
        
        # Optimization model
        self.model = None
        self.results = None
        
        # Track if model has been built and solved
        self.model_built = False
        self.model_solved = False
    
    def add_bus(self, bus_id, phases=None, v_min=0.95, v_max=1.05, is_slack=False):
        """
        Add a bus to the network.
        
        Args:
            bus_id: Bus identifier
            phases: List of phases present at this bus (default: ['a', 'b', 'c'])
            v_min: Minimum voltage magnitude in per-unit (default: 0.95)
            v_max: Maximum voltage magnitude in per-unit (default: 1.05)
            is_slack: Boolean indicating if this is the slack/reference bus (default: False)
        """
        if phases is None:
            phases = ['a', 'b', 'c']  # Default to three-phase
        
        # Validate phases
        for p in phases:
            if p not in self.phases:
                raise ValueError(f"Invalid phase: {p}. Must be one of {self.phases}")
        
        self.buses[bus_id] = {
            'phases': phases,
            'v_min': v_min,
            'v_max': v_max,
            'is_slack': is_slack
        }
        
        # Initialize mapping dictionaries
        self.bus_to_branches[bus_id] = []
        self.bus_to_generators[bus_id] = []
        self.bus_to_loads[bus_id] = []
        
        self.model_built = False  # Model needs to be rebuilt
    
    def add_branch(self, branch_id, from_bus, to_bus, z_matrix, phases=None, s_max=None):
        """
        Add a branch (line or transformer) to the network.
        
        Args:
            branch_id: Branch identifier
            from_bus: Bus ID at the sending end ("from" bus)
            to_bus: Bus ID at the receiving end ("to" bus)
            z_matrix: 3x3 complex impedance matrix [ohms]
            phases: List of phases present in this branch (default: ['a', 'b', 'c'])
            s_max: Maximum apparent power flow limit in MVA (optional)
        """
        # Verify that buses exist
        if from_bus not in self.buses:
            raise ValueError(f"From bus {from_bus} does not exist")
        if to_bus not in self.buses:
            raise ValueError(f"To bus {to_bus} does not exist")
        
        # If phases not specified, use intersection of from_bus and to_bus phases
        if phases is None:
            phases = list(set(self.buses[from_bus]['phases']) & set(self.buses[to_bus]['phases']))
            if not phases:
                raise ValueError(f"Buses {from_bus} and {to_bus} have no common phases")
        
        # Validate phases
        for p in phases:
            if p not in self.phases:
                raise ValueError(f"Invalid phase: {p}. Must be one of {self.phases}")
            if p not in self.buses[from_bus]['phases']:
                raise ValueError(f"Phase {p} not present at from_bus {from_bus}")
            if p not in self.buses[to_bus]['phases']:
                raise ValueError(f"Phase {p} not present at to_bus {to_bus}")
        
        # Create the branch impedance matrix for the specified phases
        phase_indices = {self.phases[i]: i for i in range(len(self.phases))}
        
        # Extract the relevant submatrix based on the phases
        idx = [phase_indices[p] for p in phases]
        z_submatrix = np.array([[z_matrix[i][j] for j in idx] for i in idx])
        
        # Store branch data
        self.branches[branch_id] = {
            'from_bus': from_bus,
            'to_bus': to_bus,
            'phases': phases,
            'z_matrix': z_submatrix,
            's_max': s_max
        }
        
        # Update mapping dictionaries
        self.bus_to_branches[from_bus].append(branch_id)
        self.bus_to_branches[to_bus].append(branch_id)
        self.branch_to_buses[branch_id] = (from_bus, to_bus)
        
        self.model_built = False  # Model needs to be rebuilt
    
    def add_generator(self, gen_id, bus_id, phases=None, p_min=None, p_max=None, 
                    q_min=None, q_max=None, cost_model='linear', cost_coeffs=None):
        """
        Add a generator to the network.
        
        Args:
            gen_id: Generator identifier
            bus_id: Bus ID where the generator is connected
            phases: List of phases for this generator (default: all phases at the bus)
            p_min: Dict or scalar for minimum active power output per phase in MW
            p_max: Dict or scalar for maximum active power output per phase in MW
            q_min: Dict or scalar for minimum reactive power output per phase in MVAr
            q_max: Dict or scalar for maximum reactive power output per phase in MVAr
            cost_model: Type of cost model ('linear', 'quadratic')
            cost_coeffs: Dictionary of cost coefficients per phase
        """
        # Verify that bus exists
        if bus_id not in self.buses:
            raise ValueError(f"Bus {bus_id} does not exist")
        
        # If phases not specified, use all phases at the bus
        if phases is None:
            phases = self.buses[bus_id]['phases']
        
        # Validate phases
        for p in phases:
            if p not in self.buses[bus_id]['phases']:
                raise ValueError(f"Phase {p} not present at bus {bus_id}")
        
        # Initialize limits (convert scalar values to dictionaries if needed)
        def process_limit(limit, phases):
            if limit is None:
                return {p: None for p in phases}
            elif isinstance(limit, dict):
                return limit
            else:
                return {p: limit for p in phases}
        
        p_min_dict = process_limit(p_min, phases)
        p_max_dict = process_limit(p_max, phases)
        q_min_dict = process_limit(q_min, phases)
        q_max_dict = process_limit(q_max, phases)
        
        # Process cost coefficients
        if cost_coeffs is None:
            if cost_model == 'linear':
                cost_coeffs = {p: [10.0, 0.0] for p in phases}  # [c1, c0]
            else:  # quadratic
                cost_coeffs = {p: [0.01, 10.0, 0.0] for p in phases}  # [c2, c1, c0]
        
        # Normalize to per-unit
        for p in phases:
            if p_min_dict[p] is not None:
                p_min_dict[p] /= self.base_mva
            if p_max_dict[p] is not None:
                p_max_dict[p] /= self.base_mva
            if q_min_dict[p] is not None:
                q_min_dict[p] /= self.base_mva
            if q_max_dict[p] is not None:
                q_max_dict[p] /= self.base_mva
        
        # Store generator data
        self.generators[gen_id] = {
            'bus_id': bus_id,
            'phases': phases,
            'p_min': p_min_dict,
            'p_max': p_max_dict,
            'q_min': q_min_dict,
            'q_max': q_max_dict,
            'cost_model': cost_model,
            'cost_coeffs': cost_coeffs
        }
        
        # Update mapping
        self.bus_to_generators[bus_id].append(gen_id)
        
        self.model_built = False  # Model needs to be rebuilt
    
    def add_load(self, load_id, bus_id, p_demand=None, q_demand=None, phases=None):
        """
        Add a load to the network.
        
        Args:
            load_id: Load identifier
            bus_id: Bus ID where the load is connected
            p_demand: Dict or scalar for active power demand per phase in MW
            q_demand: Dict or scalar for reactive power demand per phase in MVAr
            phases: List of phases for this load (default: all phases at the bus)
        """
        # Verify that bus exists
        if bus_id not in self.buses:
            raise ValueError(f"Bus {bus_id} does not exist")
        
        # If phases not specified, use all phases at the bus
        if phases is None:
            phases = self.buses[bus_id]['phases']
        
        # Validate phases
        for p in phases:
            if p not in self.buses[bus_id]['phases']:
                raise ValueError(f"Phase {p} not present at bus {bus_id}")
        
        # Process demands (convert scalar values to dictionaries if needed)
        def process_demand(demand, phases):
            if demand is None:
                return {p: 0.0 for p in phases}  # Default to zero demand
            elif isinstance(demand, dict):
                return demand
            else:
                return {p: demand for p in phases}
        
        p_demand_dict = process_demand(p_demand, phases)
        q_demand_dict = process_demand(q_demand, phases)
        
        # Normalize to per-unit
        for p in phases:
            p_demand_dict[p] /= self.base_mva
            q_demand_dict[p] /= self.base_mva
        
        # Store load data
        self.loads[load_id] = {
            'bus_id': bus_id,
            'phases': phases,
            'p_demand': p_demand_dict,
            'q_demand': q_demand_dict
        }
        
        # Update mapping
        self.bus_to_loads[bus_id].append(load_id)
        
        self.model_built = False  # Model needs to be rebuilt
    
    def build_model(self):
        """
        Build the Pyomo optimization model for Three-Phase LinDistFlow OPF.
        """
        # Create a concrete model
        model = pyo.ConcreteModel()
        
        # Define sets
        model.BUSES = pyo.Set(initialize=self.buses.keys())
        model.BRANCHES = pyo.Set(initialize=self.branches.keys())
        model.GENERATORS = pyo.Set(initialize=self.generators.keys())
        model.LOADS = pyo.Set(initialize=self.loads.keys())
        model.PHASES = pyo.Set(initialize=self.phases)
        
        # Create bus-phase pairs set
        bus_phase_pairs = []
        for bus_id, bus in self.buses.items():
            for phase in bus['phases']:
                bus_phase_pairs.append((bus_id, phase))
        model.BUS_PHASE_PAIRS = pyo.Set(initialize=bus_phase_pairs)
        
        # Create branch-phase pairs set
        branch_phase_pairs = []
        for branch_id, branch in self.branches.items():
            for phase in branch['phases']:
                branch_phase_pairs.append((branch_id, phase))
        model.BRANCH_PHASE_PAIRS = pyo.Set(initialize=branch_phase_pairs)
        
        # Create generator-phase pairs set
        gen_phase_pairs = []
        for gen_id, gen in self.generators.items():
            for phase in gen['phases']:
                gen_phase_pairs.append((gen_id, phase))
        model.GEN_PHASE_PAIRS = pyo.Set(initialize=gen_phase_pairs)
        
        # Define variables
        # Voltage magnitude squared at each bus and phase
        model.v_squared = pyo.Var(model.BUS_PHASE_PAIRS, domain=pyo.NonNegativeReals)
        
        # Active and reactive power flow in each branch and phase
        model.P = pyo.Var(model.BRANCH_PHASE_PAIRS, domain=pyo.Reals)
        model.Q = pyo.Var(model.BRANCH_PHASE_PAIRS, domain=pyo.Reals)
        
        # Generator active and reactive power output for each phase
        model.Pg = pyo.Var(model.GEN_PHASE_PAIRS, domain=pyo.Reals)
        model.Qg = pyo.Var(model.GEN_PHASE_PAIRS, domain=pyo.Reals)
        
        # Define constraints
        # Voltage magnitude limits
        def voltage_limits_rule(model, b, p):
            bus = self.buses[b]
            return (bus['v_min']**2, model.v_squared[b, p], bus['v_max']**2)
        model.voltage_limits = pyo.Constraint(model.BUS_PHASE_PAIRS, rule=voltage_limits_rule)
        
        # Slack bus voltage (reference)
        def slack_voltage_rule(model, p):
            # Find slack bus
            slack_buses = [b for b, data in self.buses.items() if data['is_slack']]
            
            # If no slack bus exists, skip this constraint
            if not slack_buses:
                print("Warning: No slack bus found in the system! Adding default slack at bus 1.")
                if 1 in self.buses:
                    self.buses[1]['is_slack'] = True
                    slack_bus = 1
                else:
                    # Get the first bus in the system
                    slack_bus = next(iter(self.buses))
                    self.buses[slack_bus]['is_slack'] = True
                print(f"Bus {slack_bus} set as slack bus.")
            else:
                slack_bus = slack_buses[0]
            
            if p in self.buses[slack_bus]['phases']:
                # Phase-specific reference voltage (could be different per phase in unbalanced system)
                if p == 'a':
                    return model.v_squared[slack_bus, p] == 1.0**2  # V = 1.0 pu
                elif p == 'b':
                    return model.v_squared[slack_bus, p] == 1.0**2  # Could set different for unbalanced
                elif p == 'c':
                    return model.v_squared[slack_bus, p] == 1.0**2  # Could set different for unbalanced
            return pyo.Constraint.Skip
        
        # Generator limits
        def gen_p_limits_rule(model, g, p):
            gen = self.generators[g]
            if p in gen['phases']:
                p_min = gen['p_min'][p]
                p_max = gen['p_max'][p]
                
                if p_min is not None and p_max is not None:
                    return (p_min, model.Pg[g, p], p_max)
                elif p_min is not None:
                    return (p_min, model.Pg[g, p], None)
                elif p_max is not None:
                    return (None, model.Pg[g, p], p_max)
            return pyo.Constraint.Skip
        model.gen_p_limits = pyo.Constraint(model.GEN_PHASE_PAIRS, rule=gen_p_limits_rule)
        
        def gen_q_limits_rule(model, g, p):
            gen = self.generators[g]
            if p in gen['phases']:
                q_min = gen['q_min'][p]
                q_max = gen['q_max'][p]
                
                if q_min is not None and q_max is not None:
                    return (q_min, model.Qg[g, p], q_max)
                elif q_min is not None:
                    return (q_min, model.Qg[g, p], None)
                elif q_max is not None:
                    return (None, model.Qg[g, p], q_max)
            return pyo.Constraint.Skip
        model.gen_q_limits = pyo.Constraint(model.GEN_PHASE_PAIRS, rule=gen_q_limits_rule)
        
        # Branch flow limits (if s_max is specified)
        def branch_flow_limits_rule(model, br, p):
            branch = self.branches[br]
            if p in branch['phases'] and branch['s_max'] is not None:
                s_max_pu = branch['s_max'] / self.base_mva
                # Approximation of apparent power constraint using linear outer bound
                return model.P[br, p] <= s_max_pu
            return pyo.Constraint.Skip
        model.branch_flow_limits = pyo.Constraint(model.BRANCH_PHASE_PAIRS, rule=branch_flow_limits_rule)
        
        # Three-Phase LinDistFlow equations
        
        # 1. Power balance at each bus for each phase
        def active_power_balance_rule(model, b, p):
            # Skip if phase not present at this bus
            if p not in self.buses[b]['phases']:
                return pyo.Constraint.Skip
            
            # Sum of power flowing into the bus minus sum of power flowing out = net injection
            inflows = sum(model.P[br, p] for br in self.bus_to_branches[b] 
                         if self.branches[br]['to_bus'] == b and p in self.branches[br]['phases'])
            
            outflows = sum(model.P[br, p] for br in self.bus_to_branches[b] 
                          if self.branches[br]['from_bus'] == b and p in self.branches[br]['phases'])
            
            # Net generation at this bus for this phase
            generation = sum(model.Pg[g, p] for g in self.bus_to_generators[b] 
                            if p in self.generators[g]['phases'])
            
            # Net demand at this bus for this phase
            demand = sum(self.loads[l]['p_demand'][p] for l in self.bus_to_loads[b] 
                        if p in self.loads[l]['phases'])
            
            return inflows - outflows + generation == demand
        model.active_power_balance = pyo.Constraint(model.BUS_PHASE_PAIRS, rule=active_power_balance_rule)
        
        def reactive_power_balance_rule(model, b, p):
            # Skip if phase not present at this bus
            if p not in self.buses[b]['phases']:
                return pyo.Constraint.Skip
            
            # Sum of power flowing into the bus minus sum of power flowing out = net injection
            inflows = sum(model.Q[br, p] for br in self.bus_to_branches[b] 
                         if self.branches[br]['to_bus'] == b and p in self.branches[br]['phases'])
            
            outflows = sum(model.Q[br, p] for br in self.bus_to_branches[b] 
                          if self.branches[br]['from_bus'] == b and p in self.branches[br]['phases'])
            
            # Net generation at this bus for this phase
            generation = sum(model.Qg[g, p] for g in self.bus_to_generators[b] 
                            if p in self.generators[g]['phases'])
            
            # Net demand at this bus for this phase
            demand = sum(self.loads[l]['q_demand'][p] for l in self.bus_to_loads[b] 
                        if p in self.loads[l]['phases'])
            
            return inflows - outflows + generation == demand
        model.reactive_power_balance = pyo.Constraint(model.BUS_PHASE_PAIRS, rule=reactive_power_balance_rule)
        
        # 2. Voltage drop along branches using the LinDistFlow approximation with mutual coupling
        def voltage_drop_rule(model, br, p_to):
            branch = self.branches[br]
            from_bus = branch['from_bus']
            to_bus = branch['to_bus']
            
            # Skip if phase not present in this branch
            if p_to not in branch['phases']:
                return pyo.Constraint.Skip
            
            # Find index of p_to in branch phases
            p_to_idx = branch['phases'].index(p_to)
            
            # Compute the voltage drop including mutual coupling
            voltage_drop = 0
            for p_idx, p_from in enumerate(branch['phases']):
                # Extract impedance from matrix (real part = resistance, imag part = reactance)
                r_pq = branch['z_matrix'][p_to_idx, p_idx].real
                x_pq = branch['z_matrix'][p_to_idx, p_idx].imag
                
                # Add contribution to voltage drop (LinDistFlow approximation)
                voltage_drop += 2 * (r_pq * model.P[br, p_from] + x_pq * model.Q[br, p_from])
            
            # LinDistFlow voltage equation with mutual coupling
            return model.v_squared[to_bus, p_to] == model.v_squared[from_bus, p_to] - voltage_drop
        model.voltage_drop = pyo.Constraint(model.BRANCH_PHASE_PAIRS, rule=voltage_drop_rule)
        
        # Define the objective function
        def objective_rule(model):
            total_cost = 0
            
            for g, p in model.GEN_PHASE_PAIRS:
                gen = self.generators[g]
                cost_coeffs = gen['cost_coeffs'][p]
                
                if gen['cost_model'] == 'linear':
                    # Linear cost: c1*P + c0
                    c1, c0 = cost_coeffs
                    total_cost += c1 * model.Pg[g, p] * self.base_mva + c0
                elif gen['cost_model'] == 'quadratic':
                    # Quadratic cost: c2*P^2 + c1*P + c0
                    c2, c1, c0 = cost_coeffs
                    total_cost += c2 * (model.Pg[g, p] * self.base_mva)**2 + c1 * model.Pg[g, p] * self.base_mva + c0
            
            return total_cost
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        
        # Store the model
        self.model = model
        self.model_built = True
        
        return model
    
    def solve(self, solver='ipopt', tee=False):
        """
        Solve the Three-Phase LinDistFlow OPF model.
        
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
        Get the voltage profile at all buses for all phases.
        
        Returns:
            DataFrame with bus-phase voltage magnitudes
        """
        if not self.model_solved:
            raise ValueError("Model has not been solved yet")
        
        voltages = []
        for b in self.model.BUSES:
            for p in self.buses[b]['phases']:
                voltages.append({
                    'Bus': b,
                    'Phase': p,
                    'Voltage (pu)': np.sqrt(pyo.value(self.model.v_squared[b, p]))
                })
        
        return pd.DataFrame(voltages)
    
    def get_power_flows(self):
        """
        Get the power flows through all branches for all phases.
        
        Returns:
            DataFrame with branch-phase power flows
        """
        if not self.model_solved:
            raise ValueError("Model has not been solved yet")
        
        flows = []
        for br in self.model.BRANCHES:
            branch = self.branches[br]
            from_bus = branch['from_bus']
            to_bus = branch['to_bus']
            
            for p in branch['phases']:
                p_flow = pyo.value(self.model.P[br, p]) * self.base_mva
                q_flow = pyo.value(self.model.Q[br, p]) * self.base_mva
                s_flow = np.sqrt(p_flow**2 + q_flow**2)
                
                flows.append({
                    'Branch': br,
                    'From': from_bus,
                    'To': to_bus,
                    'Phase': p,
                    'P (MW)': p_flow,
                    'Q (MVAr)': q_flow,
                    'S (MVA)': s_flow
                })
        
        return pd.DataFrame(flows)
    
    def get_generator_outputs(self):
        """
        Get the power outputs of all generators for all phases.
        
        Returns:
            DataFrame with generator-phase outputs
        """
        if not self.model_solved:
            raise ValueError("Model has not been solved yet")
        
        outputs = []
        for g in self.model.GENERATORS:
            gen = self.generators[g]
            bus_id = gen['bus_id']
            
            for p in gen['phases']:
                p_out = pyo.value(self.model.Pg[g, p]) * self.base_mva
                q_out = pyo.value(self.model.Qg[g, p]) * self.base_mva
                
                outputs.append({
                    'Generator': g,
                    'Bus': bus_id,
                    'Phase': p,
                    'P (MW)': p_out,
                    'Q (MVAr)': q_out
                })
        
        return pd.DataFrame(outputs)
    
    def get_phase_unbalance(self):
        """
        Calculate the voltage unbalance factor for each three-phase bus.
        
        Returns:
            DataFrame with voltage unbalance factors
        """
        if not self.model_solved:
            raise ValueError("Model has not been solved yet")
        
        unbalance = []
        
        for b in self.model.BUSES:
            # Only calculate for buses with all three phases
            if set(self.buses[b]['phases']) == set(['a', 'b', 'c']):
                # Get phase voltages
                v_a = np.sqrt(pyo.value(self.model.v_squared[b, 'a']))
                v_b = np.sqrt(pyo.value(self.model.v_squared[b, 'b']))
                v_c = np.sqrt(pyo.value(self.model.v_squared[b, 'c']))
                
                # Calculate average magnitude
                v_avg = (v_a + v_b + v_c) / 3
                
                # Calculate maximum deviation
                v_dev = max(abs(v_a - v_avg), abs(v_b - v_avg), abs(v_c - v_avg))
                
                # Calculate unbalance factor (percentage)
                vuf = (v_dev / v_avg) * 100
                
                unbalance.append({
                    'Bus': b,
                    'Voltage Unbalance Factor (%)': vuf
                })
        
        return pd.DataFrame(unbalance)
    
    def get_total_losses(self):
        """
        Calculate the total active power losses in the system.
        
        Returns:
            Dictionary with total losses per phase and total in MW
        """
        if not self.model_solved:
            raise ValueError("Model has not been solved yet")
        
        # Calculate losses per phase
        losses = {p: 0.0 for p in self.phases}
        
        for br in self.model.BRANCHES:
            branch = self.branches[br]
            for p_idx, p in enumerate(branch['phases']):
                # Get current in this phase
                i_p = pyo.value(self.model.P[br, p]) / np.sqrt(pyo.value(self.model.v_squared[branch['from_bus'], p]))
                
                # Calculate loss using I²R
                r_pp = branch['z_matrix'][p_idx, p_idx].real
                losses[p] += (i_p**2) * r_pp * self.base_mva
        
        # Calculate total losses
        total_loss = sum(losses.values())
        
        return {
            'Phase Losses': losses,
            'Total Losses (MW)': total_loss
        }
    
    def plot_voltage_profile(self):
        """
        Plot the voltage profile of the system for all phases.
        """
        if not self.model_solved:
            raise ValueError("Model has not been solved yet")
        
        # Get voltage profile data
        voltage_df = self.get_voltage_profile()
        
        # Create a pivot table for easier plotting
        pivot_df = voltage_df.pivot(index='Bus', columns='Phase', values='Voltage (pu)')
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Color map for phases
        phase_colors = {'a': 'red', 'b': 'blue', 'c': 'green'}
        
        # Plot each phase
        for phase in pivot_df.columns:
            plt.plot(pivot_df.index, pivot_df[phase], marker='o', color=phase_colors[phase], label=f'Phase {phase.upper()}')
        
        # Add reference lines
        plt.axhline(y=1.0, color='black', linestyle='-', alpha=0.7, label='Nominal')
        plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='Lower Limit (-5%)')
        plt.axhline(y=1.05, color='red', linestyle='--', alpha=0.7, label='Upper Limit (+5%)')
        
        # Customize plot
        plt.grid(True, alpha=0.3)
        plt.xlabel('Bus')
        plt.ylabel('Voltage Magnitude (pu)')
        plt.title('Three-Phase Voltage Profile')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_unbalance(self):
        """
        Plot the voltage unbalance factors across the system.
        """
        if not self.model_solved:
            raise ValueError("Model has not been solved yet")
        
        # Get unbalance data
        unbalance_df = self.get_phase_unbalance()
        
        if len(unbalance_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.bar(unbalance_df['Bus'].astype(str), unbalance_df['Voltage Unbalance Factor (%)'], color='purple')
            
            # Add reference line for typical limit
            plt.axhline(y=2.0, color='red', linestyle='--', label='Typical Limit (2%)')
            
            plt.grid(True, alpha=0.3)
            plt.xlabel('Bus')
            plt.ylabel('Voltage Unbalance Factor (%)')
            plt.title('Phase Voltage Unbalance')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("No three-phase buses available for unbalance analysis")

# Example: Create a simple 4-bus unbalanced test system
def create_unbalanced_test_system():
    # Initialize the model
    opf = ThreePhaseLinDistFlow(base_mva=1.0)
    
    # Add buses with different phase configurations
    opf.add_bus(1, is_slack=True)  # Substation (slack bus) - all three phases
    opf.add_bus(2)  # Three-phase bus
    opf.add_bus(3, phases=['a', 'b'])  # Two-phase bus (a and b)
    opf.add_bus(4, phases=['a'])  # Single-phase bus (phase a)
    
    # Create impedance matrices
    # Self-impedance values
    z_aa = complex(0.3, 0.7)
    z_bb = complex(0.3, 0.7)
    z_cc = complex(0.3, 0.7)
    
    # Mutual impedance values
    z_ab = z_ba = complex(0.1, 0.3)
    z_ac = z_ca = complex(0.1, 0.3)
    z_bc = z_cb = complex(0.1, 0.3)
    
    # Create 3x3 matrix
    z_matrix = np.array([
        [z_aa, z_ab, z_ac],
        [z_ba, z_bb, z_bc],
        [z_ca, z_cb, z_cc]
    ])
    
    # Add branches
    opf.add_branch(1, 1, 2, z_matrix)  # Three-phase branch
    opf.add_branch(2, 2, 3, z_matrix, phases=['a', 'b'])  # Two-phase branch
    opf.add_branch(3, 3, 4, z_matrix, phases=['a'])  # Single-phase branch
    
    # Add generators
    # Slack bus generator (three-phase)
    opf.add_generator(1, 1, p_min=0, p_max=5.0, q_min=-2.0, q_max=2.0, 
                     cost_model='quadratic', cost_coeffs={'a': [0.01, 10, 0], 'b': [0.01, 10, 0], 'c': [0.01, 10, 0]})
    
    # DG at bus 3 (two-phase)
    opf.add_generator(2, 3, phases=['a', 'b'], p_min=0, p_max=1.0, q_min=-0.5, q_max=0.5, 
                     cost_model='linear', cost_coeffs={'a': [15, 0], 'b': [15, 0]})
    
    # Add loads
    # Three-phase balanced load at bus 2
    opf.add_load(1, 2, p_demand={'a': 0.5, 'b': 0.5, 'c': 0.5}, q_demand={'a': 0.2, 'b': 0.2, 'c': 0.2})
    
    # Two-phase unbalanced load at bus 3
    opf.add_load(2, 3, phases=['a', 'b'], p_demand={'a': 0.3, 'b': 0.5}, q_demand={'a': 0.1, 'b': 0.2})
    
    # Single-phase load at bus 4
    opf.add_load(3, 4, phases=['a'], p_demand={'a': 0.4}, q_demand={'a': 0.15})
    
    return opf

# Function to run a complete analysis with the three-phase model
def run_three_phase_analysis():
    print("Running Three-Phase Unbalanced LinDistFlow OPF...")
    
    # Create test system
    opf = create_unbalanced_test_system()
    
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
    
    print("\nPhase Unbalance:")
    unbalance_df = opf.get_phase_unbalance()
    print(unbalance_df)
    
    losses = opf.get_total_losses()
    print(f"\nTotal Losses: {losses['Total Losses (MW)']:.4f} MW")
    print("Phase Losses:")
    for phase, loss in losses['Phase Losses'].items():
        print(f"  Phase {phase.upper()}: {loss:.4f} MW")
    
    # Plot results
    opf.plot_voltage_profile()
    opf.plot_unbalance()
    
    print("\nThree-Phase LinDistFlow OPF completed successfully.")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def optimize_der_placement(base_system, candidate_buses, der_types, max_ders=3):
    """
    Determine optimal locations and sizes for DERs in a distribution system.
    
    Args:
        base_system: ThreePhaseLinDistFlow object with the base system (no DERs)
        candidate_buses: List of bus IDs where DERs can be installed
        der_types: Dictionary of DER types with their characteristics
        max_ders: Maximum number of DERs to be placed
        
    Returns:
        optimal_placements: List of (bus_id, der_type, size) tuples
        base_results: Results from the base case (no DERs)
        optimal_results: Results with optimal DER placement
    """
    # First, solve the base case (no DERs)
    base_system.build_model()
    base_system.solve()
    
    # Store base case results
    base_losses = base_system.get_total_losses()['Total Losses (MW)']
    base_min_voltage = base_system.get_voltage_profile()['Voltage (pu)'].min()
    base_results = {
        'losses': base_losses,
        'min_voltage': base_min_voltage,
        'voltage_profile': base_system.get_voltage_profile()
    }
    
    print(f"Base case - Losses: {base_losses:.4f} MW, Min Voltage: {base_min_voltage:.4f} pu")
    
    # Create a new optimization model to determine DER placements
    placement_model = pyo.ConcreteModel()
    
    # Sets
    placement_model.BUSES = pyo.Set(initialize=candidate_buses)
    placement_model.DER_TYPES = pyo.Set(initialize=der_types.keys())
    
    # Binary variables for DER placement decisions
    # x[b,d] = 1 if DER type d is placed at bus b
    placement_model.x = pyo.Var(placement_model.BUSES, placement_model.DER_TYPES, domain=pyo.Binary)
    
    # Variables for DER sizes
    placement_model.size = pyo.Var(placement_model.BUSES, placement_model.DER_TYPES, domain=pyo.NonNegativeReals)
    
    # Constraint: Maximum number of DERs
    def max_ders_rule(model):
        return sum(model.x[b,d] for b in model.BUSES for d in model.DER_TYPES) <= max_ders
    placement_model.max_ders_constraint = pyo.Constraint(rule=max_ders_rule)
    
    # Constraint: At most one DER per bus
    def one_der_per_bus_rule(model, b):
        return sum(model.x[b,d] for d in model.DER_TYPES) <= 1
    placement_model.one_der_per_bus = pyo.Constraint(placement_model.BUSES, rule=one_der_per_bus_rule)
    
    # Constraint: Size is zero if DER is not placed
    def size_constraint_rule(model, b, d):
        max_size = der_types[d]['max_size']
        return model.size[b,d] <= max_size * model.x[b,d]
    placement_model.size_constraint = pyo.Constraint(
        placement_model.BUSES, placement_model.DER_TYPES, rule=size_constraint_rule)
    
    # Evaluate each candidate placement
    results = []
    
    # Create evaluation function (using the simulation instead of explicit expressions)
    def evaluate_placement(placement_vars):
        # Create a copy of the base system
        test_system = create_system_with_ders(base_system, placement_vars)
        
        # Solve power flow with this DER configuration
        test_system.build_model()
        test_system.solve(tee=False)
        
        # Get metrics
        losses = test_system.get_total_losses()['Total Losses (MW)']
        min_voltage = test_system.get_voltage_profile()['Voltage (pu)'].min()
        
        return test_system, losses, min_voltage
    
    # Define objective function - minimize losses
    def objective_rule(model):
        # This is a placeholder - in a real implementation, you'd use 
        # cost expressions based on the placement variables
        total_cost = 0
        for b in model.BUSES:
            for d in model.DER_TYPES:
                # Cost has two components:
                # 1. Fixed installation cost (if DER is placed)
                fixed_cost = der_types[d]['fixed_cost'] * model.x[b,d]
                
                # 2. Variable cost based on size
                variable_cost = der_types[d]['variable_cost'] * model.size[b,d]
                
                total_cost += fixed_cost + variable_cost
        
        # Add a proxy for operational savings (simplified)
        # In reality, this would be based on time-series simulation
        operational_savings = sum(der_types[d]['capacity_factor'] * model.size[b,d] 
                                 for b in model.BUSES for d in model.DER_TYPES)
        
        return total_cost - operational_savings
    
    placement_model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    # Solve the placement model
    solver = SolverFactory('glpk')  # Mixed-integer programming solver
    solver.solve(placement_model, tee=True)
    
    # Extract optimal placements
    optimal_placements = []
    for b in placement_model.BUSES:
        for d in placement_model.DER_TYPES:
            if pyo.value(placement_model.x[b,d]) > 0.5:  # If DER is placed (binary var ≈ 1)
                size = pyo.value(placement_model.size[b,d])
                optimal_placements.append((b, d, size))
    
    # Evaluate the optimal solution
    optimal_system = create_system_with_ders(base_system, optimal_placements)
    optimal_system.build_model()
    optimal_system.solve()
    
    optimal_losses = optimal_system.get_total_losses()['Total Losses (MW)']
    optimal_min_voltage = optimal_system.get_voltage_profile()['Voltage (pu)'].min()
    optimal_results = {
        'losses': optimal_losses,
        'min_voltage': optimal_min_voltage,
        'voltage_profile': optimal_system.get_voltage_profile()
    }
    
    print(f"Optimal case - Losses: {optimal_losses:.4f} MW, Min Voltage: {optimal_min_voltage:.4f} pu")
    print("Optimal DER placements:")
    for b, d, s in optimal_placements:
        print(f"  Bus {b}: {d} DER with size {s:.2f} MW")
    
    return optimal_placements, base_results, optimal_results

def create_system_with_ders(base_system, der_placements):
    """
    Create a new system with specified DER placements.
    
    Args:
        base_system: Original ThreePhaseLinDistFlow object
        der_placements: List of (bus_id, der_type, size) tuples
        
    Returns:
        new_system: ThreePhaseLinDistFlow object with added DERs
    """
    # Create a copy of the base system
    new_system = ThreePhaseLinDistFlow(base_mva=base_system.base_mva)
    
    # Copy all buses
    for bus_id, bus in base_system.buses.items():
        new_system.add_bus(bus_id, phases=bus['phases'], 
                          v_min=bus['v_min'], v_max=bus['v_max'],
                          is_slack=bus['is_slack'])
    
    # Copy all branches
    for branch_id, branch in base_system.branches.items():
        new_system.add_branch(branch_id, branch['from_bus'], branch['to_bus'],
                             branch['z_matrix'], phases=branch['phases'],
                             s_max=branch['s_max'])
    
    # Copy existing generators
    for gen_id, gen in base_system.generators.items():
        new_system.add_generator(gen_id, gen['bus_id'], phases=gen['phases'],
                                p_min=gen['p_min'], p_max=gen['p_max'],
                                q_min=gen['q_min'], q_max=gen['q_max'],
                                cost_model=gen['cost_model'], cost_coeffs=gen['cost_coeffs'])
    
    # Copy loads
    for load_id, load in base_system.loads.items():
        new_system.add_load(load_id, load['bus_id'], 
                           p_demand=load['p_demand'], q_demand=load['q_demand'],
                           phases=load['phases'])
    
    # Add DERs based on placements
    next_gen_id = max(base_system.generators.keys()) + 1 if base_system.generators else 1
    
    for bus_id, der_type, size in der_placements:
        # Get phases available at this bus
        phases = new_system.buses[bus_id]['phases']
        
        # Determine the DER characteristics based on type
        if der_type == 'solar_pv':
            # Solar PV - can only inject active power, no reactive support
            p_max = {p: size/len(phases) for p in phases}  # Divide among phases
            q_min = {p: 0 for p in phases}
            q_max = {p: 0 for p in phases}
            cost_coeffs = {p: [-5, 0] for p in phases}  # Negative cost (incentive)
            
        elif der_type == 'battery':
            # Battery - can inject or absorb both P and Q
            p_max = {p: size/len(phases) for p in phases}
            p_min = {p: -size/len(phases) for p in phases}
            q_max = {p: 0.5*size/len(phases) for p in phases}
            q_min = {p: -0.5*size/len(phases) for p in phases}
            cost_coeffs = {p: [0, 0] for p in phases}  # Simplification
            
        elif der_type == 'wind':
            # Wind turbine - can inject P, limited Q support
            p_max = {p: size/len(phases) for p in phases}
            q_max = {p: 0.3*size/len(phases) for p in phases}
            q_min = {p: -0.3*size/len(phases) for p in phases}
            cost_coeffs = {p: [-3, 0] for p in phases}  # Negative cost (incentive)
            
        # Add the DER as a generator
        new_system.add_generator(next_gen_id, bus_id, phases=phases,
                               p_min=p_min if 'p_min' in locals() else None, 
                               p_max=p_max,
                               q_min=q_min, q_max=q_max,
                               cost_model='linear', cost_coeffs=cost_coeffs)
        next_gen_id += 1
    
    return new_system

def plot_der_impact(base_results, optimal_results, optimal_placements):
    """
    Plot comparison of voltage profiles and losses with and without DERs.
    """
    # Prepare voltage profile comparison
    base_voltages = base_results['voltage_profile']
    optimal_voltages = optimal_results['voltage_profile']
    
    # Merge dataframes for comparison
    base_voltages['Case'] = 'Base'
    optimal_voltages['Case'] = 'With DERs'
    
    comparison = pd.concat([base_voltages, optimal_voltages])
    
    # Create pivot table for phase-specific comparison
    pivot = comparison.pivot_table(
        index=['Bus', 'Phase'], columns='Case', values='Voltage (pu)'
    ).reset_index()
    
    # Plot voltage profiles
    plt.figure(figsize=(12, 8))
    
    # Group by phase for clearer visualization
    for phase in ['a', 'b', 'c']:
        phase_data = pivot[pivot['Phase'] == phase]
        
        if not phase_data.empty:
            plt.plot(phase_data['Bus'], phase_data['Base'], 
                    marker='o', linestyle='-', label=f'Phase {phase.upper()} - Base')
            plt.plot(phase_data['Bus'], phase_data['With DERs'], 
                    marker='x', linestyle='--', label=f'Phase {phase.upper()} - With DERs')
    
    # Highlight buses with DERs
    der_buses = [placement[0] for placement in optimal_placements]
    for bus in der_buses:
        plt.axvline(x=bus, color='green', alpha=0.3)
    
    # Add reference lines
    plt.axhline(y=1.0, color='black', linestyle='-', alpha=0.7, label='Nominal')
    plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='Lower Limit')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Bus')
    plt.ylabel('Voltage Magnitude (pu)')
    plt.title('Voltage Profile Comparison: Base vs. With DERs')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot loss reduction
    plt.figure(figsize=(8, 6))
    losses = [base_results['losses'], optimal_results['losses']]
    plt.bar(['Base Case', 'With DERs'], losses, color=['skyblue', 'lightgreen'])
    plt.grid(True, alpha=0.3)
    plt.ylabel('Total System Losses (MW)')
    plt.title('Impact of DERs on System Losses')
    
    # Add text labels
    for i, loss in enumerate(losses):
        plt.text(i, loss/2, f'{loss:.4f} MW', ha='center')
    
    # Add reduction percentage
    reduction = (base_results['losses'] - optimal_results['losses']) / base_results['losses'] * 100
    plt.text(0.5, max(losses)*0.9, f'Reduction: {reduction:.2f}%', 
            ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Example: Define DER types
der_types = {
    'solar_pv': {
        'max_size': 2.0,     # Maximum size per installation (MW)
        'fixed_cost': 100000,  # Fixed installation cost ($)
        'variable_cost': 500000,  # Variable cost ($/MW)
        'capacity_factor': 0.2,  # Average capacity factor
    },
    'battery': {
        'max_size': 1.0,     # Maximum size per installation (MW)
        'fixed_cost': 200000,  # Fixed installation cost ($)
        'variable_cost': 800000,  # Variable cost ($/MW)
        'capacity_factor': 0.9,  # Average availability
    },
    'wind': {
        'max_size': 1.5,     # Maximum size per installation (MW)
        'fixed_cost': 120000,  # Fixed installation cost ($)
        'variable_cost': 600000,  # Variable cost ($/MW)
        'capacity_factor': 0.3,  # Average capacity factor
    }
}

# Example usage
def optimize_ieee37_ders():
    # Create IEEE 37-node test feeder (simplified)
    base_system = create_ieee37_test_system()
    
    # Potential buses for DER placement
    candidate_buses = [4, 8, 12, 16, 20, 25, 30, 35]
    
    # Run optimization
    optimal_placements, base_results, optimal_results = optimize_der_placement(
        base_system, candidate_buses, der_types, max_ders=3)
    
    # Plot results
    plot_der_impact(base_results, optimal_results, optimal_placements)
    
    return optimal_placements, base_results, optimal_results

# This function would create the IEEE 37-node test system
def create_ieee37_test_system():
    """
    Create a simplified model of the IEEE 37-node test feeder.
    This is a three-phase unbalanced test system.
    """
    system = ThreePhaseLinDistFlow(base_mva=2.5)
    
    # Add buses - using simplified numbering for clarity
    # First, add the substation (slack bus)
    system.add_bus(1, is_slack=True)  # Substation (slack bus)
    
    # Then add the rest of the buses
    for i in range(2, 38):
        system.add_bus(i)
    
    # Create a simplified impedance matrix for the feeders
    # Self-impedance values
    z_aa = complex(0.3, 0.7)
    z_bb = complex(0.3, 0.7)
    z_cc = complex(0.3, 0.7)
    
    # Mutual impedance values
    z_ab = z_ba = complex(0.1, 0.3)
    z_ac = z_ca = complex(0.1, 0.3)
    z_bc = z_cb = complex(0.1, 0.3)
    
    # Create 3x3 matrix
    z_matrix = np.array([
        [z_aa, z_ab, z_ac],
        [z_ba, z_bb, z_bc],
        [z_ca, z_cb, z_cc]
    ])
    
    # Add branches to create a simplified radial topology
    # Main feeder branches
    system.add_branch(1, 1, 2, z_matrix)  # Substation to first node
    
    # Continue main path
    for i in range(2, 20):
        system.add_branch(i, i, i+1, z_matrix)
    
    # Add laterals
    system.add_branch(20, 2, 21, z_matrix)
    system.add_branch(21, 21, 22, z_matrix)
    system.add_branch(22, 22, 23, z_matrix)
    
    system.add_branch(23, 5, 24, z_matrix)
    system.add_branch(24, 24, 25, z_matrix)
    system.add_branch(25, 25, 26, z_matrix)
    
    system.add_branch(26, 8, 27, z_matrix)
    system.add_branch(27, 27, 28, z_matrix)
    system.add_branch(28, 28, 29, z_matrix)
    
    system.add_branch(29, 12, 30, z_matrix)
    system.add_branch(30, 30, 31, z_matrix)
    
    system.add_branch(31, 15, 32, z_matrix)
    system.add_branch(32, 32, 33, z_matrix)
    system.add_branch(33, 33, 34, z_matrix)
    
    system.add_branch(34, 17, 35, z_matrix)
    system.add_branch(35, 35, 36, z_matrix)
    system.add_branch(36, 36, 37, z_matrix)
    
    # Add the slack bus generator
    system.add_generator(1, 1, p_min=0, p_max=10.0, q_min=-5.0, q_max=5.0, 
                        cost_model='quadratic', 
                        cost_coeffs={'a': [0.01, 20, 0], 'b': [0.01, 20, 0], 'c': [0.01, 20, 0]})
    
    # Add loads to all nodes except the substation
    # Vary the load to create an unbalanced system
    for i in range(2, 38):
        # Create unbalanced load
        if i % 3 == 0:
            # Heavy on phase A
            p_demand = {'a': 0.15, 'b': 0.10, 'c': 0.10}
            q_demand = {'a': 0.075, 'b': 0.05, 'c': 0.05}
        elif i % 3 == 1:
            # Heavy on phase B
            p_demand = {'a': 0.10, 'b': 0.15, 'c': 0.10}
            q_demand = {'a': 0.05, 'b': 0.075, 'c': 0.05}
        else:
            # Heavy on phase C
            p_demand = {'a': 0.10, 'b': 0.10, 'c': 0.15}
            q_demand = {'a': 0.05, 'b': 0.05, 'c': 0.075}
        
        # Vary the load magnitude based on distance from substation for more realism
        # Nodes farther from substation have smaller loads
        scaling = 1.0 - (i / 50)  # Scaling factor decreases with distance
        p_demand = {phase: value * scaling for phase, value in p_demand.items()}
        q_demand = {phase: value * scaling for phase, value in q_demand.items()}
        
        system.add_load(i, i, p_demand=p_demand, q_demand=q_demand)
    
    return system

if __name__ == "__main__":
    optimize_ieee37_ders()
