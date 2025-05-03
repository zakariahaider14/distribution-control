import numpy as np
import matplotlib.pyplot as plt

# IEEE 4-node test feeder
# Simple radial distribution system representation

class Node:
    def __init__(self, id, P_load=0, Q_load=0, is_substation=False):
        self.id = id
        self.P_load = P_load  # Real power demand (kW)
        self.Q_load = Q_load  # Reactive power demand (kVAr)
        self.V = complex(1.0, 0.0) if is_substation else complex(0.0, 0.0)  # per unit voltage
        self.I = complex(0.0, 0.0)  # Current injection
        self.is_substation = is_substation

class Branch:
    def __init__(self, id, from_node, to_node, r, x):
        self.id = id
        self.from_node = from_node  # Source node
        self.to_node = to_node      # Destination node
        self.r = r  # Resistance (pu)
        self.x = x  # Reactance (pu)
        self.z = complex(r, x)  # Impedance
        self.I = complex(0.0, 0.0)  # Current flowing through branch

class DistributionSystem:
    def __init__(self, base_kV=12.47, base_MVA=1.0):
        self.nodes = {}
        self.branches = {}
        self.base_kV = base_kV
        self.base_MVA = base_MVA
        self.base_Z = (base_kV**2) / base_MVA
        self.convergence_threshold = 1e-4
        self.max_iterations = 100
        
    def add_node(self, id, P_load_kW=0, Q_load_kVAr=0, is_substation=False):
        # Convert loads to per unit
        P_load_pu = P_load_kW / (self.base_MVA * 1000)
        Q_load_pu = Q_load_kVAr / (self.base_MVA * 1000)
        
        self.nodes[id] = Node(id, P_load_pu, Q_load_pu, is_substation)
        return self.nodes[id]
    
    def add_branch(self, id, from_node_id, to_node_id, r_ohm, x_ohm):
        # Convert impedance to per unit
        r_pu = r_ohm / self.base_Z
        x_pu = x_ohm / self.base_Z
        
        branch = Branch(id, self.nodes[from_node_id], self.nodes[to_node_id], r_pu, x_pu)
        self.branches[id] = branch
        return branch
    
    def build_topology(self):
        """Build radial topology - identify downstream nodes and branches"""
        self.downstream_nodes = {node_id: [] for node_id in self.nodes}
        self.downstream_branches = {node_id: [] for node_id in self.nodes}
        
        for branch_id, branch in self.branches.items():
            from_node_id = branch.from_node.id
            to_node_id = branch.to_node.id
            self.downstream_nodes[from_node_id].append(to_node_id)
            self.downstream_branches[from_node_id].append(branch_id)
    
    def calculate_initial_voltages(self):
        """Set all non-substation node voltages to 1.0 per unit for initial guess"""
        for node_id, node in self.nodes.items():
            if not node.is_substation:
                node.V = complex(1.0, 0.0)
    
    def backward_sweep(self):
        """Perform backward sweep - compute branch currents from loads"""
        # Start from leaf nodes and work towards root
        # Reset branch currents
        for branch in self.branches.values():
            branch.I = complex(0.0, 0.0)
        
        # Calculate load currents
        for node_id, node in self.nodes.items():
            if not node.is_substation:
                S = complex(node.P_load, node.Q_load)
                # I = S*/V* (complex conjugate)
                node.I = np.conj(S / np.conj(node.V))
        
        # Process nodes from leaf to root
        processed_branches = set()
        while len(processed_branches) < len(self.branches):
            for branch_id, branch in self.branches.items():
                if branch_id in processed_branches:
                    continue
                
                to_node = branch.to_node
                to_node_id = to_node.id
                
                # Check if all downstream branches of this node have been processed
                all_downstream_processed = True
                for downstream_branch_id in self.downstream_branches[to_node_id]:
                    if downstream_branch_id not in processed_branches:
                        all_downstream_processed = False
                        break
                
                if all_downstream_processed:
                    # Calculate current in this branch
                    branch.I = to_node.I
                    
                    # Add currents from all branches connected downstream from to_node
                    for downstream_branch_id in self.downstream_branches[to_node_id]:
                        downstream_branch = self.branches[downstream_branch_id]
                        branch.I += downstream_branch.I
                    
                    processed_branches.add(branch_id)
    
    def forward_sweep(self):
        """Perform forward sweep - update node voltages"""
        # Start from root (substation) and work towards leaf nodes
        # Find substation node
        substation_node = None
        for node in self.nodes.values():
            if node.is_substation:
                substation_node = node
                break
        
        if not substation_node:
            raise ValueError("No substation node found in the system")
        
        # Process nodes from root to leaf
        processed_nodes = {substation_node.id}
        while len(processed_nodes) < len(self.nodes):
            for branch_id, branch in self.branches.items():
                from_node_id = branch.from_node.id
                to_node_id = branch.to_node.id
                
                if from_node_id in processed_nodes and to_node_id not in processed_nodes:
                    # Update voltage: V_to = V_from - Z * I
                    branch.to_node.V = branch.from_node.V - branch.z * branch.I
                    processed_nodes.add(to_node_id)
    
    def calculate_voltage_mismatch(self, prev_voltages):
        """Calculate maximum voltage mismatch between iterations"""
        max_mismatch = 0.0
        for node_id, node in self.nodes.items():
            if not node.is_substation:
                mismatch = abs(node.V - prev_voltages[node_id])
                max_mismatch = max(max_mismatch, mismatch)
        return max_mismatch
    
    def run_power_flow(self):
        """Execute the backward/forward sweep algorithm"""
        self.build_topology()
        self.calculate_initial_voltages()
        
        iteration = 0
        converged = False
        
        while not converged and iteration < self.max_iterations:
            # Save current voltages for convergence check
            prev_voltages = {node_id: node.V for node_id, node in self.nodes.items()}
            
            # Perform backward sweep
            self.backward_sweep()
            
            # Perform forward sweep
            self.forward_sweep()
            
            # Check convergence
            max_mismatch = self.calculate_voltage_mismatch(prev_voltages)
            
            print(f"Iteration {iteration+1}, Max Voltage Mismatch: {max_mismatch:.6f}")
            
            if max_mismatch < self.convergence_threshold:
                converged = True
                print("Power flow converged!")
            
            iteration += 1
        
        if not converged:
            print("Power flow did not converge within the maximum number of iterations")
        
        return converged
    
    def print_results(self):
        """Print voltage and current results"""
        print("\n--- Voltage Results ---")
        print("Node ID | Voltage Magnitude (pu) | Voltage Angle (degrees)")
        print("-" * 60)
        for node_id, node in self.nodes.items():
            v_mag = abs(node.V)
            v_angle = np.angle(node.V, deg=True)
            print(f"{node_id:6} | {v_mag:21.4f} | {v_angle:20.4f}")
        
        print("\n--- Branch Results ---")
        print("Branch ID | From Node | To Node | Current Magnitude (pu) | Power Loss (kW)")
        print("-" * 75)
        for branch_id, branch in self.branches.items():
            i_mag = abs(branch.I)
            p_loss = (branch.r * i_mag**2) * (self.base_MVA * 1000)  # Convert to kW
            print(f"{branch_id:9} | {branch.from_node.id:9} | {branch.to_node.id:7} | {i_mag:21.4f} | {p_loss:14.4f}")
    
    def plot_voltage_profile(self):
        """Plot voltage profile of the system"""
        node_ids = list(self.nodes.keys())
        v_mag = [abs(self.nodes[id].V) for id in node_ids]
        
        plt.figure(figsize=(10, 6))
        plt.bar(node_ids, v_mag, color='skyblue')
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Node ID')
        plt.ylabel('Voltage Magnitude (pu)')
        plt.title('Voltage Profile')
        plt.ylim([0.9, 1.05])
        
        # Add text annotations
        for i, v in enumerate(v_mag):
            plt.text(i, v + 0.005, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.show()

# Example: IEEE 4-node test feeder
def create_ieee_4_node_system():
    system = DistributionSystem(base_kV=12.47, base_MVA=1.0)
    
    # Add nodes (buses)
    system.add_node(1, is_substation=True)  # Substation/slack bus
    system.add_node(2, P_load_kW=1000, Q_load_kVAr=500)
    system.add_node(3, P_load_kW=1500, Q_load_kVAr=750)
    system.add_node(4, P_load_kW=2000, Q_load_kVAr=1000)
    
    # Add branches (lines)
    system.add_branch(1, 1, 2, 0.1, 0.2)  # Branch 1: From node 1 to node 2
    system.add_branch(2, 2, 3, 0.15, 0.25)  # Branch 2: From node 2 to node 3
    system.add_branch(3, 3, 4, 0.2, 0.3)  # Branch 3: From node 3 to node 4
    
    return system

# Run the simulation
if __name__ == "__main__":
    system = create_ieee_4_node_system()
    
    # Run power flow analysis
    system.run_power_flow()
    
    # Print results
    system.print_results()
    
    # Plot voltage profile
    system.plot_voltage_profile()