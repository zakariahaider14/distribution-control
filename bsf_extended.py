import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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
    def __init__(self, base_kV=4.16, base_MVA=10.0):
        self.nodes = {}
        self.branches = {}
        self.base_kV = base_kV
        self.base_MVA = base_MVA
        self.base_Z = (base_kV**2) / base_MVA
        self.convergence_threshold = 1e-5
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
    
    def build_radial_topology(self):
        """Build radial topology - identify parent-child relationships and levels"""
        # Find substation node
        self.substation_node = None
        for node in self.nodes.values():
            if node.is_substation:
                self.substation_node = node
                break
        
        if not self.substation_node:
            raise ValueError("No substation node found in the system")
        
        # Create a directed graph to represent the system
        G = nx.DiGraph()
        
        # Add nodes to the graph
        for node_id in self.nodes:
            G.add_node(node_id)
        
        # Add edges to the graph
        for branch in self.branches.values():
            G.add_edge(branch.from_node.id, branch.to_node.id)
        
        # Check if graph is a tree (no cycles)
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("The system contains loops and is not radial")
        
        # Get levels for each node (distance from root)
        self.node_levels = nx.shortest_path_length(G, source=self.substation_node.id)
        
        # Identify parent and children for each node
        self.parent = {}  # node_id -> parent_node_id
        self.children = {node_id: [] for node_id in self.nodes}  # node_id -> list of child node_ids
        self.parent_branch = {}  # node_id -> branch_id connecting parent to this node
        
        for branch in self.branches.values():
            from_id = branch.from_node.id
            to_id = branch.to_node.id
            
            self.parent[to_id] = from_id
            self.children[from_id].append(to_id)
            self.parent_branch[to_id] = branch.id
        
        # Find leaf nodes
        self.leaf_nodes = [node_id for node_id, children in self.children.items() if not children]
        
        # Sort nodes by level (for efficient traversal)
        self.nodes_by_level = {}
        for node_id, level in self.node_levels.items():
            if level not in self.nodes_by_level:
                self.nodes_by_level[level] = []
            self.nodes_by_level[level].append(node_id)
        
        self.max_level = max(self.node_levels.values())
    
    def calculate_initial_voltages(self):
        """Set all non-substation node voltages to 1.0 per unit for initial guess"""
        for node_id, node in self.nodes.items():
            if not node.is_substation:
                node.V = complex(1.0, 0.0)
    
    def backward_sweep(self):
        """Perform backward sweep - compute branch currents from loads"""
        # First calculate load currents at each node
        for node_id, node in self.nodes.items():
            if not node.is_substation:
                S = complex(node.P_load, node.Q_load)
                # I = S*/V* (complex conjugate)
                node.I = np.conj(S / np.conj(node.V))
        
        # Reset branch currents
        for branch in self.branches.values():
            branch.I = complex(0.0, 0.0)
        
        # Process from bottom (leaf nodes) to top (substation)
        for level in range(self.max_level, 0, -1):
            for node_id in self.nodes_by_level.get(level, []):
                # Get the branch connecting this node to its parent
                if node_id in self.parent_branch:
                    branch_id = self.parent_branch[node_id]
                    branch = self.branches[branch_id]
                    
                    # Start with the load current at this node
                    branch.I = self.nodes[node_id].I
                    
                    # Add currents from all branches connected to child nodes
                    for child_id in self.children[node_id]:
                        child_branch_id = self.parent_branch[child_id]
                        child_branch = self.branches[child_branch_id]
                        branch.I += child_branch.I
    
    def forward_sweep(self):
        """Perform forward sweep - update node voltages"""
        # Process from top (substation) to bottom (leaf nodes)
        for level in range(1, self.max_level + 1):
            for node_id in self.nodes_by_level.get(level, []):
                # Get parent node and branch
                parent_id = self.parent[node_id]
                branch_id = self.parent_branch[node_id]
                branch = self.branches[branch_id]
                
                # Update voltage: V_child = V_parent - Z * I
                self.nodes[node_id].V = self.nodes[parent_id].V - branch.z * branch.I
    
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
        self.build_radial_topology()
        self.calculate_initial_voltages()
        
        iteration = 0
        converged = False
        
        voltage_history = {node_id: [] for node_id in self.nodes}
        for node_id in self.nodes:
            voltage_history[node_id].append(abs(self.nodes[node_id].V))
        
        while not converged and iteration < self.max_iterations:
            # Save current voltages for convergence check
            prev_voltages = {node_id: node.V for node_id, node in self.nodes.items()}
            
            # Perform backward sweep
            self.backward_sweep()
            
            # Perform forward sweep
            self.forward_sweep()
            
            # Save voltage magnitudes for plotting
            for node_id in self.nodes:
                voltage_history[node_id].append(abs(self.nodes[node_id].V))
            
            # Check convergence
            max_mismatch = self.calculate_voltage_mismatch(prev_voltages)
            
            print(f"Iteration {iteration+1}, Max Voltage Mismatch: {max_mismatch:.8f}")
            
            if max_mismatch < self.convergence_threshold:
                converged = True
                print("Power flow converged!")
            
            iteration += 1
        
        if not converged:
            print("Power flow did not converge within the maximum number of iterations")
        
        return converged, voltage_history
    
    def print_results(self):
        """Print voltage and current results"""
        print("\n--- Voltage Results ---")
        print("Node ID | Voltage Magnitude (pu) | Voltage Angle (degrees)")
        print("-" * 60)
        for node_id, node in sorted(self.nodes.items()):
            v_mag = abs(node.V)
            v_angle = np.angle(node.V, deg=True)
            print(f"{node_id:6} | {v_mag:21.4f} | {v_angle:20.4f}")
        
        print("\n--- Branch Results ---")
        print("Branch ID | From Node | To Node | Current Magnitude (pu) | Power Loss (kW)")
        print("-" * 75)
        total_loss = 0
        for branch_id, branch in sorted(self.branches.items()):
            i_mag = abs(branch.I)
            p_loss = (branch.r * i_mag**2) * (self.base_MVA * 1000)  # Convert to kW
            total_loss += p_loss
            print(f"{branch_id:9} | {branch.from_node.id:9} | {branch.to_node.id:7} | {i_mag:21.4f} | {p_loss:14.4f}")
        
        print(f"\nTotal System Losses: {total_loss:.4f} kW")
    
    def plot_voltage_profile(self):
        """Plot voltage profile of the system"""
        node_ids = list(self.nodes.keys())
        v_mag = [abs(self.nodes[id].V) for id in node_ids]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(node_ids, v_mag, color='skyblue')
        plt.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, label='Nominal Voltage')
        plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='Lower Limit (-5%)')
        plt.axhline(y=1.05, color='red', linestyle='--', alpha=0.7, label='Upper Limit (+5%)')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Node ID')
        plt.ylabel('Voltage Magnitude (pu)')
        plt.title('Voltage Profile')
        plt.ylim([0.9, 1.1])
        plt.legend()
        
        # Add text annotations
        for bar, v in zip(bars, v_mag):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_network_graph(self):
        """Plot the network graph showing the topology"""
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np
        
        G = nx.DiGraph()
        
        # Add nodes to the graph
        for node_id, node in self.nodes.items():
            G.add_node(node_id, voltage=abs(node.V))
        
        # Add edges to the graph
        edge_labels = {}
        for branch_id, branch in self.branches.items():
            from_id = branch.from_node.id
            to_id = branch.to_node.id
            G.add_edge(from_id, to_id, current=abs(branch.I))
            edge_labels[(from_id, to_id)] = f"{branch_id}"
        
        # Set up node colors based on voltage
        node_colors = [abs(self.nodes[n].V) for n in G.nodes()]
        
        # Set up positions for nodes
        pos = nx.kamada_kawai_layout(G)
        
        plt.figure(figsize=(12, 10))
        
        # Draw network
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, 
                            cmap=plt.cm.RdYlGn, vmin=0.9, vmax=1.1)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        # Add colorbar - fixed version
        plt.colorbar(nodes, label='Voltage Magnitude (pu)')
        
        plt.title('Network Graph with Voltage Profile')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_convergence(self, voltage_history):
        """Plot voltage convergence during iterations"""
        iterations = range(len(voltage_history[list(voltage_history.keys())[0]]))
        
        plt.figure(figsize=(12, 6))
        
        for node_id, voltages in voltage_history.items():
            if not self.nodes[node_id].is_substation:  # Skip substation (always 1.0)
                plt.plot(iterations, voltages, marker='o', label=f'Node {node_id}')
        
        plt.xlabel('Iteration')
        plt.ylabel('Voltage Magnitude (pu)')
        plt.title('Voltage Convergence')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

# IEEE 13-node test feeder - simplified version
def create_ieee_13_node_system():
    system = DistributionSystem(base_kV=4.16, base_MVA=5.0)
    
    # Add nodes (buses)
    system.add_node(650, is_substation=True)  # Substation (slack bus)
    system.add_node(632, P_load_kW=0,      Q_load_kVAr=0  )
    system.add_node(645, P_load_kW=170,    Q_load_kVAr=125)
    system.add_node(646, P_load_kW=230,    Q_load_kVAr=132)
    system.add_node(633, P_load_kW=0,      Q_load_kVAr=0  )
    system.add_node(634, P_load_kW=400,    Q_load_kVAr=290)
    system.add_node(671, P_load_kW=1155,   Q_load_kVAr=660)
    system.add_node(684, P_load_kW=0,      Q_load_kVAr=0  )
    system.add_node(611, P_load_kW=170,    Q_load_kVAr=80 )
    system.add_node(652, P_load_kW=128,    Q_load_kVAr=86)
    system.add_node(680, P_load_kW=0,      Q_load_kVAr=0)
    system.add_node(692, P_load_kW=0,      Q_load_kVAr=0)
    system.add_node(675, P_load_kW=843,    Q_load_kVAr=462)
    
    # Add branches (lines) - simplified impedance values
    system.add_branch(1, 650, 632, 0.186, 0.421)    # Line 1: 650-632
    system.add_branch(2, 632, 645, 0.262, 0.261)    # Line 2: 632-645
    system.add_branch(3, 645, 646, 0.194, 0.194)    # Line 3: 645-646
    system.add_branch(4, 632, 633, 0.096, 0.201)    # Line 4: 632-633
    system.add_branch(5, 633, 634, 0.751, 0.535)    # Line 5: 633-634 (transformer)
    system.add_branch(6, 632, 671, 0.381, 0.450)    # Line 6: 632-671
    system.add_branch(7, 671, 684, 0.152, 0.209)    # Line 7: 671-684
    system.add_branch(8, 684, 611, 0.152, 0.076)    # Line 8: 684-611
    system.add_branch(9, 671, 680, 0.061, 0.061)    # Line 9: 671-680
    system.add_branch(10, 671, 692, 0.000, 0.000)   # Line 10: 671-692 (zero impedance switch)
    system.add_branch(11, 692, 675, 0.174, 0.131)   # Line 11: 692-675
    system.add_branch(12, 684, 652, 0.359, 0.308)   # Line 12: 684-652
    
    return system

# Function to run a complete analysis of the system
def run_full_analysis(system):
    print("Running Backward/Forward Sweep Power Flow Analysis...")
    converged, voltage_history = system.run_power_flow()
    
    if converged:
        # Print detailed results
        system.print_results()
        
        # Plot voltage profile
        system.plot_voltage_profile()
        
        # Plot network topology with voltages
        system.plot_network_graph()
        
        # Plot convergence
        system.plot_convergence(voltage_history)
    else:
        print("Analysis failed to converge!")

# Main execution
if __name__ == "__main__":
    # Create IEEE 13-node test feeder
    system = create_ieee_13_node_system()
    
    # Run the analysis
    run_full_analysis(system)
    
    print("\nAnalysis complete!")