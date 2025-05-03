import opendssdirect as dss
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap

class DistributionGridAnalysis:
    def __init__(self, dss_file_path):
        """Initialize the DSS engine and load the circuit"""
        # Store the absolute path to the DSS file
        self.dss_file_path = os.path.abspath(dss_file_path)
        
        # Create the line codes file if it doesn't exist
        line_codes_path = os.path.join(os.path.dirname(self.dss_file_path), "IEEELineCodes.dss")
        if not os.path.exists(line_codes_path):
            self.create_ieee_line_codes_file(line_codes_path)
        
        # Compile the DSS file
        dss.Basic.ClearAll()
        dss.run_command(f"Compile '{self.dss_file_path}'")
        
        # Solve the initial power flow
        dss.Solution.Solve()
        
        # Store initial system state
        self.base_voltages = {}
        self.base_powers = {}
        self.store_system_state(self.base_voltages, self.base_powers)
        
        # Get bus coordinates for plotting the network
        self.bus_coords = self.get_bus_coordinates()
        
        # Store the critical loads (can be customized)
        self.critical_loads = ["S860", "S844", "S848", "S890"]
        
        # Storage for PV scenarios
        self.pv_scenarios = {}

    def create_ieee_line_codes_file(self, file_path):
        """Create the IEEE line codes file that's referenced in the main DSS file"""
        ieee_line_codes = """
            ! this file was corrected 9/16/2010 to match the values in Kersting's files



            ! These line codes are used in the 123-bus circuit

            New linecode.1 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.088205 | 0.0312137 0.0901946 | 0.0306264 0.0316143 0.0889665 )
            !!!~ xmatrix = (0.20744 | 0.0935314 0.200783 | 0.0760312 0.0855879 0.204877 )
            !!!~ cmatrix = (2.90301 | -0.679335 3.15896 | -0.22313 -0.481416 2.8965 )
            ~ rmatrix = [0.086666667 | 0.029545455 0.088371212 | 0.02907197 0.029924242 0.087405303]
            ~ xmatrix = [0.204166667 | 0.095018939 0.198522727 | 0.072897727 0.080227273 0.201723485]
            ~ cmatrix = [2.851710072 | -0.920293787  3.004631862 | -0.350755566  -0.585011253 2.71134756]

            New linecode.2 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.0901946 | 0.0316143 0.0889665 | 0.0312137 0.0306264 0.088205 )
            !!!~ xmatrix = (0.200783 | 0.0855879 0.204877 | 0.0935314 0.0760312 0.20744 )
            !!!~ cmatrix = (3.15896 | -0.481416 2.8965 | -0.679335 -0.22313 2.90301 )
            ~ rmatrix = [0.088371212 | 0.02992424  0.087405303 | 0.029545455 0.02907197 0.086666667]
            ~ xmatrix = [0.198522727 | 0.080227273  0.201723485 | 0.095018939 0.072897727 0.204166667]
            ~ cmatrix = [3.004631862 | -0.585011253 2.71134756 | -0.920293787  -0.350755566  2.851710072]

            New linecode.3 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.0889665 | 0.0306264 0.088205 | 0.0316143 0.0312137 0.0901946 )
            !!!~ xmatrix = (0.204877 | 0.0760312 0.20744 | 0.0855879 0.0935314 0.200783 )
            !!!~ cmatrix = (2.8965 | -0.22313 2.90301 | -0.481416 -0.679335 3.15896 )

            ~ rmatrix = [0.087405303 | 0.02907197 0.086666667  | 0.029924242 0.029545455 0.088371212]
            ~ xmatrix = [0.201723485 | 0.072897727 0.204166667 | 0.080227273 0.095018939 0.198522727]
            ~ cmatrix = [2.71134756  | -0.350755566 2.851710072 | -0.585011253 -0.920293787 3.004631862]

            New linecode.4 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.0889665 | 0.0316143 0.0901946 | 0.0306264 0.0312137 0.088205 )
            !!!~ xmatrix = (0.204877 | 0.0855879 0.200783 | 0.0760312 0.0935314 0.20744 )
            !!!~ cmatrix = (2.8965 | -0.481416 3.15896 | -0.22313 -0.679335 2.90301 )
            ~ rmatrix = [0.087405303 | 0.029924242 0.088371212 | 0.02907197   0.029545455 0.086666667]
            ~ xmatrix = [0.201723485 | 0.080227273 0.198522727 | 0.072897727 0.095018939 0.204166667]
            ~ cmatrix = [2.71134756  | -0.585011253 3.004631862 | -0.350755566 -0.920293787 2.851710072]

            New linecode.5 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.0901946 | 0.0312137 0.088205 | 0.0316143 0.0306264 0.0889665 )
            !!!~ xmatrix = (0.200783 | 0.0935314 0.20744 | 0.0855879 0.0760312 0.204877 )
            !!!~ cmatrix = (3.15896 | -0.679335 2.90301 | -0.481416 -0.22313 2.8965 )

            ~ rmatrix = [0.088371212  |  0.029545455  0.086666667  |  0.029924242  0.02907197  0.087405303]
            ~ xmatrix = [0.198522727  |  0.095018939  0.204166667  |  0.080227273  0.072897727  0.201723485]
            ~ cmatrix = [3.004631862  | -0.920293787  2.851710072  |  -0.585011253  -0.350755566  2.71134756]

            New linecode.6 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.088205 | 0.0306264 0.0889665 | 0.0312137 0.0316143 0.0901946 )
            !!!~ xmatrix = (0.20744 | 0.0760312 0.204877 | 0.0935314 0.0855879 0.200783 )
            !!!~ cmatrix = (2.90301 | -0.22313 2.8965 | -0.679335 -0.481416 3.15896 )
            ~ rmatrix = [0.086666667 | 0.02907197  0.087405303 | 0.029545455  0.029924242  0.088371212]
            ~ xmatrix = [0.204166667 | 0.072897727  0.201723485 | 0.095018939  0.080227273  0.198522727]
            ~ cmatrix = [2.851710072 | -0.350755566  2.71134756 | -0.920293787  -0.585011253  3.004631862]
            New linecode.7 nphases=2 BaseFreq=60
            !!!~ rmatrix = (0.088205 | 0.0306264 0.0889665 )
            !!!~ xmatrix = (0.20744 | 0.0760312 0.204877 )
            !!!~ cmatrix = (2.75692 | -0.326659 2.82313 )
            ~ rmatrix = [0.086666667 | 0.02907197  0.087405303]
            ~ xmatrix = [0.204166667 | 0.072897727  0.201723485]
            ~ cmatrix = [2.569829596 | -0.52995137  2.597460011]
            New linecode.8 nphases=2 BaseFreq=60
            !!!~ rmatrix = (0.088205 | 0.0306264 0.0889665 )
            !!!~ xmatrix = (0.20744 | 0.0760312 0.204877 )
            !!!~ cmatrix = (2.75692 | -0.326659 2.82313 )
            ~ rmatrix = [0.086666667 | 0.02907197  0.087405303]
            ~ xmatrix = [0.204166667 | 0.072897727  0.201723485]
            ~ cmatrix = [2.569829596 | -0.52995137  2.597460011]
            New linecode.9 nphases=1 BaseFreq=60
            !!!~ rmatrix = (0.254428 )
            !!!~ xmatrix = (0.259546 )
            !!!~ cmatrix = (2.50575 )
            ~ rmatrix = [0.251742424]
            ~ xmatrix = [0.255208333]
            ~ cmatrix = [2.270366128]
            New linecode.10 nphases=1 BaseFreq=60
            !!!~ rmatrix = (0.254428 )
            !!!~ xmatrix = (0.259546 )
            !!!~ cmatrix = (2.50575 )
            ~ rmatrix = [0.251742424]
            ~ xmatrix = [0.255208333]
            ~ cmatrix = [2.270366128]
            New linecode.11 nphases=1 BaseFreq=60
            !!!~ rmatrix = (0.254428 )
            !!!~ xmatrix = (0.259546 )
            !!!~ cmatrix = (2.50575 )
            ~ rmatrix = [0.251742424]
            ~ xmatrix = [0.255208333]
            ~ cmatrix = [2.270366128]
            New linecode.12 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.291814 | 0.101656 0.294012 | 0.096494 0.101656 0.291814 )
            !!!~ xmatrix = (0.141848 | 0.0517936 0.13483 | 0.0401881 0.0517936 0.141848 )
            !!!~ cmatrix = (53.4924 | 0 53.4924 | 0 0 53.4924 )
            ~ rmatrix = [0.288049242 | 0.09844697  0.29032197 | 0.093257576  0.09844697  0.288049242]
            ~ xmatrix = [0.142443182 | 0.052556818  0.135643939 | 0.040852273  0.052556818  0.142443182]
            ~ cmatrix = [33.77150149 | 0  33.77150149 | 0  0  33.77150149]

            ! These line codes are used in the 34-node test feeder

            New linecode.300 nphases=3 basefreq=60   units=kft   ! ohms per 1000ft  Corrected 11/30/05
            ~ rmatrix = [0.253181818   |  0.039791667     0.250719697  |   0.040340909      0.039128788     0.251780303]  !ABC ORDER
            ~ xmatrix = [0.252708333   |  0.109450758     0.256988636  |   0.094981061      0.086950758     0.255132576]
            ~ CMATRIX = [2.680150309   | -0.769281006     2.5610381    |  -0.499507676     -0.312072984     2.455590387]
            New linecode.301 nphases=3 basefreq=60   units=kft
            ~ rmatrix = [0.365530303   |   0.04407197      0.36282197   |   0.04467803       0.043333333     0.363996212]
            ~ xmatrix = [0.267329545   |   0.122007576     0.270473485  |   0.107784091      0.099204545     0.269109848] 
            ~ cmatrix = [2.572492163   |  -0.72160598      2.464381882  |  -0.472329395     -0.298961096     2.368881119]
            New linecode.302 nphases=1 basefreq=60   units=kft
            ~ rmatrix = (0.530208 )
            ~ xmatrix = (0.281345 )
            ~ cmatrix = (2.12257 )
            New linecode.303 nphases=1 basefreq=60   units=kft
            ~ rmatrix = (0.530208 )
            ~ xmatrix = (0.281345 )
            ~ cmatrix = (2.12257 )
            New linecode.304 nphases=1 basefreq=60   units=kft
            ~ rmatrix = (0.363958 )
            ~ xmatrix = (0.269167 )
            ~ cmatrix = (2.1922 )


            ! This may be for the 4-node test feeder, but is not actually referenced.
            !  instead, the 4Bus*.dss files all use the wiredata and linegeometry inputs
            !  to calculate these matrices from physical data.

            New linecode.400 nphases=3 BaseFreq=60
            ~ rmatrix = (0.088205 | 0.0312137 0.0901946 | 0.0306264 0.0316143 0.0889665 )
            ~ xmatrix = (0.20744 | 0.0935314 0.200783 | 0.0760312 0.0855879 0.204877 )
            ~ cmatrix = (2.90301 | -0.679335 3.15896 | -0.22313 -0.481416 2.8965 )

            ! These are for the 13-node test feeder

            New linecode.601 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.0674673 | 0.0312137 0.0654777 | 0.0316143 0.0306264 0.0662392 )
            !!!~ xmatrix = (0.195204  | 0.0935314 0.201861 | 0.0855879 0.0760312 0.199298 )
            !!!~ cmatrix = (3.32591   | -0.743055 3.04217 | -0.525237 -0.238111 3.03116 )
            ~ rmatrix = [0.065625    | 0.029545455  0.063920455  | 0.029924242  0.02907197  0.064659091]
            ~ xmatrix = [0.192784091 | 0.095018939  0.19844697   | 0.080227273  0.072897727  0.195984848]
            ~ cmatrix = [3.164838036 | -1.002632425  2.993981593 | -0.632736516  -0.372608713  2.832670203]
            New linecode.602 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.144361 | 0.0316143 0.143133 | 0.0312137 0.0306264 0.142372 )
            !!!~ xmatrix = (0.226028 | 0.0855879 0.230122 | 0.0935314 0.0760312 0.232686 )
            !!!~ cmatrix = (3.01091  | -0.443561 2.77543  | -0.624494 -0.209615 2.77847 )
            ~ rmatrix = [0.142537879 | 0.029924242  0.14157197   | 0.029545455  0.02907197  0.140833333]
            ~ xmatrix = [0.22375     | 0.080227273  0.226950758  | 0.095018939  0.072897727  0.229393939]
            ~ cmatrix = [2.863013423 | -0.543414918  2.602031589 | -0.8492585  -0.330962141  2.725162768]
            New linecode.603 nphases=2 BaseFreq=60
            !!!~ rmatrix = (0.254472 | 0.0417943 0.253371 )
            !!!~ xmatrix = (0.259467 | 0.0912376 0.261431 )
            !!!~ cmatrix = (2.54676  | -0.28882 2.49502 )
            ~ rmatrix = [0.251780303 | 0.039128788  0.250719697]
            ~ xmatrix = [0.255132576 | 0.086950758  0.256988636]
            ~ cmatrix = [2.366017603 | -0.452083836  2.343963508]
            New linecode.604 nphases=2 BaseFreq=60
            !!!~ rmatrix = (0.253371 | 0.0417943 0.254472 )
            !!!~ xmatrix = (0.261431 | 0.0912376 0.259467 )
            !!!~ cmatrix = (2.49502 | -0.28882 2.54676 )
            ~ rmatrix = [0.250719697 | 0.039128788   0.251780303]
            ~ xmatrix = [0.256988636  | 0.086950758  0.255132576]
            ~ cmatrix = [2.343963508 | -0.452083836 2.366017603]
            New linecode.605 nphases=1 BaseFreq=60
            !!!~ rmatrix = (0.254428 )
            !!!~ xmatrix = (0.259546 )
            !!!~ cmatrix = (2.50575 )
            ~ rmatrix = [0.251742424]
            ~ xmatrix = [0.255208333]
            ~ cmatrix = [2.270366128]
            New linecode.606 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.152193 | 0.0611362 0.15035 | 0.0546992 0.0611362 0.152193 )
            !!!~ xmatrix = (0.0825685 | 0.00548281 0.0745027 | -0.00339824 0.00548281 0.0825685 )
            !!!~ cmatrix = (72.7203 | 0 72.7203 | 0 0 72.7203 )
            ~ rmatrix = [0.151174242 | 0.060454545  0.149450758 | 0.053958333  0.060454545  0.151174242]
            ~ xmatrix = [0.084526515 | 0.006212121  0.076534091 | -0.002708333  0.006212121  0.084526515]
            ~ cmatrix = [48.67459408 | 0  48.67459408 | 0  0  48.67459408]
            New linecode.607 nphases=1 BaseFreq=60
            !!!~ rmatrix = (0.255799 )
            !!!~ xmatrix = (0.092284 )
            !!!~ cmatrix = (50.7067 )
            ~ rmatrix = [0.254261364]
            ~ xmatrix = [0.097045455]
            ~ cmatrix = [44.70661522]

            ! These are for the 37-node test feeder, all underground

            New linecode.721 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.0554906 | 0.0127467 0.0501597 | 0.00640446 0.0127467 0.0554906 )
            !!!~ xmatrix = (0.0372331 | -0.00704588 0.0358645 | -0.00796424 -0.00704588 0.0372331 )
            !!!~ cmatrix = (124.851 | 0 124.851 | 0 0 124.851 )
            ~ rmatrix = [0.055416667 | 0.012746212  0.050113636  | 0.006382576  0.012746212  0.055416667]
            ~ xmatrix = [0.037367424 | -0.006969697  0.035984848 | -0.007897727  -0.006969697  0.037367424]
            ~ cmatrix = [80.27484728 | 0  80.27484728            | 0  0  80.27484728]
            New linecode.722 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.0902251 | 0.0309584 0.0851482 | 0.0234946 0.0309584 0.0902251 )
            !!!~ xmatrix = (0.055991 | -0.00646552 0.0504025 | -0.0117669 -0.00646552 0.055991 )
            !!!~ cmatrix = (93.4896 | 0 93.4896 | 0 0 93.4896 )
            ~ rmatrix = [0.089981061 | 0.030852273  0.085        | 0.023371212  0.030852273  0.089981061]
            ~ xmatrix = [0.056306818 | -0.006174242  0.050719697 | -0.011496212  -0.006174242  0.056306818]
            ~ cmatrix = [64.2184109 | 0  64.2184109              | 0  0  64.2184109]
            New linecode.723 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.247572 | 0.0947678 0.249104 | 0.0893782 0.0947678 0.247572 )
            !!!~ xmatrix = (0.126339 | 0.0390337 0.118816 | 0.0279344 0.0390337 0.126339 )
            !!!~ cmatrix = (58.108 | 0 58.108 | 0 0 58.108 )
            ~ rmatrix = [0.245 | 0.092253788  0.246628788 | 0.086837121  0.092253788  0.245]
            ~ xmatrix = [0.127140152 | 0.039981061  0.119810606 | 0.028806818  0.039981061  0.127140152]
            ~ cmatrix = [37.5977112 | 0  37.5977112 | 0  0  37.5977112]
            New linecode.724 nphases=3 BaseFreq=60
            !!!~ rmatrix = (0.399883 | 0.101765 0.402011 | 0.0965199 0.101765 0.399883 )
            !!!~ xmatrix = (0.146325 | 0.0510963 0.139305 | 0.0395402 0.0510963 0.146325 )
            !!!~ cmatrix = (46.9685 | 0 46.9685 | 0 0 46.9685 )
            ~ rmatrix = [0.396818182 | 0.098560606  0.399015152 | 0.093295455  0.098560606  0.396818182]
            ~ xmatrix = [0.146931818 | 0.051856061  0.140113636 | 0.040208333  0.051856061  0.146931818]
            ~ cmatrix = [30.26701029 | 0  30.26701029 | 0  0  30.26701029]

        """
        
        with open(file_path, 'w') as f:
            f.write(ieee_line_codes)

    def store_system_state(self, voltage_dict, power_dict):
        """Store the current system state (voltages and powers)"""
        # Get all buses
        buses = dss.Circuit.AllBusNames()
        
        for bus in buses:
            dss.Circuit.SetActiveBus(bus)
            voltages = dss.Bus.puVmagAngle()
            # Convert from per unit to actual values
            voltages_real = []
            for i in range(0, len(voltages), 2):
                voltages_real.append(voltages[i] * dss.Bus.kVBase() * 1000)  # Convert to V
                voltages_real.append(voltages[i+1])  # Angle
            voltage_dict[bus] = voltages_real
            
            # For powers, we need to sum up all connected elements
            powers = [0] * (len(voltages) // 2 * 2)  # Initialize with zeros
            power_dict[bus] = powers
    
    def get_bus_coordinates(self):
        """Get bus coordinates for plotting the network"""
        # For IEEE 34-bus, we'll use relative positions since the file doesn't include coordinates
        # This is a simplified layout - for real coordinates, you would need to use GIS data
        bus_coords = {}
        
        # Get all buses
        buses = dss.Circuit.AllBusNames()
        
        # Create a simple layout based on bus numbers
        for i, bus in enumerate(buses):
            # Extract the bus number for positioning
            try:
                bus_num = int(''.join(filter(str.isdigit, bus.split('.')[0])))
                bus_coords[bus] = (bus_num % 10, bus_num // 10)
            except:
                bus_coords[bus] = (i % 10, i // 10)
        
        return bus_coords
    
    def plot_network(self, highlight_buses=None, title="Distribution Network"):
        """Plot the network with optional bus highlighting"""
        G = nx.Graph()
        
        # Get all lines
        dss.Circuit.SetActiveClass('Line')
        line_names = dss.ActiveClass.AllNames()
        
        # Add nodes and edges to the graph
        for line in line_names:
            dss.Circuit.SetActiveElement(f"Line.{line}")
            buses = [dss.CktElement.BusNames()[0].split('.')[0], 
                     dss.CktElement.BusNames()[1].split('.')[0]]
            
            G.add_node(buses[0])
            G.add_node(buses[1])
            G.add_edge(buses[0], buses[1], name=line)
        
        # Use bus coordinates for layout if available
        pos = {}
        for node in G.nodes():
            base_node = node.split('.')[0]
            if base_node in self.bus_coords:
                pos[node] = self.bus_coords[base_node]
            else:
                # If no coordinates, use spring layout
                temp_pos = nx.spring_layout(G)
                pos.update(temp_pos)
                break
        
        # Draw the network
        plt.figure(figsize=(12, 8))
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        
        # Draw regular nodes
        regular_nodes = [n for n in G.nodes() if highlight_buses is None or n not in highlight_buses]
        nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, node_size=100, node_color='blue', alpha=0.7)
        
        # Draw highlighted nodes
        if highlight_buses:
            nx.draw_networkx_nodes(G, pos, nodelist=highlight_buses, node_size=200, node_color='red', alpha=0.9)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        return plt
    
    def add_pv_system(self, bus, kw, kvar=0, pf=1.0, name=None):
        """Add a PV system to the specified bus"""
        if name is None:
            name = f"PV_{bus}"
        
        dss.run_command(f"""
        New PVSystem.{name} 
            phases=3 
            bus1={bus} 
            kV=24.9 
            pmpp={kw} 
            kvar={kvar}
            pf={pf} 
            irradiance=1.0 
            yearly=''
            daily=''
            duty=''
            conn=wye 
            model=1
        """)
        return name
    
    def create_pv_scenarios(self):
        """Create different PV penetration scenarios"""
        # Scenario 1: PV at substation
        # Scenario 1: PV at substation
        dss.Solution.Solve()
        dss.Basic.ClearAll()
        dss.run_command(f"Compile '{self.dss_file_path}'")
        dss.Solution.Solve()
        self.add_pv_system("800", 500, name="PV_Substation")
        dss.Solution.Solve()
        
        # Initialize the voltages dictionary first
        self.pv_scenarios["high_penetration"] = {"voltages": {}, "powers": {}}
        self.store_system_state(self.pv_scenarios["high_penetration"]["voltages"], 
                            self.pv_scenarios["high_penetration"]["powers"])
                
        # Repeat the same pattern for other scenarios
        # Scenario 2: Distributed PV systems
        dss.Solution.Solve()
        dss.Basic.ClearAll()
        dss.run_command(f"Compile '{self.dss_file_path}'")
        dss.Solution.Solve()
        self.add_pv_system("816", 100, name="PV_816")
        self.add_pv_system("832", 150, name="PV_832")
        self.add_pv_system("848", 100, name="PV_848")
        self.add_pv_system("860", 150, name="PV_860")
        dss.Solution.Solve()
        
        self.pv_scenarios["distributed"] = {"voltages": {}, "powers": {}}
        self.store_system_state(self.pv_scenarios["distributed"]["voltages"], 
                            self.pv_scenarios["distributed"]["powers"])
        
        # Scenario 3: High PV penetration
        dss.Solution.Solve()
        dss.Basic.ClearAll()
        dss.run_command(f"Compile '{self.dss_file_path}'")

        dss.Solution.Solve()
        self.add_pv_system("816", 200, name="PV_816")
        self.add_pv_system("824", 150, name="PV_824")
        self.add_pv_system("828", 100, name="PV_828")
        self.add_pv_system("832", 300, name="PV_832")
        self.add_pv_system("844", 200, name="PV_844")
        self.add_pv_system("848", 200, name="PV_848")
        self.add_pv_system("860", 300, name="PV_860")
        self.add_pv_system("890", 150, name="PV_890")
        dss.Solution.Solve()
        self.pv_scenarios["high_penetration"] = {"voltages": {}, "powers": {}}
        self.store_system_state(self.pv_scenarios["high_penetration"]["voltages"], 
                            self.pv_scenarios["high_penetration"]["powers"])
    
    def simulate_black_start(self, pv_buses):
        """Simulate a black start scenario using PV systems"""
        # Clear circuit and reload base case
        dss.Basic.ClearAll()
        dss.run_command(f"Compile '{self.dss_file_path}'")
        
        # Disable main source (simulate blackout)
        dss.run_command("Disable Transformer.SubXF")
        
        # Add PV systems for black start with batteries
        for i, bus in enumerate(pv_buses):
            self.add_pv_system(bus, 300, kvar=100, name=f"BS_PV_{i}")
            # Add a storage element to each PV system for black start capability
            dss.run_command(f"""
            New Storage.BESS_{i} 
                phases=3 
                bus1={bus} 
                kW=300 
                kWrated=500
                kWhrated=2000
                %stored=80
                %reserve=10
                model=1
                conn=wye
                kV=24.9
                dischargetrigger=0
                chargetrigger=0
                state=discharging
            """)
        
        # Only enable critical loads
        dss.Circuit.SetActiveClass("Load")
        all_loads = dss.ActiveClass.AllNames()
        for load in all_loads:
            if load in self.critical_loads:
                dss.run_command(f"Load.{load}.enabled=True")
            else:
                dss.run_command(f"Load.{load}.enabled=False")
        
        # Try to solve the circuit
        dss.run_command("set maxiterations=100")
        dss.Solution.Solve()
        
        # Store results
        black_start_results = {
            "converged": dss.Solution.Converged(),
            "voltages": {},
            "powers": {}
        }
        self.store_system_state(black_start_results["voltages"], black_start_results["powers"])
        
        return black_start_results
    
    def plot_voltage_profile(self, scenario_name=None):
        """Plot voltage profile for the current circuit state or a specific scenario"""
        plt.figure(figsize=(12, 6))
        
        if scenario_name:
            voltages = self.pv_scenarios[scenario_name]["voltages"]
            title = f"Voltage Profile: {scenario_name.replace('_', ' ').title()} Scenario"
        else:
            # Get current state
            voltages = {}
            self.store_system_state(voltages, {})
            title = "Current Voltage Profile"
        
        # Prepare data for plotting
        bus_names = []
        v_pu_a = []
        v_pu_b = []
        v_pu_c = []
        
        for bus, v_data in voltages.items():
            # Skip source bus
            if bus.startswith("source"):
                continue
                
            # Extract base bus name without node IDs
            base_bus = bus.split('.')[0]
            
            # Only process each base bus once
            if base_bus in bus_names:
                continue
                
            bus_names.append(base_bus)
            
            # Get per-unit voltages for each phase if available
            # IEEE 34-bus nominal voltage is 24.9kV for most buses
            nominal_v = 24900 / np.sqrt(3)  # Line-to-neutral voltage
            
            if len(v_data) >= 2:  # We have voltage magnitudes
                # Calculate number of phases from voltage data
                num_phases = len(v_data) // 2
                
                # Initialize with NaN for missing phases
                v_a, v_b, v_c = float('nan'), float('nan'), float('nan')
                
                # Assign actual values where available
                if num_phases >= 1:
                    v_a = v_data[0] / nominal_v
                if num_phases >= 2:
                    v_b = v_data[2] / nominal_v
                if num_phases >= 3:
                    v_c = v_data[4] / nominal_v
                
                v_pu_a.append(v_a)
                v_pu_b.append(v_b)
                v_pu_c.append(v_c)
        
        # Sort buses by name/number for better visualization
        sorted_indices = sorted(range(len(bus_names)), key=lambda i: int(''.join(filter(str.isdigit, bus_names[i]))) if any(c.isdigit() for c in bus_names[i]) else 0)
        bus_names = [bus_names[i] for i in sorted_indices]
        v_pu_a = [v_pu_a[i] for i in sorted_indices]
        v_pu_b = [v_pu_b[i] for i in sorted_indices]
        v_pu_c = [v_pu_c[i] for i in sorted_indices]
        
        # Plotting
        x = range(len(bus_names))
        
        plt.plot(x, v_pu_a, 'ro-', label='Phase A', markersize=4)
        plt.plot(x, v_pu_b, 'go-', label='Phase B', markersize=4)
        plt.plot(x, v_pu_c, 'bo-', label='Phase C', markersize=4)
        
        # Add reference lines
        plt.axhline(y=1.05, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.3)
        
        plt.xticks(x, bus_names, rotation=90)
        plt.ylabel('Voltage (p.u.)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        return plt
    
    def simulate_load_restoration(self, pv_locations):
        """Simulate load restoration with PV support"""
        # Clear circuit and reload base case
        dss.Basic.ClearAll()
        dss.run_command(f"Compile '{self.dss_file_path}'")

        
        # Start with only critical loads enabled
        dss.Circuit.SetActiveClass("Load")
        all_loads = dss.ActiveClass.AllNames()
        for load in all_loads:
            if load in self.critical_loads:
                dss.run_command(f"Load.{load}.enabled=True")
            else:
                dss.run_command(f"Load.{load}.enabled=False")
        
        # Add PV systems
        for i, location in enumerate(pv_locations):
            self.add_pv_system(location, 200, name=f"Rest_PV_{i}")
        
        # Solve with only critical loads
        dss.Solution.Solve()
        
        critical_results = {
            "converged": dss.Solution.Converged(),
            "voltages": {},
            "powers": {}
        }
        self.store_system_state(critical_results["voltages"], critical_results["powers"])
        
        # Now restore all loads step by step
        restoration_steps = []
        load_groups = [
            # Group 1: Some medium priority loads
            ["S840", "S830a", "S830b", "S830c"],
            # Group 2: More loads
            ["D828_830sa", "D828_830ra", "D824_826sb", "D824_826rb"],
            # Group 3: All remaining loads
            [load for load in all_loads if load not in self.critical_loads and 
                           load not in ["S840", "S830a", "S830b", "S830c", 
                                        "D828_830sa", "D828_830ra", "D824_826sb", "D824_826rb"]]
        ]
        
        for i, group in enumerate(load_groups):
            # Enable this group of loads
            for load in group:
                dss.run_command(f"Load.{load}.enabled=True")
            
            # Try to solve
            dss.Solution.Solve()
            
            # Record results
            step_result = {
                "step": i+1,
                "loads_added": len(group),
                "converged": dss.Solution.Converged(),
                "voltages": {},
                "powers": {}
            }
            self.store_system_state(step_result["voltages"], step_result["powers"])
            restoration_steps.append(step_result)
        
        return critical_results, restoration_steps
    
    def analyze_load_dispatch(self):
        """Analyze load dispatch in different PV scenarios"""
        scenarios = ["No PV", "Distributed PV", "High PV Penetration"]
        
        # Get base case (no PV)
        dss.Basic.ClearAll()
        dss.run_command(f"Compile '{self.dss_file_path}'")

        dss.Solution.Solve()
        base_case_losses = dss.Circuit.Losses()[0] / 1000  # Convert W to kW
        
        # Create distributed PV scenario
        dss.Basic.ClearAll()
        dss.run_command(f"Compile '{self.dss_file_path}'")

        dss.Solution.Solve()
        self.add_pv_system("816", 100)
        self.add_pv_system("832", 150)
        self.add_pv_system("848", 100)
        self.add_pv_system("860", 150)
        dss.Solution.Solve()
        dist_pv_losses = dss.Circuit.Losses()[0] / 1000  # Convert W to kW
        
        # Create high penetration PV scenario
        dss.Basic.ClearAll()
        dss.run_command(f"Compile '{self.dss_file_path}'")

        dss.Solution.Solve()
        self.add_pv_system("816", 200)
        self.add_pv_system("824", 150)
        self.add_pv_system("828", 100)
        self.add_pv_system("832", 300)
        self.add_pv_system("844", 200)
        self.add_pv_system("848", 200)
        self.add_pv_system("860", 300)
        self.add_pv_system("890", 150)
        dss.Solution.Solve()
        high_pv_losses = dss.Circuit.Losses()[0] / 1000  # Convert W to kW
        
        # Plot results
        plt.figure(figsize=(10, 6))
        
        losses = [base_case_losses, dist_pv_losses, high_pv_losses]
        colors = ['crimson', 'royalblue', 'forestgreen']
        
        plt.bar(scenarios, losses, color=colors)
        plt.ylabel('Total System Losses (kW)')
        plt.title('System Losses Comparison Across PV Scenarios')
        plt.grid(axis='y', alpha=0.3)
        
        # Add loss reduction percentages
        for i in range(1, len(losses)):
            reduction = ((base_case_losses - losses[i]) / base_case_losses) * 100
            plt.text(i, losses[i] + 10, f"{reduction:.1f}% reduction", 
                     ha='center', va='bottom', color='black', fontweight='bold')
        
        plt.tight_layout()
        
        return plt, [base_case_losses, dist_pv_losses, high_pv_losses]


def main():
    # Initialize the distribution grid analysis with the IEEE 34-bus test system
    # The file path is relative to the current working directory
    dss_file_path = "paste.txt"  # Adjust this path as needed
    
    grid = DistributionGridAnalysis(dss_file_path)
    
    # 1. Plot the network highlighting critical loads
    # First get the buses where critical loads are connected
    critical_load_buses = []
    for load_name in grid.critical_loads:
        # Get the bus for each critical load
        dss.Circuit.SetActiveElement(f"Load.{load_name}")
        bus_name = dss.CktElement.BusNames()[0].split('.')[0]  # Get base bus name
        critical_load_buses.append(bus_name)

    # Now plot with the proper bus names
    plt_network = grid.plot_network(highlight_buses=critical_load_buses, 
                            title="IEEE 34-Bus System with Critical Loads Highlighted")
    plt_network.savefig("network_critical_loads.png", dpi=300, bbox_inches='tight')
    plt_network.close()
    
    # 2. Create and analyze PV scenarios
    grid.create_pv_scenarios()
    
    # Plot voltage profiles for different scenarios
    plt_base = grid.plot_voltage_profile()
    plt_base.savefig("voltage_profile_base.png", dpi=300, bbox_inches='tight')
    plt_base.close()
    
    plt_dist = grid.plot_voltage_profile("distributed")
    plt_dist.savefig("voltage_profile_distributed_pv.png", dpi=300, bbox_inches='tight')
    plt_dist.close()
    
    plt_high = grid.plot_voltage_profile("high_penetration")
    plt_high.savefig("voltage_profile_high_pv.png", dpi=300, bbox_inches='tight')
    plt_high.close()
    
    # 3. Analyze load dispatch with PV
    plt_losses, losses = grid.analyze_load_dispatch()
    plt_losses.savefig("system_losses_comparison.png", dpi=300, bbox_inches='tight')
    plt_losses.close()
    
    print(f"System Losses Comparison:")
    print(f"Base Case: {losses[0]:.2f} kW")
    print(f"Distributed PV: {losses[1]:.2f} kW ({(losses[0]-losses[1])/losses[0]*100:.2f}% reduction)")
    print(f"High PV Penetration: {losses[2]:.2f} kW ({(losses[0]-losses[2])/losses[0]*100:.2f}% reduction)")
    
    # 4. Simulate black start
    black_start_pv_buses = ["832", "848", "860", "890"]
    black_start_results = grid.simulate_black_start(black_start_pv_buses)
    
    print("\nBlack Start Simulation Results:")
    print(f"Converged: {black_start_results['converged']}")
    
    # Plot network with black start PV locations
    plt_bs = grid.plot_network(highlight_buses=black_start_pv_buses, 
                         title="Black Start Scenario with PV+Storage Locations")
    plt_bs.savefig("black_start_scenario.png", dpi=300, bbox_inches='tight')
    plt_bs.close()
    
    # 5. Simulate load restoration
    pv_locations = ["832", "848", "860", "890"]
    critical_results, restoration_steps = grid.simulate_load_restoration(pv_locations)
    
    print("\nLoad Restoration Simulation:")
    print(f"Critical Loads Restoration Converged: {critical_results['converged']}")
    
    for step in restoration_steps:
        print(f"Step {step['step']}: Added {step['loads_added']} loads, Converged: {step['converged']}")
    
    # Create a comprehensive report on the restoration process
    # Plot voltage changes during restoration
    plt.figure(figsize=(12, 6))
    
    x_labels = ["Critical Only"] + [f"Step {step['step']}" for step in restoration_steps]
    # Calculate average voltages for each step
    avg_voltages = []
    
    # Critical loads only
    sum_v = 0
    count = 0
    for bus, v_data in critical_results["voltages"].items():
        if len(v_data) >= 2 and not bus.startswith("source"):
            for i in range(0, len(v_data), 2):  # Only voltage magnitudes
                sum_v += v_data[i]
                count += 1
    avg_voltages.append(sum_v / count if count > 0 else 0)
    
    # Each restoration step
    for step in restoration_steps:
        sum_v = 0
        count = 0
        for bus, v_data in step["voltages"].items():
            if len(v_data) >= 2 and not bus.startswith("source"):
                for i in range(0, len(v_data), 2):  # Only voltage magnitudes
                    sum_v += v_data[i]
                    count += 1
        avg_voltages.append(sum_v / count if count > 0 else 0)
    
    # Convert to per unit (approximately)
    avg_voltages = [v / 14376 for v in avg_voltages]  # Using 14.376kV as base
    
    # Create a color gradient
    colors = []
    for v in avg_voltages:
        if v < 0.95:
            colors.append('red')
        elif v > 1.05:
            colors.append('orange')
        else:
            colors.append('green')
    
    plt.bar(x_labels, avg_voltages, color=colors)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=1.05, color='red', linestyle='--', alpha=0.5)
    
    plt.ylabel('Average Bus Voltage (p.u.)')
    plt.title('Voltage Profile During Load Restoration')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("restoration_voltage_profile.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()