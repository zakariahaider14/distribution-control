import opendssdirect as dss
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import os
import itertools
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

class PVOptimization:
    def __init__(self, dss_file_path):
        """Initialize the DSS engine and load the circuit"""
        # Store the absolute path to the DSS file
        self.dss_file_path = os.path.abspath(dss_file_path)
        
        # Create the line codes file if it doesn't exist
        line_codes_path = os.path.join(os.path.dirname(self.dss_file_path), "IEEELineCodes.dss")
        if not os.path.exists(line_codes_path):
            self.create_ieee_line_codes_file(line_codes_path)
        
        # Clear any existing circuits and compile the DSS file
        dss.Basic.ClearAll()
        dss.Text.Command(f"Compile '{self.dss_file_path}'")
        
        # Solve the initial power flow
        dss.Solution.Solve()
        
        # Get all buses for analysis
        buses = dss.Circuit.AllBusNames()
        self.all_buses = buses
        
        # Get a clean list of bus names without node IDs
        self.base_buses = list(set([bus.split('.')[0] for bus in self.all_buses]))
        
        # Define the critical loads (can be customized)
        self.critical_loads = ["S860", "S844", "S848", "S890"]
        
        # Storage for optimization results
        self.optimization_results = {}

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
            
    def find_optimal_pv_placement(self, num_pv_systems=3, pv_size=300):
        """Find optimal PV placement for black start and critical load restoration"""
        potential_locations = [
            # List of good candidate buses for PV systems (3-phase buses)
            "816", "824", "828", "830", "832", "834", "836", "840", 
            "842", "844", "846", "848", "858", "860", "890"
        ]
        
        # Dictionary to store results for different combinations
        placement_scores = {}
        
        # Try different combinations of PV system placements
        for combo in itertools.combinations(potential_locations, num_pv_systems):
            score = self.evaluate_pv_placement(combo, pv_size)
            placement_scores[combo] = score
        
        # Find the best placement
        best_placement = max(placement_scores.items(), key=lambda x: x[1]['total_score'])
        
        # Store the results
        self.optimization_results['best_placement'] = {
            'buses': best_placement[0],
            'score': best_placement[1]
        }
        
        return best_placement
    
    def evaluate_pv_placement(self, pv_buses, pv_size):
        """Evaluate a specific PV placement configuration for black start"""
        # Clear circuit and reload base case
        dss.Basic.ClearAll()
        dss.Text.Command(f"Compile '{self.dss_file_path}'")
        
        # Disable main source (simulate blackout)
        dss.Text.Command("Disable Transformer.SubXF")

        # Add PV systems with storage for black start
        for i, bus in enumerate(pv_buses):
            dss.Text.Command(f"New PVSystem.BS_PV_{i} phases=3 bus1={bus} kV=24.9 Pmpp={pv_size} kvar=100 pf=1.0 model=1 conn=wye")
            
            # Add storage for black start capability
            dss.Text.Command(f"New Storage.BESS_{i} phases=3 bus1={bus} Kw={pv_size} kWrated={pv_size*1.5} kWhrated={pv_size*5} %stored=80 %reserve=10 model=1 conn=wye kV=24.9 state=discharging")
        
        # Only enable critical loads
        dss.Circuit.SetActiveClass("Load")
        all_loads = dss.ActiveClass.AllNames()
        for load in all_loads:
            if load in self.critical_loads:
                dss.Text.Command(f"Load.{load}.enabled=True")
            else:
                dss.Text.Command(f"Load.{load}.enabled=False")
        
        # Try to solve the circuit
        dss.Text.Command("set maxiterations=100")
        dss.Solution.Solve()
        
        # Check if solution converged
        converged = dss.Solution.Converged()
        
        # Calculate voltage deviation from ideal (1.0 pu)
        voltage_deviation = 0
        count = 0
        
        if converged:
            for load in self.critical_loads:
                # Get the bus associated with this load
                if dss.Circuit.SetActiveElement(f"Load.{load}"):
                    load_bus = dss.CktElement.BusNames()[0].split('.')[0]
                    
                    # Get voltage at this bus
                    dss.Circuit.SetActiveBus(load_bus)
                    pu_voltages = dss.Bus.puVmagAngle()
                    
                    # Calculate voltage deviation for each phase
                    for i in range(0, len(pu_voltages), 2):  # Skip angle values
                        v_pu = pu_voltages[i]  # Already in per-unit
                        voltage_deviation += abs(v_pu - 1.0)
                        count += 1
            
            # Average voltage deviation
            if count > 0:
                voltage_deviation /= count
        else:
            # If not converged, set a high voltage deviation
            voltage_deviation = 1.0  # Maximum deviation
        
        # Calculate losses if converged
        if converged:
            losses = dss.Circuit.Losses()[0] / 1000  # Convert W to kW
        else:
            losses = float('inf')
        
        # Calculate total score - lower is better for voltage_deviation and losses
        # Higher is better for ability to meet critical loads (converged)
        score = {
            'converged': 1 if converged else 0,
            'voltage_deviation': voltage_deviation,
            'losses': losses,
            'total_score': (1 if converged else 0) * (1 - voltage_deviation) * (1000 / (losses + 1))
        }
        
        return score
        
    def analyze_power_flow_with_pv(self, pv_buses, pv_size=300):
        """Analyze power flow patterns with PV systems"""
        # Clear circuit and reload base case
        dss.Basic.ClearAll()
        dss.Text.Command(f"Compile '{self.dss_file_path}'")
        
        # Add PV systems 
        for i, bus in enumerate(pv_buses):
            dss.Text.Command(f"New PVSystem.PV_{i} phases=3 bus1={bus} kV=24.9 Pmpp={pv_size} pf=1.0 model=1 conn=wye") 
        
        # Solve the circuit
        dss.Solution.Solve()
        
        # Check if solution converged
        converged = dss.Solution.Converged()
        
        if not converged:
            return {'converged': False}
        
        # Get line flows
        dss.Circuit.SetActiveClass("Line")
        line_names = dss.ActiveClass.AllNames()
        
        line_flows = {}
        for line in line_names:
            dss.Circuit.SetActiveElement(f"Line.{line}")
            powers = dss.CktElement.Powers()
            
            # Calculate total power flow magnitude
            total_power = 0
            for i in range(0, len(powers), 2):
                p = powers[i]
                q = powers[i+1]
                s = np.sqrt(p**2 + q**2)
                total_power += s
            
            line_flows[line] = total_power
        
        # Get bus voltages
        bus_voltages = {}
        buses = dss.Circuit.AllBusNames()
        
        for bus in buses:
            dss.Circuit.SetActiveBus(bus)
            voltages = dss.Bus.puVmagAngle()
            
            # Only take magnitude values (already in per-unit)
            v_mags = voltages[0::2]  # Every other value starting from 0
            
            bus_voltages[bus] = v_mags
        
        return {
            'converged': converged,
            'line_flows': line_flows,
            'bus_voltages': bus_voltages
        }
    
    def plot_network_power_flow(self, pv_buses, pv_size=300, title="Power Flow with PV Integration"):
        """Plot the network with power flow visualization"""
        # First analyze the power flow
        flow_results = self.analyze_power_flow_with_pv(pv_buses, pv_size)
        
        if not flow_results['converged']:
            return None
        
        # Create a network graph
        G = nx.Graph()
        
        # Get all lines
        dss.Circuit.SetActiveClass("Line")
        line_names = dss.ActiveClass.AllNames()
        
        # Add nodes and edges to the graph
        for line in line_names:
            dss.Circuit.SetActiveElement(f"Line.{line}")
            buses = dss.CktElement.BusNames()
            
            # Extract bus names without node numbers
            bus1 = buses[0].split('.')[0]
            bus2 = buses[1].split('.')[0]
            
            G.add_node(bus1)
            G.add_node(bus2)
            
            # Add the edge with the power flow as an attribute
            power_flow = flow_results['line_flows'].get(line, 0)
            G.add_edge(bus1, bus2, weight=power_flow, name=line)
        
        # Create a simple layout based on bus numbers for visualization
        pos = {}
        for node in G.nodes():
            try:
                bus_num = int(''.join(filter(str.isdigit, node)))
                pos[node] = (bus_num % 10, bus_num // 10)
            except:
                pos[node] = (0, 0)
        
        # Draw the network
        plt.figure(figsize=(12, 8))
        
        # Normalize edge weights for width visualization
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        normalized_weights = [w / max_weight * 5 for w in edge_weights]
        
        # Draw edges with width proportional to power flow
        nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.7, 
                              edge_color='blue')
        
        # Draw regular nodes
        regular_nodes = [n for n in G.nodes() if n not in pv_buses]
        nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, node_size=100, 
                              node_color='blue', alpha=0.7)
        
        # Draw PV nodes
        nx.draw_networkx_nodes(G, pos, nodelist=pv_buses, node_size=200, 
                              node_color='green', alpha=0.9)
        
        # Draw critical load nodes
        critical_load_buses = []
        for load in self.critical_loads:
            if dss.Circuit.SetActiveElement(f"Load.{load}"):
                bus = dss.CktElement.BusNames()[0].split('.')[0]
                critical_load_buses.append(bus)
        
        nx.draw_networkx_nodes(G, pos, nodelist=critical_load_buses, node_size=200, 
                              node_color='red', alpha=0.9)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
        
        # Create a legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='PV Systems'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Critical Loads'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Other Buses'),
            Line2D([0], [0], color='blue', lw=2, label='Power Flow (Line Width = Magnitude)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        return plt
        
    def analyze_voltage_profile_with_optimal_pv(self, save_file=None):
        """Analyze voltage profile with optimal PV placement"""
        if 'best_placement' not in self.optimization_results:
            print("Run find_optimal_pv_placement first!")
            return None
        
        best_buses = self.optimization_results['best_placement']['buses']
        
        # Scenarios to compare
        scenarios = [
            {"name": "No PV (Blackout)", "pv_buses": [], "disable_source": True},
            {"name": "Optimal PV Placement", "pv_buses": best_buses, "disable_source": True},
            {"name": "Normal Operation", "pv_buses": [], "disable_source": False},
            {"name": "Normal + PV", "pv_buses": best_buses, "disable_source": False}
        ]
        
        scenario_results = {}
        
        # Analyze each scenario
        for scenario in scenarios:
            # Clear circuit and reload base case
            dss.Basic.ClearAll()
            dss.Text.Command(f"Compile '{self.dss_file_path}'")
            
            # Disable source if required
            if scenario["disable_source"]:
                dss.Text.Command("Disable Transformer.SubXF")
            
            # Add PV systems
            for i, bus in enumerate(scenario["pv_buses"]):
                # Create PV system with individual commands
                dss.Text.Command(f"New PVSystem.PV_{i} bus1={bus} phases=3 kV=24.9")
                dss.Text.Command(f"PVSystem.PV_{i}.irradiance=1.0")
                dss.Text.Command(f"PVSystem.PV_{i}.pmpp=300")
                dss.Text.Command(f"PVSystem.PV_{i}.pf=1.0")
                dss.Text.Command(f"PVSystem.PV_{i}.conn=wye")
                dss.Text.Command(f"PVSystem.PV_{i}.model=1")
                
                # Add storage if in blackout scenario
                if scenario["disable_source"]:
                    # Create storage with individual commands
                    dss.Text.Command(f"New Storage.BESS_{i} bus1={bus} phases=3 kV=24.9")
                    dss.Text.Command(f"Storage.BESS_{i}.kW=300")
                    dss.Text.Command(f"Storage.BESS_{i}.kWrated=500")
                    dss.Text.Command(f"Storage.BESS_{i}.kWhrated=1500")
                    dss.Text.Command(f"Storage.BESS_{i}.%stored=80")
                    dss.Text.Command(f"Storage.BESS_{i}.%reserve=10")
                    dss.Text.Command(f"Storage.BESS_{i}.model=1")
                    dss.Text.Command(f"Storage.BESS_{i}.conn=wye")
                    dss.Text.Command(f"Storage.BESS_{i}.state=discharging")
            
            # If in blackout scenario, only enable critical loads
            if scenario["disable_source"]:
                dss.Circuit.SetActiveClass("Load")
                all_loads = dss.ActiveClass.AllNames()
                for load in all_loads:
                    if load in self.critical_loads:
                        dss.Text.Command(f"Load.{load}.enabled=True")
                    else:
                        dss.Text.Command(f"Load.{load}.enabled=False")
            
            # Try to solve the circuit
            dss.Text.Command("set maxiterations=100")
            dss.Solution.Solve()
            
            # Calculate voltage profile
            bus_voltages = {}
            if dss.Solution.Converged():
                buses = dss.Circuit.AllBusNames()
                
                for bus in buses:
                    dss.Circuit.SetActiveBus(bus)
                    voltages = dss.Bus.puVmagAngle()
                    
                    # Only take magnitude values (already in per-unit)
                    v_mags = voltages[0::2]  # Every other value starting from 0
                    
                    base_bus = bus.split('.')[0]
                    if base_bus not in bus_voltages:
                        bus_voltages[base_bus] = v_mags
            
            scenario_results[scenario["name"]] = {
                "converged": dss.Solution.Converged(),
                "voltages": bus_voltages
            }
        
        # Plot comparison
        # Get a common set of buses across all scenarios
        common_buses = []
        for scenario_name, data in scenario_results.items():
            if data["converged"]:
                for bus in data["voltages"].keys():
                    if bus not in common_buses and not bus.startswith("source"):
                        common_buses.append(bus)
        
        # Sort buses by number for better visualization
        common_buses.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (scenario_name, data) in enumerate(scenario_results.items()):
            ax = axes[i]
            
            if not data["converged"]:
                ax.text(0.5, 0.5, f"Solution did not converge for\n{scenario_name}", 
                       ha='center', va='center', fontsize=12)
                ax.set_title(scenario_name)
                ax.set_axis_off()
                continue
            
            # Collect voltages for plotting
            x = []
            y_phase_a = []
            y_phase_b = []
            y_phase_c = []
            
            for j, bus in enumerate(common_buses):
                if bus in data["voltages"]:
                    x.append(j)
                    v_data = data["voltages"][bus]
                    
                    # Handle buses with different numbers of phases
                    if len(v_data) >= 1:
                        y_phase_a.append(v_data[0])
                    else:
                        y_phase_a.append(None)
                        
                    if len(v_data) >= 2:
                        y_phase_b.append(v_data[1])
                    else:
                        y_phase_b.append(None)
                        
                    if len(v_data) >= 3:
                        y_phase_c.append(v_data[2])
                    else:
                        y_phase_c.append(None)
            
            # Plot voltage profile
            ax.plot(x, y_phase_a, 'ro-', label='Phase A', markersize=3)
            ax.plot(x, y_phase_b, 'go-', label='Phase B', markersize=3)
            ax.plot(x, y_phase_c, 'bo-', label='Phase C', markersize=3)
            
            # Add reference lines
            ax.axhline(y=1.05, color='r', linestyle='--', alpha=0.3)
            ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.3)
            
            # Only add bus labels on the bottom plots
            if i >= 2:
                ax.set_xticks(x)
                ax.set_xticklabels(common_buses, rotation=90, fontsize=8)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            
            ax.set_ylabel('Voltage (p.u.)')
            ax.set_title(scenario_name)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
        
        return plt
    
    def evaluate_pv_sizing(self, pv_buses, pv_sizes):
        """Evaluate different PV system sizes for black start"""
        results = {}
        
        for size in pv_sizes:
            score = self.evaluate_pv_placement(pv_buses, size)
            results[size] = score
        
        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        sizes = list(results.keys())
        scores = [results[size]['total_score'] for size in sizes]
        converged = [results[size]['converged'] for size in sizes]
        
        # Plot total score
        ax1.plot(sizes, scores, 'bo-')
        ax1.set_xlabel('PV System Size (kW)')
        ax1.set_ylabel('Total Performance Score')
        ax1.set_title('PV System Sizing Performance')
        ax1.grid(True, alpha=0.3)
        
        # Plot voltage deviation
        v_deviation = [results[size]['voltage_deviation'] for size in sizes]
        losses = [results[size]['losses'] if results[size]['converged'] else 0 for size in sizes]
        
        ax2.bar(sizes, v_deviation, color='green', alpha=0.6, label='Voltage Deviation')
        ax2.set_xlabel('PV System Size (kW)')
        ax2.set_ylabel('Voltage Deviation (p.u.)')
        ax2.set_title('Voltage Quality vs. PV Size')
        
        # Create a secondary axis for losses
        ax3 = ax2.twinx()
        ax3.plot(sizes, losses, 'r-', label='System Losses')
        ax3.set_ylabel('System Losses (kW)')
        
        # Add legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        return plt, results


def main():
    """Main function to demonstrate PV optimization"""
    # Initialize with the IEEE 34-bus test system
    dss_file_path = "paste.txt"  # Adjust as needed
    
    pv_opt = PVOptimization(dss_file_path)
    
    # 1. Find optimal PV placement for black start
    print("Finding optimal PV placement for black start...")
    best_placement = pv_opt.find_optimal_pv_placement(num_pv_systems=3, pv_size=300)
    
    print(f"Best PV placement buses: {best_placement[0]}")
    print(f"Score: {best_placement[1]['total_score']:.2f}")
    print(f"Converged: {best_placement[1]['converged']}")
    
    # 2. Plot power flow with optimal PV placement
    print("Plotting power flow with optimal PV placement...")
    pf_plot = pv_opt.plot_network_power_flow(best_placement[0], pv_size=300, 
                                       title="Power Flow with Optimal PV Placement")
    if pf_plot:
        pf_plot.savefig("optimal_pv_power_flow.png", dpi=300, bbox_inches='tight')
        pf_plot.close()
    
    # 3. Analyze voltage profiles with different scenarios
    print("Analyzing voltage profiles...")
    v_plot = pv_opt.analyze_voltage_profile_with_optimal_pv(save_file="voltage_profile_comparison.png")
    if v_plot:
        v_plot.close()
    
    # 4. Evaluate different PV sizes
    print("Evaluating PV sizing...")
    pv_sizes = [100, 200, 300, 400, 500]
    sizing_plot, sizing_results = pv_opt.evaluate_pv_sizing(best_placement[0], pv_sizes)
    sizing_plot.savefig("pv_sizing_analysis.png", dpi=300, bbox_inches='tight')
    sizing_plot.close()
    
    # Find optimal size
    best_size = max(sizing_results.items(), key=lambda x: x[1]['total_score'] if x[1]['converged'] else 0)
    print(f"Optimal PV size: {best_size[0]} kW")
    
    # 5. Summary of results
    print("\nSummary of Black Start and Critical Load Restoration Analysis:")
    print(f"Optimal PV Placement: Buses {', '.join(best_placement[0])}")
    print(f"Optimal PV Size: {best_size[0]} kW per system")
    print(f"Total Required PV Capacity: {best_size[0] * len(best_placement[0])} kW")
    
    # 6. Save results to CSV file
    results_df = pd.DataFrame({
        'bus': list(best_placement[0]),
        'pv_size_kw': [best_size[0]] * len(best_placement[0]),
        'load_served': ['Critical'] * len(best_placement[0])
    })
    results_df.to_csv("optimal_pv_placement.csv", index=False)
    print("Results saved to 'optimal_pv_placement.csv'")


if __name__ == "__main__":
    main()