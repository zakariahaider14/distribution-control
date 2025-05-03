import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import opendssdirect as dss

class ComprehensiveAnalysis:
    def __init__(self, dss_file_path):
        """Initialize the comprehensive analysis environment"""
        # Store absolute path to the DSS file
        self.dss_file_path = os.path.abspath(dss_file_path)
        
        # Check if the file exists
        if not os.path.exists(self.dss_file_path):
            raise FileNotFoundError(f"DSS file not found: {self.dss_file_path}")
        
        # Create IEEE Line Codes file if needed
        self.create_ieee_line_codes_file()

        print(f"Trying to compile file: {os.path.abspath(dss_file_path)}")

        print(f"Current working directory: {os.getcwd()}")
        
        # Initialize OpenDSS
        dss.Basic.ClearAll()
        dss.Text.Command(f"Compile '{self.dss_file_path}'")  # Then compile the file


        
        # Check if circuit was successfully loaded
        if dss.Circuit.NumBuses() == 0:
            raise RuntimeError(f"Failed to load circuit from {self.dss_file_path}")
        
        # Create output directory for results
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Critical loads for black start and restoration
        self.critical_loads = ["S860", "S844", "S848", "S890"]
        
        # Import specific analysis modules
        from pv_grid_integration import DistributionGridAnalysis
        from pv_optimization import PVOptimization
        from load_dispatch_analysis import LoadDispatchAnalysis
        
        # Initialize analysis modules
        self.grid_analysis = DistributionGridAnalysis(self.dss_file_path)
        self.pv_optimization = PVOptimization(self.dss_file_path)
        self.load_dispatch = LoadDispatchAnalysis(self.dss_file_path)
        
        print(f"Comprehensive analysis initialized for {os.path.basename(self.dss_file_path)}")
        print(f"Circuit has {dss.Circuit.NumBuses()} buses and {dss.Circuit.NumCktElements()} elements")
    
    def create_ieee_line_codes_file(self):
        """Create the IEEE line codes file if it doesn't exist"""
        line_codes_path = os.path.join(os.path.dirname(self.dss_file_path), "IEEELineCodes.dss")
        if not os.path.exists(line_codes_path):
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
            
            with open(line_codes_path, 'w') as f:
                f.write(ieee_line_codes)
            
            print(f"Created IEEE line codes file: {line_codes_path}")
    
    def map_loads_to_buses(self):
        """Map critical loads to their respective buses"""
        critical_load_buses = []
        
        for load_name in self.critical_loads:
            if dss.Circuit.SetActiveElement(f"Load.{load_name}"):
                bus_name = dss.CktElement.BusNames()[0].split('.')[0]  # Get base bus name
                critical_load_buses.append(bus_name)
                print(f"Load {load_name} is connected to bus {bus_name}")
            else:
                print(f"Warning: Critical load {load_name} not found in circuit")
        
        return critical_load_buses
    
    def run_base_case_analysis(self):
        """Run base case analysis of the distribution grid"""
        print("\n=== Running Base Case Analysis ===")
        
        # Map loads to buses for plotting
        critical_load_buses = self.map_loads_to_buses()
        
        # Plot network with critical loads highlighted
        plt_network = self.grid_analysis.plot_network(
            highlight_buses=critical_load_buses,
            title="IEEE 34-Bus System with Critical Loads Highlighted"
        )
        plt_network.savefig(os.path.join(self.output_dir, "network_critical_loads.png"), 
                       dpi=300, bbox_inches='tight')
        plt_network.close()
        
        # Get base case voltage profile
        plt_base = self.grid_analysis.plot_voltage_profile()
        plt_base.savefig(os.path.join(self.output_dir, "base_voltage_profile.png"),
                    dpi=300, bbox_inches='tight')
        plt_base.close()
        
        print("Base case analysis completed")
        print(f"Results saved to {self.output_dir}")
    
    def run_pv_optimization(self):
        """Run PV placement optimization for black start"""
        print("\n=== Running PV Placement Optimization ===")
        
        # Find optimal PV placement
        best_placement = self.pv_optimization.find_optimal_pv_placement(num_pv_systems=3, pv_size=300)
        
        print(f"Optimal PV placement found: {', '.join(best_placement[0])}")
        print(f"Score: {best_placement[1]['total_score']:.2f}")
        print(f"Converged: {best_placement[1]['converged']}")
        
        # Plot power flow with optimal PV
        pf_plot = self.pv_optimization.plot_network_power_flow(
            best_placement[0], pv_size=300,
            title="Power Flow with Optimal PV Placement"
        )
        if pf_plot:
            pf_plot.savefig(os.path.join(self.output_dir, "optimal_pv_power_flow.png"), 
                       dpi=300, bbox_inches='tight')
            pf_plot.close()
        
        # Analyze voltage profiles
        v_plot = self.pv_optimization.analyze_voltage_profile_with_optimal_pv(
            save_file=os.path.join(self.output_dir, "voltage_profile_comparison.png")
        )
        if v_plot:
            v_plot.close()
        
        # Evaluate different PV sizes
        pv_sizes = [100, 200, 300, 400, 500]
        sizing_plot, sizing_results = self.pv_optimization.evaluate_pv_sizing(best_placement[0], pv_sizes)
        sizing_plot.savefig(os.path.join(self.output_dir, "pv_sizing_analysis.png"), 
                       dpi=300, bbox_inches='tight')
        sizing_plot.close()
        
        # Find optimal size
        best_size = max(sizing_results.items(), 
                        key=lambda x: x[1]['total_score'] if x[1]['converged'] else 0)
        
        print(f"Optimal PV size: {best_size[0]} kW per system")
        print(f"Total required capacity: {best_size[0] * len(best_placement[0])} kW")
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'bus': list(best_placement[0]),
            'pv_size_kw': [best_size[0]] * len(best_placement[0]),
            'storage_size_kwh': [best_size[0] * 4] * len(best_placement[0]),  # 4 hours storage
            'purpose': ['Black Start & Critical Load Support'] * len(best_placement[0])
        })
        results_df.to_csv(os.path.join(self.output_dir, "optimal_pv_configuration.csv"), index=False)
        
        print(f"PV optimization results saved to {self.output_dir}")
        
        return {
            'optimal_buses': best_placement[0],
            'optimal_size': best_size[0]
        }
    
    def run_load_dispatch_analysis(self, pv_results):
        """Run load dispatch analysis with optimal PV configuration"""
        print("\n=== Running Load Dispatch Analysis ===")
        
        # Create PV and storage configurations based on optimization results
        optimal_buses = pv_results['optimal_buses']
        optimal_size = pv_results['optimal_size']
        
        pv_config = [
            {'bus': bus, 'kw': optimal_size} for bus in optimal_buses
        ]
        
        storage_config = [
            {'bus': bus, 'kw': optimal_size, 'kwh': optimal_size * 4,  # 4 hours of storage
             'discharge_trigger': 0.9, 'charge_trigger': 0.4} for bus in optimal_buses
        ]
        
        # Analyze dispatch scenarios
        dispatch_results = self.load_dispatch.create_dispatch_scenarios()
        
        print("System Losses by Scenario:")
        for scenario, losses in dispatch_results.items():
            if losses == float('inf'):
                print(f"  {scenario.replace('_', ' ').title()}: Did not converge")
            else:
                print(f"  {scenario.replace('_', ' ').title()}: {losses:.2f} kW")
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        scenarios = list(dispatch_results.keys())
        losses = []
        
        for scenario in scenarios:
            if dispatch_results[scenario] == float('inf'):
                losses.append(0)  # For visualization
            else:
                losses.append(dispatch_results[scenario])
        
        colors = ['crimson', 'royalblue', 'forestgreen', 'darkorange']
        
        plt.bar(range(len(scenarios)), losses, color=colors)
        plt.xticks(range(len(scenarios)), [s.replace('_', ' ').title() for s in scenarios])
        plt.ylabel('System Losses (kW)')
        plt.title('Loss Comparison Across Different Scenarios')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "dispatch_scenario_losses.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyze daily profile
        print("Analyzing daily load profile with PV and storage...")
        daily_results = self.load_dispatch.analyze_daily_load_profile(pv_config, storage_config)
        
        # Plot daily profile
        daily_plot = self.load_dispatch.plot_daily_profile(daily_results)
        daily_plot.savefig(os.path.join(self.output_dir, "daily_power_profile.png"), 
                      dpi=300, bbox_inches='tight')
        daily_plot.close()
        
        # Economic analysis
        print("Conducting economic analysis...")
        economic_metrics = self.load_dispatch.calculate_economic_metrics(
            pv_config, storage_config, daily_results
        )
        
        # Print economic results
        print("\nEconomic Analysis Results:")
        print(f"Total PV Capacity: {economic_metrics['total_pv_kw']} kW")
        print(f"Total Storage Capacity: {economic_metrics['total_storage_kwh']} kWh")
        print(f"Total Capital Cost: ${economic_metrics['total_capital_cost']:,.2f}")
        print(f"Annual Energy Savings: ${economic_metrics['annual_energy_savings']:,.2f}/year")
        print(f"Annual Resilience Benefit: ${economic_metrics['annual_resilience_benefit']:,.2f}/year")
        print(f"Net Present Value: ${economic_metrics['total_npv']:,.2f}")
        print(f"Simple Payback Period: {economic_metrics['simple_payback']:.2f} years")
        print(f"PV LCOE: ${economic_metrics['pv_lcoe']:.4f}/kWh")
        
        # Plot economic results
        econ_plot = self.load_dispatch.plot_economic_analysis(economic_metrics)
        econ_plot.savefig(os.path.join(self.output_dir, "economic_analysis.png"), 
                     dpi=300, bbox_inches='tight')
        econ_plot.close()
        
        print(f"Load dispatch analysis results saved to {self.output_dir}")
        
        return {
            'dispatch_results': dispatch_results,
            'daily_results': daily_results,
            'economic_metrics': economic_metrics
        }
    
    def generate_summary_report(self, pv_results, dispatch_results):
        """Generate a comprehensive summary report"""
        print("\n=== Generating Summary Report ===")
        
        # Create a summary DataFrame
        summary_data = {
            'Metric': [
                'Optimal PV Locations',
                'Optimal PV Size per Location',
                'Total PV Capacity',
                'Storage Capacity',
                'System Losses Reduction',
                'Black Start Capability',
                'Total Capital Cost',
                'Annual Energy Savings',
                'Annual Resilience Benefit',
                'Net Present Value (NPV)',
                'Simple Payback Period',
                'PV Levelized Cost of Energy (LCOE)'
            ],
            'Value': [
                ', '.join(pv_results['optimal_buses']),
                f"{pv_results['optimal_size']} kW",
                f"{pv_results['optimal_size'] * len(pv_results['optimal_buses'])} kW",
                f"{pv_results['optimal_size'] * len(pv_results['optimal_buses']) * 4} kWh",  # 4h storage
                f"{(1 - dispatch_results['economic_metrics']['pv_lcoe'] / 0.12) * 100:.1f}%",
                "Yes" if dispatch_results['dispatch_results']['black_start'] != float('inf') else "No",
                f"${dispatch_results['economic_metrics']['total_capital_cost']:,.2f}",
                f"${dispatch_results['economic_metrics']['annual_energy_savings']:,.2f}/year",
                f"${dispatch_results['economic_metrics']['annual_resilience_benefit']:,.2f}/year",
                f"${dispatch_results['economic_metrics']['total_npv']:,.2f}",
                f"{dispatch_results['economic_metrics']['simple_payback']:.2f} years",
                f"${dispatch_results['economic_metrics']['pv_lcoe']:.4f}/kWh"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.output_dir, "analysis_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        # Generate a text report
        report_file = os.path.join(self.output_dir, "comprehensive_report.txt")
        with open(report_file, 'w') as f:
            f.write("=============================================================\n")
            f.write("      COMPREHENSIVE PV INTEGRATION ANALYSIS REPORT\n")
            f.write("=============================================================\n\n")
            
            f.write("SYSTEM OVERVIEW:\n")
            f.write(f"IEEE 34-Bus Test System\n")
            f.write(f"Total Buses: {dss.Circuit.NumBuses()}\n")
            f.write(f"Total Circuit Elements: {dss.Circuit.NumCktElements()}\n")
            f.write(f"Critical Loads: {', '.join(self.critical_loads)}\n\n")
            
            f.write("PV INTEGRATION OPTIMIZATION RESULTS:\n")
            f.write(f"Optimal PV Locations: {', '.join(pv_results['optimal_buses'])}\n")
            f.write(f"Optimal PV Size: {pv_results['optimal_size']} kW per location\n")
            f.write(f"Total PV Capacity: {pv_results['optimal_size'] * len(pv_results['optimal_buses'])} kW\n")
            f.write(f"Storage Capacity: {pv_results['optimal_size'] * len(pv_results['optimal_buses']) * 4} kWh\n\n")
            
            f.write("SYSTEM PERFORMANCE RESULTS:\n")
            f.write("Losses by Scenario:\n")
            for scenario, losses in dispatch_results['dispatch_results'].items():
                if losses == float('inf'):
                    f.write(f"  {scenario.replace('_', ' ').title()}: Did not converge\n")
                else:
                    f.write(f"  {scenario.replace('_', ' ').title()}: {losses:.2f} kW\n")
            
            f.write("\nECONOMIC ANALYSIS RESULTS:\n")
            f.write(f"Total Capital Cost: ${dispatch_results['economic_metrics']['total_capital_cost']:,.2f}\n")
            f.write(f"Annual Energy Savings: ${dispatch_results['economic_metrics']['annual_energy_savings']:,.2f}/year\n")
            f.write(f"Annual Resilience Benefit: ${dispatch_results['economic_metrics']['annual_resilience_benefit']:,.2f}/year\n")
            f.write(f"Net Present Value: ${dispatch_results['economic_metrics']['total_npv']:,.2f}\n")
            f.write(f"Simple Payback Period: {dispatch_results['economic_metrics']['simple_payback']:.2f} years\n")
            f.write(f"PV LCOE: ${dispatch_results['economic_metrics']['pv_lcoe']:.4f}/kWh\n\n")
            
            f.write("CONCLUSION:\n")
            if dispatch_results['economic_metrics']['total_npv'] > 0:
                f.write("The proposed PV and storage integration is economically viable and provides ")
                f.write("significant benefits in terms of loss reduction, energy savings, and resilience.\n")
                
                if dispatch_results['dispatch_results']['black_start'] != float('inf'):
                    f.write("The system is capable of black start operation during grid outages, ")
                    f.write("providing power to critical loads and enhancing system resilience.\n")
                else:
                    f.write("The system could not achieve black start capability with the current configuration. ")
                    f.write("Additional PV and storage capacity may be required.\n")
            else:
                f.write("The proposed PV and storage integration is not economically viable with the current parameters. ")
                f.write("Consider revising the economic parameters or exploring alternative configurations.\n")
        
        print(f"Summary report saved to {report_file}")
        print(f"Summary data saved to {summary_file}")
    
    def run_comprehensive_analysis(self):
        """Run all analyses in sequence"""
        print("\n=== Starting Comprehensive Analysis ===")
        
        # 1. Run base case analysis
        self.run_base_case_analysis()
        
        # 2. Run PV optimization
        pv_results = self.run_pv_optimization()
        
        # 3. Run load dispatch analysis
        dispatch_results = self.run_load_dispatch_analysis(pv_results)
        
        # 4. Generate summary report
        self.generate_summary_report(pv_results, dispatch_results)
        
        print("\n=== Comprehensive Analysis Complete ===")
        print(f"All results saved to {os.path.abspath(self.output_dir)}")


import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Add this to the PVOptimization class
def create_pyomo_optimization_model(self, num_pv_systems, pv_size):
    """Create a Pyomo optimization model for PV placement"""
    # Initialize OpenDSS for baseline metrics
    dss.Basic.ClearAll()
    dss.Text.Command(f"Compile '{self.dss_file_path}'")
    
    # Get all potential buses for PV placement
    all_buses = []
    dss.Circuit.SetActiveBus("")
    for i in range(dss.Circuit.NumBuses()):
        bus_name = dss.Circuit.NextBus()
        # Filter for suitable buses (voltage level, etc.)
        if dss.Bus.kVBase() >= 0.4 and dss.Bus.NumNodes() >= 3:
            all_buses.append(bus_name)
    
    # Create Pyomo model
    model = pyo.ConcreteModel()
    
    # Decision variables - binary variables for PV placement
    model.pv_placement = pyo.Var(all_buses, domain=pyo.Binary)
    
    # Constraint - limit number of PV systems
    model.num_pv_constraint = pyo.Constraint(
        expr=sum(model.pv_placement[bus] for bus in all_buses) == num_pv_systems
    )
    
    # Add minimum distance constraint between PV systems
    # This would require bus coordinate information
    # model.distance_constraints = ...
    
    # Objective function parameters based on OpenDSS simulations
    # These would be calculated for each potential bus
    voltage_improvement = self._calculate_voltage_improvements(all_buses, pv_size)
    loss_reduction = self._calculate_loss_reductions(all_buses, pv_size)
    critical_load_support = self._calculate_critical_load_support(all_buses)
    
    # Objective function - maximize combined benefits
    def objective_rule(model):
        return sum(
            model.pv_placement[bus] * (
                0.4 * voltage_improvement[bus] +
                0.3 * loss_reduction[bus] +
                0.3 * critical_load_support[bus]
            ) for bus in all_buses
        )
    
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    return model, all_buses

def _calculate_voltage_improvements(self, buses, pv_size):
    """Calculate voltage improvement metric for each potential bus"""
    improvements = {}
    
    # Get base case voltage profile
    dss.Basic.ClearAll()
    dss.Text.Command(f"Compile '{self.dss_file_path}'")
    dss.Solution.Solve()
    base_voltages = self._get_voltage_profile()
    
    # Calculate improvement for each bus
    for bus in buses:
        # Add PV to this bus
        dss.Basic.ClearAll()
        dss.Text.Command(f"Compile '{self.dss_file_path}'")
        self._add_pv_to_bus(bus, pv_size)
        
        # Solve and get new voltage profile
        try:
            dss.Solution.Solve()
            if dss.Solution.Converged():
                new_voltages = self._get_voltage_profile()
                
                # Calculate improvement metric
                v_improvement = 0
                for b in new_voltages:
                    v_base = base_voltages.get(b, 1.0)
                    v_new = new_voltages[b]
                    
                    # Penalize voltage outside 0.95-1.05 range
                    if v_base < 0.95:
                        v_improvement += min(0, v_new - v_base) * 10
                    elif v_base > 1.05:
                        v_improvement += min(0, v_base - v_new) * 10
                    else:
                        # Encourage stable voltages
                        v_improvement += (1 - abs(1.0 - v_new)) * 2
                
                improvements[bus] = v_improvement
            else:
                improvements[bus] = -100  # Penalize non-convergence
        except:
            improvements[bus] = -100  # Penalize errors
    
    # Normalize improvements
    max_imp = max(improvements.values()) if improvements else 1
    min_imp = min(improvements.values()) if improvements else 0
    range_imp = max_imp - min_imp if max_imp > min_imp else 1
    
    normalized_improvements = {
        bus: (improvements[bus] - min_imp) / range_imp 
        for bus in improvements
    }
    
    return normalized_improvements

def solve_pyomo_model(self, model, all_buses, num_pv_systems, pv_size):
    """Solve the Pyomo optimization model"""
    # Select solver (GLPK, CBC, CPLEX, Gurobi, etc.)
    solver = SolverFactory('glpk')  # or 'cbc', 'cplex', 'gurobi'
    
    # Solve the model
    results = solver.solve(model, tee=True)
    
    # Check if solver found an optimal solution
    if results.solver.status == pyo.SolverStatus.ok and \
       results.solver.termination_condition == pyo.TerminationCondition.optimal:
        
        # Extract solution
        selected_buses = []
        for bus in all_buses:
            if pyo.value(model.pv_placement[bus]) > 0.5:  # Binary variable is 1
                selected_buses.append(bus)
        
        # Validate the solution with OpenDSS
        dss.Basic.ClearAll()
        dss.Text.Command(f"Compile '{self.dss_file_path}'")
        
        # Add all PV systems from solution
        for bus in selected_buses:
            self._add_pv_to_bus(bus, pv_size)
        
        # Check system performance
        dss.Solution.Solve()
        converged = dss.Solution.Converged()
        
        # Calculate final metrics
        total_score = 0
        if converged:
            voltage_profile = self._get_voltage_profile()
            losses = dss.Circuit.Losses()[0] / 1000  # Convert W to kW
            
            # Calculate score components
            voltage_score = self._calculate_voltage_score(voltage_profile)
            loss_score = self._calculate_loss_score(losses)
            critical_load_score = self._calculate_critical_load_score(selected_buses)
            
            # Calculate total score
            total_score = 0.4 * voltage_score + 0.3 * loss_score + 0.3 * critical_load_score
        
        return selected_buses, {
            'converged': converged,
            'total_score': total_score
        }
    else:
        print(f"Solver status: {results.solver.status}")
        print(f"Termination condition: {results.solver.termination_condition}")
        return [], {'converged': False, 'total_score': 0}


def main():
    """Main function"""
    # Check command line arguments
    if len(sys.argv) > 1:
        dss_file_path = sys.argv[1]
    else:
        dss_file_path = "paste.txt"  # Default file name
    
    # Check if file exists
    if not os.path.exists(dss_file_path):
        print(f"Error: DSS file '{dss_file_path}' not found.")
        print("Please provide a valid path to the OpenDSS file.")
        return
    
    try:
        # Initialize comprehensive analysis
        analysis = ComprehensiveAnalysis(dss_file_path)
        
        # Run all analyses
        analysis.run_comprehensive_analysis()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()