import opendssdirect as dss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap

class LoadDispatchAnalysis:
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
        
        # Define critical loads BEFORE calling get_load_summary
        self.critical_loads = ["S860", "S844", "S848", "S890"]
        
        # THEN get the load summary
        self.base_loads = self.get_load_summary()
        self.base_system_losses = dss.Circuit.Losses()[0] / 1000  # kW
        
        # Economic parameters
        self.parameters = {
            'pv_cost_per_kw': 1500,  # $ per kW of PV installed
            'storage_cost_per_kwh': 400,  # $ per kWh of storage
            'electricity_cost': 0.12,  # $ per kWh
            'pv_life_years': 25,
            'storage_life_years': 15,
            'discount_rate': 0.05,  # 5%
            'pv_degradation': 0.005,  # 0.5% per year
            'blackout_hours_per_year': 8,  # Average hours of blackout per year
            'value_of_lost_load': 10.0,  # $ per kWh not served during blackout
        }

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
    
    def get_load_summary(self):
        """Get a summary of all loads in the system"""
        dss.Circuit.SetActiveClass("Load")
        load_names = dss.ActiveClass.AllNames()
        
        loads_summary = []
        for load in load_names:
            dss.Circuit.SetActiveElement(f"Load.{load}")
            kw = dss.Properties.Value("kW")
            kvar = dss.Properties.Value("kvar")
            bus = dss.CktElement.BusNames()[0]
            is_critical = load in self.critical_loads
            
            loads_summary.append({
                'name': load,
                'kw': float(kw),
                'kvar': float(kvar),
                'bus': bus,
                'is_critical': is_critical
            })
        
        return loads_summary
    
    def add_pv_systems(self, pv_config):
        """Add PV systems based on configuration"""
        # pv_config is a list of dicts with 'bus', 'kw', etc.
        for i, config in enumerate(pv_config):
            # Create base PV system without optional parameters
            dss.Text.Command(f"New PVSystem.PV_{i} phases=3 bus1={config['bus']} kV=24.9 Pmpp={config['kw']} pf={config.get('pf', 1.0)} model=1 conn=wye")
            
            # Add optional parameters only if they exist and are not empty
            if config.get('daily', '') != '':
                dss.Text.Command(f"PVSystem.PV_{i}.daily={config['daily']}")
            
            if config.get('yearly', '') != '':
                dss.Text.Command(f"PVSystem.PV_{i}.yearly={config['yearly']}")
    
    def add_storage_systems(self, storage_config):
        """Add storage systems based on configuration"""
        # storage_config is a list of dicts with 'bus', 'kw', 'kwh', etc.
        for i, config in enumerate(storage_config):
            # Create base storage system
            dss.Text.Command(f"New Storage.BESS_{i} phases=3 bus1={config['bus']} kW={config['kw']} kWrated={config['kw']} kWhrated={config['kwh']} %stored={config.get('initial_soc', 80)} %reserve={config.get('reserve', 20)} model=1 conn=wye kV=24.9 state={config.get('state', 'idling')}")
            
            # Add optional parameters only if they exist and are not empty
            if config.get('daily', '') != '':
                dss.Text.Command(f"Storage.BESS_{i}.daily={config['daily']}")
            
            if config.get('yearly', '') != '':
                dss.Text.Command(f"Storage.BESS_{i}.yearly={config['yearly']}")

    def create_dispatch_scenarios(self):
        """Create and analyze different dispatch scenarios"""
        # 1. Base case - no PV or storage
        dss.Basic.ClearAll()
        dss.Text.Command(f"Compile '{self.dss_file_path}'")
        dss.Solution.Solve()
        base_losses = dss.Circuit.Losses()[0] / 1000  # kW
        
        # 2. PV only at optimal locations (from previous analysis)
        dss.Basic.ClearAll()
        dss.Text.Command(f"Compile '{self.dss_file_path}'")
        pv_config = [
            {'bus': '832', 'kw': 300},
            {'bus': '848', 'kw': 300},
            {'bus': '860', 'kw': 300}
        ]
        self.add_pv_systems(pv_config)
        dss.Solution.Solve()
        pv_only_losses = dss.Circuit.Losses()[0] / 1000  # kW
        
        # 3. PV with storage dispatch
        dss.Basic.ClearAll()
        dss.Text.Command(f"Compile '{self.dss_file_path}'")
        self.add_pv_systems(pv_config)
        storage_config = [
            {'bus': '832', 'kw': 300, 'kwh': 1200, 'state': 'discharging'},
            {'bus': '848', 'kw': 300, 'kwh': 1200, 'state': 'discharging'},
            {'bus': '860', 'kw': 300, 'kwh': 1200, 'state': 'discharging'}
        ]
        self.add_storage_systems(storage_config)
        dss.Solution.Solve()
        pv_storage_losses = dss.Circuit.Losses()[0] / 1000  # kW
        
        # 4. Black start (only critical loads, no grid, PV+storage)
        dss.Basic.ClearAll()
        dss.Text.Command(f"Compile '{self.dss_file_path}'")
        dss.Text.Command("Disable Transformer.SubXF")  # Disable main source
        
        # Only enable critical loads
        dss.Circuit.SetActiveClass("Load")
        all_loads = dss.ActiveClass.AllNames()
        for load in all_loads:
            if load in self.critical_loads:
                dss.Text.Command(f"Load.{load}.enabled=True")
            else:
                dss.Text.Command(f"Load.{load}.enabled=False")
        
        self.add_pv_systems(pv_config)
        self.add_storage_systems(storage_config)
        dss.Solution.Solve()
        black_start_losses = float('inf')
        if dss.Solution.Converged():
            black_start_losses = dss.Circuit.Losses()[0] / 1000  # kW
        
        return {
            'base_case': base_losses,
            'pv_only': pv_only_losses,
            'pv_with_storage': pv_storage_losses,
            'black_start': black_start_losses
        }
    
    def analyze_daily_load_profile(self, pv_config, storage_config=None, time_steps=24):
        """Analyze daily load profile with PV and storage integration"""
        # Create a daily load shape
        daily_load_shape = [0.6, 0.55, 0.5, 0.5, 0.52, 0.57, 0.6, 0.68, 0.75, 0.84, 0.9, 
                            0.94, 0.96, 0.95, 0.93, 0.94, 0.98, 1.0, 0.97, 0.92, 0.85, 
                            0.8, 0.7, 0.65]
        
        # Create a PV generation shape
        pv_shape = [0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 0.98, 0.9, 
                    0.8, 0.65, 0.4, 0.2, 0.05, 0, 0, 0, 0]
        
        # Use the simplest approach - skip time-series and solve a single case
        # Set up simulation
        dss.Basic.ClearAll()
        dss.Text.Command(f"Compile '{self.dss_file_path}'")
        
        # Add PV systems without daily shapes
        for i, config in enumerate(pv_config):
            dss.Text.Command(f"New PVSystem.PV_{i} phases=3 bus1={config['bus']} kV=24.9 Pmpp={config['kw']} pf={config.get('pf', 1.0)} model=1 conn=wye")
        
        # Add storage if provided
        if storage_config:
            for i, config in enumerate(storage_config):
                dss.Text.Command(f"New Storage.BESS_{i} phases=3 bus1={config['bus']} kV=24.9")
                dss.Text.Command(f"Storage.BESS_{i}.kW={config['kw']}")
                dss.Text.Command(f"Storage.BESS_{i}.kWrated={config['kw']}")
                dss.Text.Command(f"Storage.BESS_{i}.kWhrated={config['kwh']}")
                dss.Text.Command(f"Storage.BESS_{i}.%stored=70")
                dss.Text.Command(f"Storage.BESS_{i}.%reserve=20")
                dss.Text.Command(f"Storage.BESS_{i}.model=1")
                dss.Text.Command(f"Storage.BESS_{i}.conn=wye")
        
        # Solve a single case
        dss.Solution.Solve()
        
        # Create synthetic time-series results using the load and PV shapes
        results = {
            'time': list(range(1, time_steps + 1)),
            'total_load': [],
            'pv_generation': [],
            'storage_power': [],
            'grid_power': [],
            'losses': []
        }
        
        # Get the base values from a single solution
        dss.Circuit.SetActiveClass("Load")
        all_loads = dss.ActiveClass.AllNames()
        
        base_load = 0
        for load in all_loads:
            dss.Circuit.SetActiveElement(f"Load.{load}")
            load_kw = float(dss.Properties.Value("kW"))
            base_load += load_kw
        
        base_pv_gen = 0
        for i in range(len(pv_config)):
            if dss.Circuit.SetActiveElement(f"PVSystem.PV_{i}"):
                powers = dss.CktElement.Powers()
                # PV is a negative load, so powers are negative when generating
                base_pv_gen -= sum(powers[0::2])  # Sum all phases
        
        base_storage_power = 0
        if storage_config:
            for i in range(len(storage_config)):
                if dss.Circuit.SetActiveElement(f"Storage.BESS_{i}"):
                    powers = dss.CktElement.Powers()
                    # Negative when discharging, positive when charging
                    base_storage_power -= sum(powers[0::2])
        
        # Substation power (grid import/export)
        base_grid_power = 0
        if dss.Circuit.SetActiveElement("Transformer.SubXF"):
            powers = dss.CktElement.Powers()
            base_grid_power = sum(powers[0::2])  # Positive when importing, negative when exporting
        
        # System losses
        base_losses = dss.Circuit.Losses()[0] / 1000  # kW
        
        # Now create synthetic time-series results by scaling the base values
        for hour in range(time_steps):
            load_factor = daily_load_shape[hour]
            pv_factor = pv_shape[hour]
            
            results['total_load'].append(base_load * load_factor)
            results['pv_generation'].append(base_pv_gen * pv_factor)
            
            # Storage and grid power calculations would be more complex in reality
            # For simplicity, we'll keep storage constant and adjust grid to balance
            results['storage_power'].append(base_storage_power)
            
            # Grid power balances load, PV, and storage
            grid_power = results['total_load'][-1] - results['pv_generation'][-1] - results['storage_power'][-1]
            results['grid_power'].append(grid_power)
            
            # Losses are roughly proportional to the square of the current (power)
            power_ratio = abs(grid_power / base_grid_power) if base_grid_power != 0 else 1
            results['losses'].append(base_losses * power_ratio * power_ratio)
        
        return results
    def plot_daily_profile(self, results):
        """Plot daily power profile"""
        plt.figure(figsize=(12, 8))
        
        # Create stacked area plot
        plt.fill_between(results['time'], 0, results['total_load'], 
                        color='red', alpha=0.6, label='Load')
        
        plt.fill_between(results['time'], 0, -np.array(results['pv_generation']), 
                        color='green', alpha=0.6, label='PV Generation')
        
        plt.fill_between(results['time'], 0, -np.array(results['storage_power']), 
                        color='blue', alpha=0.6, label='Storage (discharge/charge)')
        
        plt.plot(results['time'], results['grid_power'], 'k--', 
                label='Grid Import/Export')
        
        plt.plot(results['time'], results['losses'], 'm-', 
                label='System Losses')
        
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        plt.xlabel('Hour of Day')
        plt.ylabel('Power (kW)')
        plt.title('Daily Power Profile with PV and Storage')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.xticks(range(1, 25))
        
        return plt
    
    def calculate_economic_metrics(self, pv_config, storage_config, daily_results):
        """Calculate economic metrics for the PV and storage investment"""
        # Calculate total PV and storage capacity
        total_pv_kw = sum(config['kw'] for config in pv_config)
        total_storage_kwh = sum(config['kwh'] for config in storage_config)
        
        # Calculate capital costs
        pv_capital_cost = total_pv_kw * self.parameters['pv_cost_per_kw']
        storage_capital_cost = total_storage_kwh * self.parameters['storage_cost_per_kwh']
        total_capital_cost = pv_capital_cost + storage_capital_cost
        
        # Calculate annual energy savings
        daily_pv_energy = sum(daily_results['pv_generation'])  # kWh for a day
        annual_pv_energy = daily_pv_energy * 365  # Simplified
        annual_energy_savings = annual_pv_energy * self.parameters['electricity_cost']
        
        # Calculate resilience benefit (value of lost load during outages)
        critical_load_energy = 0
        for load in self.base_loads:
            if load['is_critical']:
                critical_load_energy += load['kw']  # kW
        
        # Assuming critical loads are served during blackout hours
        annual_resilience_benefit = (critical_load_energy * 
                                    self.parameters['blackout_hours_per_year'] * 
                                    self.parameters['value_of_lost_load'])
        
        # Calculate Net Present Value (NPV)
        discount_rate = self.parameters['discount_rate']
        pv_cash_flows = []
        storage_cash_flows = []
        
        # Initial investment (year 0)
        pv_cash_flows.append(-pv_capital_cost)
        storage_cash_flows.append(-storage_capital_cost)
        
        # Annual benefits over project life
        for year in range(1, max(self.parameters['pv_life_years'], 
                                self.parameters['storage_life_years']) + 1):
            
            # PV degradation factor
            degradation_factor = (1 - self.parameters['pv_degradation'])**year
            
            # Annual PV energy savings (adjusted for degradation)
            annual_pv_benefit = annual_energy_savings * degradation_factor
            
            # Resilience benefit (assumed constant)
            year_resilience_benefit = annual_resilience_benefit
            
            # Allocate benefits proportionally to PV and storage
            pv_benefit_share = 0.7  # 70% of benefit attributed to PV
            storage_benefit_share = 0.3  # 30% to storage
            
            if year <= self.parameters['pv_life_years']:
                pv_cash_flows.append((annual_pv_benefit * pv_benefit_share + 
                                    year_resilience_benefit * pv_benefit_share))
            else:
                pv_cash_flows.append(0)
                
            if year <= self.parameters['storage_life_years']:
                storage_cash_flows.append((annual_pv_benefit * storage_benefit_share + 
                                        year_resilience_benefit * storage_benefit_share))
                
                # Storage replacement in year 15
                if year == self.parameters['storage_life_years'] and self.parameters['pv_life_years'] > year:
                    storage_cash_flows.append(-storage_capital_cost * 0.8)  # Assuming 20% cost reduction
            else:
                storage_cash_flows.append(0)
        
        # Calculate NPV
        pv_npv = sum(cf / (1 + discount_rate)**year for year, cf in enumerate(pv_cash_flows))
        storage_npv = sum(cf / (1 + discount_rate)**year for year, cf in enumerate(storage_cash_flows))
        total_npv = pv_npv + storage_npv
        
        # Calculate Payback Period (simplified)
        cumulative_cash_flow = -total_capital_cost
        annual_benefit = annual_pv_energy * self.parameters['electricity_cost'] + annual_resilience_benefit
        simple_payback = total_capital_cost / annual_benefit
        
        # Calculate Levelized Cost of Energy (LCOE)
        total_energy_over_life = 0
        for year in range(self.parameters['pv_life_years']):
            degradation_factor = (1 - self.parameters['pv_degradation'])**year
            total_energy_over_life += annual_pv_energy * degradation_factor
        
        pv_lcoe = pv_capital_cost / total_energy_over_life
        
        return {
            'total_pv_kw': total_pv_kw,
            'total_storage_kwh': total_storage_kwh,
            'pv_capital_cost': pv_capital_cost,
            'storage_capital_cost': storage_capital_cost,
            'total_capital_cost': total_capital_cost,
            'annual_pv_energy': annual_pv_energy,
            'annual_energy_savings': annual_energy_savings,
            'annual_resilience_benefit': annual_resilience_benefit,
            'pv_npv': pv_npv,
            'storage_npv': storage_npv,
            'total_npv': total_npv,
            'simple_payback': simple_payback,
            'pv_lcoe': pv_lcoe
        }
    
    def plot_economic_analysis(self, metrics):
        """Plot key economic metrics"""
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Capital cost breakdown
        cost_labels = ['PV System', 'Storage System']
        cost_values = [metrics['pv_capital_cost'], metrics['storage_capital_cost']]
        
        axes[0, 0].bar(cost_labels, cost_values, color=['green', 'blue'])
        axes[0, 0].set_title('Capital Cost Breakdown')
        axes[0, 0].set_ylabel('Cost ($)')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels on top of bars
        for i, v in enumerate(cost_values):
            axes[0, 0].text(i, v + 0.1, f'${v:,.0f}', ha='center')
        
        # 2. Annual benefits
        benefit_labels = ['Energy Savings', 'Resilience Benefit']
        benefit_values = [metrics['annual_energy_savings'], metrics['annual_resilience_benefit']]
        
        axes[0, 1].bar(benefit_labels, benefit_values, color=['orange', 'purple'])
        axes[0, 1].set_title('Annual Benefits')
        axes[0, 1].set_ylabel('Value ($/year)')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(benefit_values):
            axes[0, 1].text(i, v + 0.1, f'${v:,.0f}/year', ha='center')
        
        # 3. NPV breakdown
        npv_labels = ['PV System', 'Storage System', 'Total']
        npv_values = [metrics['pv_npv'], metrics['storage_npv'], metrics['total_npv']]
        npv_colors = ['green', 'blue', 'gray']
        
        axes[1, 0].bar(npv_labels, npv_values, color=npv_colors)
        axes[1, 0].set_title('Net Present Value (NPV)')
        axes[1, 0].set_ylabel('NPV ($)')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(npv_values):
            if v >= 0:
                axes[1, 0].text(i, v + 0.1, f'${v:,.0f}', ha='center')
            else:
                axes[1, 0].text(i, v - 0.1, f'${v:,.0f}', ha='center', va='top')
        
        # 4. Payback and LCOE
        metric_labels = ['Simple Payback', 'PV LCOE']
        metric_values = [metrics['simple_payback'], metrics['pv_lcoe']]
        metric_units = [' years', ' $/kWh']
        
        axes[1, 1].bar(metric_labels, metric_values, color=['red', 'teal'])
        axes[1, 1].set_title('Payback Period and LCOE')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(metric_values):
            axes[1, 1].text(i, v + 0.01, f'{v:.2f}{metric_units[i]}', ha='center')
        
        plt.tight_layout()
        return plt


def main():
    """Main function to demonstrate load dispatch and economic analysis"""
    # Initialize with the IEEE 34-bus test system
    dss_file_path = "paste.txt"  # Adjust as needed
    
    dispatch_analysis = LoadDispatchAnalysis(dss_file_path)
    
    # 1. Analyze dispatch scenarios
    print("Analyzing dispatch scenarios...")
    dispatch_results = dispatch_analysis.create_dispatch_scenarios()
    
    print("System Losses by Scenario:")
    for scenario, losses in dispatch_results.items():
        if losses == float('inf'):
            print(f"  {scenario.replace('_', ' ').title()}: Did not converge")
        else:
            print(f"  {scenario.replace('_', ' ').title()}: {losses:.2f} kW")
    
    # Plot loss comparison
    plt.figure(figsize=(10, 6))
    scenarios = list(dispatch_results.keys())
    losses = []
    
    for scenario in scenarios:
        if dispatch_results[scenario] == float('inf'):
            losses.append(0)  # For visualization purposes
        else:
            losses.append(dispatch_results[scenario])
    
    colors = ['crimson', 'royalblue', 'forestgreen', 'darkorange']
    
    plt.bar(range(len(scenarios)), losses, color=colors)
    plt.xticks(range(len(scenarios)), [s.replace('_', ' ').title() for s in scenarios])
    plt.ylabel('System Losses (kW)')
    plt.title('Loss Comparison Across Different Scenarios')
    plt.grid(axis='y', alpha=0.3)
    
    # Add convergence note for black start if needed
    if dispatch_results['black_start'] == float('inf'):
        plt.text(3, 5, "Did not converge", ha='center', va='bottom', 
                rotation=0, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("dispatch_scenario_losses.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Analyze daily profile with PV and storage
    print("\nAnalyzing daily load profile with PV and storage...")
    
    # Define PV and storage configurations
    pv_config = [
        {'bus': '832', 'kw': 300},
        {'bus': '848', 'kw': 300},
        {'bus': '860', 'kw': 300}
    ]
    
    storage_config = [
        {'bus': '832', 'kw': 200, 'kwh': 800, 'discharge_trigger': 0.9, 'charge_trigger': 0.4},
        {'bus': '848', 'kw': 200, 'kwh': 800, 'discharge_trigger': 0.85, 'charge_trigger': 0.4},
        {'bus': '860', 'kw': 200, 'kwh': 800, 'discharge_trigger': 0.95, 'charge_trigger': 0.4}
    ]
    
    # Analyze daily profile
    daily_results = dispatch_analysis.analyze_daily_load_profile(pv_config, storage_config)
    
    # Plot daily profile
    daily_plot = dispatch_analysis.plot_daily_profile(daily_results)
    daily_plot.savefig("daily_power_profile.png", dpi=300, bbox_inches='tight')
    daily_plot.close()
    
    # 3. Conduct economic analysis
    print("\nConducting economic analysis...")
    economic_metrics = dispatch_analysis.calculate_economic_metrics(pv_config, storage_config, daily_results)
    
    # Print key economic results
    print("\nEconomic Analysis Results:")
    print(f"Total PV Capacity: {economic_metrics['total_pv_kw']} kW")
    print(f"Total Storage Capacity: {economic_metrics['total_storage_kwh']} kWh")
    print(f"Total Capital Cost: ${economic_metrics['total_capital_cost']:,.2f}")
    print(f"Annual Energy Savings: ${economic_metrics['annual_energy_savings']:,.2f}/year")
    print(f"Annual Resilience Benefit: ${economic_metrics['annual_resilience_benefit']:,.2f}/year")
    print(f"Net Present Value: ${economic_metrics['total_npv']:,.2f}")
    print(f"Simple Payback Period: {economic_metrics['simple_payback']:.2f} years")
    print(f"PV Levelized Cost of Energy: ${economic_metrics['pv_lcoe']:.4f}/kWh")
    
    # Plot economic results
    econ_plot = dispatch_analysis.plot_economic_analysis(economic_metrics)
    econ_plot.savefig("economic_analysis.png", dpi=300, bbox_inches='tight')
    econ_plot.close()
    
    # 4. Create a summary table of all analyses
    summary_data = {
        'Scenario': ['Base Case', 'PV Only', 'PV with Storage', 'Black Start (Critical Loads)'],
        'System Losses (kW)': [
            dispatch_results['base_case'],
            dispatch_results['pv_only'],
            dispatch_results['pv_with_storage'],
            'N/A' if dispatch_results['black_start'] == float('inf') else dispatch_results['black_start']
        ],
        'Peak Grid Import (kW)': [
            max(daily_results['total_load']),
            max(np.array(daily_results['grid_power'])[np.array(daily_results['grid_power']) > 0]),
            max(np.array(daily_results['grid_power'])[np.array(daily_results['grid_power']) > 0]),
            0
        ],
        'PV Self-Consumption (%)': [
            0,
            100 * (1 - abs(min(0, min(daily_results['grid_power']))) / sum(daily_results['pv_generation'])),
            100 * (1 - abs(min(0, min(daily_results['grid_power']))) / sum(daily_results['pv_generation'])),
            100
        ]
    }
    
    # Create and save the summary table
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("dispatch_summary.csv", index=False)
    print("\nSummary saved to 'dispatch_summary.csv'")
    
    # Create a function to help analyze islanding scenarios for different critical load levels
    def analyze_islanding_capability(pv_kw, storage_kwh):
        critical_loads = [sum(load['kw']) for load in dispatch_analysis.base_loads if load['is_critical']]
        if not critical_loads:
            return 0  # or some default value
        min_hours = min((storage_kwh / (load_kw + 1e-8)) for load_kw in critical_loads)
        islanding_hours = min(min_hours, 24)
        return islanding_hours
    
    # Analyze different PV and storage capacities for islanding capability
    pv_sizes = [300, 600, 900, 1200]
    storage_sizes = [1200, 2400, 3600, 4800]
    
    islanding_hours = np.zeros((len(pv_sizes), len(storage_sizes)))
    for i, pv_kw in enumerate(pv_sizes):
        for j, storage_kwh in enumerate(storage_sizes):
            islanding_hours[i, j] = analyze_islanding_capability(pv_kw, storage_kwh)
    
    # Plot islanding capability
    plt.figure(figsize=(10, 8))
    
    # Create a heatmap
    plt.imshow(islanding_hours, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Islanding Duration (hours)')
    
    # Set x and y ticks
    plt.xticks(np.arange(len(storage_sizes)), [f"{s} kWh" for s in storage_sizes])
    plt.yticks(np.arange(len(pv_sizes)), [f"{s} kW" for s in pv_sizes])
    
    plt.xlabel('Storage Capacity')
    plt.ylabel('PV Capacity')
    plt.title('Islanding Capability for Critical Loads')
    
    # Add text annotations
    for i in range(len(pv_sizes)):
        for j in range(len(storage_sizes)):
            plt.text(j, i, f"{islanding_hours[i, j]:.1f}h", 
                    ha="center", va="center", 
                    color="w" if islanding_hours[i, j] < 12 else "black")
    
    plt.tight_layout()
    plt.savefig("islanding_capability.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nIslanding capability analysis complete!")


if __name__ == "__main__":
    main()