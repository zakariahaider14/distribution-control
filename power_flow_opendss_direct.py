import opendssdirect as dss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Clear any existing circuit
# dss('clear')  # Use the callable shortcut instead of run_command

# Define the IEEE 33-bus system
def create_ieee33_case():
    # Create a new circuit
    dss('New Circuit.IEEE33 bus1=1 basekv=12.66 pu=1.0 phases=3 MVAsc3=80 MVAsc1=80')
    
    # Define line parameters as R and X in ohms per length
    r_ohm_per_km = 0.0922  # Default R value
    x_ohm_per_km = 0.0470  # Default X value
    
    # Line data for IEEE 33-bus system (from_bus, to_bus, length_km, r_ohm_per_km, x_ohm_per_km)
    line_data = [
        (1, 2, 0.5, 0.0922, 0.0470),
        (2, 3, 0.5, 0.493, 0.2511),
        (3, 4, 0.5, 0.366, 0.1864),
        (4, 5, 0.5, 0.3811, 0.1941),
        (5, 6, 0.5, 0.819, 0.707),
        (6, 7, 0.5, 0.1872, 0.6188),
        (7, 8, 0.5, 1.7114, 1.2351),
        (8, 9, 0.5, 1.03, 0.74),
        (9, 10, 0.5, 1.04, 0.74),
        (10, 11, 0.5, 0.1966, 0.065),
        (11, 12, 0.5, 0.3744, 0.1238),
        (12, 13, 0.5, 1.468, 1.155),
        (13, 14, 0.5, 0.5416, 0.7129),
        (14, 15, 0.5, 0.591, 0.526),
        (15, 16, 0.5, 0.7463, 0.545),
        (16, 17, 0.5, 1.289, 1.721),
        (17, 18, 0.5, 0.732, 0.574),
        (2, 19, 0.5, 0.164, 0.1565),
        (19, 20, 0.5, 1.5042, 1.3554),
        (20, 21, 0.5, 0.4095, 0.4784),
        (21, 22, 0.5, 0.7089, 0.9373),
        (3, 23, 0.5, 0.4512, 0.3083),
        (23, 24, 0.5, 0.898, 0.7091),
        (24, 25, 0.5, 0.896, 0.7011),
        (6, 26, 0.5, 0.203, 0.1034),
        (26, 27, 0.5, 0.2842, 0.1447),
        (27, 28, 0.5, 1.059, 0.9337),
        (28, 29, 0.5, 0.8042, 0.7006),
        (29, 30, 0.5, 0.5075, 0.2585),
        (30, 31, 0.5, 0.9744, 0.963),
        (31, 32, 0.5, 0.3105, 0.3619),
        (32, 33, 0.5, 0.341, 0.5302)
    ]
    
    # Create lines
    for from_bus, to_bus, length, r, x in line_data:
        dss(f"""
            New Line.Line_{from_bus}_{to_bus} bus1={from_bus} bus2={to_bus} length={length} units=km
            ~ rmatrix=[{r} | 0 {r} | 0 0 {r}] xmatrix=[{x} | 0 {x} | 0 0 {x}] cmatrix=[0 | 0 0 | 0 0 0]
        """)
    
    # Load data for IEEE 33-bus system (bus, P_kW, Q_kVAr)
    load_data = [
        (2, 100, 60),
        (3, 90, 40),
        (4, 120, 80),
        (5, 60, 30),
        (6, 60, 20),
        (7, 200, 100),
        (8, 200, 100),
        (9, 60, 20),
        (10, 60, 20),
        (11, 45, 30),
        (12, 60, 35),
        (13, 60, 35),
        (14, 120, 80),
        (15, 60, 10),
        (16, 60, 20),
        (17, 60, 20),
        (18, 90, 40),
        (19, 90, 40),
        (20, 90, 40),
        (21, 90, 40),
        (22, 90, 40),
        (23, 90, 50),
        (24, 420, 200),
        (25, 420, 200),
        (26, 60, 25),
        (27, 60, 25),
        (28, 60, 20),
        (29, 120, 70),
        (30, 200, 600),
        (31, 150, 70),
        (32, 210, 100),
        (33, 60, 40)
    ]
    
    # Create loads
    for bus, p, q in load_data:
        dss(f"""
            New Load.Load_{bus} bus1={bus} phases=3 kv=12.66 kw={p} kvar={q} model=1 conn=wye
        """)

    # Set voltage bases
    dss('Set VoltageBases=[12.66]')
    dss('CalcVoltageBases')

# Create the IEEE 33-bus system
create_ieee33_case()

# Set some basic solution parameters
dss('Set MaxControlIter=100')
dss('Set MaxIterations=100')

# Solve the power flow
dss.Solution.Solve()

if dss.Solution.Converged():
    print("Power flow solution converged")
else:
    print("Power flow solution did not converge")

# Collect voltage profile data
bus_names = []
voltage_pu = []
voltage_actual = []
distances = []

for i in range(dss.Circuit.NumBuses()):
    # Use SetActiveBusi instead of SetActiveBusIndex
    dss.Circuit.SetActiveBusi(i)
    bus_name = dss.Bus.Name()
    bus_names.append(bus_name)
    
    # Get voltage in per unit
    v_pu = dss.Bus.puVmagAngle()[::2]  # Get just the magnitudes (every other value)
    voltage_pu.append(np.mean(v_pu))   # Average of all phases
    
    # Get actual voltage in kV
    v_actual = dss.Bus.VMagAngle()[::2]  # Get just the magnitudes
    voltage_actual.append(np.mean(v_actual)/1000)  # Convert from V to kV
    
    # For plotting purposes, use bus number as distance
    distances.append(int(bus_name) if bus_name.isdigit() else i+1)

# Calculate total losses
losses = dss.Circuit.Losses()
total_losses_kw = losses[0]/1000  # Convert from W to kW
total_losses_kvar = losses[1]/1000  # Convert from VAr to kVAr

print(f"Total circuit losses: {total_losses_kw:.2f} kW, {total_losses_kvar:.2f} kVAr")

# Create voltage profile plot
plt.figure(figsize=(12, 8))

# Sort data by distance for better plotting
sorted_indices = np.argsort(distances)
sorted_distances = [distances[i] for i in sorted_indices]
sorted_voltages = [voltage_pu[i] for i in sorted_indices]
sorted_bus_names = [bus_names[i] for i in sorted_indices]

plt.plot(sorted_distances, sorted_voltages, 'o-', linewidth=2, markersize=8)
plt.xlabel('Bus Number', fontsize=12)
plt.ylabel('Voltage (pu)', fontsize=12)
plt.title('IEEE 33-Bus System Voltage Profile', fontsize=14)
plt.grid(True)
plt.xticks(sorted_distances, sorted_bus_names, rotation=90)
plt.axhline(y=0.95, color='r', linestyle='--', label='Lower Limit (0.95 pu)')
plt.axhline(y=1.05, color='g', linestyle='--', label='Upper Limit (1.05 pu)')
plt.legend()
plt.tight_layout()
plt.show()

# Create a summary dataframe
results_df = pd.DataFrame({
    'Bus': bus_names,
    'Voltage_pu': voltage_pu,
    'Voltage_kV': voltage_actual,
    'Distance': distances
})

# Sort by bus number for better readability
results_df['Bus_Num'] = results_df['Bus'].apply(lambda x: int(x) if x.isdigit() else 999)
results_df = results_df.sort_values('Bus_Num')
results_df = results_df.drop('Bus_Num', axis=1)

print("\nVoltage Profile Summary:")
print(results_df)

# Add some PV generation to improve voltage profile
print("\nAdding PV generation at bus 18...")
dss("""
    New Generator.PV1 bus1=18 phases=3 kv=12.66 kw=300 kvar=100 model=3
""")

# Solve again
dss.Solution.Solve()

# Collect voltage profile after adding PV
voltage_with_pv = []
for i in range(dss.Circuit.NumBuses()):
    # Use SetActiveBusi instead of SetActiveBusIndex
    dss.Circuit.SetActiveBusi(i)
    v_pu = dss.Bus.puVmagAngle()[::2]  # Get just the magnitudes
    voltage_with_pv.append(np.mean(v_pu))  # Average of all phases

# Update results dataframe
results_df['Voltage_with_PV'] = [voltage_with_pv[bus_names.index(bus)] for bus in results_df['Bus']]
results_df['Voltage_Improvement'] = results_df['Voltage_with_PV'] - results_df['Voltage_pu']

print("\nVoltage Profile After Adding PV:")
print(results_df[['Bus', 'Voltage_pu', 'Voltage_with_PV', 'Voltage_Improvement']])

# Calculate new losses
new_losses = dss.Circuit.Losses()
new_losses_kw = new_losses[0]/1000
new_losses_kvar = new_losses[1]/1000

print(f"\nLosses after adding PV: {new_losses_kw:.2f} kW, {new_losses_kvar:.2f} kVAr")
print(f"Reduction in losses: {total_losses_kw - new_losses_kw:.2f} kW, {total_losses_kvar - new_losses_kvar:.2f} kVAr")