import opendssdirect as dss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load your DSS circuit ===
dss.Basic.ClearAll()
dss.Text.Command("Redirect ieee34Mod1.dss")  # Update path as needed
dss.Text.Command("Set mode=daily")
dss.Text.Command("Set stepsize=1m")
dss.Text.Command("Set number=1440")  # 24 hours * 60 minutes

# === Load shape definitions and load variation ===
# Store original load values
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

# Generate load profiles with variation
def generate_load_profiles(original_loads):
    """Generate 24-hour load profiles for all loads with minute-by-minute resolution"""
    profiles = {}
    
    # Create a basic load profile multiplier (24 hours, converted to minutes)
    # Morning peak and evening peak pattern
    hourly_profile = np.array([
        0.6, 0.55, 0.5, 0.5, 0.55, 0.6,       # 12am-6am
        0.85, 1.0, 1.05, 0.95, 0.9, 0.9,      # 6am-12pm
        0.85, 0.85, 0.85, 0.9, 1.0, 1.1,      # 12pm-6pm
        1.05, 1.0, 0.9, 0.8, 0.7, 0.65        # 6pm-12am
    ])
    
    # Convert hourly profile to minute-by-minute (1440 points)
    minute_profile = np.zeros(1440)
    for hour in range(24):
        start_min = hour * 60
        end_min = start_min + 60
        minute_profile[start_min:end_min] = hourly_profile[hour]
    
    # Add some randomness to each load
    for load_name in original_loads:
        # Add random variation (+/- 10%) and some noise
        variation = np.random.uniform(0.9, 1.1, 1440)
        # Add some random spikes to simulate sudden load changes
        num_spikes = np.random.randint(5, 15)  # 5-15 spikes throughout the day
        spike_indices = np.random.choice(range(1440), num_spikes, replace=False)
        spike_magnitudes = np.random.uniform(1.2, 1.5, num_spikes)  # 20-50% increase
        
        # Apply spikes
        profile_with_spikes = minute_profile * variation
        for i, idx in enumerate(spike_indices):
            # Create a spike that lasts 5-15 minutes
            spike_duration = np.random.randint(5, 15)
            end_idx = min(idx + spike_duration, 1440)
            profile_with_spikes[idx:end_idx] *= spike_magnitudes[i]
        
        profiles[load_name] = profile_with_spikes
    
    return profiles

# Set loads according to the profiles for a specific time step
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

# Get original load values
original_loads = get_original_loads()

# Generate load profiles with variation
load_profiles = generate_load_profiles(original_loads)

# === Solve QSTS with load variation ===
voltages = []
times = []
losses = []
load_multipliers = []
bus_to_monitor = "806"  # Change this to your bus of interest

# Add more buses to monitor if needed
additional_buses = ["810", "822", "848", "890"]  # Example buses across the feeder
all_buses_to_monitor = [bus_to_monitor] + additional_buses

# Dictionary to store voltages for each monitored bus and phase
bus_voltages = {bus: {'A': [], 'B': [], 'C': []} for bus in all_buses_to_monitor}

for i in range(1440):
    # Apply load variation for this time step
    set_loads_for_time(i, original_loads, load_profiles)
    
    # Solve power flow
    dss.Solution.Solve()
    
    # Capture time
    hr = dss.Solution.Hour()
    sec = dss.Solution.Seconds()
    times.append(f"{hr}:{sec//60}")
    
    # Capture voltages at all monitored buses
    for bus in all_buses_to_monitor:
        dss.Circuit.SetActiveBus(bus)
        vmag = dss.Bus.puVmagAngle()[::2]  # Only magnitudes (PU)
        
        # Store voltages for each phase of this bus
        # Handle cases where we might have fewer than 3 phases
        for p, phase in enumerate(['A', 'B', 'C']):
            if p < len(vmag):
                bus_voltages[bus][phase].append(vmag[p])
            else:
                bus_voltages[bus][phase].append(None)  # No voltage for this phase
        
        # For backward compatibility, also store the main bus voltage in the original list
        if bus == bus_to_monitor:
            voltages.append(vmag)
    
    # Capture total losses (kW)
    loss_kw = dss.Circuit.Losses()[0] / 1000
    losses.append(loss_kw)
    
    # Store average load multiplier for visualization
    avg_multiplier = np.mean([load_profiles[load][i] for load in load_profiles])
    load_multipliers.append(avg_multiplier)

# === Postprocess ===
# Create DataFrames for each bus
bus_voltage_dfs = {}
for bus in all_buses_to_monitor:
    # Create DataFrame with each phase as a column
    bus_voltage_dfs[bus] = pd.DataFrame({
        f"{bus} Phase A": bus_voltages[bus]['A'],
        f"{bus} Phase B": bus_voltages[bus]['B'],
        f"{bus} Phase C": bus_voltages[bus]['C'],
        "Time": times
    })

# For backward compatibility
voltage_df = pd.DataFrame(voltages, columns=["Phase A", "Phase B", "Phase C"])
voltage_df["Time"] = times

# Create loss and load multiplier DataFrames
loss_df = pd.DataFrame({"Time": times, "Losses (kW)": losses})
load_df = pd.DataFrame({"Time": times, "Load Multiplier": load_multipliers})

# Create output directory if it doesn't exist
output_dir = "voltage_variation_results"
os.makedirs(output_dir, exist_ok=True)

# === Plot voltages with load variation ===
plt.figure(figsize=(15, 10))

# Plot main bus voltage
plt.subplot(3, 1, 1)
for phase in ["Phase A", "Phase B", "Phase C"]:
    plt.plot(voltage_df["Time"][::20], voltage_df[phase][::20], label=phase)  # Plot every 20th point for clarity
plt.axhline(1.05, color="r", linestyle="--", label="Upper limit")
plt.axhline(0.95, color="r", linestyle="--", label="Lower limit")
plt.xticks(rotation=45)
plt.ylabel("Voltage (PU)")
plt.title(f"Bus {bus_to_monitor} Voltage Profile With Load Variation")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot load multiplier
plt.subplot(3, 1, 2)
plt.plot(load_df["Time"][::20], load_df["Load Multiplier"][::20], color='green')
plt.xticks(rotation=45)
plt.ylabel("Load Multiplier")
plt.title("Average Load Variation Over 24 Hours")
plt.grid(True, alpha=0.3)

# Plot losses
plt.subplot(3, 1, 3)
plt.plot(loss_df["Time"][::20], loss_df["Losses (kW)"][::20], color='orange')
plt.xticks(rotation=45)
plt.ylabel("System Losses (kW)")
plt.title("Total System Losses With Load Variation")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "voltage_load_losses.png"), dpi=300)
plt.show()

# === Plot all monitored buses for comparison ===
plt.figure(figsize=(15, 8))

# Plot Phase A voltages for all monitored buses
plt.subplot(3, 1, 1)
for bus in all_buses_to_monitor:
    plt.plot(bus_voltage_dfs[bus]["Time"][::20], bus_voltage_dfs[bus][f"{bus} Phase A"][::20], label=f"Bus {bus}")
plt.axhline(1.05, color="r", linestyle="--", label="Limits")
plt.axhline(0.95, color="r", linestyle="--")
plt.ylabel("Phase A Voltage (PU)")
plt.title("Phase A Voltage Comparison Across Multiple Buses")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Phase B voltages for all monitored buses
plt.subplot(3, 1, 2)
for bus in all_buses_to_monitor:
    plt.plot(bus_voltage_dfs[bus]["Time"][::20], bus_voltage_dfs[bus][f"{bus} Phase B"][::20], label=f"Bus {bus}")
plt.axhline(1.05, color="r", linestyle="--", label="Limits")
plt.axhline(0.95, color="r", linestyle="--")
plt.ylabel("Phase B Voltage (PU)")
plt.title("Phase B Voltage Comparison Across Multiple Buses")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Phase C voltages for all monitored buses
plt.subplot(3, 1, 3)
for bus in all_buses_to_monitor:
    plt.plot(bus_voltage_dfs[bus]["Time"][::20], bus_voltage_dfs[bus][f"{bus} Phase C"][::20], label=f"Bus {bus}")
plt.axhline(1.05, color="r", linestyle="--", label="Limits")
plt.axhline(0.95, color="r", linestyle="--")
plt.ylabel("Phase C Voltage (PU)")
plt.title("Phase C Voltage Comparison Across Multiple Buses")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "all_buses_voltage_comparison.png"), dpi=300)
plt.show()

# === Create a heatmap of voltage variations across buses and time ===
# Extract Phase A voltages for all buses at specific time intervals
time_samples = range(0, 1440, 60)  # Sample every hour
phase_a_voltages = np.zeros((len(all_buses_to_monitor), len(time_samples)))

for i, bus in enumerate(all_buses_to_monitor):
    for j, t in enumerate(time_samples):
        phase_a_voltages[i, j] = bus_voltages[bus]['A'][t]  # Phase A

plt.figure(figsize=(15, 6))
plt.imshow(phase_a_voltages, aspect='auto', cmap='viridis')
plt.colorbar(label='Voltage (PU)')
plt.xlabel('Time (hour)')
plt.ylabel('Bus')
plt.title('Phase A Voltage Heatmap Across Buses and Time')
plt.xticks(range(len(time_samples)), [f"{t//60}:00" for t in time_samples])
plt.yticks(range(len(all_buses_to_monitor)), all_buses_to_monitor)
plt.grid(False)
plt.savefig(os.path.join(output_dir, "voltage_heatmap.png"), dpi=300)
plt.show()

# Save the data to CSV files for further analysis
voltage_df.to_csv(os.path.join(output_dir, f"bus_{bus_to_monitor}_voltages.csv"), index=False)
loss_df.to_csv(os.path.join(output_dir, "system_losses.csv"), index=False)
load_df.to_csv(os.path.join(output_dir, "load_variation.csv"), index=False)

print(f"Analysis complete. Results saved to {output_dir}/")
