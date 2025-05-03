"""
Reinforcement Learning for Voltage Regulation in a 34-Bus Distribution System
Using OpenDSSDirect.py for OpenDSS interface and DQN for RL implementation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
import gymnasium as gym
from gymnasium import spaces
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import opendssdirect as dss
dss.dss_lib.DSS_Set_AllowForms(0) 

# Global settings
VOLTAGE_MIN = 0.95  # p.u.
VOLTAGE_MAX = 1.05  # p.u.
NOMINAL_VOLTAGE = 1.0  # p.u.

class IEEE34BusEnv(gym.Env):
    """
    OpenDSS Environment for IEEE 34 Bus Distribution System
    This environment simulates the power flow and voltage regulation in a 34-bus distribution system
    """
    
    def __init__(self, dss_file_path, max_episodes=1000):
        super(IEEE34BusEnv, self).__init__()
        
        # Initialize OpenDSS
        self.dss_file_path = dss_file_path
        self.initialize_opendss()
        
        # Get system information
        self.num_buses = len(dss.Circuit.AllBusNames())
        self.num_loads = len(dss.Loads.AllNames())
        self.num_capacitors = len(dss.Capacitors.AllNames())
        self.num_regulators = len(dss.RegControls.AllNames())
        
        print(f"System loaded: {self.num_buses} buses, {self.num_loads} loads, "
              f"{self.num_capacitors} capacitors, {self.num_regulators} regulators")
        
        # Define action and observation spaces
        # Actions: Capacitor banks (on/off) and regulator tap positions
        self.action_space = spaces.Dict({
            'capacitors': spaces.MultiBinary(self.num_capacitors),
            'regulators': spaces.MultiDiscrete([33] * self.num_regulators)  # Taps from -16 to +16
        })
        
        # Observations: Bus voltages and power flows
        self.observation_space = spaces.Box(
            low=0.8, high=1.2, shape=(self.num_buses * 3,),  # 3-phase voltages for each bus
            dtype=np.float32
        )
        
        # Store original load profiles
        self.original_loads = self.get_loads()
        
        # Store bus names and nominal voltages
        self.bus_names = dss.Circuit.AllBusNames()
        self.nominal_voltages = self.get_nominal_voltages()
        
        # Episode tracking
        self.current_episode = 0
        self.max_episodes = max_episodes
        self.time_step = 0
        self.max_time_steps = 24  # 24 hours simulation
        
        # Load a typical load profile (24-hour)
        self.load_profiles = self.generate_load_profiles()
    
    def initialize_opendss(self):
        """Initialize OpenDSS with IEEE 34-bus system"""
        dss.Basic.ClearAll()
        dss.Basic.Start(0)
        
        import os
        abs_path = os.path.abspath(self.dss_file_path)
        
        # Verify file exists
        if not os.path.exists(abs_path):
            print(f"ERROR: DSS file not found at: {abs_path}")
            raise Exception(f"DSS file does not exist: {abs_path}")
        
        print(f"Compiling DSS file: {abs_path}")
        
        # Compile the file
        dss.Text.Command(f'Compile "{abs_path}"')
        
        # Disable all controls to avoid convergence issues
        dss.Text.Command("Set ControlMode=OFF")
        
        # Check if circuit was created successfully
        if not dss.Circuit.Name():
            print("ERROR: Failed to create circuit")
            error_desc = dss.Error.Description()
            if error_desc:
                print(f"OpenDSS Error: {error_desc}")
            raise Exception("Failed to load DSS circuit")
        
        # Solve the initial power flow
        dss.Solution.Solve()
        
    def get_loads(self):
        """Get current load settings"""
        loads = {}
        for load_name in dss.Loads.AllNames():
            dss.Loads.Name(load_name)
            loads[load_name] = {
                'kW': dss.Loads.kW(),
                'kvar': dss.Loads.kvar(),
                'model': dss.Loads.Model(),
                # Use dss.CktElement.BusNames() to get bus information of the active load
                'bus': dss.CktElement.BusNames()[0]  # Get the first bus name
            }
        return loads
    
    def get_nominal_voltages(self):
        """Get nominal voltages for all buses"""
        nominal_volts = {}
        for bus in self.bus_names:
            dss.Circuit.SetActiveBus(bus)
            nominal_volts[bus] = dss.Bus.kVBase() * 1000  # Convert to V
        return nominal_volts
    
    def generate_load_profiles(self):
        """Generate 24-hour load profiles for all loads with more realistic variations"""
        profiles = {}
        load_variation_data = []
        
        # Create different load profile patterns
        # 1. Residential pattern (morning and evening peaks)
        residential_profile = np.array([
            0.45, 0.40, 0.35, 0.35, 0.40, 0.55,    # 12am-6am
            0.75, 0.90, 0.95, 0.90, 0.85, 0.85,    # 6am-12pm
            0.80, 0.80, 0.85, 0.90, 1.00, 1.15,    # 12pm-6pm
            1.20, 1.10, 0.95, 0.80, 0.65, 0.55     # 6pm-12am
        ])
        
        # 2. Commercial pattern (workday peak)
        commercial_profile = np.array([
            0.30, 0.25, 0.25, 0.25, 0.30, 0.40,    # 12am-6am
            0.60, 0.85, 1.00, 1.05, 1.10, 1.10,    # 6am-12pm
            1.05, 1.05, 1.00, 1.00, 0.95, 0.85,    # 12pm-6pm
            0.70, 0.60, 0.50, 0.45, 0.40, 0.35     # 6pm-12am
        ])
        
        # 3. Industrial pattern (flat with slight daytime increase)
        industrial_profile = np.array([
            0.70, 0.70, 0.70, 0.70, 0.75, 0.80,    # 12am-6am
            0.85, 0.90, 0.95, 1.00, 1.00, 1.00,    # 6am-12pm
            1.00, 1.00, 1.00, 0.95, 0.95, 0.90,    # 12pm-6pm
            0.85, 0.80, 0.75, 0.75, 0.70, 0.70     # 6pm-12am
        ])
        
        # Assign different profiles to different loads based on their location/characteristics
        # For this example, we'll distribute them based on load name
        load_names = list(self.original_loads.keys())
        num_loads = len(load_names)
        
        # Distribute load types: 60% residential, 30% commercial, 10% industrial
        residential_count = int(num_loads * 0.6)
        commercial_count = int(num_loads * 0.3)
        industrial_count = num_loads - residential_count - commercial_count
        
        # Shuffle load names to randomize assignment
        np.random.shuffle(load_names)
        
        # Assign load types
        load_types = {}
        for i, load_name in enumerate(load_names):
            if i < residential_count:
                load_types[load_name] = 'residential'
            elif i < residential_count + commercial_count:
                load_types[load_name] = 'commercial'
            else:
                load_types[load_name] = 'industrial'
        
        # Create profiles with variations
        for load_name in self.original_loads:
            # Select base profile based on load type
            if load_types[load_name] == 'residential':
                base_profile = residential_profile
                # Add more variation to residential loads
                variation = np.random.uniform(0.85, 1.15, 24)
            elif load_types[load_name] == 'commercial':
                base_profile = commercial_profile
                variation = np.random.uniform(0.90, 1.10, 24)
            else:  # industrial
                base_profile = industrial_profile
                # Less variation for industrial loads
                variation = np.random.uniform(0.95, 1.05, 24)
            
            # Apply variation and add some noise
            load_profile = base_profile * variation
            
            # Add small random fluctuations
            noise = np.random.normal(0, 0.02, 24)  # Small Gaussian noise
            load_profile = np.maximum(0.2, load_profile + noise)  # Ensure no negative or very small values
            
            profiles[load_name] = load_profile
            
            # Store data for visualization and analysis
            for hour, multiplier in enumerate(load_profile):
                load_variation_data.append({
                    'Load': load_name,
                    'Hour': hour,
                    'Type': load_types[load_name],
                    'Load Multiplier': multiplier
                })
        
        # Save load variation data to CSV
        output_dir = "rl_voltage_regulation_results"
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(load_variation_data).to_csv(f"{output_dir}/load_variation.csv", index=False)
        
        # Create a visualization of the load profiles
        self._visualize_load_profiles(profiles, load_types, output_dir)
        
        return profiles
        
    def _visualize_load_profiles(self, profiles, load_types, output_dir):
        """Create visualizations of the load profiles"""
        plt.figure(figsize=(12, 8))
        
        # Plot by load type
        residential_profiles = []
        commercial_profiles = []
        industrial_profiles = []
        
        for load_name, profile in profiles.items():
            if load_types[load_name] == 'residential':
                residential_profiles.append(profile)
            elif load_types[load_name] == 'commercial':
                commercial_profiles.append(profile)
            else:  # industrial
                industrial_profiles.append(profile)
        
        hours = range(24)
        
        # Plot average profiles by type
        if residential_profiles:
            avg_residential = np.mean(residential_profiles, axis=0)
            plt.plot(hours, avg_residential, 'b-', label='Residential (avg)', linewidth=2)
        
        if commercial_profiles:
            avg_commercial = np.mean(commercial_profiles, axis=0)
            plt.plot(hours, avg_commercial, 'r-', label='Commercial (avg)', linewidth=2)
        
        if industrial_profiles:
            avg_industrial = np.mean(industrial_profiles, axis=0)
            plt.plot(hours, avg_industrial, 'g-', label='Industrial (avg)', linewidth=2)
        
        # Plot individual profiles with transparency
        for profile in residential_profiles[:5]:  # Limit to 5 for clarity
            plt.plot(hours, profile, 'b-', alpha=0.2)
            
        for profile in commercial_profiles[:5]:
            plt.plot(hours, profile, 'r-', alpha=0.2)
            
        for profile in industrial_profiles[:5]:
            plt.plot(hours, profile, 'g-', alpha=0.2)
        
        plt.xlabel('Hour of Day')
        plt.ylabel('Load Multiplier')
        plt.title('Load Profiles by Type')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(range(0, 24, 2))
        plt.tight_layout()
        plt.savefig(f"{output_dir}/load_profiles.png", dpi=300)
    
    def set_loads_for_time(self, time_step):
        """Set loads according to the profiles for a specific time step"""
        for load_name, profile in self.load_profiles.items():
            multiplier = profile[time_step]
            dss.Loads.Name(load_name)
            original_kw = self.original_loads[load_name]['kW']
            original_kvar = self.original_loads[load_name]['kvar']
            
            # Set new load values
            dss.Loads.kW(original_kw * multiplier)
            dss.Loads.kvar(original_kvar * multiplier)
    
    def get_state(self):
        """Get bus voltages as the state"""
        voltages = []
        
        for bus in self.bus_names:
            dss.Circuit.SetActiveBus(bus)
            v = dss.Bus.puVmagAngle()[::2]  # Get only magnitudes, not angles
            
            # Handle buses with less than 3 phases
            v_padded = np.zeros(3)
            v_padded[:len(v)] = v
            voltages.extend(v_padded)
            
        return np.array(voltages, dtype=np.float32)
    
    def calculate_reward(self, voltages):
        """
        Calculate reward based on voltage deviation from nominal
        Penalize voltages outside the acceptable range (0.95-1.05 p.u.)
        Enhanced to provide more nuanced feedback to the agent
        """
        # Convert to numpy array and consider only non-zero voltages
        v_array = np.array(voltages)
        valid_v = v_array[v_array > 0.1]  # Filter out zero voltages (non-existing phases)
        
        # Calculate voltage deviations
        deviations = np.abs(valid_v - NOMINAL_VOLTAGE)
        mean_deviation = np.mean(deviations) if len(deviations) > 0 else 0
        max_deviation = np.max(deviations) if len(deviations) > 0 else 0
        
        # Calculate violations with severity
        under_violations = np.sum(valid_v < VOLTAGE_MIN)
        over_violations = np.sum(valid_v > VOLTAGE_MAX)
        total_violations = under_violations + over_violations
        
        # Calculate severity of violations
        if under_violations > 0:
            under_severity = np.sum(VOLTAGE_MIN - valid_v[valid_v < VOLTAGE_MIN])
        else:
            under_severity = 0
            
        if over_violations > 0:
            over_severity = np.sum(valid_v[valid_v > VOLTAGE_MAX] - VOLTAGE_MAX)
        else:
            over_severity = 0
            
        violation_severity = under_severity + over_severity
        
        # Reward components
        violation_penalty = -150 * total_violations  # Base penalty for any violation
        severity_penalty = -200 * violation_severity  # Additional penalty based on how far outside limits
        max_deviation_penalty = -50 * max_deviation  # Penalty for maximum deviation
        mean_deviation_penalty = -30 * mean_deviation  # Penalty for average deviation
        
        # Reward for staying within limits and close to nominal
        if total_violations > 0:
            # Heavy penalties for violations
            reward = violation_penalty + severity_penalty + max_deviation_penalty
        else:
            # Reward for staying within limits, with bonus for being close to nominal
            base_reward = 20  # Base reward for no violations
            deviation_reward = -40 * mean_deviation  # Smaller penalty for average deviation
            closeness_reward = 10 * (1 - max_deviation/0.05)  # Bonus for staying close to nominal
            reward = base_reward + deviation_reward + closeness_reward
        
        # Track metrics for analysis
        self.last_reward_components = {
            'violations': total_violations,
            'under_violations': under_violations,
            'over_violations': over_violations,
            'violation_severity': violation_severity,
            'max_deviation': max_deviation,
            'mean_deviation': mean_deviation,
            'reward': reward
        }
        
        return reward
        
    def get_voltage_profile(self):
        """Get detailed voltage profile for all buses"""
        voltage_data = []
        time_str = f"{self.time_step}:0"  # Format as hour:minute
        
        for bus in self.bus_names:
            dss.Circuit.SetActiveBus(bus)
            v_pu = dss.Bus.puVmagAngle()[::2]  # Get only magnitudes, not angles
            
            # Handle different number of phases
            phases = len(v_pu)
            
            # Store voltage for each phase
            for p in range(phases):
                phase_name = chr(65 + p)  # Convert to A, B, C
                voltage_data.append({
                    'Bus': bus,
                    'Phase': phase_name,
                    'Voltage (pu)': v_pu[p],
                    'Time': time_str
                })
        
        return voltage_data
    
    def step(self, action):
        """
        Execute one step in the environment
        Action: Dict with capacitor settings and regulator tap positions
        """
        # Set loads for the current time step
        self.set_loads_for_time(self.time_step)
        
        # Apply actions to the system
        self._apply_actions(action)
        
        # Force control mode to be off
        dss.Text.Command("Set ControlMode=OFF")
        
        # Try to solve with error handling
        try:
            dss.Solution.Solve()
        except Exception as e:
            print(f"Warning during solution: {e}")
            # Continue despite the warning
            pass
        
        # Get the new state (bus voltages)
        state = self.get_state()
        
        # Calculate reward
        reward = self.calculate_reward(state)
        
        # Get detailed voltage profile for analysis
        voltage_profile = self.get_voltage_profile()
        
        # Get system losses
        losses = dss.Circuit.Losses()[0] / 1000  # Convert W to kW
        
        # Track metrics for this step
        metrics = {
            'time_step': self.time_step,
            'action': action,
            'reward': reward,
            'losses': losses,
            **self.last_reward_components  # Include all reward components
        }
        
        # If this is the first step of the episode, initialize metrics storage
        if self.time_step == 0:
            self.episode_metrics = []
            self.episode_voltage_profiles = []
        
        # Store metrics and voltage profiles
        self.episode_metrics.append(metrics)
        self.episode_voltage_profiles.extend(voltage_profile)
        
        # Check if episode is done
        self.time_step += 1
        done = self.time_step >= self.max_time_steps
        
        # If episode is done, save metrics and voltage profiles
        if done and self.current_episode % 10 == 0:  # Save every 10 episodes to avoid excessive I/O
            self._save_episode_data()
        
        # Return state, reward, done, truncated, info
        return state, reward, done, False, {
            'voltages': state,
            'metrics': metrics,
            'voltage_profile': voltage_profile
        }
        
    def _save_episode_data(self):
        """Save episode metrics and voltage profiles"""
        output_dir = "rl_voltage_regulation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame(self.episode_metrics)
        metrics_df.to_csv(f"{output_dir}/episode_{self.current_episode}_metrics.csv", index=False)
        
        # Save voltage profiles
        voltage_df = pd.DataFrame(self.episode_voltage_profiles)
        voltage_df.to_csv(f"{output_dir}/episode_{self.current_episode}_voltages.csv", index=False)
    
    def _apply_actions(self, action):
        """Apply the agent's actions to the OpenDSS model"""
        # Set capacitor states (on/off)
        cap_names = dss.Capacitors.AllNames()
        for i, cap_name in enumerate(cap_names):
            if i < len(action['capacitors']):
                dss.Capacitors.Name(cap_name)
                if action['capacitors'][i] == 1:
                    dss.Capacitors.States([1])  # Turn on
                else:
                    dss.Capacitors.States([0])  # Turn off
        
        # Set regulator tap positions
        reg_names = dss.RegControls.AllNames()
        for i, reg_name in enumerate(reg_names):
            if i < len(action['regulators']):
                dss.RegControls.Name(reg_name)
                # Convert action (0-32) to tap position (-16 to +16)
                # Ensure tap is an integer
                tap = int(action['regulators'][i]) - 16
                # Set the tap position
                dss.RegControls.TapNumber(tap)
                dss.RegControls.TapWinding(1)                                               
                
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode"""
        super().reset(seed=seed)
        
        # Clear OpenDSS and start fresh
        dss.Basic.ClearAll()
        dss.Basic.Start(0)
        
        # Use the original file path that was set during initialization
        print(f"Resetting environment with DSS file: {self.dss_file_path}")
        dss.Text.Command(f'Compile "{self.dss_file_path}"')
        
        # Configure solution parameters to improve convergence
        dss.Text.Command("Set MaxControlIter=100")  # Increase control iterations
        dss.Text.Command("Set ControlMode=Static")  # Static control mode
        dss.Text.Command("Set MaxIterations=20")    # Increase power flow iterations
        
        # Reset time step
        self.time_step = 0
        
        # Set loads for initial time step
        self.set_loads_for_time(0)
        
        # Solve initial power flow
        dss.Solution.Solve()
        
        # Increment episode counter
        self.current_episode += 1
        
        # Get initial state
        state = self.get_state()
        
        return state, {}


class DQNAgent:
    """
    Deep Q-Network Agent for voltage regulation
    """
    def __init__(self, state_size, action_space, learning_rate=0.001, gamma=0.95):
        self.state_size = state_size
        self.action_space = action_space
        
        # Number of distinct actions (simplified from Dict space)
        # For simplicity, we'll flatten the action space
        # Each capacitor: 2 states (on/off)
        # Each regulator: 33 states (taps from -16 to +16)
        self.num_capacitors = len(action_space['capacitors'].shape)
        self.num_regulators = len(action_space['regulators'].shape)
        
        # Define action mapping strategy 
        # For simplicity, we'll use a limited set of predefined actions
        self.define_discrete_actions(num_actions=16)
        
        self.memory = deque(maxlen=2000)
        self.gamma = gamma  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        # Build the Q-network model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def define_discrete_actions(self, num_actions=16):
        """Define a discrete set of combined actions"""
        self.discrete_actions = []
        
        # Add a balanced set of actions (combinations of capacitor states and regulator taps)
        
        # No capacitors, regulators at different positions
        self.discrete_actions.append({'capacitors': np.zeros(self.num_capacitors), 
                                     'regulators': np.zeros(self.num_regulators)})  # All off, neutral
        
        # All capacitors on, regulators at different positions
        self.discrete_actions.append({'capacitors': np.ones(self.num_capacitors), 
                                     'regulators': np.zeros(self.num_regulators)})  # All on, neutral
        
        # Different combinations of capacitors and regulator positions
        for i in range(2, num_actions):
            # Create random combinations
            cap_states = np.random.randint(0, 2, self.num_capacitors)
            # Spread regulator taps between -16 and +16
            reg_taps = np.random.randint(0, 33, self.num_regulators)
            
            self.discrete_actions.append({
                'capacitors': cap_states,
                'regulators': reg_taps
            })
        
        print(f"Created {len(self.discrete_actions)} discrete actions")
    
    def _build_model(self):
        """Build a neural network to predict Q-values"""
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(len(self.discrete_actions), activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update the target network with the main network's weights"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action_idx, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def act(self, state):
        """Decide action based on epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            # Random action
            action_idx = np.random.randint(0, len(self.discrete_actions))
        else:
            # Predict Q-values and choose best action
            act_values = self.model.predict(state.reshape(1, -1), verbose=0)
            action_idx = np.argmax(act_values[0])
        
        return action_idx, self.discrete_actions[action_idx]
    
    def replay(self, batch_size):
        """Train the model with experiences from replay memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action_idx, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Predict future rewards with target network
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                )
            
            # Get current Q-values
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            # Update the Q-value for the chosen action
            target_f[0][action_idx] = target
            
            # Train the model
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name)
    
# In the DQNAgent class
    def save(self, name):
        """Save model weights"""
        # Make sure the filename ends with .weights.h5 as required by Keras
        if not name.endswith('.weights.h5'):
            name = name.replace('.h5', '.weights.h5')
            if not name.endswith('.weights.h5'):
                name = name + '.weights.h5'
        self.model.save_weights(name)


def create_ieee34_dss_file(output_dir):
    """
    Create the IEEE 34-bus DSS file if it doesn't exist
    Returns the path to the DSS file
    """
    import os
    
    # Ensure proper directory creation with normalized path
    output_dir = os.path.abspath(os.path.normpath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # Create path with proper OS-specific separator
    dss_file_path = os.path.join(output_dir, "IEEE34.dss")
    
    if os.path.exists(dss_file_path):
        print(f"DSS file already exists at: {dss_file_path}")
        # Let's remove the file and recreate it to ensure it has the correct settings
        os.remove(dss_file_path)
        print(f"Existing file removed. Creating new DSS file.")
    
    print(f"Creating IEEE 34-bus DSS file at: {dss_file_path}")
    
    # IEEE 34-bus system DSS script with modified regulator controls
    dss_script = """
    Clear
    
    ! IEEE 34-bus test feeder
    
    New Circuit.ieee34 bus1=800 pu=1.05 basekV=24.9 R1=0.0001 X1=0.0001 R0=0.0001 X0=0.0001
    
    ! Substation Transformer
    New Transformer.SubXF Phases=3 Windings=2 Xhl=8
    ~ wdg=1 bus=800 conn=Delta kv=24.9 kva=2500 %r=1
    ~ wdg=2 bus=802 conn=Wye kv=24.9 kva=2500 %r=1
    
    ! Load Tap Changing Transformer
    New Transformer.XFM1 Phases=3 Windings=2 Xhl=2
    ~ wdg=1 bus=832 conn=Delta kv=24.9 kva=500 %r=0.5
    ~ wdg=2 bus=888 conn=Wye kv=4.16 kva=500 %r=0.5
    ~ Tap=1 MaxTap=1.1 MinTap=0.9
    
    ! Regulator 1
    New Transformer.Reg1 phases=3 windings=2 bank=reg1 Xhl=0.01 ppm=0
    ~ wdg=1 bus=814 conn=Wye kv=24.9 kva=1000 %r=0.01
    ~ wdg=2 bus=850 conn=Wye kv=24.9 kva=1000 %r=0.01
    
    New RegControl.Reg1 transformer=Reg1 winding=2 vreg=122 band=2 ptratio=120 ctprim=100 R=3 X=9
    ! Disable the regulator control - we'll control it manually
    RegControl.Reg1.Enabled=No
    
    ! Regulator 2
    New Transformer.Reg2 phases=3 windings=2 bank=reg2 Xhl=0.01 ppm=0
    ~ wdg=1 bus=852 conn=Wye kv=24.9 kva=1000 %r=0.01
    ~ wdg=2 bus=832 conn=Wye kv=24.9 kva=1000 %r=0.01
    
    New RegControl.Reg2 transformer=Reg2 winding=2 vreg=124 band=2 ptratio=120 ctprim=100 R=3 X=9
    ! Disable the regulator control - we'll control it manually
    RegControl.Reg2.Enabled=No
    
    ! Capacitors
    New Capacitor.Cap1 Bus1=844 phases=3 kVAR=300 kV=24.9
    New Capacitor.Cap2 Bus1=848 phases=3 kVAR=450 kV=24.9
    
    ! Lines
    New Line.L1 Phases=3 Bus1=802 Bus2=806 R1=0.0922 X1=0.0470 R0=0.1883 X0=0.0982 Length=0.5 units=km
    New Line.L2 Phases=3 Bus1=806 Bus2=808 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=1.5 units=km
    New Line.L3 Phases=3 Bus1=808 Bus2=810 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=1.0 units=km
    New Line.L4 Phases=3 Bus1=808 Bus2=812 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=1.0 units=km
    New Line.L5 Phases=3 Bus1=812 Bus2=814 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L6 Phases=3 Bus1=814 Bus2=850 R1=0.1014 X1=0.1152 R0=0.3394 X0=0.3054 Length=0.1 units=km
    New Line.L7 Phases=3 Bus1=816 Bus2=818 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L8 Phases=3 Bus1=816 Bus2=824 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=1.0 units=km
    New Line.L9 Phases=3 Bus1=818 Bus2=820 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L10 Phases=3 Bus1=820 Bus2=822 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L11 Phases=3 Bus1=824 Bus2=826 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L12 Phases=3 Bus1=824 Bus2=828 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L13 Phases=3 Bus1=828 Bus2=830 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L14 Phases=3 Bus1=830 Bus2=854 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L15 Phases=3 Bus1=832 Bus2=858 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L16 Phases=3 Bus1=834 Bus2=860 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L17 Phases=3 Bus1=834 Bus2=842 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L18 Phases=3 Bus1=836 Bus2=840 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L19 Phases=3 Bus1=836 Bus2=862 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L20 Phases=3 Bus1=842 Bus2=844 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L21 Phases=3 Bus1=844 Bus2=846 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L22 Phases=3 Bus1=846 Bus2=848 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L23 Phases=3 Bus1=850 Bus2=816 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L24 Phases=3 Bus1=852 Bus2=832 R1=0.0001 X1=0.0001 R0=0.0001 X0=0.0001 Length=0.001 units=km
    New Line.L25 Phases=3 Bus1=854 Bus2=856 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L26 Phases=3 Bus1=854 Bus2=852 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=10.0 units=km
    New Line.L27 Phases=3 Bus1=858 Bus2=864 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L28 Phases=3 Bus1=858 Bus2=834 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L29 Phases=3 Bus1=860 Bus2=836 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L30 Phases=3 Bus1=862 Bus2=838 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    New Line.L31 Phases=3 Bus1=888 Bus2=890 R1=0.4720 X1=0.1814 R0=0.6982 X0=0.4582 Length=0.5 units=km
    
    ! Loads
    New Load.L1 Bus1=860 Phases=3 Conn=Wye Model=1 kV=24.9 kW=60 kvar=48 vminpu=0.85 vmaxpu=1.15
    New Load.L2 Bus1=840 Phases=3 Conn=Wye Model=1 kV=24.9 kW=27 kvar=21 vminpu=0.85 vmaxpu=1.15
    New Load.L3 Bus1=844 Phases=3 Conn=Wye Model=1 kV=24.9 kW=405 kvar=315 vminpu=0.85 vmaxpu=1.15
    New Load.L4 Bus1=848 Phases=3 Conn=Delta Model=1 kV=24.9 kW=60 kvar=48 vminpu=0.85 vmaxpu=1.15
    New Load.L5 Bus1=830 Phases=3 Conn=Wye Model=1 kV=24.9 kW=45 kvar=36 vminpu=0.85 vmaxpu=1.15
    New Load.L6 Bus1=890 Phases=3 Conn=Delta Model=1 kV=4.16 kW=150 kvar=75 vminpu=0.85 vmaxpu=1.15
    
    ! Default voltage bases
    Set VoltageBases=[24.9, 4.16]
    CalcVoltageBases
    
    ! Solution mode settings
    Set ControlMode=OFF
    Set MaxControlIter=100
    Set MaxIterations=20
    
    ! Solve initial power flow
    Solve
    """
    
    with open(dss_file_path, 'w') as f:
        f.write(dss_script)
    
    return dss_file_path



def train_rl_voltage_regulation(output_dir="./output", episodes=100, batch_size=32):
    """
    Train the RL agent for voltage regulation
    """
    # Normalize the output directory path
    import os
    output_dir = os.path.normpath(output_dir)
    
    # Create the IEEE 34-bus DSS file
    dss_file_path = create_ieee34_dss_file(output_dir)
    
    # Create environment
    env = IEEE34BusEnv(dss_file_path, max_episodes=episodes)
    
    # Get state size
    state_size = env.observation_space.shape[0]
    
    # Create agent
    agent = DQNAgent(state_size, env.action_space, learning_rate=0.001, gamma=0.95)
    
    # Training metrics
    rewards_history = []
    voltage_violations_history = []
    average_deviation_history = []
    
    # Training loop
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        violations_count = 0
        deviations_sum = 0
        steps = 0
        
        # Set up progress tracking
        if episode % 5 == 0:
            print(f"\nEpisode {episode}/{episodes}")
        
        done = False
        while not done:
            # Select action
            action_idx, action = agent.act(state)
            
            # Execute action
            next_state, reward, done, _, info = env.step(action)
            
            # Store experience
            agent.remember(state, action_idx, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Update metrics
            total_reward += reward
            
            # Count voltage violations and calculate average deviation
            voltages = info['voltages']
            v_array = np.array(voltages)
            valid_v = v_array[v_array > 0.1]  # Filter out zero voltages
            
            violations = np.sum((valid_v < VOLTAGE_MIN) | (valid_v > VOLTAGE_MAX))
            violations_count += violations
            
            deviations = np.abs(valid_v - NOMINAL_VOLTAGE)
            deviations_sum += np.mean(deviations)
            
            # Set loads for next time step if not done
            if not done:
                env.set_loads_for_time(env.time_step)
            
            steps += 1
            
            # Train the agent
            agent.replay(batch_size)
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_model()
        
        # Record metrics
        rewards_history.append(total_reward)
        voltage_violations_history.append(violations_count / steps)
        average_deviation_history.append(deviations_sum / steps)
        
        # Print progress
        if episode % 5 == 0:
            avg_reward = np.mean(rewards_history[-5:]) if len(rewards_history) >= 5 else total_reward
            avg_violations = np.mean(voltage_violations_history[-5:]) if len(voltage_violations_history) >= 5 else violations_count / steps
            avg_deviation = np.mean(average_deviation_history[-5:]) if len(average_deviation_history) >= 5 else deviations_sum / steps
            
            print(f"  Episode: {episode}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Violations: {avg_violations:.2f}")
            print(f"  Avg Voltage Deviation: {avg_deviation:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
    
    # Save model weights
# Save model weights
    model_path = os.path.join(output_dir, "voltage_reg_model.weights.h5")
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training performance
    plot_training_performance(rewards_history, voltage_violations_history, average_deviation_history, output_dir)
    
    return agent, env


def plot_training_performance(rewards, violations, deviations, output_dir):
    """Plot training performance metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot voltage violations
    ax2.plot(violations, color='red')
    ax2.set_title('Voltage Violations per Step')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Avg Violations per Step')
    ax2.grid(True)
    
    # Plot voltage deviations
    ax3.plot(deviations, color='green')
    ax3.set_title('Average Voltage Deviation from Nominal')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Avg Deviation (p.u.)')
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, "training_performance.png")
    plt.savefig(fig_path)
    print(f"Performance plot saved to {fig_path}")
    
    # Close figure to free memory
    plt.close(fig)


def evaluate_agent(agent, env, episodes=5):
    """
    Evaluate the trained agent's performance
    """
    print("\nEvaluating agent performance...")
    
    total_rewards = []
    total_violations = []
    total_deviations = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_violations = 0
        episode_deviations = 0
        steps = 0
        
        done = False
        while not done:
            # Use trained policy (no exploration)
            agent.epsilon = 0
            action_idx, action = agent.act(state)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            
            # Update state
            state = next_state
            
            # Update metrics
            episode_reward += reward
            
            # Calculate violations and deviations
            voltages = info['voltages']
            v_array = np.array(voltages)
            valid_v = v_array[v_array > 0.1]  # Filter out zero voltages
            
            violations = np.sum((valid_v < VOLTAGE_MIN) | (valid_v > VOLTAGE_MAX))
            episode_violations += violations
            
            deviations = np.abs(valid_v - NOMINAL_VOLTAGE)
            episode_deviations += np.mean(deviations)
            
            # Set loads for next time step if not done
            if not done:
                env.set_loads_for_time(env.time_step)
            
            steps += 1
        
        # Record metrics
        total_rewards.append(episode_reward)
        total_violations.append(episode_violations / steps)
        total_deviations.append(episode_deviations / steps)
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
              f"Violations = {episode_violations / steps:.2f}, "
              f"Deviation = {episode_deviations / steps:.4f}")
    
    # Calculate averages
    avg_reward = np.mean(total_rewards)
    avg_violations = np.mean(total_violations)
    avg_deviation = np.mean(total_deviations)
    
    print("\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Violations per Step: {avg_violations:.2f}")
    print(f"Average Voltage Deviation: {avg_deviation:.4f}")
    
    return avg_reward, avg_violations, avg_deviation


def compare_with_baseline(env, episodes=5):
    """
    Compare the RL approach with a baseline approach (no control actions)
    """
    print("\nComparing with baseline (no control)...")
    
    baseline_violations = []
    baseline_deviations = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_violations = 0
        episode_deviations = 0
        steps = 0
        
        # Create a "do nothing" action
        do_nothing_action = {
            'capacitors': np.zeros(env.num_capacitors),
            'regulators': np.ones(env.num_regulators) * 16  # Neutral position (0 tap)
        }
        
        done = False
        while not done:
            # Take "do nothing" action
            next_state, _, done, _, info = env.step(do_nothing_action)
            
            # Update state
            state = next_state
            
            # Calculate violations and deviations
            voltages = info['voltages']
            v_array = np.array(voltages)
            valid_v = v_array[v_array > 0.1]  # Filter out zero voltages
            
            violations = np.sum((valid_v < VOLTAGE_MIN) | (valid_v > VOLTAGE_MAX))
            episode_violations += violations
            
            deviations = np.abs(valid_v - NOMINAL_VOLTAGE)
            episode_deviations += np.mean(deviations)
            
            # Set loads for next time step if not done
            if not done:
                env.set_loads_for_time(env.time_step)
            
            steps += 1
        
        # Record metrics
        baseline_violations.append(episode_violations / steps)
        baseline_deviations.append(episode_deviations / steps)
        
        print(f"Baseline Episode {episode}: "
              f"Violations = {episode_violations / steps:.2f}, "
              f"Deviation = {episode_deviations / steps:.4f}")
    
    # Calculate averages
    avg_violations = np.mean(baseline_violations)
    avg_deviation = np.mean(baseline_deviations)
    
    print("\nBaseline Results (No Control):")
    print(f"Average Violations per Step: {avg_violations:.2f}")
    print(f"Average Voltage Deviation: {avg_deviation:.4f}")
    
    return avg_violations, avg_deviation


def visualize_voltage_profiles(agent, env, num_steps=24):
    """
    Visualize voltage profiles with and without RL control
    """
    print("\nVisualizing voltage profiles...")
    
    # Create output directory for plots
    output_dir = "voltage_profiles"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run RL agent
    state, _ = env.reset()
    done = False
    
    rl_voltages = []
    rl_actions = []
    rl_metrics = []
    time_steps = []
    
    while not done:
        # The act method returns both action_idx and action
        action_idx, action = agent.act(state)
        next_state, reward, done, _, info = env.step(action)
        
        # Store data
        time_steps.append(env.time_step - 1)  # -1 because time_step is incremented in step()
        rl_voltages.append(info['voltages'])
        rl_actions.append(action)
        # Store metrics from the info dictionary
        rl_metrics.append(info['metrics'])
        
        state = next_state
        
        if env.time_step >= num_steps:
            break
    
    # Run baseline (no control)
    state, _ = env.reset()
    done = False
    
    baseline_voltages = []
    baseline_metrics = []
    
    # Set all capacitors to off and regulators to neutral position
    baseline_action = {
        'capacitors': np.zeros(env.num_capacitors),
        'regulators': np.zeros(env.num_regulators)
    }
    
    while not done:
        next_state, reward, done, _, info = env.step(baseline_action)
        
        # Store data
        baseline_voltages.append(info['voltages'])
        # Store metrics from the info dictionary
        baseline_metrics.append(info['metrics'])
        
        state = next_state
        
        if env.time_step >= num_steps:
            break
    
    # Extract metrics for easier plotting
    rl_losses = [m.get('losses', 0) for m in rl_metrics]
    baseline_losses = [m.get('losses', 0) for m in baseline_metrics]
    rl_violations = [m.get('violations', 0) for m in rl_metrics]
    baseline_violations = [m.get('violations', 0) for m in baseline_metrics]
    rl_max_deviations = [m.get('max_deviation', 0) for m in rl_metrics]
    baseline_max_deviations = [m.get('max_deviation', 0) for m in baseline_metrics]
    
    # Create DataFrames for easier analysis
    rl_data = pd.DataFrame({
        'Time': time_steps,
        'RL_Losses': rl_losses,
        'RL_Violations': rl_violations,
        'RL_Max_Deviation': rl_max_deviations
    })
    
    baseline_data = pd.DataFrame({
        'Time': time_steps,
        'Baseline_Losses': baseline_losses,
        'Baseline_Violations': baseline_violations,
        'Baseline_Max_Deviation': baseline_max_deviations
    })
    
    # Merge data
    results = pd.merge(rl_data, baseline_data, on='Time')
    
    # Save results to CSV
    results_path = os.path.join(output_dir, "comparison_results.csv")
    results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Plot voltage profiles for key buses
    key_buses = [0, 5, 10, 15, 20, 25]  # Example key buses to monitor
    
    plt.figure(figsize=(12, 8))
    for i, bus_idx in enumerate(key_buses):
        if bus_idx < len(rl_voltages[0]):
            plt.subplot(len(key_buses), 1, i+1)
            
            # RL control voltages
            rl_bus_voltages = [v[bus_idx] for v in rl_voltages]
            plt.plot(time_steps, rl_bus_voltages, 'b-', label='RL Control')
            
            # Baseline voltages
            baseline_bus_voltages = [v[bus_idx] for v in baseline_voltages]
            plt.plot(time_steps, baseline_bus_voltages, 'r--', label='No Control')
            
            # Voltage limits
            plt.axhline(y=VOLTAGE_MIN, color='k', linestyle=':', alpha=0.5)
            plt.axhline(y=VOLTAGE_MAX, color='k', linestyle=':', alpha=0.5)
            
            plt.title(f'Bus {bus_idx+1} Voltage Profile')
            plt.ylabel('Voltage (p.u.)')
            if i == len(key_buses) - 1:
                plt.xlabel('Time Step (Hour)')
            plt.legend()
            plt.grid(True)
    
    plt.tight_layout()
    voltage_plot_path = os.path.join(output_dir, "voltage_profiles.png")
    plt.savefig(voltage_plot_path)
    print(f"Voltage profiles saved to {voltage_plot_path}")
    plt.close()
    
    # Plot control actions
    plt.figure(figsize=(12, 8))
    
    # Capacitor states
    plt.subplot(2, 1, 1)
    
    # Check if we have any capacitors to plot
    if env.num_capacitors > 0:
        # Safely extract capacitor states, handling different action structures
        for cap_idx in range(min(env.num_capacitors, len(rl_actions[0]['capacitors']))):
            try:
                cap_states = [float(action['capacitors'][cap_idx]) for action in rl_actions]
                plt.step(time_steps, cap_states, label=f'Capacitor {cap_idx+1}', where='post', linewidth=2)
            except (IndexError, KeyError, TypeError) as e:
                print(f"Warning: Could not plot capacitor {cap_idx+1}: {e}")
                continue
    else:
        plt.text(0.5, 0.5, 'No capacitors in system', horizontalalignment='center', verticalalignment='center')
    
    plt.title('Capacitor Control Actions')
    plt.xlabel('Time Step (Hour)')
    plt.ylabel('State (0=Off, 1=On)')
    plt.yticks([0, 1])
    plt.legend()
    
    # Regulator tap positions
    plt.subplot(2, 1, 2)
    
    # Check if we have any regulators to plot
    if env.num_regulators > 0:
        # Safely extract regulator tap positions, handling different action structures
        for reg_idx in range(min(env.num_regulators, len(rl_actions[0]['regulators']))):
            try:
                reg_taps = [float(action['regulators'][reg_idx]) for action in rl_actions]
                plt.step(time_steps, reg_taps, label=f'Regulator {reg_idx+1}', where='post', linewidth=2)
            except (IndexError, KeyError, TypeError) as e:
                print(f"Warning: Could not plot regulator {reg_idx+1}: {e}")
                continue
    else:
        plt.text(0.5, 0.5, 'No regulators in system', horizontalalignment='center', verticalalignment='center')
    
    plt.title('Voltage Regulator Tap Positions')
    plt.xlabel('Time Step (Hour)')
    plt.ylabel('Tap Position')
    plt.legend()
    
    plt.tight_layout()
    control_plot_path = os.path.join(output_dir, "control_actions.png")
    plt.savefig(control_plot_path)
    print(f"Control actions saved to {control_plot_path}")
    plt.close()
    
    # Plot performance metrics
    plt.figure(figsize=(15, 10))
    
    # Losses
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, results['RL_Losses'], 'b-', label='RL Control')
    plt.plot(time_steps, results['Baseline_Losses'], 'r--', label='No Control')
    plt.title('System Losses')
    plt.ylabel('Losses (kW)')
    plt.legend()
    plt.grid(True)
    
    # Violations
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, results['RL_Violations'], 'b-', label='RL Control')
    plt.plot(time_steps, results['Baseline_Violations'], 'r--', label='No Control')
    plt.title('Voltage Violations')
    plt.ylabel('Number of Violations')
    plt.legend()
    plt.grid(True)
    
    # Max Deviation
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, results['RL_Max_Deviation'], 'b-', label='RL Control')
    plt.plot(time_steps, results['Baseline_Max_Deviation'], 'r--', label='No Control')
    plt.title('Maximum Voltage Deviation from Nominal')
    plt.xlabel('Time Step (Hour)')
    plt.ylabel('Max Deviation (p.u.)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    metrics_plot_path = os.path.join(output_dir, "performance_metrics.png")
    plt.savefig(metrics_plot_path)
    print(f"Performance metrics saved to {metrics_plot_path}")
    plt.close()
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print(f"Average Losses: RL = {results['RL_Losses'].mean():.2f} kW, Baseline = {results['Baseline_Losses'].mean():.2f} kW")
    print(f"Total Violations: RL = {results['RL_Violations'].sum()}, Baseline = {results['Baseline_Violations'].sum()}")
    print(f"Average Max Deviation: RL = {results['RL_Max_Deviation'].mean():.4f}, Baseline = {results['Baseline_Max_Deviation'].mean():.4f}")
    
    return results
    print(f"Violation Reduction: {violation_improvement:.2f}%")
    print(f"Deviation Reduction: {deviation_improvement:.2f}%")
    
    return comparison_df


def main():
    """
    Main function to run the voltage regulation reinforcement learning
    with enhanced load variations and voltage profile analysis
    """
    print("Starting Enhanced Voltage Regulation RL System for IEEE 34-Bus Distribution System")
    
    # Output directory
    output_dir = "rl_voltage_regulation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DSS file if it doesn't exist
    dss_file_path = create_ieee34_dss_file(output_dir)
    
    # Create the environment with enhanced load variations
    env = IEEE34BusEnv(dss_file_path)
    
    # Print information about the environment
    print(f"\nEnvironment created with:")
    print(f"- {env.num_buses} buses")
    print(f"- {env.num_loads} loads")
    print(f"- {env.num_capacitors} capacitors")
    print(f"- {env.num_regulators} voltage regulators")
    
    # Choose mode: 'train' for training a new agent, 'load' to load a pre-trained agent
    mode = 'train'  # Change to 'load' to use a pre-trained model
    
    if mode == 'train':
        # Training parameters
        episodes = 500  # Reduced for faster training, increase for better performance
        batch_size = 32
        
        # Train the agent
        print(f"\nTraining RL agent for {episodes} episodes...")
        agent, env = train_rl_voltage_regulation(output_dir, episodes, batch_size)
    else:
        # Load a pre-trained agent
        model_path = os.path.join(output_dir, "voltage_reg_model.weights.h5")
        if os.path.exists(model_path):
            print(f"\nLoading pre-trained agent from {model_path}...")
            state_size = env.observation_space.shape[0]
            agent = DQNAgent(state_size, env.action_space)
            agent.load(model_path)
            agent.epsilon = 0  # No exploration for evaluation
        else:
            print(f"\nNo pre-trained model found at {model_path}. Training a new agent...")
            episodes = 500  # Quick training
            batch_size = 32
            agent, env = train_rl_voltage_regulation(output_dir, episodes, batch_size)
    
    # Evaluate the agent with the enhanced load variations
    print("\nEvaluating agent with enhanced load variations...")
    evaluate_agent(agent, env)
    
    # Compare with baseline
    print("\nComparing with baseline (no control)...")
    compare_with_baseline(env)
    
    # Visualize voltage profiles with detailed analysis
    print("\nGenerating detailed voltage profile analysis...")
    results = visualize_voltage_profiles(agent, env)
    
    # Generate a summary report
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("===== Voltage Regulation RL System Performance Summary =====\n\n")
        f.write(f"Number of buses: {env.num_buses}\n")
        f.write(f"Number of loads: {env.num_loads}\n")
        f.write(f"Number of capacitors: {env.num_capacitors}\n")
        f.write(f"Number of regulators: {env.num_regulators}\n\n")
        
        # Add performance metrics
        f.write("Performance Metrics:\n")
        f.write(f"Average Losses: RL = {results['RL_Losses'].mean():.2f} kW, Baseline = {results['Baseline_Losses'].mean():.2f} kW\n")
        f.write(f"Total Violations: RL = {results['RL_Violations'].sum()}, Baseline = {results['Baseline_Violations'].sum()}\n")
        f.write(f"Average Max Deviation: RL = {results['RL_Max_Deviation'].mean():.4f}, Baseline = {results['Baseline_Max_Deviation'].mean():.4f}\n\n")
        
        # Calculate improvement percentages
        loss_improvement = ((results['Baseline_Losses'].mean() - results['RL_Losses'].mean()) / results['Baseline_Losses'].mean()) * 100
        violation_improvement = ((results['Baseline_Violations'].sum() - results['RL_Violations'].sum()) / max(1, results['Baseline_Violations'].sum())) * 100
        deviation_improvement = ((results['Baseline_Max_Deviation'].mean() - results['RL_Max_Deviation'].mean()) / results['Baseline_Max_Deviation'].mean()) * 100
        
        f.write("Improvement with RL Control:\n")
        f.write(f"Loss Reduction: {loss_improvement:.2f}%\n")
        f.write(f"Violation Reduction: {violation_improvement:.2f}%\n")
        f.write(f"Deviation Reduction: {deviation_improvement:.2f}%\n")
    
    print(f"\nPerformance summary saved to {summary_path}")
    print("\nEnhanced Voltage Regulation RL System completed successfully!")
    print(f"All results saved to {output_dir}/")
    
    return agent, env, results


if __name__ == "__main__":
    main()