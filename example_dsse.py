import os
import numpy as np
import matplotlib.pyplot as plt
from distribution_state_estimation import DistributionStateEstimation, logger
import logging
import traceback
import opendssdirect as dss

# Set logging level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Example of using the Distribution State Estimation class
    """
    # Path to the OpenDSS file describing the distribution system
    # Replace with your actual OpenDSS file path
    dss_file_path = "ieee34_feeder.dss"
    
    # Check if the file exists
    if not os.path.exists(dss_file_path):
        print(f"Error: OpenDSS file not found at {dss_file_path}")
        print("Please provide a valid path to an OpenDSS file")
        return
    
    # Create output directory for results
    output_dir = "dsse_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Running Distribution State Estimation on {dss_file_path}")
    print(f"{'='*80}\n")
    
    try:
        # Check if OpenDSS is properly initialized
        print("Checking OpenDSS installation...")
        try:
            dss.Basic.Version()
            print(f"OpenDSS version: {dss.Basic.Version()}")
        except Exception as e:
            print(f"Error with OpenDSS: {str(e)}")
            return
            
        # Initialize the distribution state estimation object
        print("Initializing DistributionStateEstimation object...")
        dsse = DistributionStateEstimation(dss_file_path)
        print("DistributionStateEstimation object initialized successfully")
        
        # Add SCADA measurements at the feeder head (substation)
        # These would typically come from real-time SCADA systems
        print("Adding voltage measurements at sourcebus...")
        try:
            dsse.add_voltage_measurement("sourcebus.1", 1.0, 0.0001, phase=1)
            dsse.add_voltage_measurement("sourcebus.2", 1.0, 0.0001, phase=2)
            dsse.add_voltage_measurement("sourcebus.3", 1.0, 0.0001, phase=3)
            print("Voltage measurements added successfully")
        except Exception as e:
            print(f"Error adding voltage measurements: {str(e)}")
            traceback.print_exc()
        
        # Add power measurements at feeder head
        print("Adding power measurements at sourcebus...")
        try:
            dsse.add_power_measurement("sourcebus.1", 1000, 0.01, 300, 0.01, phase=1)
            dsse.add_power_measurement("sourcebus.2", 900, 0.01, 250, 0.01, phase=2)
            dsse.add_power_measurement("sourcebus.3", 1100, 0.01, 320, 0.01, phase=3)
            print("Power measurements added successfully")
        except Exception as e:
            print(f"Error adding power measurements: {str(e)}")
            traceback.print_exc()
        
        # Add voltage measurements from smart meters (end points)
        # These would typically come from AMI systems
        print("Adding voltage measurements from smart meters...")
        try:
            # Use actual nodes from the IEEE 34 bus system instead of generic endnode names
            dsse.add_voltage_measurement("848.1", 0.97, 0.0004, phase=1)  # Using actual node from IEEE 34 bus
            dsse.add_voltage_measurement("848.2", 0.96, 0.0004, phase=2)  # Using actual node from IEEE 34 bus
            print("Smart meter measurements added successfully")
        except Exception as e:
            print(f"Error adding smart meter measurements: {str(e)}")
            traceback.print_exc()
        
        # Add a PMU measurement (if available)
        # dsse.add_pmu_measurement("midpoint.1", 0.98, 0.1, 0.0001, 0.0001, phase=1)
        
        # Add a bad measurement to demonstrate bad data detection
        # This is deliberately incorrect to trigger bad data detection
        print("Adding bad measurement...")
        try:
            # Use an actual node from the IEEE 34 bus system
            dsse.add_voltage_measurement("830.3", 1.2, 0.0004, phase=3)  # Bad measurement using actual node
            print("Bad measurement added successfully")
        except Exception as e:
            print(f"Error adding bad measurement: {str(e)}")
            traceback.print_exc()
        
        # Add pseudo-measurements for loads without real-time measurements
        print("Adding pseudo-measurements...")
        try:
            dsse.add_pseudo_measurements(error_variance=0.5)
            print("Pseudo-measurements added successfully")
        except Exception as e:
            print(f"Error adding pseudo-measurements: {str(e)}")
            traceback.print_exc()
        
        print(f"\n{'='*40}")
        print("Running state estimation...")
        print(f"{'='*40}\n")
        
        # Run state estimation
        print("Running state estimation...")
        try:
            estimated_state, obj_value, performance = dsse.run_state_estimation(max_iter=20, tolerance=1e-5)
            print(f"State estimation completed successfully with objective value: {obj_value}")
        except Exception as e:
            print(f"Error running state estimation: {str(e)}")
            traceback.print_exc()
        
        # Bad data detection
        print(f"\n{'='*40}")
        print("Performing bad data detection...")
        print(f"{'='*40}\n")
        
        print("Performing bad data detection...")
        try:
            bad_data_indices, detection_stats = dsse.bad_data_detection(confidence_level=0.95)
            print(f"Bad data detection completed with {len(bad_data_indices)} bad measurements found")
        except Exception as e:
            print(f"Error in bad data detection: {str(e)}")
            traceback.print_exc()
        
        if bad_data_indices:
            print(f"Detected {len(bad_data_indices)} bad measurements")
            
            # Remove bad data
            print("Removing bad measurements...")
            try:
                dsse.remove_bad_measurements(bad_data_indices)
                print("Bad measurements removed successfully")
                
                # Run state estimation again after removing bad data
                print(f"\n{'='*40}")
                print("Re-running state estimation after removing bad data...")
                print(f"{'='*40}\n")
                
                estimated_state, obj_value, performance = dsse.run_state_estimation(max_iter=20, tolerance=1e-5)
                print(f"Re-run of state estimation completed with objective value: {obj_value}")
            except Exception as e:
                print(f"Error removing bad data or re-running state estimation: {str(e)}")
                traceback.print_exc()
        
        # Generate and display report
        print(f"\n{'='*40}")
        print("Generating report...")
        print(f"{'='*40}\n")
        
        print("Generating report...")
        try:
            dsse.generate_report(output_dir)
            print(f"Report generated successfully and saved to {output_dir}")
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            traceback.print_exc()
        
        # Plot voltage profile
        print("Plotting voltage profile...")
        try:
            dsse.plot_voltage_profile(os.path.join(output_dir, "voltage_profile.png"))
            print("Voltage profile plot created successfully")
        except Exception as e:
            print(f"Error plotting voltage profile: {str(e)}")
            traceback.print_exc()
        
        # Plot measurement residuals
        print("Plotting measurement residuals...")
        try:
            dsse.plot_measurement_residuals(os.path.join(output_dir, "measurement_residuals.png"))
            print("Measurement residuals plot created successfully")
        except Exception as e:
            print(f"Error plotting measurement residuals: {str(e)}")
            traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f"Distribution State Estimation completed successfully")
        print(f"Results saved to {os.path.abspath(output_dir)}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"Error running distribution state estimation: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
