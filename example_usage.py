"""
Example script demonstrating how to use the data_preprocessing module.

This script creates sample glucose time series data and demonstrates
all the exploration features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_preprocessing import GlucoseDataExplorer, explore_glucose_data


def create_sample_glucose_data(output_file: str = 'sample_glucose_data.csv'):
    """
    Create a sample glucose time series dataset for demonstration.
    
    Args:
        output_file: Path to save the sample data
    """
    np.random.seed(42)
    
    # Generate sample data for 5 patients
    patients = []
    
    for patient_id in range(1, 6):
        # Each patient has different number of readings and duration
        num_readings = np.random.randint(50, 200)
        start_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 30))
        
        # Generate timestamps (5-minute intervals with some gaps)
        timestamps = []
        current_time = start_date
        for _ in range(num_readings):
            timestamps.append(current_time)
            # Add 5-15 minutes between readings
            current_time += timedelta(minutes=np.random.randint(5, 16))
        
        # Generate glucose values (typical range 70-180 mg/dL with some variation)
        base_glucose = np.random.uniform(90, 120)
        glucose_values = base_glucose + np.random.normal(0, 20, num_readings)
        glucose_values = np.clip(glucose_values, 50, 300)  # Clip to realistic range
        
        # Add some missing values randomly (2-5%)
        missing_indices = np.random.choice(num_readings, size=int(num_readings * 0.03), replace=False)
        glucose_values[missing_indices] = np.nan
        
        # Create patient data
        patient_data = pd.DataFrame({
            'patient_id': patient_id,
            'timestamp': timestamps,
            'glucose': glucose_values,
            'age': np.random.randint(25, 75),
            'gender': np.random.choice(['M', 'F'])
        })
        
        patients.append(patient_data)
    
    # Combine all patient data
    all_data = pd.concat(patients, ignore_index=True)
    
    # Save to CSV
    all_data.to_csv(output_file, index=False)
    print(f"Sample data created and saved to {output_file}")
    print(f"Total records: {len(all_data)}")
    
    return all_data


def main():
    """
    Main function demonstrating the data preprocessing module.
    """
    print("=" * 80)
    print("GLUCOSE TIME SERIES DATA PREPROCESSING - DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Step 1: Create sample data
    print("Step 1: Creating sample glucose time series data...")
    print("-" * 80)
    sample_data = create_sample_glucose_data('sample_glucose_data.csv')
    print()
    
    # Step 2: Load and explore the data using the convenience function
    print("Step 2: Exploring the data using convenience function...")
    print("-" * 80)
    explorer = explore_glucose_data(
        data_path='sample_glucose_data.csv',
        patient_id_col='patient_id',
        timestamp_col='timestamp',
        glucose_col='glucose',
        save_report=True,
        report_path='glucose_exploration_report.txt'
    )
    print()
    
    # Step 3: Demonstrate individual functions
    print("Step 3: Demonstrating individual analysis functions...")
    print("-" * 80)
    print()
    
    # Get summary statistics
    print("Summary Statistics:")
    print("-" * 40)
    stats = explorer.get_summary_statistics()
    print(f"Total records: {stats['total_records']}")
    if 'glucose_stats' in stats:
        print(f"Glucose mean: {stats['glucose_stats']['mean']:.2f}")
        print(f"Glucose std: {stats['glucose_stats']['std']:.2f}")
        print(f"Glucose range: [{stats['glucose_stats']['min']:.2f}, {stats['glucose_stats']['max']:.2f}]")
    print()
    
    # Get unique patients
    print("Patient Analysis:")
    print("-" * 40)
    unique_patients = explorer.get_unique_patients()
    print()
    
    # Get recordings per patient
    print("Recordings per Patient:")
    print("-" * 40)
    recordings_per_patient = explorer.get_recordings_per_patient()
    print(recordings_per_patient)
    print()
    
    # Get time duration per patient
    print("Time Duration per Patient:")
    print("-" * 40)
    time_duration = explorer.get_time_duration_per_patient()
    print(time_duration[['patient_id', 'duration_days', 'duration_hours']])
    print()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Files created:")
    print("  - sample_glucose_data.csv: Sample glucose time series data")
    print("  - glucose_exploration_report.txt: Detailed exploration report")
    print()
    print("You can now use the data_preprocessing module with your own data!")
    print()


if __name__ == "__main__":
    main()
