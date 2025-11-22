"""
Data Preprocessing and Exploration for Glucose Time Series Dataset

This module provides functions for:
- Loading and exploring glucose time series data
- Computing summary statistics
- Analyzing patient-level information
- Data cleaning and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class GlucoseDataExplorer:
    """
    A class for exploring and preprocessing glucose time series data.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the GlucoseDataExplorer.
        
        Args:
            data_path: Path to the glucose data file (CSV or other format)
        """
        self.data_path = data_path
        self.data = None
        self.patient_id_col = None
        self.timestamp_col = None
        self.glucose_col = None
        
    def load_data(self, data_path: Optional[str] = None, 
                  patient_id_col: str = 'patient_id',
                  timestamp_col: str = 'timestamp',
                  glucose_col: str = 'glucose') -> pd.DataFrame:
        """
        Load glucose time series data from a file.
        
        Args:
            data_path: Path to the data file
            patient_id_col: Name of the patient ID column
            timestamp_col: Name of the timestamp column
            glucose_col: Name of the glucose value column
            
        Returns:
            Loaded DataFrame
        """
        if data_path is None:
            data_path = self.data_path
            
        if data_path is None:
            raise ValueError("No data path provided")
            
        # Load data based on file extension
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            self.data = pd.read_excel(data_path)
        elif data_path.endswith('.json'):
            self.data = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
            
        self.patient_id_col = patient_id_col
        self.timestamp_col = timestamp_col
        self.glucose_col = glucose_col
        
        # Convert timestamp to datetime if not already
        if timestamp_col in self.data.columns:
            self.data[timestamp_col] = pd.to_datetime(self.data[timestamp_col])
            
        print(f"Data loaded successfully from {data_path}")
        print(f"Shape: {self.data.shape}")
        
        return self.data
    
    def get_summary_statistics(self) -> Dict:
        """
        Calculate comprehensive summary statistics for the glucose data.
        
        Returns:
            Dictionary containing various summary statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        summary = {}
        
        # Basic dataset info
        summary['total_records'] = len(self.data)
        summary['total_columns'] = len(self.data.columns)
        summary['columns'] = list(self.data.columns)
        
        # Glucose statistics (if glucose column exists)
        if self.glucose_col and self.glucose_col in self.data.columns:
            glucose_data = self.data[self.glucose_col].dropna()
            summary['glucose_stats'] = {
                'mean': glucose_data.mean(),
                'median': glucose_data.median(),
                'std': glucose_data.std(),
                'min': glucose_data.min(),
                'max': glucose_data.max(),
                'q25': glucose_data.quantile(0.25),
                'q75': glucose_data.quantile(0.75),
                'missing_values': self.data[self.glucose_col].isna().sum(),
                'missing_percentage': (self.data[self.glucose_col].isna().sum() / len(self.data)) * 100
            }
        
        # Missing values summary
        summary['missing_values'] = self.data.isna().sum().to_dict()
        
        # Data types
        summary['data_types'] = self.data.dtypes.astype(str).to_dict()
        
        return summary
    
    def get_unique_patients(self) -> int:
        """
        Get the total number of unique patients in the dataset.
        
        Returns:
            Number of unique patients
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        if self.patient_id_col not in self.data.columns:
            raise ValueError(f"Patient ID column '{self.patient_id_col}' not found in data")
            
        unique_patients = self.data[self.patient_id_col].nunique()
        print(f"Total unique patients: {unique_patients}")
        
        return unique_patients
    
    def get_total_recordings(self) -> int:
        """
        Get the total number of glucose recordings in the dataset.
        
        Returns:
            Total number of recordings
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        total_recordings = len(self.data)
        print(f"Total recordings: {total_recordings}")
        
        return total_recordings
    
    def get_recordings_per_patient(self) -> pd.Series:
        """
        Get the number of recordings for each patient.
        
        Returns:
            Series with patient IDs as index and recording counts as values
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        if self.patient_id_col not in self.data.columns:
            raise ValueError(f"Patient ID column '{self.patient_id_col}' not found in data")
            
        recordings_per_patient = self.data[self.patient_id_col].value_counts().sort_index()
        
        return recordings_per_patient
    
    def get_time_duration_per_patient(self) -> pd.DataFrame:
        """
        Calculate the total time duration for each patient.
        
        Returns:
            DataFrame with patient ID, start time, end time, and duration
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        if self.patient_id_col not in self.data.columns:
            raise ValueError(f"Patient ID column '{self.patient_id_col}' not found in data")
            
        if self.timestamp_col not in self.data.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' not found in data")
            
        # Group by patient and calculate time range
        patient_time_info = self.data.groupby(self.patient_id_col)[self.timestamp_col].agg([
            ('start_time', 'min'),
            ('end_time', 'max')
        ]).reset_index()
        
        # Calculate duration
        patient_time_info['duration'] = patient_time_info['end_time'] - patient_time_info['start_time']
        patient_time_info['duration_days'] = patient_time_info['duration'].dt.total_seconds() / (24 * 3600)
        patient_time_info['duration_hours'] = patient_time_info['duration'].dt.total_seconds() / 3600
        
        return patient_time_info
    
    def generate_exploration_report(self) -> str:
        """
        Generate a comprehensive exploration report.
        
        Returns:
            Formatted string containing the exploration report
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        report = []
        report.append("=" * 80)
        report.append("GLUCOSE TIME SERIES DATA EXPLORATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Basic info
        report.append("1. DATASET OVERVIEW")
        report.append("-" * 80)
        report.append(f"Total Records: {len(self.data):,}")
        report.append(f"Total Columns: {len(self.data.columns)}")
        report.append(f"Columns: {', '.join(self.data.columns)}")
        report.append("")
        
        # Patient info
        report.append("2. PATIENT INFORMATION")
        report.append("-" * 80)
        if self.patient_id_col in self.data.columns:
            unique_patients = self.get_unique_patients()
            recordings_per_patient = self.get_recordings_per_patient()
            
            report.append(f"Total Unique Patients: {unique_patients}")
            report.append(f"Average Recordings per Patient: {recordings_per_patient.mean():.2f}")
            report.append(f"Median Recordings per Patient: {recordings_per_patient.median():.2f}")
            report.append(f"Min Recordings per Patient: {recordings_per_patient.min()}")
            report.append(f"Max Recordings per Patient: {recordings_per_patient.max()}")
        else:
            report.append(f"Patient ID column '{self.patient_id_col}' not found")
        report.append("")
        
        # Recording info
        report.append("3. RECORDING STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Recordings: {self.get_total_recordings():,}")
        report.append("")
        
        # Time duration info
        report.append("4. TIME DURATION PER PATIENT")
        report.append("-" * 80)
        if self.timestamp_col in self.data.columns and self.patient_id_col in self.data.columns:
            time_info = self.get_time_duration_per_patient()
            
            report.append(f"Average Duration: {time_info['duration_days'].mean():.2f} days ({time_info['duration_hours'].mean():.2f} hours)")
            report.append(f"Median Duration: {time_info['duration_days'].median():.2f} days ({time_info['duration_hours'].median():.2f} hours)")
            report.append(f"Min Duration: {time_info['duration_days'].min():.2f} days ({time_info['duration_hours'].min():.2f} hours)")
            report.append(f"Max Duration: {time_info['duration_days'].max():.2f} days ({time_info['duration_hours'].max():.2f} hours)")
            report.append("")
            report.append("Per Patient Duration:")
            for _, row in time_info.iterrows():
                report.append(f"  Patient {row[self.patient_id_col]}: {row['duration_days']:.2f} days ({row['duration_hours']:.2f} hours)")
        else:
            report.append("Timestamp or Patient ID column not found")
        report.append("")
        
        # Glucose statistics
        report.append("5. GLUCOSE STATISTICS")
        report.append("-" * 80)
        if self.glucose_col and self.glucose_col in self.data.columns:
            glucose_data = self.data[self.glucose_col].dropna()
            report.append(f"Mean: {glucose_data.mean():.2f}")
            report.append(f"Median: {glucose_data.median():.2f}")
            report.append(f"Std Dev: {glucose_data.std():.2f}")
            report.append(f"Min: {glucose_data.min():.2f}")
            report.append(f"Max: {glucose_data.max():.2f}")
            report.append(f"Q25: {glucose_data.quantile(0.25):.2f}")
            report.append(f"Q75: {glucose_data.quantile(0.75):.2f}")
            report.append(f"Missing Values: {self.data[self.glucose_col].isna().sum()} ({(self.data[self.glucose_col].isna().sum()/len(self.data)*100):.2f}%)")
        else:
            report.append(f"Glucose column '{self.glucose_col}' not found")
        report.append("")
        
        # Missing values
        report.append("6. MISSING VALUES SUMMARY")
        report.append("-" * 80)
        missing = self.data.isna().sum()
        if missing.sum() > 0:
            for col, count in missing[missing > 0].items():
                percentage = (count / len(self.data)) * 100
                report.append(f"{col}: {count} ({percentage:.2f}%)")
        else:
            report.append("No missing values found")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def print_exploration_report(self):
        """
        Print the exploration report to console.
        """
        print(self.generate_exploration_report())
    
    def save_exploration_report(self, output_path: str):
        """
        Save the exploration report to a file.
        
        Args:
            output_path: Path to save the report
        """
        report = self.generate_exploration_report()
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_path}")


def explore_glucose_data(data_path: str,
                        patient_id_col: str = 'patient_id',
                        timestamp_col: str = 'timestamp',
                        glucose_col: str = 'glucose',
                        save_report: bool = False,
                        report_path: str = 'exploration_report.txt') -> GlucoseDataExplorer:
    """
    Convenience function to explore glucose data in one call.
    
    Args:
        data_path: Path to the glucose data file
        patient_id_col: Name of the patient ID column
        timestamp_col: Name of the timestamp column
        glucose_col: Name of the glucose value column
        save_report: Whether to save the report to a file
        report_path: Path to save the report
        
    Returns:
        GlucoseDataExplorer instance with loaded data
    """
    explorer = GlucoseDataExplorer(data_path)
    explorer.load_data(patient_id_col=patient_id_col,
                      timestamp_col=timestamp_col,
                      glucose_col=glucose_col)
    
    explorer.print_exploration_report()
    
    if save_report:
        explorer.save_exploration_report(report_path)
    
    return explorer


if __name__ == "__main__":
    """
    Example usage of the GlucoseDataExplorer class.
    """
    print("Glucose Time Series Data Preprocessing Module")
    print("=" * 80)
    print("\nThis module provides tools for exploring glucose time series data.")
    print("\nExample usage:")
    print("-" * 80)
    print("""
# Load and explore data
from data_preprocessing import GlucoseDataExplorer, explore_glucose_data

# Method 1: Using the convenience function
explorer = explore_glucose_data(
    data_path='glucose_data.csv',
    patient_id_col='patient_id',
    timestamp_col='timestamp',
    glucose_col='glucose',
    save_report=True
)

# Method 2: Using the class directly
explorer = GlucoseDataExplorer('glucose_data.csv')
explorer.load_data(patient_id_col='patient_id', 
                   timestamp_col='timestamp', 
                   glucose_col='glucose')

# Get summary statistics
stats = explorer.get_summary_statistics()
print(stats)

# Get unique patients
unique_patients = explorer.get_unique_patients()

# Get total recordings
total_recordings = explorer.get_total_recordings()

# Get recordings per patient
recordings_per_patient = explorer.get_recordings_per_patient()

# Get time duration per patient
time_duration = explorer.get_time_duration_per_patient()
print(time_duration)

# Generate and print report
explorer.print_exploration_report()

# Save report to file
explorer.save_exploration_report('glucose_exploration_report.txt')
    """)
    print("-" * 80)
    print("\nNote: Make sure your data file has columns for patient_id, timestamp, and glucose values.")
    print("The column names can be customized when loading the data.")
