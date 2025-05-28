import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import os
import shutil
import tempfile
import atexit

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.acc_data = None
        self.gyro_data = None
        self.subject_info = None
        self.temp_dir = None
        self._cleanup_registered = False
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {self.temp_dir}")

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            print("\nCleaning up temporary files...")
            shutil.rmtree(self.temp_dir)
            print("Cleanup complete!")

    def register_cleanup(self):
        """Register cleanup only after data is loaded."""
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
            print("Cleanup registered for program exit")

    def extract_zip_to_temp(self, zip_path):
        """Extract zip file to temporary directory."""
        # Create a subdirectory in temp for this zip
        extract_dir = Path(self.temp_dir) / zip_path.stem
        os.makedirs(extract_dir, exist_ok=True)
        
        print(f"Extracting {zip_path.name} to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print(f"Extraction complete for {zip_path.name}")
        return extract_dir / zip_path.stem  # Return path including the inner directory

    def load_csv_safely(self, file_path):
        """Load CSV file with error handling and proper numeric conversion."""
        try:
            # Read CSV with header row and skip it
            df = pd.read_csv(file_path, skiprows=1, names=['x', 'y', 'z'])
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None

    def load_data(self):
        """Load accelerometer and gyroscope data."""
        # Extract zip files to temporary directory
        acc_base_path = self.extract_zip_to_temp(self.data_dir / 'B_Accelerometer_data.zip')
        gyro_base_path = self.extract_zip_to_temp(self.data_dir / 'C_Gyroscope_data.zip')

        # Load subject info if available
        subject_info_path = self.data_dir / 'data_subjects_info.csv'
        if subject_info_path.exists():
            self.subject_info = pd.read_csv(subject_info_path)

        # Initialize data dictionaries
        acc_data_dict = {}
        gyro_data_dict = {}

        # Activities to process
        base_activities = ['dws', 'ups', 'sit', 'std', 'wlk', 'jog']

        # Load accelerometer data
        print("Loading accelerometer data...")
        for base_activity in base_activities:
            # Look for numbered activity folders (e.g., dws_1, dws_2)
            activity_folders = list(acc_base_path.glob(f'{base_activity}_*'))
            
            for activity_folder in activity_folders:
                # Process each subject file in the activity folder
                subject_files = list(activity_folder.glob('sub_*.csv'))
                for subject_file in subject_files:
                    subject_id = int(subject_file.stem.split('_')[1])
                    df = self.load_csv_safely(subject_file)
                    if df is not None:
                        # Add row index as measurement_id for synchronization
                        df['measurement_id'] = range(len(df))
                        df['subject_id'] = subject_id
                        df['activity'] = base_activity
                        acc_data_dict[f"{base_activity}_{subject_id}_{activity_folder.name}"] = df

        # Load gyroscope data
        print("Loading gyroscope data...")
        for base_activity in base_activities:
            # Look for numbered activity folders
            activity_folders = list(gyro_base_path.glob(f'{base_activity}_*'))
            
            for activity_folder in activity_folders:
                # Process each subject file in the activity folder
                subject_files = list(activity_folder.glob('sub_*.csv'))
                for subject_file in subject_files:
                    subject_id = int(subject_file.stem.split('_')[1])
                    df = self.load_csv_safely(subject_file)
                    if df is not None:
                        # Add row index as measurement_id for synchronization
                        df['measurement_id'] = range(len(df))
                        df['subject_id'] = subject_id
                        df['activity'] = base_activity
                        gyro_data_dict[f"{base_activity}_{subject_id}_{activity_folder.name}"] = df

        # Combine all data
        if acc_data_dict:
            self.acc_data = pd.concat(acc_data_dict.values(), ignore_index=True)
        else:
            raise ValueError("No accelerometer data found in the zip file")

        if gyro_data_dict:
            self.gyro_data = pd.concat(gyro_data_dict.values(), ignore_index=True)
        else:
            raise ValueError("No gyroscope data found in the zip file")

        print(f"Loaded data for {len(set(self.acc_data['subject_id']))} subjects")
        print(f"Activities found: {sorted(set(self.acc_data['activity']))}")

        return self.acc_data, self.gyro_data

    def merge_sensor_data(self):
        """Merge accelerometer and gyroscope data based on measurement_id and subject_id."""
        if self.acc_data is None or self.gyro_data is None:
            self.load_data()

        print("Merging sensor data...")
        # Merge based on measurement_id and subject_id
        merged_data = pd.merge(
            self.acc_data,
            self.gyro_data,
            on=['measurement_id', 'subject_id', 'activity'],
            suffixes=('_acc', '_gyro')
        )
        
        # Register cleanup only after we're done with the data
        self.register_cleanup()
        return merged_data

    def normalize_per_subject(self, data):
        """Apply z-score normalization per subject."""
        print("Normalizing data per subject...")
        normalized_data = data.copy()
        sensor_cols = [col for col in data.columns if col.startswith(('x', 'y', 'z'))]
        
        for subject in data['subject_id'].unique():
            mask = data['subject_id'] == subject
            subject_data = data.loc[mask, sensor_cols]
            
            # Check for numeric data and handle errors
            if not subject_data.apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().all().all():
                print(f"Warning: Non-numeric values found for subject {subject}")
                continue
                
            normalized_data.loc[mask, sensor_cols] = (
                subject_data - subject_data.mean()
            ) / subject_data.std()
            
        return normalized_data 