"""
Data processing module for exoplanet detection
Handles loading, cleaning, and preprocessing of NASA datasets (Kepler, K2, TESS)
Standardizes diverse column formats into uniform feature set for ML models
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ExoplanetDataProcessor:
    """
    Standardizes NASA exoplanet datasets (Kepler/K2/TESS) into uniform format
    
    Standard Features:
    - orbital_period: Orbital period in days
    - transit_duration: Transit duration in hours
    - planetary_radius: Planet radius in Earth radii
    - stellar_magnitude: Visual/TESS magnitude
    - transit_depth: Transit depth in parts per million (ppm)
    - impact_parameter: Impact parameter (0-1)
    - equilibrium_temperature: Equilibrium temperature in Kelvin
    - stellar_radius: Stellar radius in solar radii
    - stellar_mass: Stellar mass in solar masses
    - disposition: CONFIRMED, CANDIDATE, FALSE_POSITIVE, or NO_PREDICT
    
    Disposition Classes:
    - CONFIRMED: Confirmed exoplanet
    - CANDIDATE: Planet candidate (awaiting confirmation)
    - FALSE_POSITIVE: False positive detection
    - NO_PREDICT: Insufficient data for prediction (used in ML inference)
    """
    
    # Column mapping for Kepler cumulative dataset
    KEPLER_MAPPING = {
        'orbital_period': 'koi_period',
        'transit_duration': 'koi_duration',
        'planetary_radius': 'koi_prad',
        'stellar_magnitude': 'koi_kepmag',
        'transit_depth': 'koi_depth',
        'impact_parameter': 'koi_impact',
        'equilibrium_temperature': 'koi_teq',
        'stellar_radius': 'koi_srad',
        'stellar_mass': 'koi_smass',
        'disposition_raw': 'koi_disposition',
        'disposition_score': 'koi_score',
        'object_id': 'kepoi_name',
        'dataset': lambda: 'KEPLER'
    }
    
    # Column mapping for K2 dataset
    K2_MAPPING = {
        'orbital_period': 'pl_orbper',
        'transit_duration': None,  # Not directly available
        'planetary_radius': 'pl_rade',
        'stellar_magnitude': None,  # Use computed magnitude
        'transit_depth': None,  # Not available
        'impact_parameter': None,  # Not available
        'equilibrium_temperature': 'pl_eqt',
        'stellar_radius': 'st_rad',
        'stellar_mass': 'st_mass',
        'disposition_raw': 'disposition',
        'disposition_score': None,  # Not available in K2
        'object_id': 'pl_name',
        'dataset': lambda: 'K2'
    }
    
    # Column mapping for TESS (TOI) dataset
    TESS_MAPPING = {
        'orbital_period': 'pl_orbper',
        'transit_duration': 'pl_trandurh',
        'planetary_radius': 'pl_rade',
        'stellar_magnitude': 'st_tmag',
        'transit_depth': 'pl_trandep',
        'impact_parameter': None,  # Not available
        'equilibrium_temperature': 'pl_eqt',
        'stellar_radius': 'st_rad',
        'stellar_mass': None,  # Not available
        'disposition_raw': 'tfopwg_disp',
        'disposition_score': None,  # Not available in TESS
        'object_id': 'toi',
        'dataset': lambda: 'TESS'
    }
    
    def __init__(self, data_dir: str = '../data'):
        """
        Initialize data processor
        
        Args:
            data_dir: Base directory containing raw/ and processed/ folders
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_raw_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load NASA CSV file, skipping comment lines
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with raw data
        """
        print(f"Loading {os.path.basename(filepath)}...")
        
        # Skip comment lines starting with #
        df = pd.read_csv(filepath, comment='#', low_memory=False)
        
        print(f"  ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    def standardize_disposition(self, disposition: str, dataset: str) -> str:
        """
        Standardize disposition labels across datasets
        
        Args:
            disposition: Raw disposition string
            dataset: Dataset name (KEPLER, K2, TESS)
            
        Returns:
            Standardized label: CONFIRMED, CANDIDATE, FALSE_POSITIVE, or UNKNOWN
            
        Note:
            NO_PREDICT is used during ML inference when insufficient data is provided.
            It is not used in training data.
        """
        if pd.isna(disposition):
            return 'UNKNOWN'
        
        disposition = str(disposition).upper().strip()
        
        # Kepler dispositions
        if dataset == 'KEPLER':
            if disposition == 'CONFIRMED':
                return 'CONFIRMED'
            elif disposition == 'CANDIDATE':
                return 'CANDIDATE'
            elif disposition in ['FALSE POSITIVE', 'FALSE_POSITIVE']:
                return 'FALSE_POSITIVE'
        
        # K2 dispositions
        elif dataset == 'K2':
            if disposition == 'CONFIRMED':
                return 'CONFIRMED'
            elif disposition == 'CANDIDATE':
                return 'CANDIDATE'
            elif disposition in ['FALSE POSITIVE', 'FALSE_POSITIVE']:
                return 'FALSE_POSITIVE'
        
        # TESS dispositions (TFOPWG codes)
        elif dataset == 'TESS':
            if disposition == 'CP':  # Confirmed Planet
                return 'CONFIRMED'
            elif disposition in ['PC', 'KP']:  # Planet Candidate, Known Planet
                return 'CANDIDATE'
            elif disposition == 'FP':  # False Positive
                return 'FALSE_POSITIVE'
        
        return 'UNKNOWN'
    
    def process_kepler_data(self, filepath: str) -> pd.DataFrame:
        """
        Process Kepler cumulative dataset
        
        Args:
            filepath: Path to Kepler CSV
            
        Returns:
            Standardized DataFrame
        """
        print("\n" + "="*60)
        print("PROCESSING KEPLER CUMULATIVE DATASET")
        print("="*60)
        
        df = self.load_raw_csv(filepath)
        
        # Create standardized dataframe
        standardized = pd.DataFrame()
        
        for std_col, raw_col in self.KEPLER_MAPPING.items():
            if callable(raw_col):
                standardized[std_col] = raw_col()
            elif raw_col is None:
                standardized[std_col] = np.nan
            elif raw_col in df.columns:
                standardized[std_col] = df[raw_col]
            else:
                standardized[std_col] = np.nan
        
        # Standardize disposition
        standardized['disposition'] = standardized['disposition_raw'].apply(
            lambda x: self.standardize_disposition(x, 'KEPLER')
        )
        standardized = standardized.drop('disposition_raw', axis=1)
        
        print(f"\n  Disposition distribution:")
        print(standardized['disposition'].value_counts().to_string())
        
        return standardized
    
    def process_k2_data(self, filepath: str) -> pd.DataFrame:
        """
        Process K2 dataset
        
        Args:
            filepath: Path to K2 CSV
            
        Returns:
            Standardized DataFrame
        """
        print("\n" + "="*60)
        print("PROCESSING K2 DATASET")
        print("="*60)
        
        df = self.load_raw_csv(filepath)
        
        # Create standardized dataframe
        standardized = pd.DataFrame()
        
        for std_col, raw_col in self.K2_MAPPING.items():
            if callable(raw_col):
                standardized[std_col] = raw_col()
            elif raw_col is None:
                standardized[std_col] = np.nan
            elif raw_col in df.columns:
                standardized[std_col] = df[raw_col]
            else:
                standardized[std_col] = np.nan
        
        # Standardize disposition
        standardized['disposition'] = standardized['disposition_raw'].apply(
            lambda x: self.standardize_disposition(x, 'K2')
        )
        standardized = standardized.drop('disposition_raw', axis=1)
        
        print(f"\n  Disposition distribution:")
        print(standardized['disposition'].value_counts().to_string())
        
        return standardized
    
    def process_tess_data(self, filepath: str) -> pd.DataFrame:
        """
        Process TESS (TOI) dataset
        
        Args:
            filepath: Path to TESS CSV
            
        Returns:
            Standardized DataFrame
        """
        print("\n" + "="*60)
        print("PROCESSING TESS (TOI) DATASET")
        print("="*60)
        
        df = self.load_raw_csv(filepath)
        
        # Create standardized dataframe
        standardized = pd.DataFrame()
        
        for std_col, raw_col in self.TESS_MAPPING.items():
            if callable(raw_col):
                standardized[std_col] = raw_col()
            elif raw_col is None:
                standardized[std_col] = np.nan
            elif raw_col in df.columns:
                standardized[std_col] = df[raw_col]
            else:
                standardized[std_col] = np.nan
        
        # Standardize disposition
        standardized['disposition'] = standardized['disposition_raw'].apply(
            lambda x: self.standardize_disposition(x, 'TESS')
        )
        standardized = standardized.drop('disposition_raw', axis=1)
        
        print(f"\n  Disposition distribution:")
        print(standardized['disposition'].value_counts().to_string())
        
        return standardized
    
    def clean_data(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Clean and validate standardized data
        
        Args:
            df: Standardized DataFrame
            dataset_name: Name for reporting
            
        Returns:
            Cleaned DataFrame
        """
        print(f"\n  Cleaning {dataset_name} data...")
        
        initial_count = len(df)
        
        # Remove rows with UNKNOWN disposition
        df = df[df['disposition'] != 'UNKNOWN'].copy()
        print(f"    - Removed {initial_count - len(df)} rows with UNKNOWN disposition")
        
        # Remove duplicate object IDs
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['object_id'], keep='first')
        print(f"    - Removed {before_dedup - len(df)} duplicate objects")
        
        # Convert numeric columns to proper types
        numeric_cols = [
            'orbital_period', 'transit_duration', 'planetary_radius',
            'stellar_magnitude', 'transit_depth', 'impact_parameter',
            'equilibrium_temperature', 'stellar_radius', 'stellar_mass'
        ]
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove physically impossible values
        if 'orbital_period' in df.columns:
            df = df[(df['orbital_period'].isna()) | (df['orbital_period'] > 0)]
        if 'planetary_radius' in df.columns:
            df = df[(df['planetary_radius'].isna()) | (df['planetary_radius'] > 0)]
        if 'stellar_radius' in df.columns:
            df = df[(df['stellar_radius'].isna()) | (df['stellar_radius'] > 0)]
        if 'impact_parameter' in df.columns:
            df = df[(df['impact_parameter'].isna()) | 
                   ((df['impact_parameter'] >= 0) & (df['impact_parameter'] <= 2))]
        
        print(f"    ✓ Final count: {len(df):,} rows")
        
        return df
    
    def generate_summary_report(self, df: pd.DataFrame, dataset_name: str) -> str:
        """
        Generate summary statistics for dataset
        
        Args:
            df: DataFrame to summarize
            dataset_name: Dataset name
            
        Returns:
            Summary report string
        """
        report = f"\n{'='*60}\n"
        report += f"SUMMARY: {dataset_name}\n"
        report += f"{'='*60}\n\n"
        
        report += f"Total Records: {len(df):,}\n\n"
        
        # Disposition breakdown
        report += "Disposition Distribution:\n"
        disp_counts = df['disposition'].value_counts()
        for disp, count in disp_counts.items():
            pct = (count / len(df)) * 100
            report += f"  {disp:20s}: {count:6,} ({pct:5.1f}%)\n"
        
        # Feature completeness
        report += "\nFeature Completeness (% non-null):\n"
        feature_cols = [
            'orbital_period', 'transit_duration', 'planetary_radius',
            'stellar_magnitude', 'transit_depth', 'impact_parameter',
            'equilibrium_temperature', 'stellar_radius', 'stellar_mass',
            'disposition_score'
        ]
        
        for col in feature_cols:
            if col in df.columns:
                completeness = (df[col].notna().sum() / len(df)) * 100
                report += f"  {col:25s}: {completeness:5.1f}%\n"
        
        # Feature statistics (for non-null values)
        report += "\nFeature Statistics (non-null values):\n"
        report += f"{'Feature':<25s} {'Min':>12s} {'Mean':>12s} {'Max':>12s} {'Std':>12s}\n"
        report += "-" * 73 + "\n"
        
        for col in feature_cols:
            if col in df.columns and df[col].notna().any():
                stats = df[col].describe()
                report += f"{col:<25s} {stats['min']:12.2e} {stats['mean']:12.2e} {stats['max']:12.2e} {stats['std']:12.2e}\n"
        
        return report
    
    def process_all_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process all three datasets and save standardized versions
        
        Returns:
            Tuple of (kepler_df, k2_df, tess_df)
        """
        print("\n" + "█"*60)
        print("█" + " "*58 + "█")
        print("█" + "  EXOPLANET DATA STANDARDIZATION PIPELINE".center(58) + "█")
        print("█" + " "*58 + "█")
        print("█"*60)
        
        # Process Kepler
        kepler_path = os.path.join(self.raw_dir, 'cumulative_2025.09.18_13.24.09.csv')
        kepler_df = self.process_kepler_data(kepler_path)
        kepler_df = self.clean_data(kepler_df, 'KEPLER')
        
        # Process K2
        k2_path = os.path.join(self.raw_dir, 'k2pandc_2025.09.18_13.24.20.csv')
        k2_df = self.process_k2_data(k2_path)
        k2_df = self.clean_data(k2_df, 'K2')
        
        # Process TESS
        tess_path = os.path.join(self.raw_dir, 'TOI_2025.09.18_13.24.15.csv')
        tess_df = self.process_tess_data(tess_path)
        tess_df = self.clean_data(tess_df, 'TESS')
        
        # Save standardized datasets
        print("\n" + "="*60)
        print("SAVING STANDARDIZED DATASETS")
        print("="*60)
        
        kepler_output = os.path.join(self.processed_dir, 'kepler_standardized.csv')
        kepler_df.to_csv(kepler_output, index=False)
        print(f"  ✓ Saved: {kepler_output}")
        
        k2_output = os.path.join(self.processed_dir, 'k2_standardized.csv')
        k2_df.to_csv(k2_output, index=False)
        print(f"  ✓ Saved: {k2_output}")
        
        tess_output = os.path.join(self.processed_dir, 'tess_standardized.csv')
        tess_df.to_csv(tess_output, index=False)
        print(f"  ✓ Saved: {tess_output}")
        
        # Generate reports
        print("\n" + "="*60)
        print("GENERATING SUMMARY REPORTS")
        print("="*60)
        
        report = ""
        report += self.generate_summary_report(kepler_df, "KEPLER CUMULATIVE")
        report += self.generate_summary_report(k2_df, "K2 PLANETS & CANDIDATES")
        report += self.generate_summary_report(tess_df, "TESS TOI")
        
        # Save report
        report_path = os.path.join(self.processed_dir, 'standardization_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"  ✓ Saved: {report_path}")
        
        print("\n" + report)
        
        print("\n" + "█"*60)
        print("█" + " "*58 + "█")
        print("█" + "  STANDARDIZATION COMPLETE!".center(58) + "█")
        print("█" + " "*58 + "█")
        print("█"*60 + "\n")
        
        return kepler_df, k2_df, tess_df


def main():
    """Main execution function"""
    # Initialize processor (use absolute path)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    processor = ExoplanetDataProcessor(data_dir=data_dir)
    
    # Process all datasets
    kepler_df, k2_df, tess_df = processor.process_all_datasets()
    
    total_records = len(kepler_df) + len(k2_df) + len(tess_df)
    print(f"\n✨ Successfully processed {total_records:,} total exoplanet observations!")
    print(f"\n📁 Standardized files saved to: {processor.processed_dir}")
    print("\n🔍 All 3 datasets now have the SAME standardized columns:")
    print("  • orbital_period (days)")
    print("  • transit_duration (hours)")
    print("  • planetary_radius (Earth radii)")
    print("  • stellar_magnitude")
    print("  • transit_depth (ppm)")
    print("  • impact_parameter")
    print("  • equilibrium_temperature (K)")
    print("  • stellar_radius (Solar radii)")
    print("  • stellar_mass (Solar masses)")
    print("  • disposition_score (0-1, confidence score)")
    print("  • disposition (CONFIRMED/CANDIDATE/FALSE_POSITIVE)")
    print("  • dataset (KEPLER/K2/TESS)")
    print("  • object_id (unique identifier)")
    print("\n⚠️  Note: Some columns may be null in K2/TESS where data is not available")


if __name__ == "__main__":
    main()
