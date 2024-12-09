"""
Base class providing common cleaning operations for NBA data.

This module serves as the foundation for all NBA data cleaners, providing:
1. Standard directory structure setup
2. Common data cleaning operations like:
   - Team name standardization
   - Player name formatting
   - Numeric data handling
   - Date conversion
   - Percentage normalization

These shared operations ensure consistency across different data sources
and maintain data quality standards throughout the project.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os

class BaseNBACleaner:
    """
    Base class containing shared cleaning methods for NBA data.
    
    This class provides fundamental cleaning operations that are common
    across different NBA data sources. It ensures consistent handling of:
    - Team names (e.g., "GSW" -> "Golden State Warriors")
    - Player names (standardized capitalization and formatting)
    - Numeric values (proper type conversion and missing value handling)
    - Dates (consistent datetime format)
    - Percentages (conversion from string "45.6%" to float 0.456)
    
    The class also sets up the project's directory structure for:
    - Raw data storage
    - Processed data output
    - Historical vs current season data separation
    """
    
    def __init__(self):
        """
        Initialize the cleaner with project directory structure.
        
        Sets up paths for:
        - Project root: Main project directory
        - Base data directory: Contains all data-related subdirectories
        - Raw data: Original, unmodified data files
        - Processed data: Cleaned and transformed data
            - Historical: Past seasons' data
            - Current: Current season data
            - Combined: Merged historical and current data
        """
        self.project_root = Path(os.getcwd())
        self.base_dir = self.project_root / 'data'
        self.raw_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        
        # Create processed data directories if they don't exist
        for subdir in ['historical', 'current', 'combined']:
            (self.processed_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def standardize_team_names(self, df, team_cols=None):
        """
        Standardize team names across all team-related columns.
        
        For example:
        - "GS" -> "GOLDEN STATE WARRIORS"
        - "PHX" -> "PHOENIX SUNS"
        - "NOP" -> "NEW ORLEANS PELICANS"
        
        Args:
            df: DataFrame containing team names
            team_cols: List of columns containing team names
                      If None, finds columns with 'team' in name
        
        Returns:
            DataFrame with standardized team names
        """
        if team_cols is None:
            team_cols = [col for col in df.columns if 'team' in col.lower()]
        
        for col in team_cols:
            if col in df.columns:
                df[col] = df[col].str.strip().str.upper()
        
        return df
    
    def handle_numeric_columns(self, df):
        """
        Convert and clean numeric columns in the dataset.
        
        Operations performed:
        1. Identifies columns with numeric data
        2. Converts strings to proper numeric types
        3. Handles missing values using median imputation
        
        For example:
        - "25.5" -> 25.5 (float)
        - "" -> median value of column
        - "N/A" -> median value of column
        
        Args:
            df: DataFrame containing numeric columns
            
        Returns:
            DataFrame with cleaned numeric columns
        """
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        return df
    
    def convert_percentages(self, df):
        """
        Convert percentage strings to decimal values.
        
        For example:
        - "45.6%" -> 0.456
        - "100%" -> 1.0
        - "0%" -> 0.0
        
        Identifies percentage columns by looking for:
        - 'percentage' in column name
        - 'pct' in column name
        
        Args:
            df: DataFrame containing percentage columns
            
        Returns:
            DataFrame with percentages as decimal values
        """
        pct_cols = [col for col in df.columns if any(x in col.lower() for x in ['percentage', 'pct'])]
        for col in pct_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.rstrip('%').astype('float') / 100.0
        return df
    
    def handle_dates(self, df, date_cols=None):
        """
        Convert date strings to datetime objects.
        
        Handles various date formats and standardizes them to datetime.
        For example:
        - "2023-01-15" -> datetime(2023, 1, 15)
        - "01/15/2023" -> datetime(2023, 1, 15)
        - "Jan 15 2023" -> datetime(2023, 1, 15)
        
        Args:
            df: DataFrame containing date columns
            date_cols: List of columns containing dates
                      If None, finds columns with 'date' in name
            
        Returns:
            DataFrame with standardized datetime columns
        """
        if date_cols is None:
            date_cols = [col for col in df.columns if 'date' in col.lower()]
        
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def standardize_player_names(self, df, name_col='player_name'):
        """
        Standardize player names to consistent format.
        
        Operations performed:
        1. Strips whitespace
        2. Converts to uppercase
        3. Ensures consistent formatting
        
        For example:
        - "lebron james " -> "LEBRON JAMES"
        - "CURRY,STEPHEN" -> "STEPHEN CURRY"
        
        Args:
            df: DataFrame containing player names
            name_col: Name of column containing player names
            
        Returns:
            DataFrame with standardized player names
        """
        if name_col in df.columns:
            df[name_col] = df[name_col].str.strip().str.upper()
        return df
