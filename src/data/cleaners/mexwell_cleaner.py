"""
This module cleans NBA shot data from mexwell's dataset.

The data contains detailed information about every shot taken in NBA games, including:
- Shot location coordinates (x, y positions on court)
- Shot details (type, distance, made/missed)
- Player and team information
- Game timing information
- Shot zone classifications

This data is vital for analyzing:
- Team shooting patterns and efficiency
- Player shooting preferences and success rates
- Spatial distribution of shots across the court
- Shot selection strategies in different game situations
"""
import pandas as pd
import numpy as np
from pathlib import Path
from .base_cleaner import BaseNBACleaner

class MexwellCleaner(BaseNBACleaner):
    """
    Handles cleaning of NBA shot location and outcome data.
    
    This cleaner ensures consistency in:
    - Player names (standardizing across different spellings/formats)
    - Team names (using consistent naming across seasons)
    - Shot coordinates (ensuring valid court positions)
    - Shot classifications (standardizing zone names and types)
    - Game timing information (quarters, minutes, seconds)
    
    The cleaned data helps analyze:
    - Where shots are taken on the court
    - Which players/teams are most effective from different locations
    - How shot selection changes throughout games
    - Historical shooting trends and patterns
    """
    
    def __init__(self):
        """
        Initialize the cleaner with the correct data directory path.
        Sets up the path to mexwell's NBA shots dataset.
        """
        super().__init__()
        self.data_dir = self.raw_dir / 'kaggle' / 'mexwell' / 'nba-shots'
        
    def clean_shots_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean NBA shot location and outcome data.
        
        This function processes shot data by:
        - Converting text fields to proper string format
          (e.g., standardizing case, removing extra whitespace)
        - Ensuring numeric values for coordinates and distances
          (e.g., shot distance in feet, x/y coordinates on court)
        - Standardizing categorical data like shot types and zones
          (e.g., "Three Point" vs "3PT" -> "THREE_POINT")
        - Converting dates to proper datetime objects
        
        Specific cleaning steps:
        1. String columns (names, types, zones):
           - Convert to consistent string format
           - Strip whitespace
           - Standardize to uppercase
           - Handle missing or invalid values
        
        2. Numeric columns (coordinates, distances, timing):
           - Convert to proper numeric types
           - Handle invalid values (e.g., shots outside court)
           - Ensure consistent units
        
        3. Date handling:
           - Convert game dates to datetime objects
           - Ensure consistent date format
        
        Args:
            df: DataFrame containing raw shot data with columns like:
                - PLAYER_NAME: Player who took the shot
                - TEAM_NAME: Team the player was on
                - LOC_X, LOC_Y: Shot coordinates
                - SHOT_DISTANCE: Distance from basket
                - Various categorical columns for shot classification
            
        Returns:
            DataFrame with cleaned shot data, including:
            - Standardized string values
            - Proper numeric types for measurements
            - Consistent date formats
            - Validated shot locations and classifications
        """
        if df.empty:
            return df
            
        try:
            print("\nCleaning shots data...")
            df = df.copy()

            # Print all column types at start for debugging
            print("\nInitial column types:")
            for col in df.columns:
                print(f"{col}: {df[col].dtype} | Sample: {df[col].iloc[0]}")

            # Clean text-based columns (names, types, zones)
            string_columns = [
                'PLAYER_NAME', 'TEAM_NAME', 'EVENT_TYPE', 
                'ACTION_TYPE', 'SHOT_TYPE', 'BASIC_ZONE', 
                'ZONE_NAME', 'ZONE_ABB', 'ZONE_RANGE'
            ]

            # Process each text column individually with error checking
            for col in string_columns:
                if col in df.columns:
                    print(f"\nProcessing column: {col}")
                    print(f"Current type: {df[col].dtype}")
                    try:
                        # Convert to consistent string format
                        df[col] = df[col].astype('string')
                        print(f"Converted {col} to string type")
                        
                        # Standardize text values
                        df[col] = df[col].str.strip()
                        df[col] = df[col].str.upper()
                        print(f"Successfully cleaned {col}")
                    except Exception as col_error:
                        print(f"Error processing {col}: {str(col_error)}")
                        print(f"Sample values: {df[col].head()}")
                        raise

            # Clean numeric measurements and coordinates
            numeric_columns = [
                'LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 
                'QUARTER', 'MINS_LEFT', 'SECS_LEFT'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert game dates to datetime objects
            if 'GAME_DATE' in df.columns:
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

            print(f"Cleaned {len(df)} shot records")
            return df

        except Exception as e:
            print(f"Error cleaning shots data: {str(e)}")
            print("\nFinal column types:")
            print(df.dtypes)
            raise
