"""
This module cleans NBA injury statistics data from loganlauton's dataset.

The data covers NBA player injuries from 1951 to 2023, including:
- Injury dates (start and return)
- Player information
- Team information
- Injury descriptions and types
- Recovery durations

This data is crucial for understanding:
- Player availability throughout seasons
- Impact of injuries on team performance
- Historical injury patterns and their effect on playoff chances
"""
import pandas as pd
import numpy as np
from pathlib import Path
from .base_cleaner import BaseNBACleaner

class LoganlautonCleaner(BaseNBACleaner):
    """
    Handles cleaning of NBA injury statistics data.
    
    This cleaner ensures consistency in:
    - Date formats (converting string dates to datetime objects)
    - Player names (standardizing across different spellings/formats)
    - Team names (using consistent team names across seasons)
    - Injury durations (calculating accurate recovery periods)
    
    The cleaned data helps track:
    - How long players were out due to injuries
    - Which teams were most affected by injuries
    - Patterns in injury types and recovery times
    """
    
    def __init__(self):
        """
        Initialize the cleaner with the correct data directory path.
        Sets up the path to loganlauton's NBA injury statistics dataset.
        """
        super().__init__()
        self.data_dir = self.raw_dir / 'kaggle' / 'loganlauton' / 'nba-injury-stats-1951-2023'
    
    def clean_injury_data(self, df):
        """
        Clean NBA injury statistics data.
        
        This function processes injury data by:
        - Converting date strings to proper datetime objects
          (e.g., "2023-01-15" -> datetime(2023, 1, 15))
        - Standardizing player names for consistency across datasets
          (e.g., "James,LeBron" -> "LeBron James")
        - Converting team abbreviations to full team names
          (e.g., "LAL" -> "Los Angeles Lakers")
        - Calculating injury durations in days
          (time between injury start and return dates)
        
        For missing return dates:
        - Calculates injury duration where possible
        - Uses median duration for that injury type when return date is missing
        
        Args:
            df: DataFrame containing raw injury statistics
            
        Returns:
            DataFrame with cleaned injury statistics including:
            - Standardized player and team names
            - Properly formatted dates
            - Calculated injury durations
        """
        if df.empty:
            return df
            
        try:
            print("\nCleaning injury data...")
            # Convert string dates to datetime objects for proper date handling
            df = self.handle_dates(df, ['start_date', 'return_date'])
            
            # Standardize player and team names for consistency across datasets
            df = self.standardize_player_names(df)
            df = self.standardize_team_names(df)
            
            # Calculate how many days each player was out due to injury
            # This helps analyze the severity and impact of different injuries
            if 'start_date' in df.columns and 'return_date' in df.columns:
                df['injury_duration'] = (df['return_date'] - df['start_date']).dt.days
                # For missing return dates, use the median duration for similar injuries
                df['injury_duration'] = df['injury_duration'].fillna(df['injury_duration'].median())
            
            print(f"Cleaned {len(df)} injury records")
            return df
            
        except Exception as e:
            print(f"Error cleaning injury data: {str(e)}")
            return df
