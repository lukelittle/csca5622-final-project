"""
This module cleans historical NBA/ABA/BAA statistics data from sumitrodatta's dataset.

The data includes player season statistics and team performance metrics from 1950 to present.
This cleaner handles two main types of data:
1. Player season statistics - individual player performance metrics
2. Team statistics - aggregated team performance data

Key cleaning operations include:
- Standardizing player and team names across different eras
- Converting string-based numbers to proper numeric types
- Handling percentage values (converting from string "45.6%" to float 0.456)
- Removing or fixing invalid entries
"""
import pandas as pd
import numpy as np
from pathlib import Path
from .base_cleaner import BaseNBACleaner

class SumitrodattaCleaner(BaseNBACleaner):
    """
    Handles cleaning of NBA/ABA/BAA historical statistics data.
    
    This cleaner ensures consistency in:
    - Team names (e.g., "GS" and "GSW" both become "Golden State Warriors")
    - Player names (standardizing formatting and handling special characters)
    - Numeric data (converting strings to proper numeric types)
    - Percentage values (converting to proper decimal format)
    """
    
    def __init__(self):
        """
        Initialize the cleaner with the correct data directory path.
        Sets up the path to sumitrodatta's NBA/ABA/BAA statistics dataset.
        """
        super().__init__()
        self.data_dir = self.raw_dir / 'kaggle' / 'sumitrodatta' / 'nba-aba-baa-stats'
    
    def clean_player_season_data(self, df):
        """
        Clean individual player statistics for each season.
        
        This function handles player season statistics by:
        - Standardizing player names to ensure consistency across seasons
        - Converting team abbreviations to full team names
        - Fixing data types for statistical columns (points, rebounds, etc.)
        - Removing invalid entries (e.g., duplicate seasons, incorrect values)
        
        Args:
            df: DataFrame containing raw player season statistics
            
        Returns:
            DataFrame with cleaned player season statistics
        """
        if df.empty:
            return df
            
        try:
            print("\nCleaning player season data...")
            # Convert player names to standard format (e.g., "James, LeBron" -> "LeBron James")
            df = self.standardize_player_names(df)
            # Convert team codes to full names (e.g., "BOS" -> "Boston Celtics")
            df = self.standardize_team_names(df)
            
            # Convert string numbers to proper numeric types
            # Handles cases like "25.5" -> 25.5 and "" -> NaN
            df = self.handle_numeric_columns(df)
            
            print(f"Cleaned {len(df)} player season records")
            return df
            
        except Exception as e:
            print(f"Error cleaning player season data: {str(e)}")
            return df
    
    def clean_team_stats_data(self, df):
        """
        Clean team-level statistics data.
        
        This function processes team statistics by:
        - Converting team abbreviations to full, consistent team names
        - Ensuring proper numeric types for all statistical columns
        - Converting percentage values to proper decimal format
        - Handling missing or invalid values
        
        For example:
        - Team names: "PHO" -> "Phoenix Suns"
        - Percentages: "45.6%" -> 0.456
        - Points: "115.2" -> 115.2
        
        Args:
            df: DataFrame containing raw team statistics
            
        Returns:
            DataFrame with cleaned team statistics
        """
        if df.empty:
            return df
            
        try:
            print("\nCleaning team stats data...")
            # Convert team abbreviations to full names
            df = self.standardize_team_names(df)
            
            # Convert string numbers to proper numeric types
            df = self.handle_numeric_columns(df)
            
            # Convert percentage strings to decimal values
            # e.g., "45.6%" -> 0.456
            df = self.convert_percentages(df)
            
            print(f"Cleaned {len(df)} team statistics records")
            return df
            
        except Exception as e:
            print(f"Error cleaning team statistics data: {str(e)}")
            return df
