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
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os

class NBACleaner:
    def __init__(self):
        """Initialize the cleaner with project directory structure."""
        self.project_root = Path(os.getcwd())
        self.base_dir = self.project_root / 'data'
        self.raw_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        
        # Create processed data directories if they don't exist
        for subdir in ['historical', 'current', 'combined']:
            (self.processed_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Define team name mappings for historical teams
        self.team_mappings = {
            'BULLETS': 'WAS',
            'WASHINGTON BULLETS': 'WAS',
            'CAPITAL BULLETS': 'WAS',
            'BALTIMORE BULLETS': 'WAS',
            'WASHINGTON WIZARDS': 'WAS',
            'WIZARDS': 'WAS',
            'CHICAGO ZEPHYRS': 'WAS',
            'CHICAGO PACKERS': 'WAS',
            
            'HAWKS': 'ATL',
            'ATLANTA HAWKS': 'ATL',
            'ST. LOUIS HAWKS': 'ATL',
            'MILWAUKEE HAWKS': 'ATL',
            'TRI-CITIES BLACKHAWKS': 'ATL',
            
            'CLIPPERS': 'LAC',
            'LA CLIPPERS': 'LAC',
            'LOS ANGELES CLIPPERS': 'LAC',
            'BUFFALO BRAVES': 'LAC',
            'SAN DIEGO CLIPPERS': 'LAC',
            
            'KINGS': 'SAC',
            'SACRAMENTO KINGS': 'SAC',
            'KANSAS CITY KINGS': 'SAC',
            'CINCINNATI ROYALS': 'SAC',
            'ROCHESTER ROYALS': 'SAC',
            
            '76ERS': 'PHI',
            'SIXERS': 'PHI',
            'PHILADELPHIA 76ERS': 'PHI',
            'SYRACUSE NATIONALS': 'PHI',
            
            'LAKERS': 'LAL',
            'LA LAKERS': 'LAL',
            'LOS ANGELES LAKERS': 'LAL',
            'MINNEAPOLIS LAKERS': 'LAL',
            
            'ROCKETS': 'HOU',
            'HOUSTON ROCKETS': 'HOU',
            'SAN DIEGO ROCKETS': 'HOU',
            
            'THUNDER': 'OKC',
            'OKLAHOMA CITY THUNDER': 'OKC',
            'SEATTLE SUPERSONICS': 'OKC',
            'SONICS': 'OKC',
            'SEA': 'OKC',
            
            'GRIZZLIES': 'MEM',
            'MEMPHIS GRIZZLIES': 'MEM',
            'VANCOUVER GRIZZLIES': 'MEM',
            
            'PELICANS': 'NOP',
            'NEW ORLEANS PELICANS': 'NOP',
            'NEW ORLEANS HORNETS': 'NOP',
            'NEW ORLEANS/OKLAHOMA CITY HORNETS': 'NOP',
            'NOK': 'NOP',
            'NOH': 'NOP',
            
            'JAZZ': 'UTA',
            'UTAH JAZZ': 'UTA',
            'NEW ORLEANS JAZZ': 'UTA',
            
            'HORNETS': 'CHA',
            'CHARLOTTE HORNETS': 'CHA',
            'CHARLOTTE BOBCATS': 'CHA',
            'BOBCATS': 'CHA',
            'CHO': 'CHA',
            
            'NETS': 'BKN',
            'BROOKLYN NETS': 'BKN',
            'NEW JERSEY NETS': 'BKN',
            'NJN': 'BKN',
            'BRK': 'BKN',
            
            'WARRIORS': 'GSW',
            'GOLDEN STATE WARRIORS': 'GSW',
            'SAN FRANCISCO WARRIORS': 'GSW',
            
            'SUNS': 'PHX',
            'PHOENIX SUNS': 'PHX',
            'PHO': 'PHX',
            
            'BLAZERS': 'POR',
            'TRAIL BLAZERS': 'POR',
            'PORTLAND TRAIL BLAZERS': 'POR',
            
            'SPURS': 'SAS',
            'SAN ANTONIO SPURS': 'SAS',
            
            'RAPTORS': 'TOR',
            'TORONTO RAPTORS': 'TOR',
            
            'BUCKS': 'MIL',
            'MILWAUKEE BUCKS': 'MIL',
            
            'TIMBERWOLVES': 'MIN',
            'MINNESOTA TIMBERWOLVES': 'MIN',
            
            'NUGGETS': 'DEN',
            'DENVER NUGGETS': 'DEN',
            
            'HEAT': 'MIA',
            'MIAMI HEAT': 'MIA',
            
            'CAVALIERS': 'CLE',
            'CLEVELAND CAVALIERS': 'CLE',
            'CAVS': 'CLE',
            
            'CELTICS': 'BOS',
            'BOSTON CELTICS': 'BOS',
            
            'PISTONS': 'DET',
            'DETROIT PISTONS': 'DET',
            
            'PACERS': 'IND',
            'INDIANA PACERS': 'IND',
            
            'BULLS': 'CHI',
            'CHICAGO BULLS': 'CHI',
            
            'MAVERICKS': 'DAL',
            'DALLAS MAVERICKS': 'DAL',
            
            'MAGIC': 'ORL',
            'ORLANDO MAGIC': 'ORL',
            
            'KNICKS': 'NYK',
            'NEW YORK KNICKS': 'NYK',
        }
    
    def standardize_team_names(self, df, team_cols=None):
        """
        Standardize team names to NBA three-letter codes.
        
        Args:
            df: DataFrame containing team names
            team_cols: List of columns containing team names
                      If None, finds columns with 'team' in name
        
        Returns:
            DataFrame with standardized team codes
        """
        if team_cols is None:
            team_cols = [col for col in df.columns if 'team' in col.lower()]
            # Also check for common team name columns
            extra_cols = ['tm', 'Team', 'TEAM_NAME']
            team_cols.extend([col for col in extra_cols if col in df.columns])
        
        df = df.copy()
        for col in team_cols:
            if col in df.columns and df[col].dtype == 'object':
                # Convert to uppercase and strip whitespace
                df[col] = df[col].str.strip().str.upper()
                
                # Map team names to standard three-letter codes
                df[col] = df[col].map(lambda x: self.team_mappings.get(x, x) if pd.notnull(x) else x)
        
        return df
    
    def handle_numeric_columns(self, df):
        """Convert and clean numeric columns in the dataset."""
        # Get all columns except obvious categorical ones
        exclude_patterns = ['name', 'team', 'position', 'date', 'season', 'location']
        numeric_candidates = [col for col in df.columns 
                            if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        df = df.copy()
        # Convert to numeric and handle missing values
        for col in numeric_candidates:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    median_val = df[col].median()
                    if pd.isnull(median_val):  # If median is also NaN
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(median_val)
            except Exception as e:
                print(f"Warning: Could not convert column {col} to numeric: {str(e)}")
                continue
        
        return df
    
    def convert_percentages(self, df):
        """Convert percentage strings to decimal values."""
        df = df.copy()
        pct_cols = [col for col in df.columns if any(x in col.lower() for x in ['percentage', 'pct'])]
        for col in pct_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.rstrip('%').astype('float') / 100.0
        return df
    
    def handle_dates(self, df, date_cols=None):
        """Convert date strings to datetime objects."""
        if date_cols is None:
            date_cols = [col for col in df.columns if 'date' in col.lower()]
        
        df = df.copy()
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def standardize_player_names(self, df, name_col='player_name'):
        """Standardize player names to consistent format."""
        df = df.copy()
        if name_col in df.columns:
            df[name_col] = df[name_col].str.strip().str.upper()
        return df
    
    def add_conference_mappings(self, df, name_col='team'):

        eastern_conf = [
            'ATL',  # Atlanta Hawks
            'BOS',  # Boston Celtics
            'BKN',  # Brooklyn Nets
            'CHA',  # Charlotte Hornets
            'CHI',  # Chicago Bulls
            'CLE',  # Cleveland Cavaliers
            'DET',  # Detroit Pistons
            'IND',  # Indiana Pacers
            'MIA',  # Miami Heat
            'MIL',  # Milwaukee Bucks
            'NYK',  # New York Knicks
            'ORL',  # Orlando Magic
            'PHI',  # Philadelphia 76ers
            'TOR',  # Toronto Raptors
            'WAS'   # Washington Wizards
        ]

        western_conf = [
            'DAL',  # Dallas Mavericks
            'DEN',  # Denver Nuggets
            'GSW',  # Golden State Warriors
            'HOU',  # Houston Rockets
            'LAC',  # Los Angeles Clippers
            'LAL',  # Los Angeles Lakers
            'MEM',  # Memphis Grizzlies
            'MIN',  # Minnesota Timberwolves
            'NOP',  # New Orleans Pelicans
            'OKC',  # Oklahoma City Thunder
            'PHX',  # Phoenix Suns
            'POR',  # Portland Trail Blazers
            'SAC',  # Sacramento Kings
            'SAS',  # San Antonio Spurs
            'UTA'   # Utah Jazz
        ]

        df['conference'] = df[name_col].apply(
            lambda x: 'EAST' if x in eastern_conf else 'WEST' if x in western_conf else 'Unknown'
        )
        return df
