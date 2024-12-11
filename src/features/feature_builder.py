"""
Feature engineering for NBA playoff prediction.

This module provides a comprehensive feature engineering pipeline for NBA playoff prediction,
creating features from multiple data sources including team statistics, player data,
injury records, and shot patterns.
"""
import pandas as pd
import numpy as np
from datetime import datetime

class FeatureBuilder:
    def __init__(self):
        """Initialize the FeatureBuilder with required column definitions."""
        self.feature_stats = {}
        
        # Define required columns for each data source
        self._required_team_cols = [
            'season', 'team', 'playoffs', 'pts_per_game', 'fg_per_game',
            'fga_per_game', 'ft_per_game', 'fta_per_game', 'x3p_per_game',
            'x3pa_per_game', 'orb_per_game', 'drb_per_game', 'ast_per_game',
            'stl_per_game', 'blk_per_game', 'tov_per_game'
        ]
        
        self._required_player_cols = [
            'season', 'tm', 'experience', 'pos', 'age'
        ]
        
        self._required_shot_cols = [
            'SEASON_1', 'TEAM_NAME', 'EVENT_TYPE', 'SHOT_TYPE',
            'SHOT_DISTANCE', 'LOC_X', 'LOC_Y'
        ]
    
    def _validate_columns(self, df: pd.DataFrame, required_cols: list, source: str) -> None:
        """Validate that required columns are present in the DataFrame."""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {source} data: {missing_cols}")
    
    def create_team_features(self, team_stats: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive team performance features."""
        self._validate_columns(team_stats, self._required_team_cols, "team")
        df = team_stats.copy()
        
        # Calculate efficiency metrics
        df['true_shooting_pct'] = df['pts_per_game'] / (2 * (df['fga_per_game'] + 0.44 * df['fta_per_game']))
        df['efg_pct'] = (df['fg_per_game'] + 0.5 * df['x3p_per_game']) / df['fga_per_game']
        df['oreb_pct'] = df['orb_per_game'] / (df['orb_per_game'] + df['drb_per_game'])
        df['ast_to_ratio'] = df['ast_per_game'] / df['tov_per_game']
        df['ast_ratio'] = df['ast_per_game'] / df['fg_per_game']
        
        # Calculate defensive metrics
        df['stocks_per_game'] = df['stl_per_game'] + df['blk_per_game']
        df['def_rating'] = df['stocks_per_game'] * 2 - df['tov_per_game']
        
        # Calculate possession metrics
        df['possessions_per_game'] = (
            df['fga_per_game'] + 
            0.44 * df['fta_per_game'] - 
            df['orb_per_game'] + 
            df['tov_per_game']
        )
        df['off_efficiency'] = df['pts_per_game'] / df['possessions_per_game']
        df['three_point_rate'] = df['x3pa_per_game'] / df['fga_per_game']
        df['ft_rate'] = df['fta_per_game'] / df['fga_per_game']
        
        # Calculate overall efficiency rating
        df['efficiency_rating'] = (
            0.3 * df['true_shooting_pct'] +
            0.2 * df['ast_to_ratio'] +
            0.15 * df['def_rating'] +
            0.15 * df['off_efficiency'] +
            0.1 * df['oreb_pct'] +
            0.1 * df['stocks_per_game']
        )
        
        self.feature_stats['team_features'] = {
            'n_features': len(df.columns),
            'n_samples': len(df)
        }
        
        return df
    
    def create_player_features(self, player_stats: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
        """Create team-level features from player statistics and injury data.
        
        Args:
            player_stats: DataFrame containing player statistics
            injuries: DataFrame containing injury data
        
        Returns:
            DataFrame with team-level features including player stats and injury counts
        """
        # Input validation
        self._validate_columns(player_stats, self._required_player_cols, "player")
        if injuries.empty:
            return pd.DataFrame()
            
        # Create copies to avoid modifying original data
        player_stats = player_stats.copy()
        injuries = injuries.copy()
        
        try:
            # Process player stats
            player_stats['team'] = player_stats['tm']  # Already standardized in cleaning
            player_stats['season'] = player_stats['season'].astype(int)
            
            # Process injuries
            injuries['team'] = injuries['Team'].str.strip().upper()
            injuries['season'] = pd.to_datetime(injuries['Date']).dt.year
            
            # Create team-level aggregations
            team_stats = player_stats.groupby(['team', 'season']).agg({
                'experience': ['mean', 'max', 'min'],
                'player': 'count',
                'age': ['mean', 'max', 'min']
            }).reset_index()
            
            # Rename columns
            team_stats.columns = [
                'team', 'season',
                'avg_experience', 'max_experience', 'min_experience',
                'roster_size',
                'avg_age', 'max_age', 'min_age'
            ]
            
            # Calculate position distributions
            pos_dist = (player_stats.groupby(['team', 'season'])['pos']
                    .value_counts(normalize=True)
                    .unstack(fill_value=0)
                    .add_prefix('pos_pct_')
                    .reset_index())
            
            # Calculate injury counts
            injury_counts = (injuries.groupby(['team', 'season'])
                            .size()
                            .reset_index(name='injury_count'))
            
            # Merge all features
            features = (team_stats.merge(pos_dist, on=['team', 'season'])
                    .merge(injury_counts, on=['team', 'season'], how='left'))
            
            # Fill missing values
            features['injury_count'] = features['injury_count'].fillna(0)
            
            # Store statistics
            self.feature_stats['player_features'] = {
                'n_features': len(features.columns),
                'n_samples': len(features)
            }
            
            return features
            
        except Exception as e:
            print(f"Error creating player features: {str(e)}")
            return pd.DataFrame()
    
    def create_shot_features(self, shots: pd.DataFrame) -> pd.DataFrame:
        """Create team-level features from shot data."""
        self._validate_columns(shots, self._required_shot_cols, "shots")
        
        # Create a copy and standardize column names
        df = shots.copy()
        df['season'] = df['SEASON_1']
        df['team'] = df['TEAM_NAME']  # Already standardized in cleaning
        
        # Convert columns to string type before using str methods
        df['EVENT_TYPE'] = df['EVENT_TYPE'].astype(str)
        df['SHOT_TYPE'] = df['SHOT_TYPE'].astype(str)
        
        df['shot_made'] = df['EVENT_TYPE'].str.contains('MADE')
        df['is_three'] = df['SHOT_TYPE'].str.contains('3PT')
        df['shot_distance'] = df['SHOT_DISTANCE']
        
        # Drop any NaN values
        df = df.dropna(subset=['shot_made', 'is_three', 'shot_distance', 'LOC_X', 'LOC_Y'])
        
        # Group shots by team and season
        shot_features = df.groupby(['team', 'season']).agg({
            'shot_made': 'mean',
            'is_three': 'mean',
            'shot_distance': ['mean', 'std'],
            'LOC_X': ['mean', 'std'],
            'LOC_Y': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        shot_features.columns = [
            'team', 'season', 'fg_pct', 'three_point_rate',
            'avg_shot_distance', 'shot_distance_std',
            'avg_loc_x', 'loc_x_std',
            'avg_loc_y', 'loc_y_std'
        ]
        
        # Add relative to league average
        for col in ['fg_pct', 'three_point_rate', 'avg_shot_distance']:
            league_avg = shot_features[col].mean()
            shot_features[f'{col}_vs_avg'] = shot_features[col] - league_avg
        
        self.feature_stats['shot_features'] = {
            'n_features': len(shot_features.columns),
            'n_samples': len(shot_features)
        }
        
        return shot_features
    
    def combine_features(self, team_features: pd.DataFrame, player_features: pd.DataFrame,
                        shot_features: pd.DataFrame) -> tuple:
        """Combine all feature sets and create target variable."""
        if 'playoffs' not in team_features.columns:
            raise ValueError("Team features must include 'playoffs' column for target variable")
        
        # Create copies to avoid modifying originals
        team_features = team_features.copy()
        player_features = player_features.copy()
        shot_features = shot_features.copy()
        
        # Ensure team and season columns are the correct type
        for df in [team_features, player_features, shot_features]:
            df['team'] = df['team'].astype(str)
            df['season'] = df['season'].astype(int)
        
        # Merge all features
        features = (
            team_features.merge(
                player_features,
                on=['team', 'season'],
                how='left'
            ).merge(
                shot_features,
                on=['team', 'season'],
                how='left'
            )
        )
        
        # Create target variable
        target = features['playoffs'].astype(int)
        
        # Remove non-feature columns
        drop_cols = ['playoffs', 'team', 'season', 'abbreviation', 'lg']
        features = features.drop(columns=[col for col in drop_cols if col in features.columns])
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        self.feature_stats['combined_features'] = {
            'n_features': len(features.columns),
            'n_samples': len(features),
            'feature_names': list(features.columns),
            'target_distribution': target.value_counts().to_dict()
        }
        
        return features, target
