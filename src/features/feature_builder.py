"""
Feature engineering for NBA playoff prediction.

This module provides a feature engineering pipeline for NBA playoff prediction,
creating features from team statistics, player data, and injury summaries.
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
            'season', 'team', 'experience', 'age'
        ]
        
        self._required_injury_cols = [
            'year', 'team', 'count'
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
            injuries: DataFrame containing injury summary data
        
        Returns:
            DataFrame with team-level features including player stats and injury counts
        """
        # Input validation
        self._validate_columns(player_stats, self._required_player_cols, "player")
        self._validate_columns(injuries, self._required_injury_cols, "injuries")
            
        # Create copies to avoid modifying original data
        player_stats = player_stats.copy()
        injuries = injuries.copy()
        
        try:
            # Process player stats
            player_stats['season'] = player_stats['season'].astype(int)
            
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
            
            # Process injuries
            injuries = injuries.rename(columns={'year': 'season'})
            injuries['season'] = injuries['season'].astype(int)
            
            # Merge all features
            features = team_stats.merge(injuries, on=['team', 'season'], how='left')
            
            # Fill missing values
            features['count'] = features['count'].fillna(0)
            
            # Store statistics
            self.feature_stats['player_features'] = {
                'n_features': len(features.columns),
                'n_samples': len(features)
            }
            
            return features
            
        except Exception as e:
            print(f"Error creating player features: {str(e)}")
            return pd.DataFrame()
    
    def combine_features(self, team_features: pd.DataFrame, player_features: pd.DataFrame) -> tuple:
        """Combine all feature sets and create target variable."""
        if 'playoffs' not in team_features.columns:
            raise ValueError("Team features must include 'playoffs' column for target variable")
        
        # Create copies to avoid modifying originals
        team_features = team_features.copy()
        player_features = player_features.copy()
        
        # Ensure team and season columns are the correct type
        for df in [team_features, player_features]:
            df['team'] = df['team'].astype(str)
            df['season'] = df['season'].astype(int)
        
        # Merge all features
        features = team_features.merge(
            player_features,
            on=['team', 'season'],
            how='left'
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
