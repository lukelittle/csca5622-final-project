"""
Feature engineering for NBA playoff prediction.

This module provides a comprehensive feature engineering pipeline for NBA playoff prediction,
creating features from multiple data sources including team statistics, player data,
injury records, and shot patterns.

Features are organized into three main categories:
1. Team Performance: Efficiency metrics and advanced statistics
2. Player Impact: Experience, position distribution, and injury effects
3. Shot Analysis: Shooting patterns and court utilization

Example:
    >>> builder = FeatureBuilder()
    >>> team_features = builder.create_team_features(team_stats)
    >>> player_features = builder.create_player_features(player_stats, injuries)
    >>> shot_features = builder.create_shot_features(shots)
    >>> feature_matrix, target = builder.combine_features(
    ...     team_features, player_features, shot_features
    ... )
"""
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

class FeatureBuilder:
    """Creates predictive features for NBA playoff qualification.
    
    This class implements a comprehensive feature engineering pipeline that creates
    features from multiple data sources and combines them into a unified feature matrix
    for playoff prediction modeling.
    
    Attributes:
        feature_stats: Dict storing statistics about created features
        _required_team_cols: List of required columns for team statistics
        _required_player_cols: List of required columns for player statistics
        _required_shot_cols: List of required columns for shot data
    """
    
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
    
    def _validate_columns(self, df: pd.DataFrame, required_cols: List[str], source: str) -> None:
        """Validate that required columns are present in the DataFrame.
        
        Args:
            df: DataFrame to validate
            required_cols: List of required column names
            source: Name of the data source for error messages
            
        Raises:
            ValueError: If any required columns are missing
        """
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {source} data: {missing_cols}"
            )
    
    def _calculate_efficiency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced efficiency metrics for teams.
        
        Args:
            df: DataFrame containing basic team statistics
            
        Returns:
            DataFrame with additional efficiency metrics
        """
        result = df.copy()
        
        # True Shooting Percentage (accounts for 2pt, 3pt, and FT)
        result['true_shooting_pct'] = (
            result['pts_per_game'] / 
            (2 * (result['fga_per_game'] + 0.44 * result['fta_per_game']))
        )
        
        # Effective Field Goal Percentage (adjusts for 3pt being worth more)
        result['efg_pct'] = (
            (result['fg_per_game'] + 0.5 * result['x3p_per_game']) / 
            result['fga_per_game']
        )
        
        # Offensive rebounding percentage
        result['oreb_pct'] = (
            result['orb_per_game'] / 
            (result['orb_per_game'] + result['drb_per_game'])
        )
        
        # Assist to Turnover ratio
        result['ast_to_ratio'] = result['ast_per_game'] / result['tov_per_game']
        
        # Assist ratio (percentage of field goals that are assisted)
        result['ast_ratio'] = result['ast_per_game'] / result['fg_per_game']
        
        return result
    
    def _calculate_defensive_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced defensive metrics for teams.
        
        Args:
            df: DataFrame containing basic team statistics
            
        Returns:
            DataFrame with additional defensive metrics
        """
        result = df.copy()
        
        # Stocks (steals + blocks)
        result['stocks_per_game'] = result['stl_per_game'] + result['blk_per_game']
        
        # Simple defensive rating
        result['def_rating'] = result['stocks_per_game'] * 2 - result['tov_per_game']
        
        return result
    
    def _calculate_possession_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate possession-based metrics for teams.
        
        Args:
            df: DataFrame containing basic team statistics
            
        Returns:
            DataFrame with additional possession metrics
        """
        result = df.copy()
        
        # Estimate possessions
        result['possessions_per_game'] = (
            result['fga_per_game'] + 
            0.44 * result['fta_per_game'] - 
            result['orb_per_game'] + 
            result['tov_per_game']
        )
        
        # Offensive efficiency (points per possession)
        result['off_efficiency'] = result['pts_per_game'] / result['possessions_per_game']
        
        # Three point reliance
        result['three_point_rate'] = result['x3pa_per_game'] / result['fga_per_game']
        
        # Free throw rate
        result['ft_rate'] = result['fta_per_game'] / result['fga_per_game']
        
        return result
    
    def create_team_features(self, team_stats: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive team performance features.
        
        Args:
            team_stats: DataFrame containing team statistics
            
        Returns:
            DataFrame containing engineered team features
            
        Raises:
            ValueError: If required columns are missing
        """
        # Validate required columns
        self._validate_columns(team_stats, self._required_team_cols, "team")
        
        # Create copy for feature engineering
        df = team_stats.copy()
        
        # Calculate various metric categories
        df = self._calculate_efficiency_metrics(df)
        df = self._calculate_defensive_metrics(df)
        df = self._calculate_possession_metrics(df)
        
        # Calculate overall efficiency rating
        df['efficiency_rating'] = (
            0.3 * df['true_shooting_pct'] +    # Shooting efficiency
            0.2 * df['ast_to_ratio'] +         # Ball control
            0.15 * df['def_rating'] +          # Defense
            0.15 * df['off_efficiency'] +      # Offensive efficiency
            0.1 * df['oreb_pct'] +            # Rebounding
            0.1 * df['stocks_per_game']        # Defensive playmaking
        )
        
        # Store feature statistics
        self.feature_stats['team_features'] = {
            'n_features': len(df.columns),
            'n_samples': len(df)
        }
        
        return df
    
    def create_player_features(
        self, 
        player_stats: pd.DataFrame, 
        injuries: pd.DataFrame
    ) -> pd.DataFrame:
        """Create team-level features from player statistics and injury data.
        
        Args:
            player_stats: DataFrame containing player statistics
            injuries: DataFrame containing injury records
            
        Returns:
            DataFrame containing engineered player impact features
            
        Raises:
            ValueError: If required columns are missing
        """
        # Validate required columns
        self._validate_columns(player_stats, self._required_player_cols, "player")
        
        # Create basic team-level aggregations
        team_player_stats = player_stats.groupby(['tm', 'season']).agg({
            'experience': ['mean', 'max', 'min'],
            'player': 'count',
            'age': ['mean', 'max', 'min']
        }).reset_index()
        
        # Flatten column names
        team_player_stats.columns = [
            'team', 'season', 
            'avg_experience', 'max_experience', 'min_experience',
            'roster_size',
            'avg_age', 'max_age', 'min_age'
        ]
        
        # Calculate position distribution
        pos_dist = (
            player_stats.groupby(['tm', 'season'])['pos']
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        pos_dist = pos_dist.add_prefix('pos_pct_').reset_index()
        pos_dist = pos_dist.rename(columns={'tm': 'team'})
        
        # Merge position distribution
        team_player_stats = team_player_stats.merge(
            pos_dist, 
            on=['team', 'season']
        )
        
        # Add injury impact if available
        if not injuries.empty and 'Date' in injuries.columns:
            # Convert date to year for season
            injuries['season'] = pd.to_datetime(injuries['Date']).dt.year
            
            # Count injuries per team per season
            injury_counts = (
                injuries.groupby(['Team', 'season'])
                .size()
                .reset_index(name='injury_count')
            )
            injury_counts.columns = ['team', 'season', 'injury_count']
            
            # Merge with team stats
            team_player_stats = team_player_stats.merge(
                injury_counts,
                on=['team', 'season'],
                how='left'
            )
            team_player_stats['injury_count'] = team_player_stats['injury_count'].fillna(0)
        else:
            team_player_stats['injury_count'] = 0
        
        # Store feature statistics
        self.feature_stats['player_features'] = {
            'n_features': len(team_player_stats.columns),
            'n_samples': len(team_player_stats)
        }
        
        return team_player_stats
    
    def create_shot_features(self, shots: pd.DataFrame) -> pd.DataFrame:
        """Create team-level features from shot data.
        
        Args:
            shots: DataFrame containing shot-by-shot data
            
        Returns:
            DataFrame containing engineered shot features
            
        Raises:
            ValueError: If required columns are missing
        """
        # Validate required columns
        self._validate_columns(shots, self._required_shot_cols, "shots")
        
        # Create a copy and standardize column names
        df = shots.copy()
        df['season'] = df['SEASON_1']
        df['team'] = df['TEAM_NAME']
        df['shot_made'] = df['EVENT_TYPE'].str.contains('MADE')
        df['is_three'] = df['SHOT_TYPE'].str.contains('3PT')
        df['shot_distance'] = df['SHOT_DISTANCE']
        
        # Group shots by team and season
        shot_features = df.groupby(['team', 'season']).agg({
            'shot_made': 'mean',                    # FG%
            'is_three': 'mean',                     # 3PT rate
            'shot_distance': ['mean', 'std'],       # Shot distance patterns
            'LOC_X': ['mean', 'std'],              # Shot location patterns
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
        
        # Store feature statistics
        self.feature_stats['shot_features'] = {
            'n_features': len(shot_features.columns),
            'n_samples': len(shot_features)
        }
        
        return shot_features
    
    def combine_features(
        self,
        team_features: pd.DataFrame,
        player_features: pd.DataFrame,
        shot_features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Combine all feature sets and create target variable.
        
        Args:
            team_features: DataFrame containing team performance features
            player_features: DataFrame containing player impact features
            shot_features: DataFrame containing shot analysis features
            
        Returns:
            Tuple containing:
                - DataFrame of combined features
                - Series containing target variable (playoff qualification)
            
        Raises:
            ValueError: If playoffs column is missing or if merge fails
        """
        # Ensure playoffs column exists
        if 'playoffs' not in team_features.columns:
            raise ValueError("Team features must include 'playoffs' column for target variable")
        
        # Standardize team names
        team_features['team'] = team_features['team'].str.upper()
        player_features['team'] = player_features['team'].str.upper()
        shot_features['team'] = shot_features['team'].str.upper()
        
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
        features = features.drop(
            columns=[col for col in drop_cols if col in features.columns]
        )
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Store feature statistics
        self.feature_stats['combined_features'] = {
            'n_features': len(features.columns),
            'n_samples': len(features),
            'feature_names': list(features.columns),
            'target_distribution': target.value_counts().to_dict()
        }
        
        return features, target
