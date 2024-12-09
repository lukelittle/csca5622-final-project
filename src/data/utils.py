"""Utility functions for data collection"""
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd

def setup_logging(logging_level=logging.INFO):
    """Configure logging with timestamp and formatting"""
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def create_data_directories(base_path='data'):
    """Create required data directories"""
    directories = {
        'raw': {
            'kaggle': [
                'sumitrodatta/nba-aba-baa-stats',
                'mexwell/nba-shots',
                'loganlauton/nba-injury-stats-1951-2023'
            ],
            'rapidapi': [
                'games',
                'standings'
            ]
        },
        'processed': [
            'historical',
            'current',
            'combined'
        ]
    }
    
    created = []
    for category, subdirs in directories.items():
        if isinstance(subdirs, list):
            for subdir in subdirs:
                path = Path(base_path) / category / subdir
                path.mkdir(parents=True, exist_ok=True)
                created.append(str(path))
        else:
            for source, source_subdirs in subdirs.items():
                for subdir in source_subdirs:
                    path = Path(base_path) / category / source / subdir
                    path.mkdir(parents=True, exist_ok=True)
                    created.append(str(path))
    
    return created

def validate_dataframe(df, name, required_columns=None):
    """Validate a DataFrame meets basic requirements"""
    if df is None or df.empty:
        raise ValueError(f"No data available for {name}")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for {name}: {missing}")
    
    return True

def save_data(df, directory, name, validate_cols=None):
    """Save data with validation and timestamping"""
    try:
        validate_dataframe(df, name, validate_cols)
        
        date_str = datetime.now().strftime('%Y%m%d')
        output_path = Path(directory) / f"{name}_{date_str}.csv"
        
        df.to_csv(output_path, index=False)
        return str(output_path), len(df)
    
    except Exception as e:
        raise Exception(f"Error saving {name}: {str(e)}")

class DataCollectionProgress:
    """Track progress of data collection tasks"""
    
    def __init__(self):
        self.tasks = {}
        self.start_time = datetime.now()
    
    def add_task(self, name, total_steps=None):
        """Add a new task to track"""
        self.tasks[name] = {
            'status': 'pending',
            'total_steps': total_steps,
            'completed_steps': 0,
            'start_time': None,
            'end_time': None,
            'error': None
        }
    
    def start_task(self, name):
        """Mark a task as started"""
        if name in self.tasks:
            self.tasks[name]['status'] = 'in_progress'
            self.tasks[name]['start_time'] = datetime.now()
    
    def complete_task(self, name, success=True, error=None):
        """Mark a task as completed"""
        if name in self.tasks:
            self.tasks[name]['status'] = 'completed' if success else 'failed'
            self.tasks[name]['end_time'] = datetime.now()
            self.tasks[name]['error'] = error
    
    def update_progress(self, name, steps_completed):
        """Update progress of a task"""
        if name in self.tasks:
            self.tasks[name]['completed_steps'] = steps_completed
    
    def get_summary(self):
        """Get summary of all tasks"""
        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for t in self.tasks.values() if t['status'] == 'completed')
        failed_tasks = sum(1 for t in self.tasks.values() if t['status'] == 'failed')
        
        duration = datetime.now() - self.start_time
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'duration': str(duration),
            'tasks': self.tasks
        }
