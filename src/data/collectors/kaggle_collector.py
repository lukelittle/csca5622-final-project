"""
KaggleCollector class for downloading NBA datasets from Kaggle.
"""

import os
import logging
from typing import Dict
from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleCollector:
    """
    A class to manage the download of datasets from Kaggle.
    """

    def __init__(self, base_dir: str):
        """
        Initialize the KaggleCollector with the target directory for downloads.

        Args:
            base_dir (str): Base directory where downloaded datasets will be stored
        """
        self.base_dir = base_dir
        self.api = KaggleApi()
        self.api.authenticate()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)

    def download_dataset(self, dataset_name: str, dataset_path: str) -> Dict:
        """
        Download a specific dataset from Kaggle.

        Args:
            dataset_name (str): Name to use for the dataset directory
            dataset_path (str): Kaggle path to the dataset (e.g., 'username/dataset-name')

        Returns:
            Dict: Result of the download operation including status and any error message
        """
        try:
            self.logger.info(f"Downloading dataset: {dataset_name}")
            
            # Create the base directory and dataset subdirectory
            dataset_dir = os.path.join(self.base_dir, dataset_path)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Download the dataset
            self.api.dataset_download_files(
                dataset_path,
                path=dataset_dir,
                unzip=True
            )
            
            self.logger.info(f"Successfully downloaded {dataset_name}")
            return {
                'status': 'success',
                'error': None,
                'path': dataset_dir
            }
            
        except Exception as e:
            error_msg = f"Error downloading {dataset_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'status': 'failed',
                'error': error_msg,
                'path': None
            }
