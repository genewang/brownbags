"""
Redis Model Loader

This script is used to load machine learning models into Redis for the CPX Prediction API.
It reads model files from the local filesystem and stores them in Redis for fast access.
"""

import argparse
import glob
import logging
import os
import pickle
from typing import Dict, Any, Optional

import joblib
import redis
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model key mappings
MODEL_KEY_MAPPING = {
    'impressions_model.pkl': 'impressions_model',
    'clicks_model.pkl': 'clicks_model',
    'purchases_model.pkl': 'purchases_model',
    'cost_per_impression_model.pkl': 'cost_per_impression_model',
    'cost_per_click_model.pkl': 'cost_per_click_model',
    'cost_per_purchase_model.pkl': 'cost_per_purchase_model',
    'impressions_columns.pkl': 'impressions_columns',
    'clicks_columns.pkl': 'clicks_columns',
    'purchases_columns.pkl': 'purchases_columns',
    'cost_per_impression_columns.pkl': 'cost_per_impression_columns',
    'cost_per_click_columns.pkl': 'cost_per_click_columns',
    'cost_per_purchase_columns.pkl': 'cost_per_purchase_columns'
}


def init_redis_connection(host: str = 'localhost', 
                         port: int = 6379, 
                         password: Optional[str] = None, 
                         db: int = 0) -> redis.Redis:
    """Initialize and return a Redis client connection.
    
    Args:
        host: Redis server host
        port: Redis server port
        password: Redis password if required
        db: Redis database number
        
    Returns:
        redis.Redis: Configured Redis client instance
    """
    try:
        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=False  # We need bytes for model storage
        )
        # Test the connection
        client.ping()
        return client
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        raise


def load_model(file_path: str) -> Any:
    """Load a model or columns from a file.
    
    Args:
        file_path: Path to the model or columns file
        
    Returns:
        The loaded model or columns
    """
    try:
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return joblib.load(f)
        return None
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise


def save_to_redis(redis_client: redis.Redis, key: str, data: Any) -> bool:
    """Save data to Redis with the given key.
    
    Args:
        redis_client: Redis client instance
        key: Key to store the data under
        data: Data to store (will be serialized)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Serialize the data using joblib
        serialized_data = joblib.dumps(data)
        # Store in Redis
        return redis_client.set(key, serialized_data)
    except Exception as e:
        logger.error(f"Error saving to Redis (key: {key}): {str(e)}")
        return False


def load_models_to_redis(models_dir: str, redis_client: redis.Redis) -> None:
    """Load all models from a directory into Redis.
    
    Args:
        models_dir: Directory containing model files
        redis_client: Redis client instance
    """
    # Find all model files
    model_files = glob.glob(os.path.join(models_dir, '*.pkl'))
    
    if not model_files:
        logger.warning(f"No model files found in {models_dir}")
        return
    
    success_count = 0
    
    # Process each model file
    for model_file in tqdm(model_files, desc="Loading models to Redis"):
        try:
            # Get the base filename
            filename = os.path.basename(model_file)
            
            # Get the Redis key from our mapping
            redis_key = MODEL_KEY_MAPPING.get(filename)
            
            if not redis_key:
                logger.warning(f"No Redis key mapping for {filename}, skipping")
                continue
            
            # Load the model
            model = load_model(model_file)
            if model is None:
                continue
            
            # Save to Redis
            if save_to_redis(redis_client, redis_key, model):
                logger.info(f"Successfully loaded {filename} to Redis as {redis_key}")
                success_count += 1
            else:
                logger.error(f"Failed to save {filename} to Redis")
                
        except Exception as e:
            logger.error(f"Error processing {model_file}: {str(e)}")
    
    logger.info(f"Successfully loaded {success_count}/{len(model_files)} models to Redis")


def main():
    """Main entry point for the Redis model loader."""
    parser = argparse.ArgumentParser(description='Load ML models into Redis')
    parser.add_argument('--models-dir', type=str, required=True,
                       help='Directory containing model files')
    parser.add_argument('--redis-host', type=str, default='localhost',
                       help='Redis server host')
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis server port')
    parser.add_argument('--redis-password', type=str, default=None,
                       help='Redis password')
    parser.add_argument('--redis-db', type=int, default=0,
                       help='Redis database number')
    
    args = parser.parse_args()
    
    try:
        # Initialize Redis connection
        redis_client = init_redis_connection(
            host=args.redis_host,
            port=args.redis_port,
            password=args.redis_password,
            db=args.redis_db
        )
        
        # Load models into Redis
        load_models_to_redis(args.models_dir, redis_client)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
