#!/usr/bin/env python3
"""Train recommendation models."""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import DataLoader
from models import create_model, get_available_models
from utils import set_seed, load_config, split_data_chronological

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train recommendation models."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    config = load_config(str(config_path))
    
    # Set random seed
    set_seed(config['seed'])
    
    # Load data
    logger.info("Loading data...")
    
    data_dir = Path(__file__).parent.parent / "data"
    loader = DataLoader(str(data_dir))
    
    users_df = loader.load_users()
    interactions_df = loader.load_interactions()
    items_df = loader.load_items()
    
    # Create interaction matrix
    interaction_matrix = loader.get_interaction_matrix()
    user_interests = loader.get_user_interests()
    
    logger.info(f"Loaded data: {interaction_matrix.shape[0]} users, {interaction_matrix.shape[1]} items")
    
    # Split data for evaluation
    logger.info("Splitting data...")
    
    # Convert interactions to list format for splitting
    interactions_list = []
    for _, row in interactions_df.iterrows():
        interactions_list.append((
            row['user_id'], 
            row['friend_id'], 
            row['weight'], 
            row['timestamp']
        ))
    
    train_data, val_data, test_data = split_data_chronological(
        interactions_list,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )
    
    # Create train interaction matrix
    train_matrix = np.zeros_like(interaction_matrix)
    for user_id, friend_id, weight, _ in train_data:
        train_matrix[user_id, friend_id] = weight
    
    logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Define models to train
    models_to_train = [
        'content_based',
        'interest_based', 
        'user_based_cf',
        'item_based_cf',
        'matrix_factorization',
        'graph_based',
        'community_based',
        'hybrid'
    ]
    
    # Train models
    logger.info("Training models...")
    
    trained_models = {}
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    for model_name in models_to_train:
        try:
            logger.info(f"Training {model_name}...")
            
            # Get model configuration
            model_config = config['models'].get(model_name, {})
            
            # Create and train model
            model = create_model(model_name, **model_config)
            
            if model_name in ['content_based', 'interest_based', 'hybrid']:
                model.fit(train_matrix, user_interests)
            else:
                model.fit(train_matrix)
            
            # Save model
            model_path = models_dir / f"{model_name}.pkl"
            model.save_model(str(model_path))
            
            trained_models[model_name] = model
            
            logger.info(f"Successfully trained and saved {model_name}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue
    
    logger.info(f"Successfully trained {len(trained_models)} models")
    
    # Save training metadata
    metadata = {
        'models_trained': list(trained_models.keys()),
        'data_shape': interaction_matrix.shape,
        'train_interactions': len(train_data),
        'val_interactions': len(val_data),
        'test_interactions': len(test_data),
        'config': config
    }
    
    metadata_path = models_dir / "training_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
