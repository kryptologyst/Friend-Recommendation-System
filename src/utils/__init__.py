"""Core utilities for the friend recommendation system."""

import random
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    import yaml
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved configuration to {config_path}")


def create_user_item_matrix(
    interactions: List[Tuple[int, int, float]], 
    n_users: int, 
    n_items: int
) -> np.ndarray:
    """Create user-item interaction matrix.
    
    Args:
        interactions: List of (user_id, item_id, rating) tuples
        n_users: Number of users
        n_items: Number of items
        
    Returns:
        User-item interaction matrix
    """
    matrix = np.zeros((n_users, n_items))
    
    for user_id, item_id, rating in interactions:
        matrix[user_id, item_id] = rating
    
    return matrix


def split_data_chronological(
    interactions: List[Tuple[int, int, float, int]], 
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[List[Tuple[int, int, float, int]], 
           List[Tuple[int, int, float, int]], 
           List[Tuple[int, int, float, int]]]:
    """Split interactions chronologically for time-aware evaluation.
    
    Args:
        interactions: List of (user_id, item_id, rating, timestamp) tuples
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        
    Returns:
        Train, validation, and test interaction lists
    """
    # Sort by timestamp
    sorted_interactions = sorted(interactions, key=lambda x: x[3])
    
    n_total = len(sorted_interactions)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = sorted_interactions[:n_train]
    val_data = sorted_interactions[n_train:n_train + n_val]
    test_data = sorted_interactions[n_train + n_val:]
    
    logger.info(f"Split data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    return train_data, val_data, test_data


def calculate_popularity_bias(
    recommendations: Dict[int, List[int]], 
    item_popularity: Dict[int, int]
) -> float:
    """Calculate popularity bias in recommendations.
    
    Args:
        recommendations: Dictionary mapping user_id to list of recommended items
        item_popularity: Dictionary mapping item_id to popularity count
        
    Returns:
        Average popularity of recommended items
    """
    total_popularity = 0
    total_recommendations = 0
    
    for user_id, recs in recommendations.items():
        for item_id in recs:
            total_popularity += item_popularity.get(item_id, 0)
            total_recommendations += 1
    
    return total_popularity / total_recommendations if total_recommendations > 0 else 0


def calculate_coverage(
    recommendations: Dict[int, List[int]], 
    total_items: int
) -> float:
    """Calculate recommendation coverage.
    
    Args:
        recommendations: Dictionary mapping user_id to list of recommended items
        total_items: Total number of items in catalog
        
    Returns:
        Coverage percentage
    """
    recommended_items = set()
    
    for recs in recommendations.values():
        recommended_items.update(recs)
    
    return len(recommended_items) / total_items * 100


def calculate_diversity(
    recommendations: Dict[int, List[int]], 
    item_features: Optional[np.ndarray] = None
) -> float:
    """Calculate intra-list diversity of recommendations.
    
    Args:
        recommendations: Dictionary mapping user_id to list of recommended items
        item_features: Item feature matrix for calculating diversity
        
    Returns:
        Average intra-list diversity
    """
    if item_features is None:
        # Simple diversity based on unique items
        diversities = []
        for recs in recommendations.values():
            if len(recs) > 1:
                diversity = len(set(recs)) / len(recs)
                diversities.append(diversity)
        return np.mean(diversities) if diversities else 0.0
    
    # Feature-based diversity using cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    diversities = []
    for recs in recommendations.values():
        if len(recs) > 1:
            rec_features = item_features[recs]
            similarities = cosine_similarity(rec_features)
            # Average pairwise similarity (lower = more diverse)
            avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            diversity = 1 - avg_similarity
            diversities.append(diversity)
    
    return np.mean(diversities) if diversities else 0.0
