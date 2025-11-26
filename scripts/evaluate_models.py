#!/usr/bin/env python3
"""Evaluate recommendation models."""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import DataLoader
from models import BaseRecommender
from utils import set_seed, load_config, split_data_chronological
from utils.metrics import RecommendationMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_models(models_dir: Path) -> Dict[str, BaseRecommender]:
    """Load all trained models.
    
    Args:
        models_dir: Directory containing trained models
        
    Returns:
        Dictionary mapping model names to model instances
    """
    models = {}
    
    for model_file in models_dir.glob("*.pkl"):
        if model_file.name == "training_metadata.pkl":
            continue
        
        model_name = model_file.stem
        try:
            model = BaseRecommender.load_model(str(model_file))
            models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
    
    return models


def create_test_data(test_interactions: List[tuple]) -> Dict[int, List[int]]:
    """Create test data dictionary for evaluation.
    
    Args:
        test_interactions: List of test interactions
        
    Returns:
        Dictionary mapping user_id to list of relevant items
    """
    test_data = {}
    
    for user_id, friend_id, weight, _ in test_interactions:
        if user_id not in test_data:
            test_data[user_id] = []
        test_data[user_id].append(friend_id)
    
    return test_data


def calculate_item_popularity(interactions_df: pd.DataFrame) -> Dict[int, int]:
    """Calculate item popularity for novelty and bias metrics.
    
    Args:
        interactions_df: Interactions DataFrame
        
    Returns:
        Dictionary mapping item_id to popularity count
    """
    popularity = {}
    
    for _, row in interactions_df.iterrows():
        friend_id = row['friend_id']
        popularity[friend_id] = popularity.get(friend_id, 0) + 1
    
    return popularity


def main():
    """Evaluate recommendation models."""
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
    
    # Split data for evaluation
    logger.info("Splitting data...")
    
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
    
    # Create test data
    test_data_dict = create_test_data(test_data)
    
    # Calculate item popularity
    item_popularity = calculate_item_popularity(interactions_df)
    
    logger.info(f"Test data: {len(test_data_dict)} users with interactions")
    
    # Load trained models
    logger.info("Loading trained models...")
    
    models_dir = Path(__file__).parent.parent / "models"
    models = load_trained_models(models_dir)
    
    if not models:
        logger.error("No trained models found!")
        return
    
    logger.info(f"Loaded {len(models)} models for evaluation")
    
    # Initialize metrics calculator
    metrics_calc = RecommendationMetrics()
    
    # Evaluate models
    logger.info("Evaluating models...")
    
    model_results = {}
    k_values = config['evaluation']['k_values']
    n_recommendations = config['evaluation']['n_recommendations']
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        try:
            # Generate recommendations for test users
            test_users = list(test_data_dict.keys())
            recommendations = model.predict_batch(test_users, n_recommendations)
            
            # Calculate metrics
            metrics = metrics_calc.evaluate_model(
                recommendations,
                test_data_dict,
                k_values=k_values,
                item_popularity=item_popularity,
                total_items=len(users_df)
            )
            
            model_results[model_name] = metrics
            
            logger.info(f"{model_name} evaluation completed")
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue
    
    # Create leaderboard
    logger.info("Creating leaderboard...")
    
    primary_metric = config['evaluation']['primary_metric']
    leaderboard = metrics_calc.create_leaderboard(model_results, primary_metric)
    
    # Print results
    print("\n" + "="*80)
    print("FRIEND RECOMMENDATION SYSTEM - EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nLeaderboard (sorted by {primary_metric}):")
    print("-" * 50)
    for i, (model_name, score) in enumerate(leaderboard, 1):
        print(f"{i:2d}. {model_name:<20} {primary_metric}: {score:.4f}")
    
    print(f"\nDetailed Results:")
    print("-" * 50)
    
    for model_name, metrics in model_results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(model_results).T
    results_df.to_csv(results_dir / "evaluation_results.csv")
    
    # Save leaderboard
    leaderboard_df = pd.DataFrame(leaderboard, columns=['model', primary_metric])
    leaderboard_df.to_csv(results_dir / "leaderboard.csv", index=False)
    
    logger.info(f"Results saved to {results_dir}")
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
