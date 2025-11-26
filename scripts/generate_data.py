#!/usr/bin/env python3
"""Generate synthetic data for friend recommendation system."""

import os
import sys
import pandas as pd
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import DataGenerator
from utils import set_seed, load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate synthetic data."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    config = load_config(str(config_path))
    
    # Set random seed
    set_seed(config['seed'])
    
    # Create data directories
    data_dir = Path(__file__).parent.parent / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    logger.info("Generating synthetic data...")
    
    generator = DataGenerator(
        n_users=config['data']['n_users'],
        n_items=config['data']['n_items'],
        seed=config['seed']
    )
    
    users_df, interactions_df, items_df = generator.generate_all_data()
    
    # Save data
    logger.info("Saving data...")
    
    users_df.to_csv(raw_dir / "users.csv", index=False)
    interactions_df.to_csv(raw_dir / "interactions.csv", index=False)
    items_df.to_csv(raw_dir / "items.csv", index=False)
    
    logger.info(f"Generated data:")
    logger.info(f"  Users: {len(users_df)}")
    logger.info(f"  Interactions: {len(interactions_df)}")
    logger.info(f"  Items: {len(items_df)}")
    
    # Create some summary statistics
    logger.info("Creating data summary...")
    
    summary = {
        'n_users': len(users_df),
        'n_interactions': len(interactions_df),
        'n_items': len(items_df),
        'avg_interactions_per_user': len(interactions_df) / len(users_df),
        'sparsity': 1 - (len(interactions_df) / (len(users_df) * len(users_df)))
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(processed_dir / "data_summary.csv", index=False)
    
    logger.info("Data generation completed successfully!")


if __name__ == "__main__":
    main()
