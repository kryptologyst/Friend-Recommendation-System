"""Data processing utilities for the friend recommendation system."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader for friend recommendation system."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.users_df: Optional[pd.DataFrame] = None
        self.interactions_df: Optional[pd.DataFrame] = None
        self.items_df: Optional[pd.DataFrame] = None
    
    def load_users(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load user data.
        
        Args:
            file_path: Path to users CSV file
            
        Returns:
            Users DataFrame
        """
        if file_path is None:
            file_path = self.data_dir / "raw" / "users.csv"
        
        self.users_df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(self.users_df)} users from {file_path}")
        return self.users_df
    
    def load_interactions(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load interaction data.
        
        Args:
            file_path: Path to interactions CSV file
            
        Returns:
            Interactions DataFrame
        """
        if file_path is None:
            file_path = self.data_dir / "raw" / "interactions.csv"
        
        self.interactions_df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(self.interactions_df)} interactions from {file_path}")
        return self.interactions_df
    
    def load_items(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load item data.
        
        Args:
            file_path: Path to items CSV file
            
        Returns:
            Items DataFrame
        """
        if file_path is None:
            file_path = self.data_dir / "raw" / "items.csv"
        
        self.items_df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(self.items_df)} items from {file_path}")
        return self.items_df
    
    def get_user_interests(self) -> Dict[int, str]:
        """Extract user interests as dictionary.
        
        Returns:
            Dictionary mapping user_id to interests string
        """
        if self.users_df is None:
            raise ValueError("Users data not loaded")
        
        return dict(zip(self.users_df['user_id'], self.users_df['interests']))
    
    def get_interaction_matrix(self) -> np.ndarray:
        """Create user-item interaction matrix.
        
        Returns:
            User-item interaction matrix
        """
        if self.interactions_df is None:
            raise ValueError("Interactions data not loaded")
        
        n_users = self.users_df['user_id'].nunique() if self.users_df is not None else self.interactions_df['user_id'].nunique()
        n_items = self.items_df['item_id'].nunique() if self.items_df is not None else self.interactions_df['friend_id'].nunique()
        
        matrix = np.zeros((n_users, n_items))
        
        for _, row in self.interactions_df.iterrows():
            user_idx = row['user_id']
            item_idx = row['friend_id']
            weight = row.get('weight', 1.0)
            matrix[user_idx, item_idx] = weight
        
        return matrix
    
    def get_user_item_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Get user and item ID to index mappings.
        
        Returns:
            Tuple of (user_id_to_idx, item_id_to_idx) dictionaries
        """
        if self.users_df is None or self.interactions_df is None:
            raise ValueError("Users and interactions data not loaded")
        
        user_ids = sorted(self.users_df['user_id'].unique())
        item_ids = sorted(self.interactions_df['friend_id'].unique())
        
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        
        return user_id_to_idx, item_id_to_idx


class DataGenerator:
    """Generate synthetic data for friend recommendation system."""
    
    def __init__(self, n_users: int = 1000, n_items: int = 500, seed: int = 42):
        """Initialize data generator.
        
        Args:
            n_users: Number of users to generate
            n_items: Number of items/interests to generate
            seed: Random seed
        """
        self.n_users = n_users
        self.n_items = n_items
        self.seed = seed
        np.random.seed(seed)
        
        # Define interest categories and activities
        self.interest_categories = [
            "sports", "music", "movies", "books", "travel", "food", 
            "technology", "art", "photography", "fitness", "gaming",
            "cooking", "reading", "hiking", "dancing", "writing"
        ]
        
        self.activities = [
            "basketball", "hiking", "watching movies", "reading", "cooking",
            "photography", "traveling", "video games", "fitness", "outdoor adventures",
            "music", "art", "dancing", "writing", "swimming", "running"
        ]
    
    def generate_users(self) -> pd.DataFrame:
        """Generate synthetic user data.
        
        Returns:
            Users DataFrame
        """
        users_data = []
        
        for i in range(self.n_users):
            # Generate random interests (3-8 interests per user)
            n_interests = np.random.randint(3, 9)
            interests = np.random.choice(self.activities, n_interests, replace=False)
            interests_str = ", ".join(interests)
            
            # Generate other user attributes
            age = np.random.randint(18, 65)
            locations = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"]
            occupations = ["Engineer", "Teacher", "Doctor", "Artist", "Student", "Manager"]
            
            users_data.append({
                'user_id': i,
                'name': f"User_{i}",
                'interests': interests_str,
                'age': age,
                'location': np.random.choice(locations),
                'occupation': np.random.choice(occupations)
            })
        
        return pd.DataFrame(users_data)
    
    def generate_interactions(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic interaction data.
        
        Args:
            users_df: Users DataFrame
            
        Returns:
            Interactions DataFrame
        """
        interactions_data = []
        
        # Generate interactions based on shared interests
        for user_id in range(self.n_users):
            user_interests = set(users_df.iloc[user_id]['interests'].split(', '))
            
            # Generate 5-20 interactions per user
            n_interactions = np.random.randint(5, 21)
            
            # Higher probability of interaction with users who share interests
            for _ in range(n_interactions):
                # Choose friend with higher probability if they share interests
                friend_id = np.random.randint(0, self.n_users)
                if friend_id == user_id:
                    continue
                
                friend_interests = set(users_df.iloc[friend_id]['interests'].split(', '))
                shared_interests = len(user_interests.intersection(friend_interests))
                
                # Weight based on shared interests
                weight = 1.0 + shared_interests * 0.5
                
                # Add some randomness
                weight += np.random.normal(0, 0.2)
                weight = max(0.1, weight)  # Ensure positive weight
                
                interactions_data.append({
                    'user_id': user_id,
                    'friend_id': friend_id,
                    'timestamp': np.random.randint(1000000000, 2000000000),  # Random timestamp
                    'interaction_type': np.random.choice(['friend_request', 'message', 'like']),
                    'weight': weight
                })
        
        return pd.DataFrame(interactions_data)
    
    def generate_items(self) -> pd.DataFrame:
        """Generate synthetic item/interest data.
        
        Returns:
            Items DataFrame
        """
        items_data = []
        
        for i, activity in enumerate(self.activities):
            category = np.random.choice(self.interest_categories)
            
            items_data.append({
                'item_id': i,
                'name': activity,
                'category': category,
                'description': f"Activity: {activity} in {category} category"
            })
        
        return pd.DataFrame(items_data)
    
    def generate_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate all synthetic data.
        
        Returns:
            Tuple of (users_df, interactions_df, items_df)
        """
        logger.info(f"Generating synthetic data: {self.n_users} users, {self.n_items} items")
        
        users_df = self.generate_users()
        interactions_df = self.generate_interactions(users_df)
        items_df = self.generate_items()
        
        logger.info(f"Generated {len(users_df)} users, {len(interactions_df)} interactions, {len(items_df)} items")
        
        return users_df, interactions_df, items_df
