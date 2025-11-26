"""Base recommendation model interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Base class for recommendation models."""
    
    def __init__(self, name: str):
        """Initialize base recommender.
        
        Args:
            name: Model name
        """
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, interactions: np.ndarray, **kwargs) -> None:
        """Fit the recommendation model.
        
        Args:
            interactions: User-item interaction matrix
            **kwargs: Additional fitting parameters
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to generate
            
        Returns:
            List of recommended item IDs
        """
        pass
    
    @abstractmethod
    def predict_batch(self, user_ids: List[int], n_recommendations: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n_recommendations: Number of recommendations per user
            
        Returns:
            Dictionary mapping user_id to list of recommendations
        """
        pass
    
    def get_similarity_score(self, user_id: int, item_id: int) -> float:
        """Get similarity score between user and item.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Similarity score
        """
        raise NotImplementedError("Similarity scoring not implemented for this model")
    
    def save_model(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save model
        """
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseRecommender':
        """Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model instance
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {filepath}")
        return model
