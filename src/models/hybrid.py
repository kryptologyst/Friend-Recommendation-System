"""Hybrid recommendation models."""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging

from .base import BaseRecommender
from .content_based import ContentBasedRecommender, InterestBasedRecommender
from .collaborative_filtering import UserBasedCF, ItemBasedCF, MatrixFactorizationCF
from .graph_based import GraphBasedRecommender, CommunityBasedRecommender

logger = logging.getLogger(__name__)


class HybridRecommender(BaseRecommender):
    """Hybrid recommendation combining multiple approaches."""
    
    def __init__(self, name: str = "Hybrid", weights: Optional[Dict[str, float]] = None):
        """Initialize hybrid recommender.
        
        Args:
            name: Model name
            weights: Dictionary mapping model names to weights
        """
        super().__init__(name)
        self.weights = weights or {
            'content': 0.3,
            'collaborative': 0.4,
            'graph': 0.3
        }
        self.models: Dict[str, BaseRecommender] = {}
        self.n_users = 0
        self.n_items = 0
    
    def fit(self, interactions: np.ndarray, user_interests: Dict[int, str], **kwargs) -> None:
        """Fit all component models.
        
        Args:
            interactions: User-item interaction matrix
            user_interests: Dictionary mapping user_id to interests string
            **kwargs: Additional parameters
        """
        self.n_users, self.n_items = interactions.shape
        
        # Initialize and fit content-based model
        self.models['content'] = ContentBasedRecommender("ContentBased")
        self.models['content'].fit(interactions, user_interests)
        
        # Initialize and fit collaborative filtering model
        self.models['collaborative'] = UserBasedCF("UserBasedCF")
        self.models['collaborative'].fit(interactions)
        
        # Initialize and fit graph-based model
        self.models['graph'] = GraphBasedRecommender("GraphBased")
        self.models['graph'].fit(interactions)
        
        self.is_fitted = True
        logger.info("Hybrid model fitted with content, collaborative, and graph components")
    
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate hybrid recommendations.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id >= self.n_users:
            raise ValueError(f"User ID {user_id} out of range")
        
        # Get recommendations from each model
        all_recommendations = {}
        
        for model_name, model in self.models.items():
            try:
                recs = model.predict(user_id, n_recommendations * 2)  # Get more for better combination
                weight = self.weights.get(model_name, 0.0)
                
                for i, item_id in enumerate(recs):
                    score = weight * (len(recs) - i) / len(recs)  # Rank-based scoring
                    all_recommendations[item_id] = all_recommendations.get(item_id, 0) + score
            except Exception as e:
                logger.warning(f"Error getting recommendations from {model_name}: {e}")
        
        # Sort by combined score and return top recommendations
        sorted_items = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, _ in sorted_items[:n_recommendations]]
        
        return recommendations
    
    def predict_batch(self, user_ids: List[int], n_recommendations: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n_recommendations: Number of recommendations per user
            
        Returns:
            Dictionary mapping user_id to recommendations
        """
        recommendations = {}
        
        for user_id in user_ids:
            recommendations[user_id] = self.predict(user_id, n_recommendations)
        
        return recommendations


class WeightedHybridRecommender(BaseRecommender):
    """Weighted hybrid recommendation with learned weights."""
    
    def __init__(self, name: str = "WeightedHybrid"):
        """Initialize weighted hybrid recommender.
        
        Args:
            name: Model name
        """
        super().__init__(name)
        self.models: Dict[str, BaseRecommender] = {}
        self.weights: Optional[np.ndarray] = None
        self.n_users = 0
        self.n_items = 0
    
    def fit(self, interactions: np.ndarray, user_interests: Dict[int, str], 
            validation_data: Optional[tuple] = None, **kwargs) -> None:
        """Fit component models and learn optimal weights.
        
        Args:
            interactions: User-item interaction matrix
            user_interests: Dictionary mapping user_id to interests string
            validation_data: Validation data for learning weights
            **kwargs: Additional parameters
        """
        self.n_users, self.n_items = interactions.shape
        
        # Initialize models
        self.models = {
            'content': ContentBasedRecommender("ContentBased"),
            'collaborative': UserBasedCF("UserBasedCF"),
            'graph': GraphBasedRecommender("GraphBased"),
            'interest': InterestBasedRecommender("InterestBased")
        }
        
        # Fit all models
        for model in self.models.values():
            model.fit(interactions, user_interests)
        
        # Learn optimal weights if validation data is provided
        if validation_data is not None:
            self._learn_weights(validation_data)
        else:
            # Use equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        self.is_fitted = True
        logger.info("Weighted hybrid model fitted with learned weights")
    
    def _learn_weights(self, validation_data: tuple) -> None:
        """Learn optimal weights using validation data.
        
        Args:
            validation_data: Tuple of (val_interactions, val_user_interests)
        """
        val_interactions, val_user_interests = validation_data
        
        # Get predictions from each model
        model_predictions = {}
        for model_name, model in self.models.items():
            predictions = {}
            for user_id in range(min(100, self.n_users)):  # Sample users for efficiency
                try:
                    predictions[user_id] = model.predict(user_id, 10)
                except:
                    predictions[user_id] = []
            model_predictions[model_name] = predictions
        
        # Simple weight learning based on prediction quality
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        weights = []
        for model_name, predictions in model_predictions.items():
            # Calculate a simple quality metric (e.g., average prediction length)
            avg_length = np.mean([len(recs) for recs in predictions.values()])
            weights.append(avg_length)
        
        # Normalize weights
        total_weight = sum(weights)
        self.weights = np.array([w / total_weight for w in weights])
        
        logger.info(f"Learned weights: {dict(zip(self.models.keys(), self.weights))}")
    
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate weighted hybrid recommendations.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id >= self.n_users:
            raise ValueError(f"User ID {user_id} out of range")
        
        # Get recommendations from each model
        all_recommendations = {}
        
        for i, (model_name, model) in enumerate(self.models.items()):
            try:
                recs = model.predict(user_id, n_recommendations * 2)
                weight = self.weights[i]
                
                for j, item_id in enumerate(recs):
                    score = weight * (len(recs) - j) / len(recs)
                    all_recommendations[item_id] = all_recommendations.get(item_id, 0) + score
            except Exception as e:
                logger.warning(f"Error getting recommendations from {model_name}: {e}")
        
        # Sort by combined score and return top recommendations
        sorted_items = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, _ in sorted_items[:n_recommendations]]
        
        return recommendations
    
    def predict_batch(self, user_ids: List[int], n_recommendations: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n_recommendations: Number of recommendations per user
            
        Returns:
            Dictionary mapping user_id to recommendations
        """
        recommendations = {}
        
        for user_id in user_ids:
            recommendations[user_id] = self.predict(user_id, n_recommendations)
        
        return recommendations


class EnsembleRecommender(BaseRecommender):
    """Ensemble recommendation using voting."""
    
    def __init__(self, name: str = "Ensemble", voting_method: str = "rank"):
        """Initialize ensemble recommender.
        
        Args:
            name: Model name
            voting_method: Voting method ('rank', 'score', 'majority')
        """
        super().__init__(name)
        self.voting_method = voting_method
        self.models: List[BaseRecommender] = []
        self.n_users = 0
        self.n_items = 0
    
    def add_model(self, model: BaseRecommender) -> None:
        """Add a model to the ensemble.
        
        Args:
            model: Recommendation model to add
        """
        self.models.append(model)
        logger.info(f"Added {model.name} to ensemble")
    
    def fit(self, interactions: np.ndarray, user_interests: Dict[int, str], **kwargs) -> None:
        """Fit all models in the ensemble.
        
        Args:
            interactions: User-item interaction matrix
            user_interests: Dictionary mapping user_id to interests string
            **kwargs: Additional parameters
        """
        self.n_users, self.n_items = interactions.shape
        
        # Fit all models
        for model in self.models:
            model.fit(interactions, user_interests)
        
        self.is_fitted = True
        logger.info(f"Ensemble model fitted with {len(self.models)} models")
    
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate ensemble recommendations.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id >= self.n_users:
            raise ValueError(f"User ID {user_id} out of range")
        
        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            try:
                predictions = model.predict(user_id, n_recommendations)
                all_predictions.append(predictions)
            except Exception as e:
                logger.warning(f"Error getting predictions from {model.name}: {e}")
        
        if not all_predictions:
            return []
        
        # Combine predictions based on voting method
        if self.voting_method == "rank":
            return self._rank_voting(all_predictions, n_recommendations)
        elif self.voting_method == "score":
            return self._score_voting(all_predictions, n_recommendations)
        elif self.voting_method == "majority":
            return self._majority_voting(all_predictions, n_recommendations)
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
    
    def _rank_voting(self, predictions: List[List[int]], n_recommendations: int) -> List[int]:
        """Combine predictions using rank-based voting.
        
        Args:
            predictions: List of prediction lists from different models
            n_recommendations: Number of final recommendations
            
        Returns:
            Combined recommendations
        """
        item_scores = {}
        
        for model_predictions in predictions:
            for rank, item_id in enumerate(model_predictions):
                score = 1.0 / (rank + 1)  # Higher rank = higher score
                item_scores[item_id] = item_scores.get(item_id, 0) + score
        
        # Sort by score and return top recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:n_recommendations]]
    
    def _score_voting(self, predictions: List[List[int]], n_recommendations: int) -> List[int]:
        """Combine predictions using score-based voting.
        
        Args:
            predictions: List of prediction lists from different models
            n_recommendations: Number of final recommendations
            
        Returns:
            Combined recommendations
        """
        item_scores = {}
        
        for model_predictions in predictions:
            for item_id in model_predictions:
                item_scores[item_id] = item_scores.get(item_id, 0) + 1
        
        # Sort by score and return top recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:n_recommendations]]
    
    def _majority_voting(self, predictions: List[List[int]], n_recommendations: int) -> List[int]:
        """Combine predictions using majority voting.
        
        Args:
            predictions: List of prediction lists from different models
            n_recommendations: Number of final recommendations
            
        Returns:
            Combined recommendations
        """
        # Count votes for each item
        item_votes = {}
        for model_predictions in predictions:
            for item_id in model_predictions:
                item_votes[item_id] = item_votes.get(item_id, 0) + 1
        
        # Sort by vote count and return top recommendations
        sorted_items = sorted(item_votes.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:n_recommendations]]
    
    def predict_batch(self, user_ids: List[int], n_recommendations: int = 10) -> Dict[int, List[int]]:
        """Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n_recommendations: Number of recommendations per user
            
        Returns:
            Dictionary mapping user_id to recommendations
        """
        recommendations = {}
        
        for user_id in user_ids:
            recommendations[user_id] = self.predict(user_id, n_recommendations)
        
        return recommendations
