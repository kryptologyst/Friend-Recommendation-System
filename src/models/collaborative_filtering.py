"""Collaborative filtering recommendation models."""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import implicit
import logging

from .base import BaseRecommender

logger = logging.getLogger(__name__)


class UserBasedCF(BaseRecommender):
    """User-based collaborative filtering."""
    
    def __init__(self, name: str = "UserBasedCF", n_neighbors: int = 50):
        """Initialize user-based CF.
        
        Args:
            name: Model name
            n_neighbors: Number of neighbors for user similarity
        """
        super().__init__(name)
        self.n_neighbors = n_neighbors
        self.knn_model: Optional[NearestNeighbors] = None
        self.interactions: Optional[np.ndarray] = None
        self.n_users = 0
        self.n_items = 0
    
    def fit(self, interactions: np.ndarray, **kwargs) -> None:
        """Fit the user-based CF model.
        
        Args:
            interactions: User-item interaction matrix
            **kwargs: Additional parameters
        """
        self.interactions = interactions
        self.n_users, self.n_items = interactions.shape
        
        # Fit k-NN model for user similarity
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, self.n_users),
            metric='cosine',
            algorithm='brute'
        )
        self.knn_model.fit(interactions)
        
        self.is_fitted = True
        logger.info(f"User-based CF fitted with {self.n_users} users")
    
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate recommendations for a user.
        
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
        
        # Find similar users
        user_vector = self.interactions[user_id].reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(user_vector)
        
        # Get similar users (excluding the user themselves)
        similar_users = indices[0][1:]  # Skip first (self)
        similarities = 1 - distances[0][1:]  # Convert distance to similarity
        
        # Calculate weighted scores for items
        scores = np.zeros(self.n_items)
        user_items = self.interactions[user_id]
        
        for i, similar_user in enumerate(similar_users):
            similarity = similarities[i]
            similar_user_items = self.interactions[similar_user]
            
            # Add weighted scores for items not already interacted with
            mask = (user_items == 0) & (similar_user_items > 0)
            scores[mask] += similarity * similar_user_items[mask]
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        return top_indices.tolist()
    
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


class ItemBasedCF(BaseRecommender):
    """Item-based collaborative filtering."""
    
    def __init__(self, name: str = "ItemBasedCF", n_neighbors: int = 50):
        """Initialize item-based CF.
        
        Args:
            name: Model name
            n_neighbors: Number of neighbors for item similarity
        """
        super().__init__(name)
        self.n_neighbors = n_neighbors
        self.knn_model: Optional[NearestNeighbors] = None
        self.interactions: Optional[np.ndarray] = None
        self.n_users = 0
        self.n_items = 0
    
    def fit(self, interactions: np.ndarray, **kwargs) -> None:
        """Fit the item-based CF model.
        
        Args:
            interactions: User-item interaction matrix
            **kwargs: Additional parameters
        """
        self.interactions = interactions
        self.n_users, self.n_items = interactions.shape
        
        # Fit k-NN model for item similarity
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, self.n_items),
            metric='cosine',
            algorithm='brute'
        )
        self.knn_model.fit(interactions.T)  # Transpose for item-based
        
        self.is_fitted = True
        logger.info(f"Item-based CF fitted with {self.n_items} items")
    
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate recommendations for a user.
        
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
        
        user_items = self.interactions[user_id]
        scores = np.zeros(self.n_items)
        
        # For each item the user has interacted with
        interacted_items = np.where(user_items > 0)[0]
        
        for item_id in interacted_items:
            # Find similar items
            item_vector = self.interactions[:, item_id].reshape(1, -1)
            distances, indices = self.knn_model.kneighbors(item_vector)
            
            similar_items = indices[0][1:]  # Skip first (self)
            similarities = 1 - distances[0][1:]  # Convert distance to similarity
            
            # Add weighted scores
            for i, similar_item in enumerate(similar_items):
                similarity = similarities[i]
                scores[similar_item] += similarity * user_items[item_id]
        
        # Set already interacted items to 0
        scores[interacted_items] = 0
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        return top_indices.tolist()
    
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


class MatrixFactorizationCF(BaseRecommender):
    """Matrix factorization using SVD."""
    
    def __init__(self, name: str = "MatrixFactorizationCF", n_components: int = 50):
        """Initialize matrix factorization CF.
        
        Args:
            name: Model name
            n_components: Number of latent factors
        """
        super().__init__(name)
        self.n_components = n_components
        self.svd_model: Optional[TruncatedSVD] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.n_users = 0
        self.n_items = 0
    
    def fit(self, interactions: np.ndarray, **kwargs) -> None:
        """Fit the matrix factorization model.
        
        Args:
            interactions: User-item interaction matrix
            **kwargs: Additional parameters
        """
        self.n_users, self.n_items = interactions.shape
        
        # Apply SVD
        self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_factors = self.svd_model.fit_transform(interactions)
        self.item_factors = self.svd_model.components_.T
        
        self.is_fitted = True
        logger.info(f"Matrix factorization CF fitted with {self.n_components} components")
    
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate recommendations for a user.
        
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
        
        # Calculate predicted scores
        user_factor = self.user_factors[user_id]
        scores = np.dot(self.item_factors, user_factor)
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        return top_indices.tolist()
    
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


class ALSRecommender(BaseRecommender):
    """Alternating Least Squares (ALS) matrix factorization."""
    
    def __init__(self, name: str = "ALS", factors: int = 50, iterations: int = 15):
        """Initialize ALS recommender.
        
        Args:
            name: Model name
            factors: Number of latent factors
            iterations: Number of iterations
        """
        super().__init__(name)
        self.factors = factors
        self.iterations = iterations
        self.model: Optional[implicit.als.AlternatingLeastSquares] = None
        self.n_users = 0
        self.n_items = 0
    
    def fit(self, interactions: np.ndarray, **kwargs) -> None:
        """Fit the ALS model.
        
        Args:
            interactions: User-item interaction matrix
            **kwargs: Additional parameters
        """
        self.n_users, self.n_items = interactions.shape
        
        # Initialize ALS model
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            random_state=42
        )
        
        # Convert to CSR format for implicit
        from scipy.sparse import csr_matrix
        interactions_sparse = csr_matrix(interactions)
        
        # Fit the model
        self.model.fit(interactions_sparse)
        
        self.is_fitted = True
        logger.info(f"ALS model fitted with {self.factors} factors")
    
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate recommendations for a user.
        
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
        
        # Get recommendations
        recommendations, scores = self.model.recommend(
            user_id, 
            self.model.user_items[user_id], 
            N=n_recommendations
        )
        
        return recommendations.tolist()
    
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
