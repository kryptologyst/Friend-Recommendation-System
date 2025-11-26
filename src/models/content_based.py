"""Content-based recommendation models."""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging

from .base import BaseRecommender

logger = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommendation using TF-IDF and cosine similarity."""
    
    def __init__(self, name: str = "ContentBased", use_sbert: bool = False):
        """Initialize content-based recommender.
        
        Args:
            name: Model name
            use_sbert: Whether to use Sentence-BERT embeddings
        """
        super().__init__(name)
        self.use_sbert = use_sbert
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.sbert_model: Optional[SentenceTransformer] = None
        self.user_features: Optional[np.ndarray] = None
        self.item_features: Optional[np.ndarray] = None
        self.n_users = 0
        self.n_items = 0
    
    def fit(self, interactions: np.ndarray, user_interests: Dict[int, str], 
            item_features: Optional[Dict[int, str]] = None, **kwargs) -> None:
        """Fit the content-based model.
        
        Args:
            interactions: User-item interaction matrix
            user_interests: Dictionary mapping user_id to interests string
            item_features: Dictionary mapping item_id to item description
            **kwargs: Additional parameters
        """
        self.n_users, self.n_items = interactions.shape
        
        if self.use_sbert:
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create user embeddings
            user_texts = [user_interests.get(i, "") for i in range(self.n_users)]
            self.user_features = self.sbert_model.encode(user_texts)
            
            # Create item embeddings if available
            if item_features:
                item_texts = [item_features.get(i, "") for i in range(self.n_items)]
                self.item_features = self.sbert_model.encode(item_texts)
            else:
                # Use user interests as item features (for friend recommendations)
                self.item_features = self.user_features
        else:
            # Use TF-IDF
            user_texts = [user_interests.get(i, "") for i in range(self.n_users)]
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            self.user_features = self.vectorizer.fit_transform(user_texts).toarray()
            
            if item_features:
                item_texts = [item_features.get(i, "") for i in range(self.n_items)]
                self.item_features = self.vectorizer.transform(item_texts).toarray()
            else:
                self.item_features = self.user_features
        
        self.is_fitted = True
        logger.info(f"Content-based model fitted with {self.n_users} users and {self.n_items} items")
    
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
        
        # Calculate similarities
        user_feature = self.user_features[user_id].reshape(1, -1)
        similarities = cosine_similarity(user_feature, self.item_features).flatten()
        
        # Exclude the user themselves (for friend recommendations)
        similarities[user_id] = -1
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
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
    
    def get_similarity_score(self, user_id: int, item_id: int) -> float:
        """Get similarity score between user and item.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Similarity score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting similarity scores")
        
        user_feature = self.user_features[user_id].reshape(1, -1)
        item_feature = self.item_features[item_id].reshape(1, -1)
        
        similarity = cosine_similarity(user_feature, item_feature)[0, 0]
        return similarity


class InterestBasedRecommender(BaseRecommender):
    """Interest-based friend recommendation using shared interests."""
    
    def __init__(self, name: str = "InterestBased"):
        """Initialize interest-based recommender.
        
        Args:
            name: Model name
        """
        super().__init__(name)
        self.user_interests: Optional[Dict[int, set]] = None
        self.n_users = 0
    
    def fit(self, interactions: np.ndarray, user_interests: Dict[int, str], **kwargs) -> None:
        """Fit the interest-based model.
        
        Args:
            interactions: User-item interaction matrix
            user_interests: Dictionary mapping user_id to interests string
            **kwargs: Additional parameters
        """
        self.n_users = interactions.shape[0]
        
        # Parse interests into sets
        self.user_interests = {}
        for user_id, interests_str in user_interests.items():
            interests = set(interests_str.split(', ')) if interests_str else set()
            self.user_interests[user_id] = interests
        
        self.is_fitted = True
        logger.info(f"Interest-based model fitted with {self.n_users} users")
    
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate recommendations based on shared interests.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommended user IDs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id >= self.n_users:
            raise ValueError(f"User ID {user_id} out of range")
        
        user_interests = self.user_interests[user_id]
        
        # Calculate shared interests with all other users
        similarities = []
        for other_user_id in range(self.n_users):
            if other_user_id == user_id:
                continue
            
            other_interests = self.user_interests[other_user_id]
            
            # Jaccard similarity
            intersection = len(user_interests.intersection(other_interests))
            union = len(user_interests.union(other_interests))
            similarity = intersection / union if union > 0 else 0
            
            similarities.append((other_user_id, similarity))
        
        # Sort by similarity and return top recommendations
        similarities.sort(key=lambda x: x[1], reverse=True)
        recommendations = [user_id for user_id, _ in similarities[:n_recommendations]]
        
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
    
    def get_similarity_score(self, user_id: int, item_id: int) -> float:
        """Get similarity score based on shared interests.
        
        Args:
            user_id: User ID
            item_id: Item ID (other user ID)
            
        Returns:
            Jaccard similarity score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting similarity scores")
        
        user_interests = self.user_interests[user_id]
        other_interests = self.user_interests[item_id]
        
        intersection = len(user_interests.intersection(other_interests))
        union = len(user_interests.union(other_interests))
        
        return intersection / union if union > 0 else 0
