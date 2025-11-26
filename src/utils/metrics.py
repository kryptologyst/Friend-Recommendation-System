"""Evaluation metrics for recommendation systems."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """Class for computing recommendation evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def precision_at_k(self, recommendations: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate Precision@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        if len(top_k_recs) == 0:
            return 0.0
        
        relevant_recommendations = sum(1 for item in top_k_recs if item in relevant_set)
        return relevant_recommendations / len(top_k_recs)
    
    def recall_at_k(self, recommendations: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate Recall@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        relevant_recommendations = sum(1 for item in top_k_recs if item in relevant_set)
        return relevant_recommendations / len(relevant_items)
    
    def map_at_k(self, recommendations: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate Mean Average Precision@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            MAP@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recs):
            if item in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_items)
    
    def ndcg_at_k(self, recommendations: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recs):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def hit_rate_at_k(self, recommendations: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Hit Rate@K score (0 or 1)
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        return 1.0 if any(item in relevant_set for item in top_k_recs) else 0.0
    
    def coverage(self, all_recommendations: Dict[int, List[int]], total_items: int) -> float:
        """Calculate recommendation coverage.
        
        Args:
            all_recommendations: Dictionary mapping user_id to recommendations
            total_items: Total number of items in catalog
            
        Returns:
            Coverage percentage
        """
        if total_items == 0:
            return 0.0
        
        recommended_items = set()
        for recommendations in all_recommendations.values():
            recommended_items.update(recommendations)
        
        return len(recommended_items) / total_items * 100
    
    def novelty(self, all_recommendations: Dict[int, List[int]], item_popularity: Dict[int, int]) -> float:
        """Calculate average novelty of recommendations.
        
        Args:
            all_recommendations: Dictionary mapping user_id to recommendations
            item_popularity: Dictionary mapping item_id to popularity count
            
        Returns:
            Average novelty score
        """
        total_novelty = 0.0
        total_recommendations = 0
        
        for recommendations in all_recommendations.values():
            for item_id in recommendations:
                popularity = item_popularity.get(item_id, 0)
                novelty = 1.0 / (1.0 + popularity)  # Higher popularity = lower novelty
                total_novelty += novelty
                total_recommendations += 1
        
        return total_novelty / total_recommendations if total_recommendations > 0 else 0.0
    
    def diversity(self, all_recommendations: Dict[int, List[int]], 
                  item_features: Optional[np.ndarray] = None) -> float:
        """Calculate intra-list diversity of recommendations.
        
        Args:
            all_recommendations: Dictionary mapping user_id to recommendations
            item_features: Item feature matrix for calculating diversity
            
        Returns:
            Average intra-list diversity
        """
        diversities = []
        
        for recommendations in all_recommendations.values():
            if len(recommendations) <= 1:
                continue
            
            if item_features is not None:
                # Feature-based diversity using cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                
                rec_features = item_features[recommendations]
                similarities = cosine_similarity(rec_features)
                
                # Average pairwise similarity (lower = more diverse)
                mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
                avg_similarity = np.mean(similarities[mask])
                diversity = 1 - avg_similarity
            else:
                # Simple diversity based on unique items
                diversity = len(set(recommendations)) / len(recommendations)
            
            diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def popularity_bias(self, all_recommendations: Dict[int, List[int]], 
                      item_popularity: Dict[int, int]) -> float:
        """Calculate popularity bias in recommendations.
        
        Args:
            all_recommendations: Dictionary mapping user_id to recommendations
            item_popularity: Dictionary mapping item_id to popularity count
            
        Returns:
            Average popularity of recommended items
        """
        total_popularity = 0
        total_recommendations = 0
        
        for recommendations in all_recommendations.values():
            for item_id in recommendations:
                total_popularity += item_popularity.get(item_id, 0)
                total_recommendations += 1
        
        return total_popularity / total_recommendations if total_recommendations > 0 else 0.0
    
    def evaluate_model(self, model_recommendations: Dict[int, List[int]], 
                      test_data: Dict[int, List[int]], 
                      k_values: List[int] = [5, 10, 20],
                      item_popularity: Optional[Dict[int, int]] = None,
                      item_features: Optional[np.ndarray] = None,
                      total_items: Optional[int] = None) -> Dict[str, float]:
        """Evaluate a model comprehensively.
        
        Args:
            model_recommendations: Dictionary mapping user_id to recommendations
            test_data: Dictionary mapping user_id to relevant items
            k_values: List of K values for evaluation
            item_popularity: Dictionary mapping item_id to popularity
            item_features: Item feature matrix
            total_items: Total number of items
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Ranking metrics
        for k in k_values:
            precisions = []
            recalls = []
            maps = []
            ndcgs = []
            hit_rates = []
            
            for user_id, recommendations in model_recommendations.items():
                if user_id in test_data:
                    relevant_items = test_data[user_id]
                    
                    precisions.append(self.precision_at_k(recommendations, relevant_items, k))
                    recalls.append(self.recall_at_k(recommendations, relevant_items, k))
                    maps.append(self.map_at_k(recommendations, relevant_items, k))
                    ndcgs.append(self.ndcg_at_k(recommendations, relevant_items, k))
                    hit_rates.append(self.hit_rate_at_k(recommendations, relevant_items, k))
            
            metrics[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
            metrics[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0
            metrics[f'map@{k}'] = np.mean(maps) if maps else 0.0
            metrics[f'ndcg@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
            metrics[f'hit_rate@{k}'] = np.mean(hit_rates) if hit_rates else 0.0
        
        # Coverage metrics
        if total_items is not None:
            metrics['coverage'] = self.coverage(model_recommendations, total_items)
        
        # Novelty and diversity metrics
        if item_popularity is not None:
            metrics['novelty'] = self.novelty(model_recommendations, item_popularity)
            metrics['popularity_bias'] = self.popularity_bias(model_recommendations, item_popularity)
        
        if item_features is not None:
            metrics['diversity'] = self.diversity(model_recommendations, item_features)
        
        return metrics
    
    def create_leaderboard(self, model_results: Dict[str, Dict[str, float]], 
                          primary_metric: str = 'ndcg@10') -> List[Tuple[str, float]]:
        """Create a model leaderboard.
        
        Args:
            model_results: Dictionary mapping model names to their metrics
            primary_metric: Primary metric for ranking
            
        Returns:
            List of (model_name, primary_metric_score) tuples sorted by score
        """
        leaderboard = []
        
        for model_name, metrics in model_results.items():
            if primary_metric in metrics:
                leaderboard.append((model_name, metrics[primary_metric]))
            else:
                logger.warning(f"Primary metric {primary_metric} not found for model {model_name}")
        
        # Sort by primary metric score (descending)
        leaderboard.sort(key=lambda x: x[1], reverse=True)
        
        return leaderboard
