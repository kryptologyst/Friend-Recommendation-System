"""Graph-based recommendation models."""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from sklearn.neighbors import NearestNeighbors
import logging

from .base import BaseRecommender

logger = logging.getLogger(__name__)


class GraphBasedRecommender(BaseRecommender):
    """Graph-based friend recommendation using social network analysis."""
    
    def __init__(self, name: str = "GraphBased", algorithm: str = "pagerank"):
        """Initialize graph-based recommender.
        
        Args:
            name: Model name
            algorithm: Graph algorithm to use ('pagerank', 'betweenness', 'closeness')
        """
        super().__init__(name)
        self.algorithm = algorithm
        self.graph: Optional[nx.Graph] = None
        self.centrality_scores: Optional[Dict[int, float]] = None
        self.n_users = 0
    
    def fit(self, interactions: np.ndarray, **kwargs) -> None:
        """Fit the graph-based model.
        
        Args:
            interactions: User-item interaction matrix
            **kwargs: Additional parameters
        """
        self.n_users = interactions.shape[0]
        
        # Create social network graph
        self.graph = nx.Graph()
        
        # Add nodes (users)
        for user_id in range(self.n_users):
            self.graph.add_node(user_id)
        
        # Add edges based on interactions
        for user_id in range(self.n_users):
            for friend_id in range(self.n_users):
                if user_id != friend_id and interactions[user_id, friend_id] > 0:
                    weight = interactions[user_id, friend_id]
                    self.graph.add_edge(user_id, friend_id, weight=weight)
        
        # Calculate centrality scores
        if self.algorithm == "pagerank":
            self.centrality_scores = nx.pagerank(self.graph, weight='weight')
        elif self.algorithm == "betweenness":
            self.centrality_scores = nx.betweenness_centrality(self.graph, weight='weight')
        elif self.algorithm == "closeness":
            self.centrality_scores = nx.closeness_centrality(self.graph, distance='weight')
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.is_fitted = True
        logger.info(f"Graph-based model fitted with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate recommendations using graph-based approach.
        
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
        
        # Get user's neighbors (already connected friends)
        neighbors = set(self.graph.neighbors(user_id))
        
        # Calculate recommendation scores
        scores = {}
        
        for candidate_user in range(self.n_users):
            if candidate_user == user_id or candidate_user in neighbors:
                continue
            
            # Calculate various graph-based scores
            score = 0
            
            # Common neighbors score
            candidate_neighbors = set(self.graph.neighbors(candidate_user))
            common_neighbors = neighbors.intersection(candidate_neighbors)
            if len(neighbors) > 0:
                score += len(common_neighbors) / len(neighbors)
            
            # Centrality score
            score += self.centrality_scores.get(candidate_user, 0) * 0.1
            
            # Shortest path score (inverse of distance)
            try:
                path_length = nx.shortest_path_length(self.graph, user_id, candidate_user, weight='weight')
                score += 1.0 / (1.0 + path_length)
            except nx.NetworkXNoPath:
                pass
            
            scores[candidate_user] = score
        
        # Sort by score and return top recommendations
        sorted_users = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [user_id for user_id, _ in sorted_users[:n_recommendations]]
        
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


class CommunityBasedRecommender(BaseRecommender):
    """Community-based friend recommendation."""
    
    def __init__(self, name: str = "CommunityBased", algorithm: str = "louvain"):
        """Initialize community-based recommender.
        
        Args:
            name: Model name
            algorithm: Community detection algorithm ('louvain', 'greedy_modularity')
        """
        super().__init__(name)
        self.algorithm = algorithm
        self.graph: Optional[nx.Graph] = None
        self.communities: Optional[Dict[int, int]] = None
        self.n_users = 0
    
    def fit(self, interactions: np.ndarray, **kwargs) -> None:
        """Fit the community-based model.
        
        Args:
            interactions: User-item interaction matrix
            **kwargs: Additional parameters
        """
        self.n_users = interactions.shape[0]
        
        # Create social network graph
        self.graph = nx.Graph()
        
        # Add nodes (users)
        for user_id in range(self.n_users):
            self.graph.add_node(user_id)
        
        # Add edges based on interactions
        for user_id in range(self.n_users):
            for friend_id in range(self.n_users):
                if user_id != friend_id and interactions[user_id, friend_id] > 0:
                    weight = interactions[user_id, friend_id]
                    self.graph.add_edge(user_id, friend_id, weight=weight)
        
        # Detect communities
        if self.algorithm == "louvain":
            try:
                import community as community_louvain
                self.communities = community_louvain.best_partition(self.graph, weight='weight')
            except ImportError:
                logger.warning("python-louvain not installed, using greedy modularity")
                communities = nx.community.greedy_modularity_communities(self.graph, weight='weight')
                self.communities = {}
                for i, community in enumerate(communities):
                    for user in community:
                        self.communities[user] = i
        elif self.algorithm == "greedy_modularity":
            communities = nx.community.greedy_modularity_communities(self.graph, weight='weight')
            self.communities = {}
            for i, community in enumerate(communities):
                for user in community:
                    self.communities[user] = i
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.is_fitted = True
        logger.info(f"Community-based model fitted with {len(set(self.communities.values()))} communities")
    
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate recommendations based on community structure.
        
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
        
        user_community = self.communities[user_id]
        
        # Get users in the same community
        same_community_users = [
            uid for uid, comm_id in self.communities.items() 
            if comm_id == user_community and uid != user_id
        ]
        
        # Calculate scores based on community membership and graph structure
        scores = {}
        
        for candidate_user in same_community_users:
            score = 0
            
            # Community membership bonus
            score += 1.0
            
            # Graph-based features
            if self.graph.has_edge(user_id, candidate_user):
                # Already connected, lower priority
                score += 0.1
            else:
                # Not connected, higher priority for recommendation
                score += 0.5
                
                # Common neighbors within community
                user_neighbors = set(self.graph.neighbors(user_id))
                candidate_neighbors = set(self.graph.neighbors(candidate_user))
                common_neighbors = user_neighbors.intersection(candidate_neighbors)
                score += len(common_neighbors) * 0.2
            
            scores[candidate_user] = score
        
        # If not enough users in same community, add users from other communities
        if len(scores) < n_recommendations:
            for candidate_user in range(self.n_users):
                if candidate_user == user_id or candidate_user in scores:
                    continue
                
                score = 0.1  # Lower score for different community
                scores[candidate_user] = score
        
        # Sort by score and return top recommendations
        sorted_users = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [user_id for user_id, _ in sorted_users[:n_recommendations]]
        
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


class RandomWalkRecommender(BaseRecommender):
    """Random walk-based friend recommendation."""
    
    def __init__(self, name: str = "RandomWalk", walk_length: int = 10, num_walks: int = 100):
        """Initialize random walk recommender.
        
        Args:
            name: Model name
            walk_length: Length of random walks
            num_walks: Number of random walks per user
        """
        super().__init__(name)
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.graph: Optional[nx.Graph] = None
        self.n_users = 0
    
    def fit(self, interactions: np.ndarray, **kwargs) -> None:
        """Fit the random walk model.
        
        Args:
            interactions: User-item interaction matrix
            **kwargs: Additional parameters
        """
        self.n_users = interactions.shape[0]
        
        # Create social network graph
        self.graph = nx.Graph()
        
        # Add nodes (users)
        for user_id in range(self.n_users):
            self.graph.add_node(user_id)
        
        # Add edges based on interactions
        for user_id in range(self.n_users):
            for friend_id in range(self.n_users):
                if user_id != friend_id and interactions[user_id, friend_id] > 0:
                    weight = interactions[user_id, friend_id]
                    self.graph.add_edge(user_id, friend_id, weight=weight)
        
        self.is_fitted = True
        logger.info(f"Random walk model fitted with {self.graph.number_of_nodes()} nodes")
    
    def predict(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate recommendations using random walks.
        
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
        
        # Perform random walks
        visit_counts = {}
        
        for _ in range(self.num_walks):
            current_node = user_id
            visited_nodes = set([current_node])
            
            for _ in range(self.walk_length):
                neighbors = list(self.graph.neighbors(current_node))
                if not neighbors:
                    break
                
                # Weighted random choice based on edge weights
                weights = [self.graph[current_node][neighbor].get('weight', 1.0) for neighbor in neighbors]
                total_weight = sum(weights)
                if total_weight > 0:
                    probabilities = [w / total_weight for w in weights]
                    current_node = np.random.choice(neighbors, p=probabilities)
                    
                    if current_node not in visited_nodes:
                        visit_counts[current_node] = visit_counts.get(current_node, 0) + 1
                        visited_nodes.add(current_node)
        
        # Sort by visit count and return top recommendations
        sorted_users = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
        recommendations = [user_id for user_id, _ in sorted_users[:n_recommendations]]
        
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
