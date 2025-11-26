"""Unit tests for the friend recommendation system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import DataLoader, DataGenerator
from models import create_model, get_available_models
from models.content_based import ContentBasedRecommender, InterestBasedRecommender
from models.collaborative_filtering import UserBasedCF, ItemBasedCF, MatrixFactorizationCF
from models.graph_based import GraphBasedRecommender
from models.hybrid import HybridRecommender
from utils import set_seed, create_user_item_matrix, split_data_chronological
from utils.metrics import RecommendationMetrics


class TestDataGenerator:
    """Test cases for DataGenerator."""
    
    def test_data_generator_init(self):
        """Test DataGenerator initialization."""
        generator = DataGenerator(n_users=100, n_items=50, seed=42)
        assert generator.n_users == 100
        assert generator.n_items == 50
        assert generator.seed == 42
    
    def test_generate_users(self):
        """Test user generation."""
        generator = DataGenerator(n_users=10, seed=42)
        users_df = generator.generate_users()
        
        assert len(users_df) == 10
        assert 'user_id' in users_df.columns
        assert 'name' in users_df.columns
        assert 'interests' in users_df.columns
        assert 'age' in users_df.columns
        assert 'location' in users_df.columns
        assert 'occupation' in users_df.columns
    
    def test_generate_items(self):
        """Test item generation."""
        generator = DataGenerator(n_items=20, seed=42)
        items_df = generator.generate_items()
        
        assert len(items_df) == 20
        assert 'item_id' in items_df.columns
        assert 'name' in items_df.columns
        assert 'category' in items_df.columns
        assert 'description' in items_df.columns
    
    def test_generate_interactions(self):
        """Test interaction generation."""
        generator = DataGenerator(n_users=10, seed=42)
        users_df = generator.generate_users()
        interactions_df = generator.generate_interactions(users_df)
        
        assert len(interactions_df) > 0
        assert 'user_id' in interactions_df.columns
        assert 'friend_id' in interactions_df.columns
        assert 'timestamp' in interactions_df.columns
        assert 'interaction_type' in interactions_df.columns
        assert 'weight' in interactions_df.columns


class TestContentBasedRecommender:
    """Test cases for ContentBasedRecommender."""
    
    def test_content_based_init(self):
        """Test ContentBasedRecommender initialization."""
        model = ContentBasedRecommender()
        assert model.name == "ContentBased"
        assert not model.is_fitted
    
    def test_content_based_fit(self):
        """Test ContentBasedRecommender fitting."""
        model = ContentBasedRecommender()
        
        # Create test data
        interactions = np.random.rand(5, 5)
        user_interests = {
            0: "basketball, hiking, movies",
            1: "reading, cooking, hiking",
            2: "photography, traveling, games",
            3: "cooking, reading, friends",
            4: "fitness, basketball, outdoor"
        }
        
        model.fit(interactions, user_interests)
        assert model.is_fitted
        assert model.user_features is not None
    
    def test_content_based_predict(self):
        """Test ContentBasedRecommender prediction."""
        model = ContentBasedRecommender()
        
        # Create test data
        interactions = np.random.rand(5, 5)
        user_interests = {
            0: "basketball, hiking, movies",
            1: "reading, cooking, hiking",
            2: "photography, traveling, games",
            3: "cooking, reading, friends",
            4: "fitness, basketball, outdoor"
        }
        
        model.fit(interactions, user_interests)
        recommendations = model.predict(0, 3)
        
        assert len(recommendations) == 3
        assert all(isinstance(rec, int) for rec in recommendations)
        assert 0 not in recommendations  # Should not recommend self


class TestInterestBasedRecommender:
    """Test cases for InterestBasedRecommender."""
    
    def test_interest_based_init(self):
        """Test InterestBasedRecommender initialization."""
        model = InterestBasedRecommender()
        assert model.name == "InterestBased"
        assert not model.is_fitted
    
    def test_interest_based_fit(self):
        """Test InterestBasedRecommender fitting."""
        model = InterestBasedRecommender()
        
        interactions = np.random.rand(5, 5)
        user_interests = {
            0: "basketball, hiking, movies",
            1: "reading, cooking, hiking",
            2: "photography, traveling, games",
            3: "cooking, reading, friends",
            4: "fitness, basketball, outdoor"
        }
        
        model.fit(interactions, user_interests)
        assert model.is_fitted
        assert model.user_interests is not None
    
    def test_interest_based_predict(self):
        """Test InterestBasedRecommender prediction."""
        model = InterestBasedRecommender()
        
        interactions = np.random.rand(5, 5)
        user_interests = {
            0: "basketball, hiking, movies",
            1: "reading, cooking, hiking",
            2: "photography, traveling, games",
            3: "cooking, reading, friends",
            4: "fitness, basketball, outdoor"
        }
        
        model.fit(interactions, user_interests)
        recommendations = model.predict(0, 3)
        
        assert len(recommendations) == 3
        assert all(isinstance(rec, int) for rec in recommendations)
        assert 0 not in recommendations


class TestCollaborativeFiltering:
    """Test cases for collaborative filtering models."""
    
    def test_user_based_cf_init(self):
        """Test UserBasedCF initialization."""
        model = UserBasedCF()
        assert model.name == "UserBasedCF"
        assert not model.is_fitted
    
    def test_user_based_cf_fit(self):
        """Test UserBasedCF fitting."""
        model = UserBasedCF()
        interactions = np.random.rand(10, 10)
        
        model.fit(interactions)
        assert model.is_fitted
        assert model.knn_model is not None
    
    def test_item_based_cf_fit(self):
        """Test ItemBasedCF fitting."""
        model = ItemBasedCF()
        interactions = np.random.rand(10, 10)
        
        model.fit(interactions)
        assert model.is_fitted
        assert model.knn_model is not None
    
    def test_matrix_factorization_fit(self):
        """Test MatrixFactorizationCF fitting."""
        model = MatrixFactorizationCF()
        interactions = np.random.rand(10, 10)
        
        model.fit(interactions)
        assert model.is_fitted
        assert model.svd_model is not None
        assert model.user_factors is not None
        assert model.item_factors is not None


class TestGraphBasedRecommender:
    """Test cases for GraphBasedRecommender."""
    
    def test_graph_based_init(self):
        """Test GraphBasedRecommender initialization."""
        model = GraphBasedRecommender()
        assert model.name == "GraphBased"
        assert not model.is_fitted
    
    def test_graph_based_fit(self):
        """Test GraphBasedRecommender fitting."""
        model = GraphBasedRecommender()
        
        # Create test interaction matrix
        interactions = np.zeros((5, 5))
        interactions[0, 1] = 1
        interactions[1, 2] = 1
        interactions[2, 3] = 1
        interactions[3, 4] = 1
        interactions[4, 0] = 1
        
        model.fit(interactions)
        assert model.is_fitted
        assert model.graph is not None
        assert model.centrality_scores is not None


class TestHybridRecommender:
    """Test cases for HybridRecommender."""
    
    def test_hybrid_init(self):
        """Test HybridRecommender initialization."""
        model = HybridRecommender()
        assert model.name == "Hybrid"
        assert not model.is_fitted
    
    def test_hybrid_fit(self):
        """Test HybridRecommender fitting."""
        model = HybridRecommender()
        
        interactions = np.random.rand(5, 5)
        user_interests = {
            0: "basketball, hiking, movies",
            1: "reading, cooking, hiking",
            2: "photography, traveling, games",
            3: "cooking, reading, friends",
            4: "fitness, basketball, outdoor"
        }
        
        model.fit(interactions, user_interests)
        assert model.is_fitted
        assert len(model.models) == 3  # content, collaborative, graph


class TestRecommendationMetrics:
    """Test cases for RecommendationMetrics."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = [1, 2, 3, 4, 5]
        relevant_items = [1, 3, 5, 7, 9]
        
        precision = metrics.precision_at_k(recommendations, relevant_items, 5)
        assert precision == 0.6  # 3 relevant out of 5 recommendations
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = [1, 2, 3, 4, 5]
        relevant_items = [1, 3, 5, 7, 9]
        
        recall = metrics.recall_at_k(recommendations, relevant_items, 5)
        assert recall == 0.6  # 3 relevant found out of 5 total relevant
    
    def test_map_at_k(self):
        """Test MAP@k calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = [1, 2, 3, 4, 5]
        relevant_items = [1, 3, 5]
        
        map_score = metrics.map_at_k(recommendations, relevant_items, 5)
        assert map_score > 0
        assert map_score <= 1
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = [1, 2, 3, 4, 5]
        relevant_items = [1, 3, 5]
        
        ndcg = metrics.ndcg_at_k(recommendations, relevant_items, 5)
        assert ndcg > 0
        assert ndcg <= 1
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        metrics = RecommendationMetrics()
        
        recommendations = [1, 2, 3, 4, 5]
        relevant_items = [1, 3, 5]
        
        hit_rate = metrics.hit_rate_at_k(recommendations, relevant_items, 5)
        assert hit_rate == 1.0  # At least one relevant item found
        
        recommendations = [2, 4, 6, 8, 10]
        hit_rate = metrics.hit_rate_at_k(recommendations, relevant_items, 5)
        assert hit_rate == 0.0  # No relevant items found


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise an error
        assert True
    
    def test_create_user_item_matrix(self):
        """Test user-item matrix creation."""
        interactions = [(0, 1, 1.0), (0, 2, 0.5), (1, 0, 1.0), (1, 2, 1.0)]
        matrix = create_user_item_matrix(interactions, 2, 3)
        
        assert matrix.shape == (2, 3)
        assert matrix[0, 1] == 1.0
        assert matrix[0, 2] == 0.5
        assert matrix[1, 0] == 1.0
        assert matrix[1, 2] == 1.0
    
    def test_split_data_chronological(self):
        """Test chronological data splitting."""
        interactions = [
            (0, 1, 1.0, 1000),
            (0, 2, 0.5, 2000),
            (1, 0, 1.0, 3000),
            (1, 2, 1.0, 4000),
            (2, 0, 0.8, 5000)
        ]
        
        train, val, test = split_data_chronological(interactions, 0.6, 0.2)
        
        assert len(train) == 3  # 60% of 5
        assert len(val) == 1    # 20% of 5
        assert len(test) == 1   # 20% of 5
        
        # Check chronological order
        train_timestamps = [interaction[3] for interaction in train]
        val_timestamps = [interaction[3] for interaction in val]
        test_timestamps = [interaction[3] for interaction in test]
        
        assert max(train_timestamps) <= min(val_timestamps)
        assert max(val_timestamps) <= min(test_timestamps)


class TestModelFactory:
    """Test cases for model factory."""
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert 'content_based' in models
        assert 'user_based_cf' in models
    
    def test_create_model(self):
        """Test model creation."""
        model = create_model('content_based')
        assert isinstance(model, ContentBasedRecommender)
        
        model = create_model('user_based_cf', n_neighbors=30)
        assert isinstance(model, UserBasedCF)
        assert model.n_neighbors == 30


if __name__ == "__main__":
    pytest.main([__file__])
