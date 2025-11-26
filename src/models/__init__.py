"""Model factory and registry for recommendation models."""

from typing import Dict, Type, Any, Optional
import logging

from .base import BaseRecommender
from .content_based import ContentBasedRecommender, InterestBasedRecommender
from .collaborative_filtering import UserBasedCF, ItemBasedCF, MatrixFactorizationCF, ALSRecommender
from .graph_based import GraphBasedRecommender, CommunityBasedRecommender, RandomWalkRecommender
from .hybrid import HybridRecommender, WeightedHybridRecommender, EnsembleRecommender

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for recommendation models."""
    
    def __init__(self):
        """Initialize model registry."""
        self.models: Dict[str, Type[BaseRecommender]] = {
            # Content-based models
            'content_based': ContentBasedRecommender,
            'interest_based': InterestBasedRecommender,
            
            # Collaborative filtering models
            'user_based_cf': UserBasedCF,
            'item_based_cf': ItemBasedCF,
            'matrix_factorization': MatrixFactorizationCF,
            'als': ALSRecommender,
            
            # Graph-based models
            'graph_based': GraphBasedRecommender,
            'community_based': CommunityBasedRecommender,
            'random_walk': RandomWalkRecommender,
            
            # Hybrid models
            'hybrid': HybridRecommender,
            'weighted_hybrid': WeightedHybridRecommender,
            'ensemble': EnsembleRecommender,
        }
    
    def register_model(self, name: str, model_class: Type[BaseRecommender]) -> None:
        """Register a new model class.
        
        Args:
            name: Model name
            model_class: Model class
        """
        self.models[name] = model_class
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> Type[BaseRecommender]:
        """Get model class by name.
        
        Args:
            name: Model name
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model name is not found
        """
        if name not in self.models:
            available_models = list(self.models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available_models}")
        
        return self.models[name]
    
    def list_models(self) -> list:
        """List all available models.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())


class ModelFactory:
    """Factory for creating recommendation models."""
    
    def __init__(self):
        """Initialize model factory."""
        self.registry = ModelRegistry()
    
    def create_model(self, model_name: str, **kwargs) -> BaseRecommender:
        """Create a model instance.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Model-specific parameters
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model name is not found
        """
        model_class = self.registry.get_model(model_name)
        
        try:
            model = model_class(**kwargs)
            logger.info(f"Created model: {model.name}")
            return model
        except Exception as e:
            logger.error(f"Error creating model {model_name}: {e}")
            raise
    
    def create_multiple_models(self, model_configs: Dict[str, Dict[str, Any]]) -> Dict[str, BaseRecommender]:
        """Create multiple models from configuration.
        
        Args:
            model_configs: Dictionary mapping model names to their configurations
            
        Returns:
            Dictionary mapping model names to model instances
        """
        models = {}
        
        for model_name, config in model_configs.items():
            try:
                models[model_name] = self.create_model(model_name, **config)
            except Exception as e:
                logger.error(f"Error creating model {model_name}: {e}")
        
        return models
    
    def get_available_models(self) -> list:
        """Get list of available models.
        
        Returns:
            List of available model names
        """
        return self.registry.list_models()


# Global model factory instance
model_factory = ModelFactory()


def create_model(model_name: str, **kwargs) -> BaseRecommender:
    """Convenience function to create a model.
    
    Args:
        model_name: Name of the model to create
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    return model_factory.create_model(model_name, **kwargs)


def get_available_models() -> list:
    """Get list of available models.
    
    Returns:
        List of available model names
    """
    return model_factory.get_available_models()
