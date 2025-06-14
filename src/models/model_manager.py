import logging
from typing import Dict, Any, List, Optional
from src.utils.redis_client import RedisClient
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, redis_client: RedisClient):
        self.redis_client = redis_client
        self.loaded_models = {}
    
    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """Load model from Redis or cache"""
        cache_key = f"{model_name}:{version}"
        
        # Check if model is already loaded
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Load from Redis
        try:
            model = self.redis_client.load_model(model_name, version)
            self.loaded_models[cache_key] = model
            logger.info(f"Model {cache_key} loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {cache_key}: {e}")
            raise
    
    def hot_swap_model(self, model_name: str, new_version: str = "latest") -> None:
        """Hot swap model with zero downtime"""
        try:
            # Load new model
            new_model = self.redis_client.load_model(model_name, new_version)
            
            # Update cache
            cache_key = f"{model_name}:{new_version}"
            self.loaded_models[cache_key] = new_model
            
            # Update latest pointer in cache
            latest_key = f"{model_name}:latest"
            if latest_key in self.loaded_models:
                del self.loaded_models[latest_key]
            self.loaded_models[latest_key] = new_model
            
            logger.info(f"Model {model_name} hot-swapped to version {new_version}")
            
        except Exception as e:
            logger.error(f"Hot swap failed for {model_name}: {e}")
            raise
    
    def list_models(self) -> List[str]:
        """List all available models"""
        # This would query Redis for available models
        # For now, return loaded models
        return list(self.loaded_models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata"""
        # This would fetch model metadata from Redis
        return {
            "name": model_name,
            "status": "loaded" if model_name in self.loaded_models else "not_loaded"
        }
