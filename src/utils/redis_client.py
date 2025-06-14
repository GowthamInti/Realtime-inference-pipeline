import redis
import json
import pickle
from typing import Dict, Any, List, Optional
import pandas as pd
from src.utils.config import Config

class RedisClient:
    def __init__(self):
        self.config = Config()
        self.client = redis.Redis(
            host=self.config.REDIS_HOST,
            port=self.config.REDIS_PORT,
            decode_responses=True
        )
        self.binary_client = redis.Redis(
            host=self.config.REDIS_HOST,
            port=self.config.REDIS_PORT,
            decode_responses=False
        )
    
    def store_features(self, data: Dict[str, Any], ttl: int = 3600) -> None:
        """Store features in Redis with TTL"""
        features = data['features']
        timestamp = data['timestamp']
        
        for i, feature_record in enumerate(features):
            key = f"features:{timestamp}:{i}"
            self.client.setex(key, ttl, json.dumps(feature_record))
        
        # Store metadata
        metadata_key = f"metadata:{timestamp}"
        metadata = {
            'feature_names': data.get('feature_names', []),
            'count': len(features),
            'timestamp': timestamp
        }
        self.client.setex(metadata_key, ttl, json.dumps(metadata))
    
    def get_latest_features(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get latest features from Redis"""
        # Get all feature keys sorted by timestamp
        keys = self.client.keys("features:*")
        if not keys:
            return []
        
        # Sort keys by timestamp (assuming timestamp is in the key)
        sorted_keys = sorted(keys, reverse=True)[:limit]
        
        features = []
        for key in sorted_keys:
            feature_data = self.client.get(key)
            if feature_data:
                features.append(json.loads(feature_data))
        
        return features
    
    def get_features_by_id(self, feature_id: str) -> Optional[Dict[str, Any]]:
        """Get specific features by ID"""
        keys = self.client.keys(f"features:*:{feature_id}")
        if keys:
            feature_data = self.client.get(keys[0])
            return json.loads(feature_data) if feature_data else None
        return None
    
    def store_model(self, model_name: str, model_object: Any, version: str = "latest") -> None:
        """Store ML model in Redis"""
        key = f"model:{model_name}:{version}"
        serialized_model = pickle.dumps(model_object)
        self.binary_client.set(key, serialized_model)
        
        # Update latest version pointer
        latest_key = f"model:{model_name}:latest"
        self.client.set(latest_key, version)
    
    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """Load ML model from Redis"""
        if version == "latest":
            latest_key = f"model:{model_name}:latest"
            version = self.client.get(latest_key)
            if not version:
                raise ValueError(f"No model found for {model_name}")
        
        key = f"model:{model_name}:{version}"
        serialized_model = self.binary_client.get(key)
        if not serialized_model:
            raise ValueError(f"Model {model_name}:{version} not found")
        
        return pickle.loads(serialized_model)
    
    def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            self.client.ping()
            return True
        except:
            return False
