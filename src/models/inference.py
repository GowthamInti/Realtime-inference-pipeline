import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.utils.redis_client import RedisClient
from src.models.model_manager import ModelManager
import logging

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self, redis_client: RedisClient, model_manager: ModelManager):
        self.redis_client = redis_client
        self.model_manager = model_manager
    
    def predict(self, features: Dict[str, Any], model_name: str = "default", 
                model_version: str = "latest") -> Dict[str, Any]:
        """Make single prediction"""
        try:
            # Load model
            model = self.model_manager.load_model(model_name, model_version)
            
            # Prepare features
            feature_array = self._prepare_features(features)
            
            # Make prediction
            prediction = model.predict(feature_array)
            
            # Calculate confidence if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_array)
                confidence = float(np.max(probabilities))
            
            return {
                "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                "confidence": confidence,
                "model_name": model_name,
                "model_version": model_version,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def batch_predict(self, feature_ids: List[str], model_name: str = "default",
                     model_version: str = "latest") -> List[Dict[str, Any]]:
        """Make batch predictions"""
        try:
            # Load model
            model = self.model_manager.load_model(model_name, model_version)
            
            # Get features from Redis
            features_list = []
            for feature_id in feature_ids:
                features = self.redis_client.get_features_by_id(feature_id)
                if features:
                    features_list.append(features)
            
            if not features_list:
                raise ValueError("No features found for provided IDs")
            
            # Prepare batch features
            feature_matrix = self._prepare_batch_features(features_list)
            
            # Make predictions
            predictions = model.predict(feature_matrix)
            
            # Prepare results
            results = []
            for i, prediction in enumerate(predictions):
                result = {
                    "feature_id": feature_ids[i] if i < len(feature_ids) else None,
                    "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                    "model_name": model_name,
                    "model_version": model_version
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare single feature record for prediction"""
        # Convert dict to DataFrame then to numpy array
        df = pd.DataFrame([features])
        return df.values
    
    def _prepare_batch_features(self, features_list: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare batch features for prediction"""
        df = pd.DataFrame(features_list)
        return df.values
