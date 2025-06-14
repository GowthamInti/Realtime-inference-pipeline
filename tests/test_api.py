import unittest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
    
    @patch('src.api.main.redis_client')
    def test_health_check(self, mock_redis):
        mock_redis.health_check.return_value = True
        
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["redis"])
    
    @patch('src.api.main.redis_client')
    def test_health_check_unhealthy(self, mock_redis):
        mock_redis.health_check.return_value = False
        
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "unhealthy")
        self.assertFalse(data["redis"])
    
    @patch('src.api.main.inference_engine')
    def test_predict_success(self, mock_inference):
        mock_inference.predict.return_value = {
            "prediction": [0.8],
            "confidence": 0.9,
            "model_name": "default",
            "model_version": "latest",
            "timestamp": "2024-01-01T10:00:00"
        }
        
        request_data = {
            "features": {"feature_1": 10, "feature_2": 0.5},
            "model_name": "default"
        }
        
        response = self.client.post("/predict", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        self.assertEqual(data["model_name"], "default")
    
    @patch('src.api.main.inference_engine')
    def test_predict_failure(self, mock_inference):
        mock_inference.predict.side_effect = Exception("Model error")
        
        request_data = {
            "features": {"feature_1": 10, "feature_2": 0.5}
        }
        
        response = self.client.post("/predict", json=request_data)
        
        self.assertEqual(response.status_code, 500)
    
    @patch('src.api.main.inference_engine')
    def test_batch_predict(self, mock_inference):
        mock_inference.batch_predict.return_value = [
            {
                "feature_id": "1",
                "prediction": [0.8],
                "model_name": "default",
                "model_version": "latest"
            },
            {
                "feature_id": "2",
                "prediction": [0.6],
                "model_name": "default",
                "model_version": "latest"
            }
        ]
        
        request_data = {
            "feature_ids": ["1", "2"],
            "model_name": "default"
        }
        
        response = self.client.post("/predict/batch", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predictions", data)
        self.assertEqual(len(data["predictions"]), 2)
    
    @patch('src.api.main.model_manager')
    def test_deploy_model(self, mock_manager):
        response = self.client.post("/models/test_model/deploy")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
    
    @patch('src.api.main.model_manager')
    def test_list_models(self, mock_manager):
        mock_manager.list_models.return_value = ["model1", "model2"]
        
        response = self.client.get("/models")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertEqual(len(data["models"]), 2)
    
    @patch('src.api.main.redis_client')
    def test_get_latest_features(self, mock_redis):
        mock_redis.get_latest_features.return_value = [
            {"feature_1": 10, "feature_2": 0.5},
            {"feature_1": 20, "feature_2": 0.7}
        ]
        
        response = self.client.get("/features/latest?limit=5")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("features", data)
        self.assertEqual(len(data["features"]), 2)

if __name__ == '__main__':
    unittest.main()
