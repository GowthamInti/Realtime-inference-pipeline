import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.extraction import DataExtractor, extract_data
from data.transformation import DataTransformer, transform_data
from data.validation import DataValidator, validate_data

class TestDataExtraction(unittest.TestCase):
    def setUp(self):
        self.extractor = DataExtractor()
    
    @patch('requests.get')
    def test_extract_from_api_success(self, mock_get):
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = [
            {'id': 1, 'name': 'John', 'age': 30},
            {'id': 2, 'name': 'Jane', 'age': 25}
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.extractor.extract_from_api('users')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('id', result.columns)
        self.assertIn('name', result.columns)
    
    @patch('requests.get')
    def test_extract_from_api_failure(self, mock_get):
        # Mock API failure
        mock_get.side_effect = Exception("API Error")
        
        with self.assertRaises(Exception):
            self.extractor.extract_from_api('users')
    
    @patch('src.data.extraction.DataExtractor.extract_from_api')
    def test_extract_data_function(self, mock_extract):
        # Mock the extractor method
        mock_df = pd.DataFrame([{'id': 1, 'name': 'Test'}])
        mock_extract.return_value = mock_df
        
        result = extract_data()
        
        self.assertIn('api_data', result)
        self.assertIn('timestamp', result)
        self.assertIsInstance(result['api_data'], list)

class TestDataTransformation(unittest.TestCase):
    def setUp(self):
        self.transformer = DataTransformer()
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 2],  # Include duplicate
            'name': ['John', 'Jane', None, 'Bob'],
            'age': [30, 25, None, 35],
            'score': [85.5, 92.0, 78.5, None]
        })
    
    def test_clean_data(self):
        cleaned = self.transformer.clean_data(self.sample_data)
        
        # Check duplicates removed
        self.assertEqual(len(cleaned), 3)
        
        # Check missing values handled
        self.assertFalse(cleaned.isnull().any().any())
    
    def test_engineer_features(self):
        data_with_timestamp = pd.DataFrame({
            'id': [1, 2],
            'timestamp': ['2024-01-01 10:30:00', '2024-01-01 15:45:00']
        })
        
        result = self.transformer.engineer_features(data_with_timestamp)
        
        self.assertIn('hour', result.columns)
        self.assertIn('day_of_week', result.columns)
    
    def test_encode_categorical(self):
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A'],
            'value': [1, 2, 3, 4]
        })
        
        encoded = self.transformer.encode_categorical(data)
        
        # Check that categorical column is now numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded['category']))
    
    def test_transform_data_function(self):
        raw_data = {
            'api_data': [
                {'id': 1, 'name': 'John', 'age': 30},
                {'id': 2, 'name': 'Jane', 'age': 25}
            ],
            'timestamp': '2024-01-01T10:00:00'
        }
        
        result = transform_data(raw_data)
        
        self.assertIn('features', result)
        self.assertIn('feature_names', result)
        self.assertIn('timestamp', result)
        self.assertIsInstance(result['features'], list)

class TestDataValidation(unittest.TestCase):
    def setUp(self):
        self.validator = DataValidator()
        self.valid_data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'feature_1': [10, 20, 30],
            'feature_2': [0.5, 0.7, 0.9]
        })
    
    def test_validate_schema_success(self):
        result = self.validator.validate_schema(self.valid_data)
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_validate_schema_missing_columns(self):
        incomplete_data = pd.DataFrame({
            'feature_1': [10, 20, 30]
        })
        
        result = self.validator.validate_schema(incomplete_data)
        
        self.assertFalse(result['success'])
        self.assertTrue(len(result['errors']) > 0)
    
    def test_validate_schema_null_critical_columns(self):
        data_with_nulls = pd.DataFrame({
            'user_id': [1, None, 3],
            'feature_1': [10, 20, 30],
            'feature_2': [0.5, 0.7, 0.9]
        })
        
        result = self.validator.validate_schema(data_with_nulls)
        
        self.assertFalse(result['success'])
    
    def test_validate_data_function(self):
        data = {
            'features': [
                {'user_id': 1, 'feature_1': 10, 'feature_2': 0.5},
                {'user_id': 2, 'feature_1': 20, 'feature_2': 0.7}
            ],
            'timestamp': '2024-01-01T10:00:00'
        }
        
        result = validate_data(data)
        
        self.assertIn('success', result)
        self.assertIn('schema_validation', result)
        self.assertIn('timestamp', result)

if __name__ == '__main__':
    unittest.main()
