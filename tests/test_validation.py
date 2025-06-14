import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.validation import DataValidator, validate_data

class TestDataValidator(unittest.TestCase):
    def setUp(self):
        self.validator = DataValidator()
    
    def test_validate_schema_with_valid_data(self):
        valid_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'feature_1': [10.5, 20.3, 30.1],
            'feature_2': [0.5, 0.7, 0.9]
        })
        
        result = self.validator.validate_schema(valid_df)
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_validate_schema_missing_required_columns(self):
        invalid_df = pd.DataFrame({
            'feature_1': [10.5, 20.3, 30.1]
        })
        
        result = self.validator.validate_schema(invalid_df)
        
        self.assertFalse(result['success'])
        self.assertTrue(any('Missing required columns' in error for error in result['errors']))
    
    def test_validate_schema_null_in_critical_columns(self):
        df_with_nulls = pd.DataFrame({
            'user_id': [1, None, 3],
            'feature_1': [10.5, 20.3, 30.1],
            'feature_2': [0.5, 0.7, 0.9]
        })
        
        result = self.validator.validate_schema(df_with_nulls)
        
        self.assertFalse(result['success'])
        self.assertTrue(any('Null values found' in error for error in result['errors']))
    
    def test_validate_schema_type_warnings(self):
        df_wrong_type = pd.DataFrame({
            'user_id': ['1', '2', '3'],  # String instead of int
            'feature_1': [10.5, 20.3, 30.1],
            'feature_2': [0.5, 0.7, 0.9]
        })
        
        result = self.validator.validate_schema(df_wrong_type)
        
        # Should succeed but have warnings
        self.assertTrue(len(result['warnings']) > 0)
    
    def test_validate_data_quality_basic(self):
        valid_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'feature_1': [10.5, 20.3, 30.1, 15.7, 25.9],
            'feature_2': [0.5, 0.7, 0.9, 0.6, 0.8]
        })
        
        result = self.validator.validate_data_quality(valid_df)
        
        # Should not fail even if we don't have detailed expectations
        self.assertIn('success', result)
    
    def test_validate_data_function_integration(self):
        data = {
            'features': [
                {'user_id': 1, 'feature_1': 10.5, 'feature_2': 0.5},
                {'user_id': 2, 'feature_1': 20.3, 'feature_2': 0.7},
                {'user_id': 3, 'feature_1': 30.1, 'feature_2': 0.9}
            ],
            'timestamp': '2024-01-01T10:00:00'
        }
        
        result = validate_data(data)
        
        self.assertIn('success', result)
        self.assertIn('schema_validation', result)
        self.assertIn('quality_validation', result)
        self.assertIn('timestamp', result)
        self.assertEqual(result['timestamp'], data['timestamp'])
    
    def test_validate_data_function_with_invalid_data(self):
        data = {
            'features': [
                {'feature_1': 10.5, 'feature_2': 0.5},  # Missing user_id
                {'feature_1': 20.3, 'feature_2': 0.7}
            ],
            'timestamp': '2024-01-01T10:00:00'
        }
        
        result = validate_data(data)
        
        self.assertFalse(result['success'])
        self.assertFalse(result['schema_validation']['success'])
    
    def test_empty_dataframe_validation(self):
        empty_df = pd.DataFrame()
        
        result = self.validator.validate_schema(empty_df)
        
        # Should fail due to missing required columns
        self.assertFalse(result['success'])
    
    def test_large_dataframe_validation(self):
        # Test with larger dataset
        large_df = pd.DataFrame({
            'user_id': range(1000),
            'feature_1': np.random.normal(20, 5, 1000),
            'feature_2': np.random.uniform(0, 1, 1000)
        })
        
        result = self.validator.validate_schema(large_df)
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['errors']), 0)

class TestDataValidationEdgeCases(unittest.TestCase):
    def setUp(self):
        self.validator = DataValidator()
    
    def test_single_row_dataframe(self):
        single_row_df = pd.DataFrame({
            'user_id': [1],
            'feature_1': [10.5],
            'feature_2': [0.5]
        })
        
        result = self.validator.validate_schema(single_row_df)
        
        self.assertTrue(result['success'])
    
    def test_all_null_feature_column(self):
        df_all_nulls = pd.DataFrame({
            'user_id': [1, 2, 3],
            'feature_1': [None, None, None],
            'feature_2': [0.5, 0.7, 0.9]
        })
        
        result = self.validator.validate_schema(df_all_nulls)
        
        # Should pass schema validation (feature_1 is not critical)
        self.assertTrue(result['success'])
    
    def test_mixed_data_types(self):
        mixed_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'feature_1': [10, 20.5, '30'],  # Mixed int, float, string
            'feature_2': [0.5, 0.7, 0.9]
        })
        
        result = self.validator.validate_schema(mixed_df)
        
        # Should not fail on mixed types in non-critical columns
        self.assertTrue(result['success'])

if __name__ == '__main__':
    unittest.main()
