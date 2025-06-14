import great_expectations as gx
import pandas as pd
from typing import Dict, Any, List

class DataValidator:
    def __init__(self):
        self.context = gx.get_context()
    
    def create_expectation_suite(self, suite_name: str) -> None:
        """Create a new expectation suite"""
        try:
            suite = self.context.add_expectation_suite(expectation_suite_name=suite_name)
        except Exception:
            # Suite already exists
            pass
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data schema and basic constraints"""
        results = {
            'success': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for required columns
        required_cols = ['user_id', 'feature_1', 'feature_2']  # Define your required columns
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            results['success'] = False
            results['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if 'user_id' in df.columns and df['user_id'].dtype != 'int64':
            results['warnings'].append("user_id should be integer type")
        
        # Check for null values in critical columns
        critical_cols = ['user_id']
        for col in critical_cols:
            if col in df.columns and df[col].isnull().any():
                results['success'] = False
                results['errors'].append(f"Null values found in critical column: {col}")
        
        return results
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality using Great Expectations"""
        # Create a data asset
        data_asset = self.context.sources.add_pandas("pandas_source").add_asset(
            name="validation_asset",
            dataframe=df
        )
        
        # Create batch request
        batch_request = data_asset.build_batch_request()
        
        # Get expectation suite
        suite_name = "data_quality_suite"
        self.create_expectation_suite(suite_name)
        
        # Run validation
        try:
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name
            )
            
            # Add expectations
            validator.expect_table_row_count_to_be_between(min_value=1, max_value=100000)
            validator.expect_table_column_count_to_equal(len(df.columns))
            
            # Validate and return results
            validation_result = validator.validate()
            
            return {
                'success': validation_result.success,
                'statistics': validation_result.statistics,
                'results': validation_result.results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

def validate_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Main validation function"""
    validator = DataValidator()
    
    # Convert to DataFrame
    df = pd.DataFrame(data['features'])
    
    # Run validations
    schema_validation = validator.validate_schema(df)
    quality_validation = validator.validate_data_quality(df)
    
    return {
        'success': schema_validation['success'] and quality_validation['success'],
        'schema_validation': schema_validation,
        'quality_validation': quality_validation,
        'timestamp': data['timestamp']
    }
