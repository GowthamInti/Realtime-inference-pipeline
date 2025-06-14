import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataTransformer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features"""
        # Example feature engineering
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        return df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

def transform_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main transformation function"""
    transformer = DataTransformer()
    
    # Convert to DataFrame
    df = pd.DataFrame(raw_data['api_data'])
    
    # Apply transformations
    df = transformer.clean_data(df)
    df = transformer.engineer_features(df)
    df = transformer.encode_categorical(df)
    df = transformer.normalize_features(df)
    
    return {
        'features': df.to_dict('records'),
        'feature_names': df.columns.tolist(),
        'timestamp': raw_data['timestamp']
    }
