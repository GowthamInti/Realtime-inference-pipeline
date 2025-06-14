import json
import boto3
import pandas as pd
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda function for additional data processing
    """
    try:
        # Parse input data
        if 'body' in event:
            data = json.loads(event['body'])
        else:
            data = event
        
        # Process data
        processed_data = process_data(data)
        
        # Store results (could be S3, DynamoDB, etc.)
        store_results(processed_data)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Data processed successfully',
                'processed_records': len(processed_data.get('features', []))
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process the data with additional transformations"""
    features = data.get('features', [])
    
    # Convert to DataFrame for processing
    df = pd.DataFrame(features)
    
    # Additional processing (aggregations, advanced transformations)
    if not df.empty:
        # Example: Calculate rolling statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[f'{col}_rolling_mean'] = df[col].rolling(window=3, min_periods=1).mean()
    
    return {
        'features': df.to_dict('records'),
        'feature_names': df.columns.tolist(),
        'timestamp': data.get('timestamp')
    }

def store_results(data: Dict[str, Any]) -> None:
    """Store processed results to AWS services"""
    # Example: Store to S3
    s3_client = boto3.client('s3')
    
    bucket_name = 'ml-pipeline-processed-data'
    key = f"processed/{data['timestamp']}.json"
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json.dumps(data),
        ContentType='application/json'
    )
