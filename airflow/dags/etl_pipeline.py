from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.lambda_function import LambdaInvokeFunctionOperator
from src.data.extraction import extract_data
from src.data.transformation import transform_data
from src.data.validation import validate_data
from src.utils.redis_client import RedisClient

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'ml_etl_pipeline',
    default_args=default_args,
    description='End-to-end ML data pipeline',
    schedule_interval='@hourly',
    catchup=False,
    tags=['ml', 'etl', 'production']
)

def extract_task(**context):
    """Extract data from various sources"""
    data = extract_data()
    return data

def transform_task(**context):
    """Transform and clean data"""
    ti = context['ti']
    raw_data = ti.xcom_pull(task_ids='extract_data')
    transformed_data = transform_data(raw_data)
    return transformed_data

def validate_task(**context):
    """Validate data quality using Great Expectations"""
    ti = context['ti']
    data = ti.xcom_pull(task_ids='transform_data')
    validation_result = validate_data(data)
    if not validation_result['success']:
        raise ValueError(f"Data validation failed: {validation_result['errors']}")
    return validation_result

def load_to_feature_store(**context):
    """Load processed data to Redis feature store"""
    ti = context['ti']
    data = ti.xcom_pull(task_ids='transform_data')
    
    redis_client = RedisClient()
    redis_client.store_features(data)
    return "Features stored successfully"

# Define tasks
extract_data_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_task,
    dag=dag
)

transform_data_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_task,
    dag=dag
)

validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_task,
    dag=dag
)

# AWS Lambda task for additional processing
lambda_process_task = LambdaInvokeFunctionOperator(
    task_id='lambda_process',
    function_name='data-processor-lambda',
    payload='{{ ti.xcom_pull(task_ids="transform_data") }}',
    dag=dag
)

load_features_task = PythonOperator(
    task_id='load_to_feature_store',
    python_callable=load_to_feature_store,
    dag=dag
)

# Set task dependencies
extract_data_task >> transform_data_task >> validate_data_task
validate_data_task >> [lambda_process_task, load_features_task]
