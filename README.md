# Realtime-inference-pipeline
A production-ready ML data pipeline with automated ETL, data validation, and real-time inference capabilities.
## Project Structure

```
ml-pipeline/
├── airflow/
│   ├── dags/
│   │   └── etl_pipeline.py
│   └── plugins/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── extraction.py
│   │   ├── transformation.py
│   │   └── validation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_manager.py
│   │   └── inference.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   └── utils/
│       ├── __init__.py
│       ├── redis_client.py
│       └── config.py
├── lambda/
│   └── data_processor.py
├── great_expectations/
│   ├── expectations/
│   │   └── data_suite.json
│   └── checkpoints/
├── tests/
│   ├── test_etl.py
│   ├── test_api.py
│   └── test_validation.py
├── docker/
│   ├── Dockerfile.api
│   └── docker-compose.yml
├── requirements.txt
├── setup.py
└── README.md
```

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/GowthamInti/Realtime-inference-pipeline.git
cd ml-pipeline

# Install dependencies
pip install -r requirements.txt

# Setup Great Expectations
great_expectations init

# Start Redis (Docker)
docker run -d -p 6379:6379 redis:alpine

# Start Airflow
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow webserver -p 8080
```
