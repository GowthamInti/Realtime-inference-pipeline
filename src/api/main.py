from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import numpy as np
import logging
from src.utils.redis_client import RedisClient
from src.models.model_manager import ModelManager
from src.models.inference import InferenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Inference API",
    description="Real-time ML inference with feature store",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
redis_client = RedisClient()
model_manager = ModelManager(redis_client)
inference_engine = InferenceEngine(redis_client, model_manager)

class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    model_name: Optional[str] = "default"
    model_version: Optional[str] = "latest"

class PredictionResponse(BaseModel):
    prediction: Any
    confidence: Optional[float] = None
    model_name: str
    model_version: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    feature_ids: List[str]
    model_name: Optional[str] = "default"
    model_version: Optional[str] = "latest"

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting ML Inference API...")
    
    # Check Redis connection
    if not redis_client.health_check():
        logger.error("Redis connection failed!")
        raise Exception("Redis connection failed")
    
    # Load default model
    try:
        model_manager.load_model("default")
        logger.info("Default model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    redis_healthy = redis_client.health_check()
    
    return {
        "status": "healthy" if redis_healthy else "unhealthy",
        "redis": redis_healthy,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""
    try:
        result = inference_engine.predict(
            features=request.features,
            model_name=request.model_name,
            model_version=request.model_version
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    try:
        results = inference_engine.batch_predict(
            feature_ids=request.feature_ids,
            model_name=request.model_name,
            model_version=request.model_version
        )
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_name}/deploy")
async def deploy_model(model_name: str, background_tasks: BackgroundTasks):
    """Deploy new model version"""
    try:
        background_tasks.add_task(model_manager.hot_swap_model, model_name)
        return {"message": f"Model {model_name} deployment initiated"}
        
    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    try:
        models = model_manager.list_models()
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/latest")
async def get_latest_features(limit: int = 10):
    """Get latest features from feature store"""
    try:
        features = redis_client.get_latest_features(limit=limit)
        return {"features": features}
        
    except Exception as e:
        logger.error(f"Failed to get features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
