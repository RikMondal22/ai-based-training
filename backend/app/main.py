from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.models import models
from app.models.schemas import (
    BSKMaster, ServiceMaster, DEOMaster, Provision
)
from app.models.database import engine, get_db
from typing import List
import os
from dotenv import load_dotenv
import logging
import pandas as pd
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="BSK Training Optimization API",
    description="API for AI-Assisted Training Optimization System",
    version="1.0.0"
)

# Configure CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in allowed_origins else allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NO STARTUP EVENT - Let the app start immediately!
# Embeddings will be computed on-demand when needed

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Welcome to BSK Training Optimization API",
        "status": "online",
        "version": "1.0.0",
        "environment": "render" if os.getenv("RENDER") == "true" else "local"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy",
        "environment": "render" if os.getenv("RENDER") == "true" else "local",
        "database_connected": True,
        "message": "API is running"
    }

# BSK Master endpoints
@app.get("/bsk/", response_model=List[BSKMaster])
def get_bsk_list(skip: int = 0, limit: int = Query(None), db: Session = Depends(get_db)):
    """Get list of BSKs"""
    query = db.query(models.BSKMaster).offset(skip)
    if limit is not None:
        query = query.limit(limit)
    bsk_list = query.all()
    return bsk_list

@app.get("/bsk/{bsk_code}", response_model=BSKMaster)
def get_bsk(bsk_code: str, db: Session = Depends(get_db)):
    """Get specific BSK by code"""
    bsk = db.query(models.BSKMaster).filter(models.BSKMaster.bsk_code == bsk_code).first()
    if bsk is None:
        raise HTTPException(status_code=404, detail="BSK not found")
    return bsk

# Service Master endpoints
@app.get("/services/", response_model=List[ServiceMaster])
def get_services(skip: int = 0, limit: int = Query(None), db: Session = Depends(get_db)):
    """Get list of services"""
    query = db.query(models.ServiceMaster).offset(skip)
    if limit is not None:
        query = query.limit(limit)
    services = query.all()
    return services

@app.get("/services/{service_id}", response_model=ServiceMaster)
def get_service(service_id: int, db: Session = Depends(get_db)):
    """Get specific service by ID"""
    service = db.query(models.ServiceMaster).filter(models.ServiceMaster.service_id == service_id).first()
    if service is None:
        raise HTTPException(status_code=404, detail="Service not found")
    return service

# DEO Master endpoints
@app.get("/deo/", response_model=List[DEOMaster])
def get_deo_list(skip: int = 0, limit: int = Query(None), db: Session = Depends(get_db)):
    """Get list of DEOs"""
    query = db.query(models.DEOMaster).offset(skip)
    if limit is not None:
        query = query.limit(limit)
    deo_list = query.all()
    return deo_list

@app.get("/deo/{agent_id}", response_model=DEOMaster)
def get_deo(agent_id: int, db: Session = Depends(get_db)):
    """Get specific DEO by agent ID"""
    deo = db.query(models.DEOMaster).filter(models.DEOMaster.agent_id == agent_id).first()
    if deo is None:
        raise HTTPException(status_code=404, detail="DEO not found")
    return deo

# Provision endpoints
@app.get("/provisions/", response_model=List[Provision])
def get_provisions(skip: int = 0, limit: int = Query(None), db: Session = Depends(get_db)):
    """Get list of provisions"""
    query = db.query(models.Provision).offset(skip)
    if limit is not None:
        query = query.limit(limit)
    provisions = query.all()
    return provisions

@app.get("/provisions/{customer_id}", response_model=Provision)
def get_provision(customer_id: str, db: Session = Depends(get_db)):
    """Get specific provision by customer ID"""
    provision = db.query(models.Provision).filter(models.Provision.customer_id == customer_id).first()
    if provision is None:
        raise HTTPException(status_code=404, detail="Provision not found")
    return provision

# Embeddings initialization endpoint
@app.post("/initialize-embeddings/")
def initialize_embeddings(force_rebuild: bool = False):
    """
    Initialize or rebuild service embeddings.
    Call this endpoint after adding data to the database.
    """
    try:
        from ai_service.service_recommendation import initialize_embeddings_from_db, get_embedding_stats
        
        logger.info(f"Initializing embeddings (force_rebuild={force_rebuild})...")
        success = initialize_embeddings_from_db(
            force_rebuild=force_rebuild,
            include_inactive=False
        )
        
        if success:
            stats = get_embedding_stats()
            return {
                "status": "success",
                "message": "Embeddings initialized successfully",
                "total_services": stats.get('total_services', 0)
            }
        else:
            return {
                "status": "warning",
                "message": "No data found in database to create embeddings",
                "total_services": 0
            }
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embeddings-status/")
def embeddings_status():
    """Check if embeddings are loaded and how many"""
    try:
        from ai_service.service_recommendation import get_embedding_stats
        stats = get_embedding_stats()
        
        return {
            "embeddings_loaded": stats.get('total_services', 0) > 0,
            "total_services": stats.get('total_services', 0),
            "db_path": stats.get('db_path', ''),
            "collection_name": stats.get('collection_name', '')
        }
    except Exception as e:
        return {
            "embeddings_loaded": False,
            "total_services": 0,
            "error": str(e)
        }

@app.get("/underperforming_bsks/")
def get_underperforming_bsks(
    num_bsks: int = 50,
    sort_order: str = 'asc',
    db: Session = Depends(get_db)
):
    """Get underperforming BSKs analysis"""
    try:
        from ai_service.bsk_analytics import find_underperforming_bsks
        
        # Fetch all data
        bsks = db.query(models.BSKMaster).all()
        provisions = db.query(models.Provision).all()
        deos = db.query(models.DEOMaster).all()
        services = db.query(models.ServiceMaster).all()
        
        # Convert to DataFrame
        bsks_df = pd.DataFrame([b.__dict__ for b in bsks])
        provisions_df = pd.DataFrame([p.__dict__ for p in provisions])
        deos_df = pd.DataFrame([d.__dict__ for d in deos])
        services_df = pd.DataFrame([s.__dict__ for s in services])
        
        # Remove SQLAlchemy internal state
        for df in [bsks_df, provisions_df, deos_df, services_df]:
            if '_sa_instance_state' in df.columns:
                df.drop('_sa_instance_state', axis=1, inplace=True)
        
        # Compute underperforming BSKs
        result_df = find_underperforming_bsks(bsks_df, provisions_df, deos_df, services_df)
        
        # Sort and return
        ascending = sort_order == 'asc'
        result_df = result_df.sort_values(by="score", ascending=ascending).head(num_bsks)
        
        return result_df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error in get_underperforming_bsks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend-bsk/")
def recommend_bsk_for_new_service(
    service_name: str,
    service_type: str = "",
    service_desc: str = "",
    top_n: int = 10,
    db: Session = Depends(get_db)
):
    """
    Recommend BSKs for a new service.
    Embeddings will be loaded automatically if not already loaded.
    """
    try:
        from ai_service.service_recommendation import (
            recommend_bsk_for_service,
            get_embedding_stats,
            initialize_embeddings_from_db
        )
        
        # Check if embeddings are loaded, if not, try to load them
        stats = get_embedding_stats()
        if stats.get('total_services', 0) == 0:
            logger.info("Embeddings not loaded, initializing...")
            initialize_embeddings_from_db(force_rebuild=False, include_inactive=False)
        
        # Fetch data
        services = db.query(models.ServiceMaster).all()
        provisions = db.query(models.Provision).all()
        bsks = db.query(models.BSKMaster).all()
        
        services_df = pd.DataFrame([s.__dict__ for s in services])
        provisions_df = pd.DataFrame([p.__dict__ for p in provisions])
        bsks_df = pd.DataFrame([b.__dict__ for b in bsks])
        
        # Remove SQLAlchemy internal state
        for df in [services_df, provisions_df, bsks_df]:
            if '_sa_instance_state' in df.columns:
                df.drop('_sa_instance_state', axis=1, inplace=True)
        
        # Create new service dict
        new_service = {
            'service_name': service_name,
            'service_type': service_type,
            'service_desc': service_desc
        }
        
        # Get recommendations
        result = recommend_bsk_for_service(
            new_service=new_service,
            services_df=services_df,
            provisions_df=provisions_df,
            bsk_df=bsks_df,
            top_n=top_n,
            use_precomputed_embeddings=True
        )
        
        # Handle tuple return (recommendations, similar_services)
        if isinstance(result, tuple):
            recommendations, similar_services = result
            return {
                "recommendations": recommendations.to_dict(orient='records'),
                "similar_services": similar_services.to_dict(orient='records')
            }
        else:
            return {
                "recommendations": result.to_dict(orient='records')
            }
            
    except Exception as e:
        logger.error(f"Error in recommend_bsk: {e}")
        raise HTTPException(status_code=500, detail=str(e))
