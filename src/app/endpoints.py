import logging
from fastapi import APIRouter, HTTPException

from services.classify import classify
from services.models.llm import predict as llm_predict
from services.telemetry import get_user_data_service, get_product_info_service
from app.requests import ReviewRequest, ReviewBatchResponse, UserDataResponse, ProductInfoResponse

_log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy"}


@router.post("/classify", response_model=ReviewBatchResponse)
async def classify_review(request: ReviewRequest):
    
    try:
        result = classify(
            user_texts=request.texts,
            threshold=request.threshold,
            product_id=request.product_id,
            specified_provider=request.specified_provider
        )
        return ReviewBatchResponse(**result)
        
    except Exception as e:
        _log.error(f"Error classifying review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get_user_data", response_model=UserDataResponse)
async def get_user_data(user_id: str):
    try:
        data = get_user_data_service(user_id)
        return UserDataResponse(**data)
    except KeyError:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        _log.error(f"Error fetching user data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get_product_info", response_model=ProductInfoResponse)
async def get_product_info(product_id: str):
    try:
        data = get_product_info_service(product_id)
        return ProductInfoResponse(**data)
    except KeyError:
        raise HTTPException(status_code=404, detail="Product not found")
    except Exception as e:
        _log.error(f"Error fetching product info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
