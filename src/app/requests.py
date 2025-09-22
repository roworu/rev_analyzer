from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ReviewRequest(BaseModel):
    """Request model for review classification"""
    texts: dict[str, str] = Field(
        ...,
        description="Dict of user_id to review text to analyze",
        example={
            "user_1": "This laptop sucks! It's heating like hell after 10 minutes!",
            "user_2": "I like that laptop, it has a bright screen and good battery. Recommended!"
        },
    )
    product_id: Optional[str] = Field(
        None,
        description="Optional product identifier",
        example="laptop_1",
    )
    threshold: Optional[float] = Field(
        0.7,
        description="Confidence threshold for classification",
        example=0.7,
    )
    specified_provider: str = Field(
        "ollama",
        description="LLM provider to use (e.g., 'ollama' or 'openai')",
        example="ollama",
    )


class ReviewResponse(BaseModel):
    """Response model for review classification"""
    grade: int = Field(..., description="Satisfaction grade from 1 to 10", example=8)
    confidence: float = Field(..., description="Confidence score between 0 and 1", example=0.92)
    tags: List[str] = Field(default_factory=list, description="Keywords/adjectives extracted", example=["bright", "quiet"])


class ReviewBatchResponse(BaseModel):
    """Batch response for multiple reviews"""
    product_id: Optional[str]
    threshold: Optional[float]
    results: List[ReviewResponse]


class UserDataResponse(BaseModel):
    """User data with generated brief portrait"""
    user: dict[str, Any]
    llm_summary: str


class ProductInfoResponse(BaseModel):
    """Product data with generated brief description"""
    product: Dict[str, Any]
    llm_summary: str
