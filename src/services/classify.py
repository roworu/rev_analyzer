import logging
from typing import Any

from services.models.llm import predict as llm_predict
from services.models.domain_model import init_models
from services.models.domain_model import predict as domain_model_predict
from services.telemetry import store_user_data, store_product_data


_log = logging.getLogger(__name__)


init_models()


def classify(user_texts: dict[str, str], threshold: float | None,
    product_id: str | None, specified_provider: str | None):
    """Main classification method implementing hybrid inference pipeline"""

    _log.info(f"Classifying {len(user_texts)} reviews with threshold={threshold}, product_id={product_id}")

    results: list[dict[str, Any]] = []


    for user_id, text in user_texts.items():
        try:
            # Step 1: domain model
            grade, confidence = domain_model_predict(text)
            tags = []

            # Step 2: escalate if needed
            if confidence < threshold if threshold else 0.6:
                _log.info(f"Escalated to LLM for text: '{text[:50]}...'")
                grade, confidence, tags = llm_predict(text, specified_provider)
                
            else:
                _log.info(f"Domain model handled text: '{text[:50]}...'")

            results.append({
                "text": text,
                "grade": grade,
                "confidence": confidence,
                "tags": tags,
            })

            if user_id:
                store_user_data(user_id, product_id, grade, text)


            if product_id:
                store_product_data(product_id)

        except Exception as e:
            _log.error(f"Failed to classify review '{text[:50]}...': {e}")
            results.append({
                "text": text,
                "grade": None,
                "confidence": 0.0,
                "tags": [],
                "error": str(e),
            })

    return {
        "product_id": product_id,
        "results": results,
    }
