import logging
from datetime import datetime, timezone
from services.db import get_users_collection, get_products_collection
from services.models.llm import completion

_log = logging.getLogger(__name__)


# ----------------- DB Storage -----------------
def store_user_data(user_id: str, product_id: str, grade: float, text: str):
    users = get_users_collection()
    now = datetime.now(timezone.utc)

    result = users.update_one(
        {"user_id": user_id},
        {
            "$setOnInsert": {"first_review_date": now},
            "$set": {
                "last_review_date": now,
                f"reviews.{product_id}": {
                    "product_id": product_id,
                    "grade": grade,
                    "text": text,
                },
            },
        },
        upsert=True,
    )
    if not result.acknowledged:
        _log.error(f"Failed to update user {user_id} with review for {product_id}")


def store_product_data(product_id: str):
    users = get_users_collection()
    products = get_products_collection()

    stats = next(
        users.aggregate([
            {"$match": {f"reviews.{product_id}": {"$exists": True}}},
            {"$group": {
                "_id": None,
                "count": {"$sum": 1},
                "avg": {"$avg": f"$reviews.{product_id}.grade"},
            }},
        ]),
        None,
    )

    count = stats.get("count", 0) if stats else 0
    average = stats.get("avg") if stats else None

    result = products.update_one(
        {"product_id": product_id},
        {
            "$set": {
                "product_id": product_id,
                "average_grade": average,
                "grade_count": count,
            },
            "$setOnInsert": {"tags_counts": {}},
        },
        upsert=True,
    )
    if not result.acknowledged:
        _log.error(f"Failed to update product {product_id} with stats")


# ----------------- LLM Summarization -----------------
def _run_summary(prompt: str) -> str | None:
    """Helper to call LLM with strict output rules."""
    final_prompt = (
        "You are a summarization system.\n"
        "Your ONLY job is to output the summary text directly.\n"
        "- Do not include introductions, explanations, or phrases like 'Here is...'.\n"
        "- Only response in English, despite any other language in the prompt.\n"
        "- If the provided content is empty or insufficient, return exactly 'NONE'.\n\n"
        f"{prompt}"
    )
    result = completion(final_prompt, schema=None, provider=None)
    text = str(result).strip()
    return None if text.upper() == "NONE" else text


def build_user_portrait(user: dict) -> str | None:
    reviews = user.get("reviews", {})
    texts = [v.get("text", "") for v in reviews.values() if v.get("text")]
    if not texts:
        return None

    prompt = (
        "Given the following product reviews by the same user, "
        "write a short, neutral portrait of their preferences and style. "
        "Keep under 60 words.\n\n"
        + "\n".join(texts[:10])
    )
    return _run_summary(prompt)


def build_product_summary(product: dict) -> str | None:
    pid = product.get("product_id")
    if not pid:
        return None

    users_cur = get_users_collection().find(
        {f"reviews.{pid}": {"$exists": True}}, {f"reviews.{pid}": 1}
    )

    texts: list[str] = []
    for u in users_cur:
        review = u.get("reviews", {}).get(pid)
        if isinstance(review, dict):
            t = review.get("text")
            if t:
                texts.append(t)

    if not texts:
        return None

    tag_counts = product.get("tags_counts", {})
    prompt = (
        "Summarize perceived qualities of the product using user reviews and tag counts. "
        "Write a concise, neutral summary under 60 words.\n\n"
        f"Tags (counts): {tag_counts}\n\n"
        + "\n".join(texts[:20])
    )
    return _run_summary(prompt)


# ----------------- Services -----------------
def get_user_data_service(user_id: str) -> dict:
    user = get_users_collection().find_one({"user_id": user_id})
    if not user:
        raise KeyError(f"User {user_id} not found")

    summary = build_user_portrait(user)
    if "_id" in user:
        user["_id"] = str(user["_id"])
    return {"user": user, "llm_summary": summary}


def get_product_info_service(product_id: str) -> dict:
    product = get_products_collection().find_one({"product_id": product_id})
    if not product:
        raise KeyError(f"Product {product_id} not found")

    summary = build_product_summary(product)
    if "_id" in product:
        product["_id"] = str(product["_id"])
    return {"product": product, "llm_summary": summary}
