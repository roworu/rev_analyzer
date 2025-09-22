import os
import logging
from pymongo import MongoClient

MONGO_CONNECTION = os.environ.get("MONGO_CONNECTION", "mongodb://localhost:27017")

DB_NAME = os.environ.get("DB_NAME", "rev_analyzer")
USERS_COLLECTION_NAME = os.environ.get("USERS_COLLECTION_NAME", "users")
PRODUCTS_COLLECTION_NAME = os.environ.get("USERS_COLLECTION_NAME", "products")

_log = logging.getLogger(__name__)

def create_db_client():
    try: 
        return MongoClient(MONGO_CONNECTION)
    
    except Exception as e:
        _log.error(
            f"Error connecting to MongoDB using connection string '{MONGO_CONNECTION}'. \
            Error message: {e}"
        )
        raise

def get_collection(collection_name):
    client = create_db_client()
    return client[DB_NAME][collection_name]

def get_users_collection():
    return get_collection(USERS_COLLECTION_NAME)
    
def get_products_collection():
    return get_collection(PRODUCTS_COLLECTION_NAME)


