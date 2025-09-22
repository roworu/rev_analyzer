db = db.getSiblingDB('rev_analyzer');

db.createCollection('cached_reviews');
db.createCollection('token_usage');

db.cached_reviews.createIndex({ "text": "text" });
db.cached_reviews.createIndex({ "created_at": -1 });

print('Database, collections and indexes initialized successfully');
