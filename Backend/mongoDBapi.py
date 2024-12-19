from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

def get_database():
    db_name = os.getenv("MONGODB_DB")
    if not db_name:
        raise ValueError("Environment variable MONGODB_DB is not set or is empty")
    client = MongoClient(os.getenv("MONGODB_URI"))
    return client[db_name]


# Insert a document into any collection
def insert_document(collection_name, document):
    db = get_database()
    collection = db[collection_name]
    result = collection.insert_one(document)
    return result.inserted_id

# Fetch all documents from a collection
def get_all_documents(collection_name):
    db = get_database()
    collection = db[collection_name]
    return list(collection.find())

# Find documents with a filter
def find_documents(collection_name, filter_criteria):
    db = get_database()
    collection = db[collection_name]
    return list(collection.find(filter_criteria))

# Update a document
def update_document(collection_name, filter_criteria, updated_data):
    db = get_database()
    collection = db[collection_name]
    result = collection.update_one(filter_criteria, {"$set": updated_data})
    return result.modified_count

# Delete a document
def delete_document(collection_name, filter_criteria):
    db = get_database()
    collection = db[collection_name]
    result = collection.delete_one(filter_criteria)
    return result.deleted_count


if __name__ == "__main__":
    # Testing the functions
    print(get_database())
    print(insert_document("products", {"name": "Keyboard", "price": 500}))
    print(get_all_documents("products"))
    print(find_documents("products", {"name": "Keyboard"}))
    print(update_document("products", {"name": "Keyboard"}, {"price": 300}))
    print(delete_document("products", {"name": "Keyboard"}))
    print(get_all_documents("products"))