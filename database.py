from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
from typing import Optional, List, Dict

class Database:
    def __init__(self, mongodb_uri: str):
        self.client = MongoClient(mongodb_uri, server_api=ServerApi('1'))
        self.db = self.client["telegram_ai_assistant"]
        print("Connected to MongoDB successfully!")

    def add_user(self, user_id: str, name: str, email: Optional[str] = None):
        users = self.db["users"]
        if not users.find_one({"_id": user_id}):
            user_data = {
                "_id": user_id,
                "name": name,
                "email": email,
                "created_at": datetime.utcnow(),
            }
            users.insert_one(user_data)

    def add_document(self, user_id: str, file_name: str, mime_type: str, text_content: str):
        documents = self.db["documents"]
        document_data = {
            "user_id": user_id,
            "file_name": file_name,
            "mime_type": mime_type,
            "text_content": text_content,
            "created_at": datetime.utcnow(),
        }
        document_id = documents.insert_one(document_data).inserted_id
        return document_id

    def add_embedding(self, user_id: str, document_id: str, vector: list, context: str):
        embeddings = self.db["embeddings"]
        embedding_data = {
            "user_id": user_id,
            "vector": vector,
            "metadata": {
                "source": document_id,
                "context": context,
            },
            "created_at": datetime.utcnow(),
        }
        embeddings.insert_one(embedding_data)

    def get_user_embeddings(self, user_id: str):
        embeddings = self.db["embeddings"]
        return list(embeddings.find({"user_id": user_id}))

    def get_user_documents(self, user_id: str):
        documents = self.db["documents"]
        return list(documents.find({"user_id": user_id}))

    def add_image(self, user_id: str, image_name: str, image_data: bytes):
        images = self.db["images"]
        image_record = {
            "user_id": user_id,
            "image_name": image_name,
            "image_data": image_data,
            "created_at": datetime.utcnow(),
        }
        image_id = images.insert_one(image_record).inserted_id
        return image_id

    def get_user_images(self, user_id: str) -> List[Dict]:
        images = self.db["images"]
        return list(images.find({"user_id": user_id}))

    # New methods for saving conversations
    def add_conversation(self, user_id: str, question: str, response: str):
        conversations = self.db["conversations"]
        conversation_record = {
            "user_id": user_id,
            "question": question,
            "response": response,
            "created_at": datetime.utcnow(),
        }
        conversation_id = conversations.insert_one(conversation_record).inserted_id
        return conversation_id

    def get_user_conversations(self, user_id: str) -> List[Dict]:
        conversations = self.db["conversations"]
        return list(conversations.find({"user_id": user_id}))