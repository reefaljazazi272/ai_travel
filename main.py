from sqlalchemy.orm import Session
from db import SessionLocal, Destination
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_recommendation(user_query):
    db: Session = SessionLocal()
    
    all_destinations = db.query(Destination).all()
    
    embeddings = np.array([d.embedding for d in all_destinations])
    
    query_vector = model.encode([user_query])
    
    similarities = cosine_similarity(query_vector, embeddings)[0]
    best_index = np.argmax(similarities)
    
    result = all_destinations[best_index]
    db.close()
    return result, similarities[best_index]

query = input("What's your dream trip? ")
res, score = get_recommendation(query)
print(f"Recommended: {res.name} | Score: {score:.2f}")