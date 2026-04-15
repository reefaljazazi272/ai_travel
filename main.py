import numpy as np
from sqlalchemy import select
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# تأكد أن اسم الملف هو db.py أو قم بتغييره لاسم ملفك الصحيح
from db import engine, destinations 

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_recommendations(user_query, top_n=3):
    with engine.connect() as conn:
        stmt = select(destinations)
        results = conn.execute(stmt).fetchall()
        
        if not results:
            return [], []

        query_embedding = model.encode([user_query])
        
        db_vectors = []
        for row in results:
            # تنظيف النص وتحويله لمصفوفة أرقام (Vector)
            clean_str = row.embedding.replace('{', '').replace('}', '')
            vector = [float(x) for x in clean_str.split(',')]
            db_vectors.append(vector)
        
        db_vectors = np.array(db_vectors)
        similarities = cosine_similarity(query_embedding, db_vectors)[0]
        
        # الإصلاح هنا: استخدام argsort لجلب ترتيب أفضل 3 نتائج مختلفة
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        recs = [results[i] for i in top_indices]
        scores = [similarities[i] for i in top_indices]
        
        return recs, scores