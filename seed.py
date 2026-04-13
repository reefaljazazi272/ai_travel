import json
from sqlalchemy.orm import Session
from db import SessionLocal, Destination, init_db
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def seed_from_json():
    init_db()
    db: Session = SessionLocal()
    
    with open('destinations.json', 'r') as f:
        data = json.load(f)
    
    try:
        print("جاري تحويل بيانات JSON وحفظها في قاعدة البيانات...")
        for item in data:
            exists = db.query(Destination).filter(Destination.name == item['name']).first()
            if not exists:
                vector = model.encode(item['description']).tolist()
                new_dest = Destination(
                    name=item['name'],
                    country=item['country'],
                    description=item['description'],
                    category=item['category'],
                    embedding=vector
                )
                db.add(new_dest)
        
        db.commit()
        print("تمت عملية الاستيراد من JSON بنجاح! ✅")
    except Exception as e:
        print(f"حدث خطأ: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_from_json()