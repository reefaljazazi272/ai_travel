import json
from db import engine, destinations, init_db
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def seed_from_json():
    init_db()
    with open('destinations.json', 'r') as f:
        data = json.load(f)

    with engine.connect() as conn:
        print(" Fall data core..")
        for item in data:
            vector = model.encode(item['description']).tolist()
            postgres_vector = "{" + ",".join(map(str, vector)) + "}"
            
            ins = destinations.insert().values(
                name=item['name'],
                country=item['country'],
                description=item['description'],
                category=item['category'],
                embedding=postgres_vector
            )
            conn.execute(ins)
        conn.commit()
        print("Success!")

if __name__ == "__main__":
    seed_from_json()