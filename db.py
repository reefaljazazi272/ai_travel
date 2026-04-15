import os
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found. Check your .env file!")

engine = create_engine(DATABASE_URL)
metadata = MetaData()

destinations = Table(
    'destinations', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String),
    Column('country', String),
    Column('description', String),
    Column('category', String),
    Column('embedding', String)
)

def init_db():
    metadata.create_all(engine)