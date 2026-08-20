import os
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

# Read from .env
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/finmaths")

# Initialize connection pool
try:
    db_pool = SimpleConnectionPool(1, 10, dsn=DATABASE_URL)
except Exception as e:
    print(f"Warning: Failed to connect to PostgreSQL: {e}")
    db_pool = None

@contextmanager
def get_db_connection():
    if db_pool is None:
        raise Exception("Database connection pool is not initialized. Please check DATABASE_URL.")
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)
