import os
import logging
from urllib.parse import quote_plus
from dotenv import load_dotenv
import psycopg2
from psycopg2 import pool
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration parameters
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "arecaguard_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# SQLAlchemy Database URL
encoded_password = quote_plus(DB_PASSWORD)
DATABASE_URL = (
    f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# SQLAlchemy engine & session factory
try:
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception as e:
    logger.warning(f"SQLAlchemy Engine initialization deferred or failed: {e}")
    engine = None
    SessionLocal = None
    Base = declarative_base()

# Optional psycopg2 connection pool for direct low-level queries
_connection_pool = None


def get_connection_pool():
    """Initializes or retrieves the psycopg2 ThreadedConnectionPool."""
    global _connection_pool
    if _connection_pool is None:
        try:
            _connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            logger.info("PostgreSQL psycopg2 connection pool established successfully.")
        except Exception as err:
            logger.error(f"Error creating psycopg2 connection pool: {err}")
            _connection_pool = None
    return _connection_pool


def get_raw_db_connection():
    """Retrieves a direct connection from the connection pool or creates a standalone one."""
    cp = get_connection_pool()
    if cp:
        return cp.getconn()
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def release_raw_db_connection(conn):
    """Releases a connection back to the connection pool."""
    cp = get_connection_pool()
    if cp and conn:
        cp.putconn(conn)
    elif conn:
        conn.close()


def get_db_session():
    """Context manager or generator for SQLAlchemy ORM sessions."""
    if SessionLocal is None:
        raise RuntimeError("Database session factory is not configured.")
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def test_db_connection():
    """Verifies whether PostgreSQL is reachable."""
    try:
        conn = get_raw_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.close()
        release_raw_db_connection(conn)
        return True, "Database connection successful."
    except Exception as e:
        return False, str(e)
