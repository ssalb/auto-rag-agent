from rag_agent.db.connection import DuckDBConnection
from rag_agent.db.models import DocumentModel


def get_connection():
    db = DuckDBConnection()
    conn = db.connect()
    return conn


def create_schema(conn):
    """Initialize database schema for vector search"""
    DocumentModel.create_table_if_not_exists(conn)
    return True


def init_db():
    conn = get_connection()
    create_schema(conn)
    return conn
