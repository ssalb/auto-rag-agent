"""
Database-specific pytest fixtures for testing RAG agent components.
"""
import pytest
from unittest.mock import patch
import duckdb
import json

@pytest.fixture
def mock_db_config():
    """Mock DuckDB configuration with test-specific settings"""
    with patch('rag_agent.config.DuckDBConfig') as mock_config:
        mock_config.DUCKDB_PATH = ":memory:"  # Use in-memory DB for tests
        mock_config.EMBEDDING_DIM = 384
        mock_config.VSS_METRIC = "cosine"
        mock_config.VSS_M = 16
        mock_config.VSS_EF_CONSTRUCTION = 100
        yield mock_config

@pytest.fixture
def mock_db_connection(mock_db_config):
    """Create a mock database connection for testing"""
    # Create an in-memory DuckDB connection for testing
    conn = duckdb.connect(mock_db_config.DUCKDB_PATH)
    
    # Setup the required extensions and types
    conn.execute("INSTALL vss;")
    conn.execute("LOAD vss;")
    
    try:
        conn.execute(f"CREATE TYPE embedding AS FLOAT[{mock_db_config.EMBEDDING_DIM}]")
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise e
    
    # Create a mock connection that returns our test connection
    with patch('rag_agent.db.connection.DuckDBConnection') as mock_connection_class:
        instance = mock_connection_class.return_value
        instance.connect.return_value = conn
        instance.conn = conn
        yield instance

@pytest.fixture
def setup_document_table(mock_db_connection):
    """Set up the document table schema for testing"""
    conn = mock_db_connection.connect()
    
    # Create the document table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        doc_name TEXT NOT NULL,
        chunk_text TEXT NOT NULL,
        named_entities JSON,
        embedding FLOAT[384] NOT NULL
    )
    """)
    
    try:
        conn.execute("""
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
        ON document_chunks 
        USING HNSW (embedding)
        WITH (
            metric = 'cosine',
            m = 16,
            ef_construction = 100
        )
        """)
    except Exception as e:
        print(f"Warning in test: Could not create HNSW index: {e}")
    
    yield conn
    
    # Clean up after the test
    conn.execute("DROP TABLE IF EXISTS document_chunks")

@pytest.fixture
def sample_document_chunks(sample_embedding):
    """Create a list of sample document chunks for testing"""
    return [
        {
            "doc_name": "test_doc_1.txt",
            "chunk_text": "This is a test document chunk number one.",
            "named_entities": {"Test": "ORG"},
            "embedding": sample_embedding
        },
        {
            "doc_name": "test_doc_2.txt",
            "chunk_text": "This is a test document chunk number two.",
            "named_entities": {"Test": "ORG", "Two": "CARDINAL"},
            "embedding": sample_embedding
        }
    ]

@pytest.fixture
def populated_document_table(setup_document_table, sample_document_chunks):
    """Populate the document table with sample data for testing"""
    conn = setup_document_table
    
    # Insert sample data
    for chunk in sample_document_chunks:
        named_entities = json.dumps(chunk["named_entities"])
        conn.execute("""
        INSERT INTO document_chunks (doc_name, chunk_text, named_entities, embedding)
        VALUES (?, ?, ?, ?)
        """, (chunk["doc_name"], chunk["chunk_text"], named_entities, chunk["embedding"]))
    
    return conn
