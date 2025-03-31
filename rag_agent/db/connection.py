import duckdb
import os
from rag_agent.config import DuckDBConfig

class DuckDBConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DuckDBConnection, cls).__new__(cls)
            cls._instance.conn = None
        return cls._instance
    
    def connect(self):
        """Create a connection to DuckDB database with vector search extension"""
        if self.conn is None:
            # Ensure directory exists
            os.makedirs(os.path.dirname(DuckDBConfig.DUCKDB_PATH), exist_ok=True)
            
            # Connect to database
            self.conn = duckdb.connect(DuckDBConfig.DUCKDB_PATH)
            
            # Install and load VSS extension
            self.conn.execute("INSTALL vss;")
            self.conn.execute("LOAD vss;")
            
            # Register array type for embeddings
            embedding_type = f"FLOAT[{DuckDBConfig.EMBEDDING_DIM}]"
            self.conn.execute(f"CREATE TYPE IF NOT EXISTS embedding AS {embedding_type}")
        
        return self.conn
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None