from rag_agent.config import DuckDBConfig

class DocumentModel:
    """Model for document chunks with vector embeddings"""
    table_name = "document_chunks"
    index_name = "document_chunks_embedding_idx"
    
    @classmethod
    def create_table_if_not_exists(cls, conn):
        """Create the document chunks table if it doesn't exist"""
        embedding_type = f"FLOAT[{DuckDBConfig.EMBEDDING_DIM}]"
        
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {cls.table_name} (
            doc_name TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            named_entities JSON,
            embedding {embedding_type} NOT NULL
        )
        """)
        
        # Create HNSW index if it doesn't exist
        try:
            conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {cls.index_name} 
            ON {cls.table_name} 
            USING HNSW (embedding)
            WITH (
                metric = '{DuckDBConfig.VSS_METRIC}',
                m = {DuckDBConfig.VSS_M},
                ef_construction = {DuckDBConfig.VSS_EF_CONSTRUCTION}
            )
            """)
        except Exception as e:
            print(f"Warning: Could not create HNSW index: {e}")
    
    @classmethod
    def insert_document_chunk(cls, conn, doc_name, chunk_text, named_entities, embedding):
        """Insert a document chunk with its embedding"""
        conn.execute(f"""
        INSERT INTO {cls.table_name} (doc_name, chunk_text, named_entities, embedding)
        VALUES (?, ?, ?, ?)
        """, (doc_name, chunk_text, named_entities, embedding))

    @classmethod
    def insert_document_chunks_batch(cls, conn, chunks):
        """
        Insert multiple document chunks in a single batch operation
        
        Args:
            conn: DuckDB connection
            chunks: List of tuples (doc_name, chunk_text, named_entities, embedding)
        
        Returns:
            Number of chunks inserted
        """
        # Start a transaction for better performance
        conn.execute("BEGIN TRANSACTION")
        
        try:
            # Prepare a parameterized query
            query = f"""
            INSERT INTO {cls.table_name} (doc_name, chunk_text, named_entities, embedding)
            VALUES (?, ?, ?, ?)
            """
            
            # Execute in batch
            count = 0
            for chunk in chunks:
                conn.execute(query, chunk)
                count += 1
            
            # Commit the transaction
            conn.execute("COMMIT")
            return count
            
        except Exception as e:
            # Rollback on error
            conn.execute("ROLLBACK")
            raise e
    
    @classmethod
    def search_similar(cls, conn, query_embedding, limit=5, doc_scope=None):
        """Search for similar document chunks using vector similarity"""
        embedding_type = f"FLOAT[{DuckDBConfig.EMBEDDING_DIM}]"
        
        result = conn.execute(f"""
        SELECT 
            doc_name,
            chunk_text,
            named_entities,
            array_distance(embedding, ?::{embedding_type}) as distance
        FROM {cls.table_name}
        {"WHERE doc_name = '"+doc_scope+"'" if doc_scope is not None else ''}
        ORDER BY array_distance(embedding, ?::{embedding_type})
        LIMIT ?
        """, (query_embedding, query_embedding, limit)).fetchall()
        
        return result