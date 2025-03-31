import json
from rag_agent.db import get_connection
from rag_agent.db.models import DocumentModel

def store_document_chunk(doc_name, chunk_text, named_entities, embedding):
    """
    Store a document chunk with its embedding in the database
    
    Args:
        doc_name: Name or identifier of the document
        chunk_text: Text content of the chunk
        named_entities: List of named entities (will be converted to JSON)
        embedding: Vector embedding as a list of floats
    
    Returns:
        Boolean indicating success
    """
    conn = get_connection()
    
    # Convert named entities to JSON if needed
    if not isinstance(named_entities, str):
        named_entities = json.dumps(named_entities)
    
    DocumentModel.insert_document_chunk(
        conn, 
        doc_name, 
        chunk_text, 
        named_entities, 
        embedding
    )
    
    return True

def search_similar_chunks(query_embedding, limit=5):
    """
    Search for similar document chunks using vector similarity
    
    Args:
        query_embedding: Vector embedding as a list of floats
        limit: Maximum number of results to return
        
    Returns:
        List of matching document chunks with similarity scores
    """
    conn = get_connection()
    results = DocumentModel.search_similar(conn, query_embedding, limit)
    
    # Format results as dictionaries
    formatted_results = []
    for row in results:
        formatted_results.append({
            "id": row[0],
            "doc_name": row[1],
            "chunk_text": row[2],
            "named_entities": json.loads(row[3]) if row[3] else [],
            "distance": row[4]
        })
    
    return formatted_results

def bulk_insert_chunks(chunks_list):
    """
    Bulk insert multiple document chunks
    
    Args:
        chunks_list: List of dictionaries, each containing:
            - doc_name: Document name/ID
            - chunk_text: Text content
            - named_entities: Dict of named entities
            - embedding: Vector embedding as list of floats
                
    Returns:
        Number of chunks inserted
    """
    conn = get_connection()
    
    # Convert list of dictionaries to list of tuples
    formatted_chunks = []
    for chunk in chunks_list:
        # Process named_entities to ensure it's JSON
        named_entities = chunk.get("named_entities", [])
        if not isinstance(named_entities, str):
            named_entities = json.dumps(named_entities)
            
        # Create tuple with correct order
        chunk_tuple = (
            chunk["doc_name"],
            chunk["chunk_text"],
            named_entities,
            chunk["embedding"]
        )
        
        formatted_chunks.append(chunk_tuple)
    
    doc_count = DocumentModel.insert_document_chunks_batch(conn, formatted_chunks)
    return doc_count