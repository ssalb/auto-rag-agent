"""
Unit tests for the DocumentModel class.
"""

import json
from rag_agent.db.models import DocumentModel


def test_create_table_if_not_exists(setup_document_table):
    """Test that the document table is created successfully"""
    conn = setup_document_table

    # Check if the table exists by running a query
    result = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_type='BASE TABLE' AND table_name='document_chunks'"
    ).fetchone()
    assert result is not None
    assert result[0] == "document_chunks"


def test_insert_document_chunk(setup_document_table, sample_embedding):
    """Test inserting a single document chunk"""
    conn = setup_document_table

    # Test data
    doc_name = "test_doc.txt"
    chunk_text = "This is a test document chunk."
    named_entities = json.dumps({"Test": "ORG"})

    # Insert the chunk
    DocumentModel.insert_document_chunk(
        conn, doc_name, chunk_text, named_entities, sample_embedding
    )

    # Query to verify insertion
    result = conn.execute("SELECT * FROM document_chunks").fetchone()

    # Check that data was inserted correctly
    assert result is not None
    assert result[0] == doc_name
    assert result[1] == chunk_text
    assert json.loads(result[2]) == {"Test": "ORG"}
    # The last item should be the embedding, which is harder to test for equality
    assert len(result[3]) == len(sample_embedding)


def test_insert_document_chunks_batch(setup_document_table, sample_embedding):
    """Test batch insertion of document chunks"""
    conn = setup_document_table

    # Test data
    chunks = [
        ("doc1.txt", "Chunk 1 text", json.dumps({"Entity1": "ORG"}), sample_embedding),
        ("doc2.txt", "Chunk 2 text", json.dumps({"Entity2": "PER"}), sample_embedding),
        ("doc3.txt", "Chunk 3 text", json.dumps({"Entity3": "LOC"}), sample_embedding),
    ]

    # Insert the chunks in batch
    count = DocumentModel.insert_document_chunks_batch(conn, chunks)

    # Verify correct count
    assert count == 3

    # Query to verify insertion
    results = conn.execute("SELECT * FROM document_chunks").fetchall()

    # Check that data was inserted correctly
    assert len(results) == 3

    # Check each row
    for i, result in enumerate(results):
        assert result[0] == chunks[i][0]  # doc_name
        assert result[1] == chunks[i][1]  # chunk_text
        assert json.loads(result[2]) == json.loads(chunks[i][2])  # named_entities


def test_search_similar(populated_document_table, sample_embedding):
    """Test searching for similar document chunks"""
    conn = populated_document_table

    # Search using the sample embedding
    results = DocumentModel.search_similar(conn, sample_embedding, limit=1)

    # Verify we got a result
    assert len(results) == 1

    # Verify the result has the expected columns
    assert len(results[0]) == 4  # doc_name, chunk_text, named_entities, distance

    # Verify we can search with a document scope
    results_with_scope = DocumentModel.search_similar(
        conn, sample_embedding, limit=1, doc_scope="test_doc_1.txt"
    )

    # Verify we got a result filtered by doc_name
    assert len(results_with_scope) == 1
    assert results_with_scope[0][0] == "test_doc_1.txt"
