"""
Unit tests for the DocumentIndexer tool.
"""
from unittest.mock import patch, MagicMock
from rag_agent.tools.indexer import DocumentIndexer


def test_indexer_initialization():
    """Test that the document indexer tool initializes correctly"""
    # Patch at the import point in the module being tested, not where it's defined
    with patch('rag_agent.tools.indexer.DocumentConverter') as MockConverter, \
         patch('rag_agent.tools.indexer.HybridChunker') as MockChunker:
        
        # Create the tool
        tool = DocumentIndexer()
        
        # Verify the tool has the expected attributes
        assert tool.name == "document_indexer"
        assert "index" in tool.description.lower()
        assert hasattr(tool, "converter")
        assert hasattr(tool, "chunker")
        
        # Verify the correct classes were instantiated
        assert MockConverter.called
        assert MockChunker.called


def test_indexer_document_conversion_failure():
    """Test handling of document conversion failure"""
    with patch('rag_agent.tools.indexer.DocumentConverter') as MockConverter, \
         patch('rag_agent.tools.indexer.HybridChunker') as MockChunker:
        
        # Configure the converter to raise an exception
        mock_converter = MagicMock()
        MockConverter.return_value = mock_converter
        mock_converter.convert.side_effect = Exception("Conversion error")
        
        # Create the tool and call forward
        tool = DocumentIndexer()
        result = tool.forward(document_path="/path/to/document.pdf")
        
        # Verify the response indicates failure
        assert result is not None
        assert "Processing document.pdf" in result
        assert "Failed to convert document" in result


def test_indexer_forward_success():
    """Test successful document indexing"""
    with patch('rag_agent.tools.indexer.DocumentConverter') as MockConverter, \
         patch('rag_agent.tools.indexer.HybridChunker') as MockChunker, \
         patch('rag_agent.tools.indexer.extract_entities') as mock_extract_entities, \
         patch('rag_agent.tools.indexer.encode') as mock_encode, \
         patch('rag_agent.tools.indexer.bulk_insert_chunks') as mock_bulk_insert:
        
        # Configure mocks for a successful indexing process
        mock_converter = MagicMock()
        MockConverter.return_value = mock_converter
        
        mock_doc = MagicMock()
        mock_converter.convert.return_value = MagicMock()
        mock_converter.convert.return_value.document = mock_doc
        
        mock_chunker = MagicMock()
        MockChunker.return_value = mock_chunker
        
        mock_chunks = [MagicMock(), MagicMock()]
        mock_chunker.chunk.return_value = mock_chunks
        mock_chunker.serialize.side_effect = lambda chunk: f"Chunk text for {chunk}"
        
        mock_extract_entities.return_value = {"Entity": "ORG"}
        
        mock_embeddings = [MagicMock()]
        mock_encode.return_value = mock_embeddings
        mock_embeddings[0].tolist.return_value = [0.1] * 384
        
        # Create the tool and call forward
        tool = DocumentIndexer()
        result = tool.forward(document_path="/path/to/document.pdf")
        
        # Verify the document was converted
        mock_converter.convert.assert_called_once_with("/path/to/document.pdf")
        
        # Verify chunking was called
        mock_chunker.chunk.assert_called_once_with(dl_doc=mock_doc)
        
        # Verify named entity extraction and embedding generation
        assert mock_extract_entities.call_count == len(mock_chunks)
        assert mock_encode.call_count >= 1
        
        # Verify bulk insertion was called
        assert mock_bulk_insert.called
        
        # Verify the response indicates success
        assert result is not None
        assert "Processing document.pdf" in result
        assert "Document indexed successfully" in result


def test_indexer_forward_chunking_failure():
    """Test handling of chunking failure"""
    with patch('rag_agent.tools.indexer.DocumentConverter') as MockConverter, \
         patch('rag_agent.tools.indexer.HybridChunker') as MockChunker, \
         patch('rag_agent.tools.indexer.extract_entities') as mock_extract_entities, \
         patch('rag_agent.tools.indexer.encode') as mock_encode, \
         patch('rag_agent.tools.indexer.bulk_insert_chunks') as mock_bulk_insert:
        
        # Configure the document conversion to succeed
        mock_converter = MagicMock()
        MockConverter.return_value = mock_converter
        
        mock_doc = MagicMock()
        mock_converter.convert.return_value = MagicMock()
        mock_converter.convert.return_value.document = mock_doc
        
        # Configure the chunker to raise an exception
        mock_chunker = MagicMock()
        MockChunker.return_value = mock_chunker
        mock_chunker.chunk.side_effect = Exception("Chunking error")
        
        # Configure the bulk insert to succeed with empty list in case the code proceeds
        mock_bulk_insert.return_value = 0
        
        # Create the tool and call forward
        tool = DocumentIndexer()
        result = tool.forward(document_path="/path/to/document.pdf")
        
        # Verify the response indicates failure but continues
        assert result is not None
        assert "Processing document.pdf" in result
        assert "Failed to process chuncks" in result


def test_indexer_forward_indexing_failure():
    """Test handling of indexing failure"""
    with patch('rag_agent.tools.indexer.DocumentConverter') as MockConverter, \
         patch('rag_agent.tools.indexer.HybridChunker') as MockChunker, \
         patch('rag_agent.tools.indexer.extract_entities') as mock_extract_entities, \
         patch('rag_agent.tools.indexer.encode') as mock_encode, \
         patch('rag_agent.tools.indexer.bulk_insert_chunks') as mock_bulk_insert:
        
        # Configure document conversion and chunking to succeed
        mock_converter = MagicMock()
        MockConverter.return_value = mock_converter
        
        mock_doc = MagicMock()
        mock_converter.convert.return_value = MagicMock()
        mock_converter.convert.return_value.document = mock_doc
        
        mock_chunker = MagicMock()
        MockChunker.return_value = mock_chunker
        
        mock_chunks = [MagicMock()]
        mock_chunker.chunk.return_value = mock_chunks
        mock_chunker.serialize.return_value = "Chunk text"
        
        # Configure named entity extraction and encoding to succeed
        mock_extract_entities.return_value = {"Entity": "ORG"}
        
        mock_embeddings = [MagicMock()]
        mock_encode.return_value = mock_embeddings
        mock_embeddings[0].tolist.return_value = [0.1] * 384
        
        # Configure bulk_insert to raise an exception
        mock_bulk_insert.side_effect = Exception("Indexing error")
        
        # Create the tool and call forward
        tool = DocumentIndexer()
        result = tool.forward(document_path="/path/to/document.pdf")
        
        # Verify the response indicates failure
        assert result is not None
        assert "Processing document.pdf" in result
        assert "Failed to index document" in result


def test_indexer_url_handling():
    """Test handling of URL documents"""
    with patch('rag_agent.tools.indexer.DocumentConverter') as MockConverter, \
         patch('rag_agent.tools.indexer.HybridChunker') as MockChunker, \
         patch('rag_agent.tools.indexer.bulk_insert_chunks') as mock_bulk_insert:
        
        # Configure document converter
        mock_converter = MagicMock()
        MockConverter.return_value = mock_converter
        
        # Configure document conversion to succeed
        mock_doc = MagicMock()
        mock_converter.convert.return_value = MagicMock()
        mock_converter.convert.return_value.document = mock_doc
        
        # Configure chunker to return empty list to skip processing
        mock_chunker = MagicMock()
        MockChunker.return_value = mock_chunker
        mock_chunker.chunk.return_value = []
        
        # Configure bulk_insert to succeed with empty list
        mock_bulk_insert.return_value = 0
        
        # URL to test
        url = "https://example.com/document.pdf"
        
        # Create the tool and call forward
        tool = DocumentIndexer()
        result = tool.forward(document_path=url)
        
        # Verify the document was converted with the URL
        mock_converter.convert.assert_called_once_with(url)
