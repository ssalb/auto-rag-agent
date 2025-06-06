"""
Unit tests for the TextRetriever tool.
"""
import pytest
from unittest.mock import patch, MagicMock
from rag_agent.tools.retriever import TextRetriever


def test_retriever_initialization():
    """Test that the text retriever tool initializes correctly"""
    tool = TextRetriever(max_results=5)
    
    assert tool.name == "search_tool"
    assert "search" in tool.description.lower()
    assert tool.max_results == 5


@pytest.mark.parametrize("max_results", [3, 5, 10])
def test_retriever_max_results_parameter(max_results):
    """Test that max_results parameter is respected"""
    tool = TextRetriever(max_results=max_results)
    assert tool.max_results == max_results


def test_retriever_forward_basic(sample_text):
    """Test basic retrieval without a doc_name filter"""
    query = "What advice does Gandalf give?"
    
    # Set up mocks for all the external functions
    with patch('rag_agent.tools.retriever.encode') as mock_encode, \
         patch('rag_agent.tools.retriever.extract_entities') as mock_extract, \
         patch('rag_agent.tools.retriever.search_similar_chunks') as mock_search:
        
        # Configure the mocks
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_encode.return_value = [mock_embedding]
        
        mock_extract.return_value = {"Gandalf": "PER"}
        
        mock_search.return_value = [
            {
                "doc_name": "lotr.txt",
                "chunk_text": "Text about Frodo and Gandalf",
                "named_entities": {"Frodo": "PER", "Gandalf": "PER"},
                "distance": 0.2
            }
        ]
        
        # Create and call the tool
        tool = TextRetriever(max_results=2)
        result = tool.forward(query=query)
        
        # Verify the function calls
        mock_encode.assert_called_once_with([query])
        mock_extract.assert_called_once_with(query)
        mock_search.assert_called_once()
        
        # Verify the result contains the expected text
        assert "Text about Frodo and Gandalf" in result


def test_retriever_forward_with_doc_name():
    """Test retrieval with a doc_name filter"""
    query = "What is the meaning of life?"
    doc_name = "hitchhikers_guide.txt"
    
    # Set up mocks for all the external functions
    with patch('rag_agent.tools.retriever.encode') as mock_encode, \
         patch('rag_agent.tools.retriever.extract_entities') as mock_extract, \
         patch('rag_agent.tools.retriever.search_similar_chunks') as mock_search:
        
        # Configure the mocks
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_encode.return_value = [mock_embedding]
        
        mock_extract.return_value = {}
        
        mock_search.return_value = [
            {
                "doc_name": doc_name,
                "chunk_text": "The answer is 42.",
                "named_entities": {},
                "distance": 0.2
            }
        ]
        
        # Create and call the tool
        tool = TextRetriever(max_results=2)
        result = tool.forward(query=query, doc_name=doc_name)
        
        # Verify the function calls
        mock_encode.assert_called_once_with([query])
        mock_extract.assert_called_once_with(query)
        mock_search.assert_called_once()
        
        # Verify that doc_scope was passed correctly
        call_args = mock_search.call_args
        assert call_args[1]['doc_scope'] == doc_name
        
        # Verify the result contains the expected text
        assert "The answer is 42" in result
        assert doc_name in result


def test_retriever_entity_reranking():
    """Test that retrieval reranks results based on named entities"""
    query = "Who is Frodo Baggins?"
    
    # Set up mocks for all the external functions
    with patch('rag_agent.tools.retriever.encode') as mock_encode, \
         patch('rag_agent.tools.retriever.extract_entities') as mock_extract, \
         patch('rag_agent.tools.retriever.search_similar_chunks') as mock_search:
        
        # Configure the mocks
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_encode.return_value = [mock_embedding]
        
        # Return specific entities for this query
        mock_extract.return_value = {"Frodo": "PER", "Baggins": "PER"}
        
        # Configure search results with different entities
        mock_search.return_value = [
            {
                "doc_name": "doc1.txt",
                "chunk_text": "Text about Frodo and Sam",
                "named_entities": {"Frodo": "PER", "Sam": "PER"},
                "distance": 0.3
            },
            {
                "doc_name": "doc2.txt",
                "chunk_text": "Text about Gandalf and Aragorn",
                "named_entities": {"Gandalf": "PER", "Aragorn": "PER"},
                "distance": 0.2
            },
            {
                "doc_name": "doc3.txt",
                "chunk_text": "Text about Frodo Baggins",
                "named_entities": {"Frodo": "PER", "Baggins": "PER"},
                "distance": 0.4
            }
        ]
        
        # Create and call the tool
        tool = TextRetriever(max_results=2)
        result = tool.forward(query=query)
        
        # Verify the function calls
        mock_encode.assert_called_once_with([query])
        mock_extract.assert_called_once_with(query)
        mock_search.assert_called_once()
        
        # Verify the result contains the expected reranked results
        # The result with worse distance but better entity match should be included
        assert "Text about Frodo Baggins" in result
        assert "Text about Frodo and Sam" in result
        
        # The result with better distance but no entity match should be excluded
        assert "Text about Gandalf and Aragorn" not in result


def test_retriever_input_validation():
    """Test input validation in the retriever"""
    # Test with non-string query
    tool = TextRetriever()
    with pytest.raises(AssertionError):
        tool.forward(query=123)
    
    # Test with valid inputs
    with patch('rag_agent.tools.retriever.encode') as mock_encode, \
         patch('rag_agent.tools.retriever.extract_entities') as mock_extract, \
         patch('rag_agent.tools.retriever.search_similar_chunks') as mock_search:
        
        # Configure the mocks
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_encode.return_value = [mock_embedding]
        
        mock_extract.return_value = {}
        mock_search.return_value = []
        
        # These should not raise any exceptions
        tool.forward(query="Valid query")
        tool.forward(query="Valid query", doc_name="doc.txt")
