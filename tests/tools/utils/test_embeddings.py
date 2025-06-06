"""
Unit tests for the embeddings utility.
"""
import pytest
import numpy as np
from unittest.mock import patch
from rag_agent.tools.utils.embeddings import encode


@patch('rag_agent.tools.utils.embeddings.emb_model')
def test_encode_single_text(mock_emb_model):
    """Test encoding a single text string"""
    text = "This is a test text"
    
    # Configure the mock to return a predictable embedding
    mock_emb_model.encode.return_value = np.random.rand(1, 384)
    
    # Call the encode function
    result = encode([text])
    
    # Verify the transformer was called
    mock_emb_model.encode.assert_called_once_with([text], truncate=True)
    
    # Verify we got a result with the right shape
    assert len(result) == 1
    assert result[0].shape == (384,)


@patch('rag_agent.tools.utils.embeddings.emb_model')
def test_encode_multiple_texts(mock_emb_model):
    """Test encoding multiple text strings"""
    texts = ["Text 1", "Text 2", "Text 3"]
    
    # Configure the mock to return predictable embeddings
    mock_emb_model.encode.return_value = np.random.rand(3, 384)
    
    # Call the encode function
    result = encode(texts)
    
    # Verify the transformer was called with all texts
    mock_emb_model.encode.assert_called_once_with(texts, truncate=True)
    
    # Verify we got results for each text
    assert len(result) == 3
    for embedding in result:
        assert embedding.shape == (384,)


@pytest.mark.parametrize("input_text", [
    "Short text",
    "Medium length text with some content",
    "A very long text " + "with lots of words " * 20,
])
@patch('rag_agent.tools.utils.embeddings.emb_model')
def test_encode_different_text_lengths(mock_emb_model, input_text):
    """Test encoding texts of different lengths"""
    # Configure the mock to return a predictable embedding
    mock_emb_model.encode.return_value = np.random.rand(1, 384)
    
    # Call the encode function
    result = encode([input_text])
    
    # Verify truncate=True was used for all text lengths
    mock_emb_model.encode.assert_called_once_with([input_text], truncate=True)
    
    # All embeddings should have the same size regardless of input length
    assert result[0].shape == (384,)
