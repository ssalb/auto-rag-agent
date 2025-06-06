"""
Root level pytest fixtures for testing RAG agent components.
"""
import pytest
from unittest.mock import MagicMock
import numpy as np
from smolagents import Model

@pytest.fixture
def mock_model():
    """
    Create a mock LLM model that can be used across all tests.
    This fixture mocks the Model class from smolagents.
    """
    mock = MagicMock(spec=Model)
    # Configure the mock to return a predictable response when called
    mock_response = MagicMock()
    mock_response.content = "This is a mock summary response"
    mock.return_value = mock_response
    return mock

@pytest.fixture
def sample_text():
    """Sample text that can be used across tests"""
    return (
        "'I wish it need not have happened in my time,' said Frodo. 'So do I,' said Gandalf, "
        "'and so do all who live to see such times. But that is not for them to decide. "
        "All we have to decide is what to do with the time that is given us.'"
    )

@pytest.fixture
def sample_embedding():
    """
    Create a sample embedding vector with the correct dimensions
    matching the model's embedding size (384 per config)
    """
    # Create a deterministic embedding for testing
    # Size matches the EMBEDDING_DIM in config
    return np.random.rand(384).tolist()
