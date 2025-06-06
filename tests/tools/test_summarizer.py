"""
Unit tests for the SummarizerTool class.
"""
import pytest
from unittest.mock import MagicMock
from rag_agent.tools.summarizer import SummarizerTool


def test_summarizer_initialization(mock_model):
    """Test that the summarizer tool initializes correctly"""
    tool = SummarizerTool(model=mock_model)
    
    assert tool.name == "summarizer_tool"
    assert "summarize" in tool.description.lower()
    assert tool.model == mock_model


def test_summarizer_forward_without_query(mock_model, sample_text):
    """Test summarizer without a query parameter"""
    tool = SummarizerTool(model=mock_model)
    
    result = tool.forward(text=sample_text)
    
    # Check that the model was called with the right prompt
    expected_prompt = f"Summarize the following text:\n\n{sample_text}\n"
    mock_model.assert_called_once()
    
    # Get the arguments the mock was called with
    call_args = mock_model.call_args[1]
    assert "messages" in call_args
    assert len(call_args["messages"]) == 1
    assert call_args["messages"][0]["role"] == "user"
    assert call_args["messages"][0]["content"][0]["text"] == expected_prompt


def test_summarizer_forward_with_query(mock_model, sample_text):
    """Test summarizer with a query parameter"""
    tool = SummarizerTool(model=mock_model)
    query = "What advice does Gandalf give?"
    
    result = tool.forward(text=sample_text, query=query)
    
    # Check that the model was called with the right prompt including the query
    expected_prompt = f"Summarize the following text:\n\n{sample_text}\n\n\nPlease focus on the following question: {query}\n"
    mock_model.assert_called_once()
    
    # Get the arguments the mock was called with
    call_args = mock_model.call_args[1]
    assert call_args["messages"][0]["content"][0]["text"] == expected_prompt


def test_summarizer_input_validation(mock_model):
    """Test that the summarizer validates inputs correctly"""
    tool = SummarizerTool(model=mock_model)
    
    # Test with invalid text type
    with pytest.raises(TypeError, match="text to summarize must be a string"):
        tool.forward(text=123)
    
    # Test with invalid query type
    with pytest.raises(TypeError, match="query must be a string or None"):
        tool.forward(text="Valid text", query=123)


def test_summarizer_response_processing(mock_model):
    """Test that the summarizer processes model responses correctly"""
    # Configure the mock to return a response with whitespace
    mock_response = MagicMock()
    mock_response.content = "  This is a summary with whitespace   "
    mock_model.return_value = mock_response
    
    tool = SummarizerTool(model=mock_model)
    result = tool.forward(text="Test text")
    
    # Check that whitespace was stripped
    assert result == "This is a summary with whitespace"
