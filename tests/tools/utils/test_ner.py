"""
Unit tests for the named entity recognition utility.
"""
from unittest.mock import patch
from rag_agent.tools.utils.ner import extract_entities


@patch('rag_agent.tools.utils.ner.ner_pipeline')
def test_extract_entities_with_entities(mock_pipeline):
    """Test extracting entities from text with named entities"""
    text = "Frodo Baggins and Gandalf the Grey went to Mordor."
    
    # Configure the mock to return specific entities
    mock_pipeline.return_value = [
        {"word": "Frodo", "entity_group": "PER", "score": 0.99},
        {"word": "Gandalf", "entity_group": "PER", "score": 0.98},
        {"word": "Mordor", "entity_group": "LOC", "score": 0.97}
    ]
    
    # Call the extract_entities function
    result = extract_entities(text)
    
    # Verify the pipeline was called with the text
    mock_pipeline.assert_called_once_with(text)
    
    # Verify we got the expected entities
    assert result == {
        "Frodo": "PER",
        "Gandalf": "PER",
        "Mordor": "LOC"
    }


@patch('rag_agent.tools.utils.ner.ner_pipeline')
def test_extract_entities_without_entities(mock_pipeline):
    """Test extracting entities from text without named entities"""
    text = "The quick brown fox jumps over the lazy dog."
    
    # Configure the mock to return no entities
    mock_pipeline.return_value = []
    
    # Call the extract_entities function
    result = extract_entities(text)
    
    # Verify the pipeline was called with the text
    mock_pipeline.assert_called_once_with(text)
    
    # Verify we got an empty dictionary
    assert result == {}


@patch('rag_agent.tools.utils.ner.ner_pipeline')
def test_extract_entities_with_duplicate_entities(mock_pipeline):
    """Test that duplicate entities are handled correctly"""
    text = "Frodo met Frodo in the mountains."
    
    # Configure the mock to return duplicate entities
    mock_pipeline.return_value = [
        {"word": "Frodo", "entity_group": "PER", "score": 0.99},
        {"word": "Frodo", "entity_group": "PER", "score": 0.98},
        {"word": "mountains", "entity_group": "LOC", "score": 0.97}
    ]
    
    # Call the extract_entities function
    result = extract_entities(text)
    
    # Verify we got each entity only once
    assert result == {
        "Frodo": "PER",
        "mountains": "LOC"
    }


@patch('rag_agent.tools.utils.ner.ner_pipeline')
def test_extract_entities_with_different_entity_types(mock_pipeline):
    """Test handling entities of different types"""
    text = "Microsoft announced a new product on January 15th in New York."
    
    # Configure the mock to return different entity types
    mock_pipeline.return_value = [
        {"word": "Microsoft", "entity_group": "ORG", "score": 0.99},
        {"word": "January 15th", "entity_group": "DATE", "score": 0.98},
        {"word": "New York", "entity_group": "LOC", "score": 0.97}
    ]
    
    # Call the extract_entities function
    result = extract_entities(text)
    
    # Verify we got all entity types
    assert result == {
        "Microsoft": "ORG",
        "January 15th": "DATE",
        "New York": "LOC"
    }
