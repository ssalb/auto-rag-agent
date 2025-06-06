"""
Unit tests for the database initialization functions.
"""
from unittest.mock import patch, MagicMock
from rag_agent.db import get_connection, create_schema, init_db


def test_get_connection():
    """Test that get_connection returns a connection from DuckDBConnection"""
    with patch('rag_agent.db.connection.DuckDBConnection', autospec=True) as MockDuckDBConnection:
        # Set up the mock
        mock_conn = MagicMock()
        mock_instance = MagicMock()
        mock_instance.connect.return_value = mock_conn
        MockDuckDBConnection.return_value = mock_instance
        
        # Call the function - this actually imports the module again
        # So we need a deeper patch
        with patch('rag_agent.db.DuckDBConnection', MockDuckDBConnection):
            conn = get_connection()
        
        # Verify the connection was retrieved correctly
        assert MockDuckDBConnection.call_count >= 1
        assert mock_instance.connect.called
        assert conn is mock_conn


def test_create_schema():
    """Test that create_schema calls the model's create_table method"""
    with patch('rag_agent.db.models.DocumentModel.create_table_if_not_exists') as mock_create_table:
        # Set up the mock
        mock_conn = MagicMock()
        
        # Call the function
        result = create_schema(mock_conn)
        
        # Verify the model's create_table method was called
        mock_create_table.assert_called_once_with(mock_conn)
        
        # Verify the function returned True
        assert result is True


def test_init_db():
    """Test that init_db initializes the database correctly"""
    with patch('rag_agent.db.get_connection') as mock_get_connection, \
         patch('rag_agent.db.create_schema') as mock_create_schema:
        # Set up the mocks
        mock_conn = MagicMock()
        mock_get_connection.return_value = mock_conn
        
        # Call the function
        conn = init_db()
        
        # Verify the connection was retrieved
        assert mock_get_connection.called
        
        # Verify experimental persistence was enabled
        mock_conn.execute.assert_called_with("SET hnsw_enable_experimental_persistence = true")
        
        # Verify the schema was created
        assert mock_create_schema.called
        
        # Verify the connection was returned
        assert conn is mock_conn
