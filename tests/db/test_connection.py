"""
Unit tests for the DuckDBConnection class.
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from rag_agent.db.connection import DuckDBConnection


def test_singleton_pattern():
    """Test that DuckDBConnection follows the singleton pattern"""
    # Get two instances
    conn1 = DuckDBConnection()
    conn2 = DuckDBConnection()
    
    # They should be the same object
    assert conn1 is conn2


@patch('rag_agent.config.DuckDBConfig')
@patch('os.makedirs')
@patch('duckdb.connect')
def test_connect_creates_directory(mock_connect, mock_makedirs, mock_config):
    """Test that connect creates the directory for the database"""
    # Configure the mocks
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    
    path_value = "/test/path/db.duckdb"
    mock_config.DUCKDB_PATH = path_value
    
    # Reset any existing connection
    DuckDBConnection._instance = None
    
    # Get a connection
    db = DuckDBConnection()
    db.conn = None  # Ensure connection will be created
    
    # This is the key part - patch the DuckDBConfig.DUCKDB_PATH lookup in the connect method
    with patch('rag_agent.db.connection.DuckDBConfig.DUCKDB_PATH', path_value):
        conn = db.connect()
    
    # Verify directory was created
    mock_makedirs.assert_called_once()
    assert "exist_ok=True" in str(mock_makedirs.call_args)
    
    # Verify connect was called with the path
    mock_connect.assert_called_once_with(path_value)
    
    # Verify extensions were installed and loaded
    assert any("INSTALL vss" in str(call) for call in mock_conn.execute.call_args_list)
    assert any("LOAD vss" in str(call) for call in mock_conn.execute.call_args_list)


@patch('rag_agent.config.DuckDBConfig')
@patch('duckdb.connect')
def test_connect_registers_embedding_type(mock_connect, mock_config):
    """Test that connect registers the embedding type"""
    # Configure the mocks
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    
    # Important: Set the attribute directly on the type, not the object itself
    type(mock_config).EMBEDDING_DIM = 384
    
    # Reset any existing connection
    DuckDBConnection._instance = None
    
    # Get a connection
    db = DuckDBConnection()
    db.conn = None  # Ensure connection will be created
    conn = db.connect()
    
    # Verify embedding type was created with a direct check
    mock_conn.execute.assert_any_call("CREATE TYPE embedding AS FLOAT[384]")


@patch('duckdb.connect')
def test_connect_handles_existing_type(mock_connect):
    """Test that connect handles the case where the embedding type already exists"""
    # Configure the mock to raise an exception with "already exists" message on the third call
    mock_conn = MagicMock()
    
    # Set up a side effect sequence for the execute method
    def side_effect(query, *args, **kwargs):
        if "CREATE TYPE embedding" in query:
            raise Exception("Type already exists")
        return None
    
    mock_conn.execute.side_effect = side_effect
    mock_connect.return_value = mock_conn
    
    # Reset any existing connection
    DuckDBConnection._instance = None
    
    # Get a connection - this should not raise an exception
    db = DuckDBConnection()
    db.conn = None  # Ensure connection will be created
    conn = db.connect()
    
    # Verify we still got a connection back (not a mock in this case)
    assert conn is db.conn


@patch('duckdb.connect')
def test_connect_handles_other_exceptions(mock_connect):
    """Test that connect re-raises other exceptions"""
    # Configure the mock to raise an exception with a different message
    mock_conn = MagicMock()
    
    # Set up a side effect for the execute method that raises an exception
    def side_effect(query, *args, **kwargs):
        if "CREATE TYPE embedding" in query:
            raise Exception("Some other error")
        return None
    
    mock_conn.execute.side_effect = side_effect
    mock_connect.return_value = mock_conn
    
    # Reset any existing connection
    DuckDBConnection._instance = None
    
    # Get a connection - this should raise the exception
    db = DuckDBConnection()
    db.conn = None  # Ensure connection will be created
    
    with pytest.raises(Exception, match="Some other error"):
        conn = db.connect()


def test_close():
    """Test that close closes the connection"""
    # Create a mock connection
    mock_conn = MagicMock()
    
    # Create a DuckDBConnection with the mock
    db = DuckDBConnection()
    db.conn = mock_conn
    
    # Close the connection
    db.close()
    
    # Verify close was called
    mock_conn.close.assert_called_once()
    
    # Verify conn was set to None
    assert db.conn is None
