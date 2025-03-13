import pytest
import json
from unittest.mock import patch, MagicMock

def test_a6_page(client):
    """Test that the A6 page loads successfully."""
    # Mock the initialize_rag_system and analysis functions
    with patch('app.core.routes.initialize_rag_system') as mock_init, \
         patch('app.core.routes.analyze_retriever_model') as mock_retriever, \
         patch('app.core.routes.analyze_generator_model') as mock_generator:
        
        # Configure mocks
        mock_rag = MagicMock()
        mock_init.return_value = mock_rag
        mock_retriever.return_value = {'performance': 'good'}
        mock_generator.return_value = {'performance': 'good'}
        
        # Test the route
        response = client.get('/a6')
        
        # Verify response
        assert response.status_code == 200
        assert b'Lets Talk Yourselves' in response.data
        assert b'Chat' in response.data
        assert b'Model Analysis' in response.data

def test_chat_missing_data(client):
    """Test error handling when no data is provided for chat."""
    response = client.post('/chat', 
                         data=json.dumps({}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_chat_empty_message(client):
    """Test error handling when message is empty."""
    response = client.post('/chat', 
                         data=json.dumps({'message': ''}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'No message provided' in data['error']

def test_chat_rag_not_initialized(client):
    """Test error handling when RAG system is not initialized."""
    # Ensure the RAG system is not initialized
    with patch('app.core.routes.models', {'a6': None}):
        response = client.post('/chat', 
                             data=json.dumps({'message': 'Hello'}),
                             content_type='application/json')
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'RAG system not initialized' in data['error']

def test_chat_valid_message(client):
    """Test chat with a valid message."""
    # Mock the RAG system and its query method
    mock_rag = MagicMock()
    mock_rag.query.return_value = {
        'question': 'Hello',
        'answer': 'Hi there!',
        'sources': [{'content': 'Greeting', 'source': 'test.txt'}]
    }
    
    with patch('app.core.routes.models', {'a6': mock_rag}), \
         patch('app.core.routes.qa_history', []), \
         patch('app.core.routes.open', MagicMock()), \
         patch('app.core.routes.json.dump', MagicMock()):
        
        response = client.post('/chat', 
                             data=json.dumps({'message': 'Hello'}),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'answer' in data
        assert data['answer'] == 'Hi there!'
        assert 'sources' in data
        assert len(data['sources']) == 1

def test_qa_history(client):
    """Test retrieving QA history."""
    # Mock the QA history
    test_history = [
        {'question': 'Hello', 'answer': 'Hi there!'},
        {'question': 'How are you?', 'answer': 'I am fine, thank you!'}
    ]
    
    with patch('app.core.routes.qa_history', test_history):
        response = client.get('/qa_history')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 2
        assert data[0]['question'] == 'Hello'
        assert data[1]['answer'] == 'I am fine, thank you!'

def test_analyze_missing_data(client):
    """Test error handling when no data is provided for analysis."""
    response = client.post('/analyze', 
                         data=json.dumps({}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_analyze_empty_question(client):
    """Test error handling when question is empty."""
    response = client.post('/analyze', 
                         data=json.dumps({'question': ''}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'No question provided' in data['error']

def test_analyze_rag_not_initialized(client):
    """Test error handling when RAG system is not initialized for analysis."""
    # Ensure the RAG system is not initialized
    with patch('app.core.routes.models', {'a6': None}):
        response = client.post('/analyze', 
                             data=json.dumps({'question': 'What is RAG?'}),
                             content_type='application/json')
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'RAG system not initialized' in data['error']

def test_analyze_valid_question(client):
    """Test analysis with a valid question."""
    # Mock the RAG system, vector store, and query method
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = [
        {'content': 'RAG stands for Retrieval-Augmented Generation', 'source': 'test.txt', 'score': 0.8}
    ]
    
    mock_rag = MagicMock()
    mock_rag.vector_store = mock_vector_store
    mock_rag.query.return_value = {
        'question': 'What is RAG?',
        'answer': 'RAG stands for Retrieval-Augmented Generation.',
        'sources': [{'content': 'RAG definition', 'source': 'test.txt'}]
    }
    
    with patch('app.core.routes.models', {'a6': mock_rag}):
        response = client.post('/analyze', 
                             data=json.dumps({'question': 'What is RAG?'}),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'retriever_results' in data
        assert 'generation_result' in data
        assert len(data['retriever_results']) == 1
        assert 'answer' in data['generation_result']
        assert 'sources' in data['generation_result']
