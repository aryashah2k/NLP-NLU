import pytest
import json

def test_home_page(client):
    """Test that the home page loads successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Arya Shah' in response.data

def test_word_embeddings_page(client):
    """Test that the word embeddings page loads successfully."""
    response = client.get('/word_embeddings')
    assert response.status_code == 200
    assert b'Search Similar Context' in response.data

def test_coming_soon_page(client):
    """Test that the coming soon page loads successfully."""
    response = client.get('/coming_soon')
    assert response.status_code == 200
    assert b'Coming Soon!' in response.data

def test_find_similar_words_missing_data(client):
    """Test error handling when no data is provided."""
    response = client.post('/find_similar_words', 
                         data=json.dumps({}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_find_similar_words_missing_word(client):
    """Test error handling when word is missing."""
    response = client.post('/find_similar_words', 
                         data=json.dumps({'model': 'skipgram'}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_find_similar_words_missing_model(client):
    """Test error handling when model is missing."""
    response = client.post('/find_similar_words', 
                         data=json.dumps({'word': 'test'}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_find_similar_words_invalid_model(client):
    """Test error handling when an invalid model is specified."""
    response = client.post('/find_similar_words', 
                         data=json.dumps({'word': 'test', 'model': 'invalid_model'}),
                         content_type='application/json')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert 'error' in data
