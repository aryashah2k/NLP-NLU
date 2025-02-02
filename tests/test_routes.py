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

def test_a2_page(client):
    """Test that the A2 language modeling page loads successfully."""
    response = client.get('/a2')
    assert response.status_code == 200
    assert b'Language Modelling' in response.data
    assert b'Generate Text' in response.data

def test_generate_text_missing_data(client):
    """Test error handling when no data is provided for text generation."""
    response = client.post('/generate_text',
                         data=json.dumps({}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_generate_text_missing_prompt(client):
    """Test error handling when prompt is missing."""
    response = client.post('/generate_text',
                         data=json.dumps({'temperature': 0.8, 'max_length': 50}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_generate_text_invalid_temperature(client):
    """Test error handling when temperature is invalid."""
    response = client.post('/generate_text',
                         data=json.dumps({'prompt': 'test', 'temperature': -1, 'max_length': 50}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_generate_text_invalid_max_length(client):
    """Test error handling when max_length is invalid."""
    response = client.post('/generate_text',
                         data=json.dumps({'prompt': 'test', 'temperature': 0.8, 'max_length': 0}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_a3_coming_soon_page(client):
    """Test that the A4 coming soon page loads successfully."""
    response = client.get('/coming_soon')
    assert response.status_code == 200
    assert b'A4: Do You AGREE?' in response.data
    assert b'Coming Soon!' in response.data

def test_a3_page(client):
    """Test that the A3 machine translation page loads successfully."""
    response = client.get('/a3')
    assert response.status_code == 200
    assert b'Machine Translation' in response.data

def test_translate_missing_data(client):
    """Test error handling when no data is provided for translation."""
    response = client.post('/translate',
                         data=json.dumps({}),
                         content_type='application/json')
    assert response.status_code == 200  # The route returns 200 even for errors
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'Invalid request data'

def test_translate_missing_text(client):
    """Test error handling when text is missing."""
    response = client.post('/translate',
                         data=json.dumps({'model': 'transformer'}),
                         content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'Invalid request data'

def test_translate_missing_model(client):
    """Test error handling when model is missing."""
    response = client.post('/translate',
                         data=json.dumps({'text': 'Hello world'}),
                         content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'Invalid request data'

def test_navigation_links(client):
    """Test that all navigation links are present in the pages."""
    pages = ['/', '/word_embeddings', '/a2', '/coming_soon']
    for page in pages:
        response = client.get(page)
        assert response.status_code == 200
        assert b'Home' in response.data
        assert b'A1: That\'s what I LIKE!' in response.data
        assert b'A2: Language Modelling' in response.data
        assert b'A3: Make Your Own Machine Translation Language' in response.data
