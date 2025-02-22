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
    # Test navigation links on home page
    response = client.get('/')
    assert response.status_code == 200
    
    # Test main navigation menu - using href patterns
    assert b'href="/home"' in response.data
    assert b'href="/word_embeddings"' in response.data
    assert b'href="/a2"' in response.data
    assert b'href="/a3"' in response.data
    assert b'href="/a4"' in response.data
    assert b'href="/coming_soon"' in response.data
    
    # Test navigation text
    assert b"That's what I LIKE!" in response.data
    assert b"Language Modelling" in response.data
    assert b"Make Your Own Machine Translation Language" in response.data
    assert b"Do you AGREE?" in response.data
    assert b"Optimization Human Preference" in response.data

    # Test A4 page specific content
    response = client.get('/a4')
    assert response.status_code == 200
    assert b'Natural Language Inference' in response.data
    assert b'Premise:' in response.data
    assert b'Hypothesis:' in response.data

def test_a4_page(client):
    """Test that the A4 NLI page loads successfully."""
    response = client.get('/a4')
    assert response.status_code == 200
    assert b'Natural Language Inference' in response.data
    assert b'Premise:' in response.data
    assert b'Hypothesis:' in response.data

def test_predict_nli_missing_data(client):
    """Test error handling when no data is provided for NLI prediction."""
    response = client.post('/predict_nli',
                         data=json.dumps({}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'premise and hypothesis are required' in data['error'].lower()

def test_predict_nli_missing_premise(client):
    """Test error handling when premise is missing."""
    response = client.post('/predict_nli',
                         data=json.dumps({'hypothesis': 'test'}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'premise' in data['error'].lower()

def test_predict_nli_missing_hypothesis(client):
    """Test error handling when hypothesis is missing."""
    response = client.post('/predict_nli',
                         data=json.dumps({'premise': 'test'}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'hypothesis' in data['error'].lower()

def test_predict_nli_empty_inputs(client):
    """Test error handling when inputs are empty strings."""
    response = client.post('/predict_nli',
                         data=json.dumps({'premise': '', 'hypothesis': ''}),
                         content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'must not be empty' in data['error'].lower()

def test_predict_nli_valid_input(client):
    """Test NLI prediction with valid input."""
    test_data = {
        'premise': 'A simple test sentence.',
        'hypothesis': 'Another simple sentence.'
    }
    response = client.post('/predict_nli',
                         data=json.dumps(test_data),
                         content_type='application/json')
    
    # If model is not loaded, we expect a 500 error
    if response.status_code == 500:
        data = json.loads(response.data)
        assert 'error' in data
        assert 'model not loaded' in data['error'].lower()
    else:
        # If model is loaded, verify the response format
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'predicted_label' in data
        assert 'probabilities' in data
        
        # Check probabilities format
        probs = data['probabilities']
        assert 'entailment' in probs
        assert 'contradiction' in probs
        assert 'neutral' in probs
        
        # Verify probabilities sum to approximately 1
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 0.01
        
        # Verify each probability is between 0 and 1
        for prob in probs.values():
            assert 0 <= prob <= 1
            
        # Verify label is one of the expected values
        assert data['predicted_label'] in ['entailment', 'contradiction', 'neutral']
