import pytest
import torch
import os
from app.assignments.a1_word_embeddings.utils import load_model, find_similar_words

@pytest.fixture
def model_path():
    """Get path to a test model."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base_dir, 'app', 'assignments', 'a1_word_embeddings', 
                       'models', 'w2_e100_skipgram.pt')

def test_load_model_file_not_found():
    """Test that load_model raises FileNotFoundError for non-existent files."""
    with pytest.raises(FileNotFoundError):
        load_model('nonexistent_model.pt')

def test_load_model_success(model_path):
    """Test successful model loading."""
    if os.path.exists(model_path):
        model, word2idx, idx2word = load_model(model_path)
        assert model is not None
        assert isinstance(word2idx, dict)
        assert isinstance(idx2word, dict)
        assert len(word2idx) == len(idx2word)
        assert model.embedding_center.weight.shape[0] == len(word2idx)

def test_find_similar_words_invalid_word(model_path):
    """Test find_similar_words with invalid word."""
    if os.path.exists(model_path):
        model, word2idx, idx2word = load_model(model_path)
        results = find_similar_words(model, word2idx, idx2word, 
                                   'thisisaninvalidword', k=10)
        assert len(results) == 0

def test_find_similar_words_valid_word(model_path):
    """Test find_similar_words with valid word."""
    if os.path.exists(model_path):
        model, word2idx, idx2word = load_model(model_path)
        # Find a word that exists in the vocabulary
        test_word = list(word2idx.keys())[0]
        results = find_similar_words(model, word2idx, idx2word, test_word, k=10)
        assert len(results) <= 10
        assert all(isinstance(word, str) and isinstance(sim, float) 
                  for word, sim in results)
        # Check that similarities are in descending order
        similarities = [sim for _, sim in results]
        assert similarities == sorted(similarities, reverse=True)

def test_find_similar_words_k_parameter(model_path):
    """Test that find_similar_words respects the k parameter."""
    if os.path.exists(model_path):
        model, word2idx, idx2word = load_model(model_path)
        test_word = list(word2idx.keys())[0]
        k_values = [1, 5, 10]
        for k in k_values:
            results = find_similar_words(model, word2idx, idx2word, test_word, k=k)
            assert len(results) <= k

def test_embedding_dimensions(model_path):
    """Test that embeddings have the expected dimensions."""
    if os.path.exists(model_path):
        model, word2idx, idx2word = load_model(model_path)
        # Get embedding dimension from model
        embedding_dim = model.embedding_center.weight.shape[1]
        # Test a few random words
        test_words = list(word2idx.keys())[:5]
        for word in test_words:
            word_idx = word2idx[word]
            embedding = model.embedding_center(torch.LongTensor([word_idx]))
            assert embedding.shape == (1, embedding_dim)

def test_model_eval_mode(model_path):
    """Test that loaded model is in evaluation mode."""
    if os.path.exists(model_path):
        model, _, _ = load_model(model_path)
        assert not model.training
