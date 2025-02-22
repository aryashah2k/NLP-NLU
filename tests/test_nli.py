import pytest
import torch
import os
from app.assignments.a4_do_you_agree.inference import NLIPredictor
from app.assignments.a4_do_you_agree.sentence_bert import SentenceBERT
from app.assignments.a4_do_you_agree.bert_scratch import BertConfig

def get_test_model_dir():
    """Get the path to the test model directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(current_dir), 'app', 'assignments', 'a4_do_you_agree', 'sbert_model')

@pytest.fixture
def nli_predictor():
    """Fixture to create an NLI predictor instance."""
    model_dir = get_test_model_dir()
    if not os.path.exists(model_dir):
        pytest.skip("Model directory not found, skipping NLI tests")
    try:
        return NLIPredictor(model_dir=model_dir)
    except Exception as e:
        pytest.skip(f"Failed to load NLI model: {str(e)}")

def test_nli_predictor_initialization(nli_predictor):
    """Test that the NLI predictor initializes correctly."""
    assert nli_predictor is not None
    assert nli_predictor.model is not None
    assert nli_predictor.tokenizer is not None
    assert hasattr(nli_predictor, 'predict')

def test_nli_tokenizer_vocabulary(nli_predictor):
    """Test that the tokenizer has the expected vocabulary size."""
    assert len(nli_predictor.tokenizer.vocab) == 30522  # Standard BERT vocab size

def test_nli_model_architecture(nli_predictor):
    """Test the architecture of the loaded model."""
    model = nli_predictor.model
    assert isinstance(model, SentenceBERT)
    assert isinstance(model.bert.config, BertConfig)
    
    # Test model configuration
    config = model.bert.config
    assert config.hidden_size == 64
    assert config.num_hidden_layers == 2
    assert config.num_attention_heads == 4
    assert config.intermediate_size == 256

def test_nli_prediction_format(nli_predictor):
    """Test that predictions have the correct format without checking specific labels."""
    premise = "A simple test sentence."
    hypothesis = "Another simple sentence."
    
    result = nli_predictor.predict(premise, hypothesis)
    
    # Check basic response format
    assert isinstance(result, dict)
    assert 'label' in result
    assert 'probabilities' in result
    
    # Check probability format
    probs = result['probabilities']
    assert isinstance(probs, dict)
    assert all(k in probs for k in ['entailment', 'contradiction', 'neutral'])
    
    # Check probability values
    assert all(isinstance(p, float) for p in probs.values())
    assert all(0 <= p <= 1 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-6  # Sum should be approximately 1
    
    # Check label is valid
    assert result['label'] in ['entailment', 'contradiction', 'neutral']
