from flask import render_template, jsonify, request
import os
import logging
import traceback
from app.assignments.a1_word_embeddings.utils import load_model, find_similar_words

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global dictionary to store loaded models
models = {}

def load_models():
    """Load all available models"""
    model_files = {
        'skipgram': 'w2_e100_skipgram.pt',
        'skipgram_neg': 'w2_e100_skipgram_neg_skipgram_neg.pt',
        'glove': 'w2_e100_glove_glove.pt'
    }
    
    # Get absolute path to models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            'assignments', 'a1_word_embeddings', 'models')
    
    logger.info(f"Current directory: {os.path.dirname(__file__)}")
    logger.info(f"Loading models from directory: {models_dir}")
    logger.info(f"Directory exists: {os.path.exists(models_dir)}")
    
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return
        
    # List all files in models directory
    try:
        model_files_in_dir = os.listdir(models_dir)
        logger.info(f"Files in models directory: {model_files_in_dir}")
    except Exception as e:
        logger.error(f"Error listing models directory: {e}")
        return
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        logger.info(f"Attempting to load {model_name} from {model_path}")
        logger.info(f"File exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading {model_name}...")
                model, word2idx, idx2word = load_model(model_path)
                models[model_name] = {
                    'model': model,
                    'word2idx': word2idx,
                    'idx2word': idx2word
                }
                logger.info(f"Successfully loaded {model_name} with vocabulary size: {len(word2idx)}")
            except Exception as e:
                logger.error(f"Error loading {model_name}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            logger.error(f"Model file not found: {model_path}")
    
    logger.info(f"Loaded models: {list(models.keys())}")
    logger.info(f"Total models loaded: {len(models)}")

def register_routes(app):
    @app.route('/')
    @app.route('/home')
    def home():
        return render_template('index.html')

    @app.route('/word_embeddings')
    def word_embeddings():
        return render_template('word_embeddings.html')

    @app.route('/coming_soon')
    def coming_soon():
        return render_template('coming_soon.html')

    @app.route('/find_similar_words', methods=['POST'])
    def get_similar_words():
        try:
            data = request.get_json()
            if not data:
                logger.error("No data provided in request")
                return jsonify({'error': 'No data provided'}), 400
                
            word = data.get('word', '').lower().strip()
            model_name = data.get('model', '')
            
            logger.info(f"Received request for word: {word}, model: {model_name}")
            logger.info(f"Available models: {list(models.keys())}")
            
            if not word or not model_name:
                logger.error("Word or model name missing")
                return jsonify({'error': 'Word and model are required'}), 400
            
            if model_name not in models:
                logger.error(f"Model {model_name} not found. Available models: {list(models.keys())}")
                return jsonify({'error': f'Model {model_name} not found. Available models: {list(models.keys())}'}), 404
                
            model_data = models[model_name]
            if word not in model_data['word2idx']:
                logger.error(f"Word '{word}' not found in vocabulary for model {model_name}")
                return jsonify({'error': f'Word "{word}" not found in vocabulary'}), 404
            
            logger.info(f"Finding similar words for '{word}' using {model_name}")
            similar_words = find_similar_words(
                model_data['model'],
                model_data['word2idx'],
                model_data['idx2word'],
                word,
                k=10
            )
            
            logger.info(f"Found {len(similar_words)} similar words")
            return jsonify({
                'similar_words': [
                    {'word': w, 'similarity': float(s)}
                    for w, s in similar_words
                ]
            })
            
        except Exception as e:
            logger.error(f"Error in get_similar_words: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500

    # Load models when registering routes
    load_models()
