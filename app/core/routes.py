from flask import render_template, jsonify, request
import os
import logging
import traceback
from app.assignments.a1_word_embeddings.utils import load_model, find_similar_words
import torch
from app.assignments.a2_language_modelling.utils import LSTMLanguageModel, Vocabulary, generate_text

# Global dictionary to store loaded models
models = {
    'a1': {},
    'a2': None
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_models():
    """Load all available models"""
    # A1 Models
    model_files = {
        'skipgram': 'w2_e100_skipgram.pt',
        'skipgram_neg': 'w2_e100_skipgram_neg_skipgram_neg.pt',
        'glove': 'w2_e100_glove_glove.pt'
    }
    
    # Get absolute path to models directories
    a1_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            'assignments', 'a1_word_embeddings', 'models')
    a2_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            'assignments', 'a2_language_modelling', 'models')
    
    logger.info(f"Current directory: {os.path.dirname(__file__)}")
    logger.info(f"Loading A1 models from directory: {a1_models_dir}")
    logger.info(f"Directory exists: {os.path.exists(a1_models_dir)}")
    
    if not os.path.exists(a1_models_dir):
        logger.error(f"A1 models directory not found: {a1_models_dir}")
        return
        
    # List all files in A1 models directory
    try:
        model_files_in_dir = os.listdir(a1_models_dir)
        logger.info(f"Files in A1 models directory: {model_files_in_dir}")
    except Exception as e:
        logger.error(f"Error listing A1 models directory: {e}")
        return
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(a1_models_dir, filename)
        logger.info(f"Attempting to load {model_name} from {model_path}")
        logger.info(f"File exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading {model_name}...")
                model, word2idx, idx2word = load_model(model_path)
                models['a1'][model_name] = {
                    'model': model,
                    'word2idx': word2idx,
                    'idx2word': idx2word
                }
                logger.info(f"Successfully loaded {model_name} with vocabulary size: {len(word2idx)}")
            except Exception as e:
                logger.error(f"Error loading {model_name}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            logger.error(f"A1 model file not found: {model_path}")
    
    logger.info(f"Loaded A1 models: {list(models['a1'].keys())}")
    logger.info(f"Total A1 models loaded: {len(models['a1'])}")
    
    # Load A2 model
    try:
        logger.info("Starting A2 model loading process...")
        
        # Load Shakespeare text and build vocabulary
        shakespeare_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      'data', 't8.shakespeare.txt')
        logger.info(f"Shakespeare file path: {shakespeare_file}")
        logger.info(f"File exists: {os.path.exists(shakespeare_file)}")
        
        with open(shakespeare_file, 'r', encoding='utf-8') as f:
            shakespeare_text = f.read()
            logger.info(f"Successfully read Shakespeare text, length: {len(shakespeare_text)} characters")
        
        # Initialize vocabulary with the text
        vocab = Vocabulary()
        vocab.build_vocab(shakespeare_text)
        logger.info(f"Built vocabulary with size: {len(vocab)}")
        
        # Initialize model with correct vocabulary size and dimensions
        vocab_size = len(vocab)
        logger.info(f"Initializing model with vocabulary size: {vocab_size}")
        model = LSTMLanguageModel(
            vocab_size=vocab_size,
            emb_dim=400,
            hid_dim=1024,
            num_layers=2,
            dropout_rate=0.5
        )
        logger.info("Model initialized successfully")
        
        # Load the trained model
        model_path = os.path.join(a2_models_dir, 'shakespeare_model.pt')
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model file exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            logger.info("Loading model checkpoint...")
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            logger.info(f"Checkpoint keys: {checkpoint.keys()}")
            
            logger.info("Loading state dict...")
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to evaluation mode
            
            models['a2'] = {'model': model, 'vocab': vocab}
            logger.info(f"Successfully loaded A2 language model with vocabulary size: {len(vocab)}")
        else:
            logger.error(f"A2 model file not found at {model_path}")
    except Exception as e:
        logger.error(f"Error loading A2 model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
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
            logger.info(f"Available models: {list(models['a1'].keys())}")
            
            if not word or not model_name:
                logger.error("Word or model name missing")
                return jsonify({'error': 'Word and model are required'}), 400
            
            if model_name not in models['a1']:
                logger.error(f"Model {model_name} not found. Available models: {list(models['a1'].keys())}")
                return jsonify({'error': f'Model {model_name} not found. Available models: {list(models["a1"].keys())}'}), 404
                
            model_data = models['a1'][model_name]
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

    @app.route('/a2')
    def a2():
        return render_template('a2.html')
    
    @app.route('/a3')
    def a3():
        """Route for Assignment 3: Make Your Own Machine Translation Model."""
        return render_template('a3.html')

    @app.route('/generate_text', methods=['POST'])
    def generate():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            # Validate required fields
            if 'prompt' not in data:
                return jsonify({'error': 'Prompt is required'}), 400
            
            prompt = data.get('prompt', '').strip()
            if not prompt:
                return jsonify({'error': 'Prompt cannot be empty'}), 400
                
            try:
                temperature = float(data.get('temperature', 0.8))
                if not 0 < temperature <= 5:
                    return jsonify({'error': 'Temperature must be between 0 and 5'}), 400
            except ValueError:
                return jsonify({'error': 'Invalid temperature value'}), 400
                
            try:
                max_length = int(data.get('max_length', 50))
                if max_length < 1:
                    return jsonify({'error': 'Max length must be greater than 0'}), 400
            except ValueError:
                return jsonify({'error': 'Invalid max_length value'}), 400
            
            logger.info(f"Received generation request - Prompt: {prompt}, Temp: {temperature}, Max Length: {max_length}")
            
            # Check if models['a2'] exists and has the required components
            if not models.get('a2') or 'model' not in models['a2'] or 'vocab' not in models['a2']:
                logger.error(f"Model not properly loaded. models['a2'] = {models.get('a2')}")
                return jsonify({'error': 'Model not loaded properly. Please try restarting the server.'}), 500
                
            model = models['a2']['model']
            vocab = models['a2']['vocab']
            
            logger.info(f"Generating text with vocabulary size: {len(vocab)}")
            
            generated_text = generate_text(
                prompt=prompt,
                model=model,
                vocab=vocab,
                max_len=max_length,
                temperature=temperature
            )
            
            if not generated_text:
                logger.error("Generated text is empty")
                return jsonify({'error': 'Failed to generate text'}), 500
                
            logger.info(f"Generated text: {generated_text}")
            return jsonify({'generated_text': generated_text})
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500

    @app.route('/translate', methods=['POST'])
    def translate():
        """Handle translation requests."""
        try:
            data = request.get_json()
            if not data or 'text' not in data or 'model' not in data:
                return jsonify({'error': 'Invalid request data'})
            
            from app.assignments.a3_make_your_own_mt_model.translation_handler import handle_translation
            return handle_translation(data['text'], data['model'])
            
        except Exception as e:
            print(f"Translation route error: {str(e)}")
            return jsonify({'error': 'Translation request failed'})

    # Load models when registering routes
    load_models()
