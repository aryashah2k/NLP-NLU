from flask import render_template, jsonify, request
import os
import logging
import traceback
from app.assignments.a1_word_embeddings.utils import load_model, find_similar_words
import torch
from app.assignments.a2_language_modelling.utils import LSTMLanguageModel, Vocabulary, generate_text
from app.assignments.a4_do_you_agree.inference import NLIPredictor
from app.assignments.a5_optimization_human_preference.inference import get_predictor
from app.assignments.a6_talk_with_yourselves.simple_rag import initialize_rag_system
from app.assignments.a6_talk_with_yourselves.model_analysis import analyze_retriever_model, analyze_generator_model
from app.assignments.a7_distillation_get_smaller_get_faster.toxic_classifier import ToxicClassifier
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Global dictionary to store loaded models
models = {
    'a1': {},
    'a2': None,
    'a4': None,
    'a5': None,
    'a6': None,
    'a7': None
}

# Store conversation history for A6
qa_history = []

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
    
    # Load A4 NLI model
    try:
        a4_model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                'assignments', 'a4_do_you_agree', 'sbert_model')
        if os.path.exists(a4_model_dir):
            logger.info(f"Loading A4 NLI model from {a4_model_dir}")
            try:
                models['a4'] = NLIPredictor(model_dir=a4_model_dir)
                logger.info("Successfully loaded A4 NLI model")
            except Exception as e:
                logger.error(f"Error initializing NLI model: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            logger.error(f"A4 model directory not found: {a4_model_dir}")
    except Exception as e:
        logger.error(f"Error loading A4 model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Load A5 DPO model
    try:
        logger.info("Loading A5 DPO model")
        try:
            # Initialize the DPO predictor with the Hugging Face Inference API
            models['a5'] = get_predictor(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            logger.info("Successfully initialized A5 DPO predictor")
        except Exception as e:
            logger.error(f"Error initializing DPO predictor: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    except Exception as e:
        logger.error(f"Error loading A5 model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Load A6 RAG model
    try:
        logger.info("Initializing A6 RAG system...")
        models['a6'] = initialize_rag_system()
        
        # Run initial model analysis with a few test questions for A6
        test_questions = [
            "How old are you?",
            "What is your highest level of education?",
            "What are your core beliefs regarding technology?",
            "What programming languages do you know?",
            "Where did you work before Google?"
        ]
        
        # Store analysis results
        models['a6_analysis'] = {
            'retriever_analysis': analyze_retriever_model(models['a6'], test_questions, verbose=False),
            'generator_analysis': analyze_generator_model(models['a6'], test_questions, verbose=False)
        }
        
        logger.info("A6 RAG system initialized and ready to use")
    except Exception as e:
        logger.error(f"Error loading A6 RAG model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Load A7 Toxic Classifier model
    try:
        logger.info("Loading A7 Toxic Classifier model")
        # Use the default model from the toxic_classifier.py script
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        models['a7'] = ToxicClassifier(model_name)
        logger.info(f"Successfully loaded A7 Toxic Classifier model using {model_name}")
    except Exception as e:
        logger.error(f"Error loading A7 Toxic Classifier model: {str(e)}")
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

    @app.route('/a4')
    def a4():
        """Route for Assignment 4: Natural Language Inference."""
        return render_template('a4.html')

    @app.route('/predict_nli', methods=['POST'])
    def predict_nli():
        """Handle NLI prediction requests."""
        try:
            data = request.get_json()
            if not data or 'premise' not in data or 'hypothesis' not in data:
                return jsonify({'error': 'Both premise and hypothesis are required'}), 400

            premise = data['premise'].strip()
            hypothesis = data['hypothesis'].strip()

            if not premise or not hypothesis:
                return jsonify({'error': 'Both premise and hypothesis must not be empty'}), 400

            if models.get('a4') is None:
                return jsonify({'error': 'NLI model not loaded'}), 500

            predictor = models['a4']
            result = predictor.predict(premise, hypothesis)

            return jsonify({
                'predicted_label': result['label'],
                'probabilities': result['probabilities']
            })

        except Exception as e:
            logger.error(f"Error in predict_nli: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/a5')
    def a5():
        """Route for Assignment 5: Optimization Human Preference."""
        return render_template('a5.html')
    
    @app.route('/api/a5/generate', methods=['POST'])
    def a5_generate():
        """API endpoint for A5 text generation."""
        try:
            data = request.get_json()
            prompt = data.get("prompt", "")
            
            # Initialize the model if it hasn't been already
            from app.assignments.a5_optimization_human_preference.inference import get_predictor
            
            # Use TinyLlama model
            model = get_predictor(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            
            # Generate response
            response = model.generate_response(prompt)
            
            return jsonify({"response": response})
        
        except Exception as e:
            app.logger.error(f"Error in A5 generation: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/generate_dpo_response', methods=['POST'])
    def generate_dpo_response():
        """Handle DPO response generation requests."""
        try:
            # Get the prompt from the request
            data = request.get_json()
            prompt = data.get('prompt', '')
            
            # Log the incoming request
            app.logger.info(f"Received DPO response generation request with prompt: {prompt[:50]}...")
            
            # Check if the model is loaded
            if 'a5' not in models:
                from app.assignments.a5_optimization_human_preference.inference import get_predictor
                models['a5'] = get_predictor(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            
            # Generate response
            response = models['a5'].generate_response(prompt)
            
            # Return the response
            return jsonify({'response': response})
            
        except Exception as e:
            app.logger.error(f"Error generating DPO response: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/a6')
    def a6():
        """Route for Assignment 6: Talk With Yourselves."""
        if 'a6' not in models or models['a6'] is None or 'a6_analysis' not in models:
            # Initialize RAG system if not already loaded
            try:
                logger.info("Initializing A6 RAG system on demand...")
                models['a6'] = initialize_rag_system()
                
                # Run initial model analysis with a few test questions
                test_questions = [
                    "How old are you?",
                    "What is your highest level of education?",
                    "What are your core beliefs regarding technology?",
                    "What programming languages do you know?",
                    "Where did you work before Google?"
                ]
                
                # Store analysis results
                models['a6_analysis'] = {
                    'retriever_analysis': analyze_retriever_model(models['a6'], test_questions, verbose=False),
                    'generator_analysis': analyze_generator_model(models['a6'], test_questions, verbose=False)
                }
                
                logger.info("A6 RAG system initialized on demand")
            except Exception as e:
                logger.error(f"Error initializing A6 RAG system on demand: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return render_template('coming_soon.html')
        
        return render_template('a6.html', 
                              retriever_analysis=models['a6_analysis']['retriever_analysis'],
                              generator_analysis=models['a6_analysis']['generator_analysis'])
    
    @app.route('/a7')
    def a7():
        """Route for Assignment 7: Distillation - Get Smaller, Get Faster."""
        # Initialize the toxic classifier if not already loaded
        if models.get('a7') is None:
            try:
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                models['a7'] = ToxicClassifier(model_name)
                logger.info(f"Initialized A7 Toxic Classifier on demand with {model_name}")
            except Exception as e:
                logger.error(f"Error initializing A7 Toxic Classifier: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return render_template('coming_soon.html')
                
        return render_template('a7.html')
    
    @app.route('/api/a7/classify', methods=['POST'])
    def classify_text():
        """API endpoint for A7 text classification."""
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            
            # Initialize the model if it hasn't been already
            if models.get('a7') is None:
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                models['a7'] = ToxicClassifier(model_name)
            
            # Classify the text
            result = models['a7'].classify(text)
            
            # Convert numpy arrays and types to Python native types for JSON serialization
            result['probabilities'] = result['probabilities'].tolist()
            result['confidence'] = float(result['confidence'])  # Convert numpy.float32 to Python float
            
            # Map model labels to toxicity labels (negative -> Toxic, positive -> Non-Toxic)
            toxicity_mapping = {
                'NEGATIVE': 'Toxic',
                'POSITIVE': 'Non-Toxic',
                0: 'Non-Toxic',  # In case labels are represented as indices
                1: 'Toxic'
            }
            
            # Override label with mapped toxicity label
            result['original_label'] = result['label']  # Store original label
            result['label'] = toxicity_mapping.get(result['label'], result['label'])
            
            # Map id2label to toxicity labels
            original_id2label = result['id2label']
            result['id2label'] = {}
            for id_key, label in original_id2label.items():
                result['id2label'][id_key] = toxicity_mapping.get(label, label)
            
            # Add model name to result
            result['model_name'] = models['a7'].model.config.name_or_path
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error in classify_text: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/a7/batch_classify', methods=['POST'])
    def batch_classify_texts():
        """API endpoint for A7 batch text classification."""
        try:
            data = request.get_json()
            texts = data.get('texts', [])
            
            if not texts:
                return jsonify({'error': 'No texts provided'}), 400
            
            if len(texts) > 10:
                return jsonify({'error': 'Too many texts. Maximum is 10.'}), 400
            
            # Initialize the model if it hasn't been already
            if models.get('a7') is None:
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                models['a7'] = ToxicClassifier(model_name)
            
            # Classify all texts
            results = models['a7'].batch_classify(texts)
            
            # Mapping for toxicity labels
            toxicity_mapping = {
                'NEGATIVE': 'Toxic',
                'POSITIVE': 'Non-Toxic',
                0: 'Non-Toxic',  # In case labels are represented as indices
                1: 'Toxic'
            }
            
            # Convert numpy arrays and types to Python native types for JSON serialization
            for result in results:
                result['probabilities'] = result['probabilities'].tolist()
                result['confidence'] = float(result['confidence'])
                
                # Map labels to toxicity labels
                result['original_label'] = result['label']
                result['label'] = toxicity_mapping.get(result['label'], result['label'])
                
                # Map id2label to toxicity labels
                original_id2label = result['id2label']
                result['id2label'] = {}
                for id_key, label in original_id2label.items():
                    result['id2label'][id_key] = toxicity_mapping.get(label, label)
            
            return jsonify({'results': results})
            
        except Exception as e:
            logger.error(f"Error in batch_classify_texts: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500

    @app.route('/chat', methods=['POST'])
    def chat():
        """Process chat messages and return responses for A6."""
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Check if RAG system is initialized
        if 'a6' not in models or models['a6'] is None:
            return jsonify({"error": "RAG system not initialized"}), 500
        
        # Query the RAG system
        result = models['a6'].query(user_message)
        
        # Add to history
        qa_pair = {
            "question": result["question"],
            "answer": result["answer"]
        }
        qa_history.append(qa_pair)
        
        # Save the updated history to a JSON file
        qa_history_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'assignments', 'a6_talk_with_yourselves', 'qa_history.json')
        with open(qa_history_path, 'w') as f:
            json.dump(qa_history, f, indent=2)
        
        # Return the result
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"]
        })

    @app.route('/qa_history', methods=['GET'])
    def get_qa_history():
        """Return the question-answer history for A6."""
        return jsonify(qa_history)

    @app.route('/analyze', methods=['POST'])
    def analyze_question():
        """Analyze a specific question with both models for A6."""
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Check if RAG system is initialized
        if 'a6' not in models or models['a6'] is None:
            return jsonify({"error": "RAG system not initialized"}), 500
        
        # Analyze with retriever model
        retriever_results = models['a6'].vector_store.similarity_search(question)
        
        # Analyze with generator model (get full response)
        generation_result = models['a6'].query(question)
        
        # Return analysis results
        return jsonify({
            "retriever_results": retriever_results,
            "generation_result": {
                "answer": generation_result["answer"],
                "sources": generation_result["sources"]
            }
        })

    # Load models when registering routes
    load_models()
