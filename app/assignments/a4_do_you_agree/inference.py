import torch
import json
from transformers import BertTokenizer
from .sentence_bert import SentenceBERT
from .bert_scratch import BertConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NLIPredictor:
    def __init__(self, model_dir='sbert_model'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer first to get vocab size
        self.tokenizer = BertTokenizer.from_pretrained(f"{model_dir}/tokenizer")
        logger.info(f"Loaded tokenizer with vocabulary size: {len(self.tokenizer.vocab)}")
        
        # Initialize BERT config with SBERT values
        config = BertConfig()
        
        # Set model architecture parameters
        config.vocab_size = 30522  # Fixed vocab size from SBERT config
        config.hidden_size = 64
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.intermediate_size = 256
        config.max_position_embeddings = 128
        config.max_len = 32
        config.type_vocab_size = 2
        
        # Set dropout and normalization parameters
        config.hidden_dropout_prob = 0.1
        config.attention_probs_dropout_prob = 0.1
        config.layer_norm_eps = 1e-12
        
        # Set special token IDs
        config.pad_token_id = 0
        config.mask_token_id = 3
        config.cls_token_id = 1
        config.sep_token_id = 2
        
        # Training parameters (required by BertConfig)
        config.learning_rate = 1e-4
        config.batch_size = 2
        config.gradient_accumulation_steps = 16
        config.weight_decay = 0.01
        config.adam_epsilon = 1e-8
        config.warmup_ratio = 0.1
        
        # Initialize SentenceBERT with the config
        self.model = SentenceBERT(None, hidden_size=config.hidden_size, config=config)
        
        # Load model weights
        try:
            model_path = f"{model_dir}/model.pt"
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded model weights from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping
        self.id2label = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
        
        logger.info("Model initialized successfully")
    
    def predict(self, premise, hypothesis):
        # Clear GPU cache before prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Tokenize inputs
        inputs = self.tokenizer(
            [premise, hypothesis],
            padding=True,
            truncation=True,
            max_length=32,  # Use fixed max_length from config
            return_tensors='pt'
        )
        
        # Split inputs for premise and hypothesis
        premise_input_ids = inputs['input_ids'][0].unsqueeze(0)
        premise_attention_mask = inputs['attention_mask'][0].unsqueeze(0)
        hypothesis_input_ids = inputs['input_ids'][1].unsqueeze(0)
        hypothesis_attention_mask = inputs['attention_mask'][1].unsqueeze(0)
        
        # Move to device
        premise_input_ids = premise_input_ids.to(self.device)
        premise_attention_mask = premise_attention_mask.to(self.device)
        hypothesis_input_ids = hypothesis_input_ids.to(self.device)
        hypothesis_attention_mask = hypothesis_attention_mask.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(
                    premise_input_ids,
                    premise_attention_mask,
                    hypothesis_input_ids,
                    hypothesis_attention_mask
                )
                prediction = torch.argmax(outputs, dim=1).item()
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        result = {
            'label': self.id2label[prediction],
            'probabilities': {
                self.id2label[i]: float(prob.item())
                for i, prob in enumerate(probabilities)
            }
        }
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result

def main():
    # Initialize predictor
    predictor = NLIPredictor()
    
    # Example usage
    examples = [
        {
            'premise': 'A man is playing a guitar on stage.',
            'hypothesis': 'The man is performing music.',
            'expected': 'entailment'
        },
        {
            'premise': 'The cat is sleeping on the couch.',
            'hypothesis': 'The dog is running in the park.',
            'expected': 'contradiction'
        },
        {
            'premise': 'A woman is reading a book.',
            'hypothesis': 'She is wearing glasses.',
            'expected': 'neutral'
        },
        {
        'premise': 'Children are playing soccer in the park.',
        'hypothesis': 'Kids are engaged in outdoor sports.',
        'expected': 'entailment'
    },
    {
        'premise': 'The restaurant is packed with customers.',
        'hypothesis': 'The restaurant is closed today.',
        'expected': 'contradiction'
    },
    {
        'premise': 'A student is writing notes in class.',
        'hypothesis': 'The student understands the material.',
        'expected': 'neutral'
    },
    {
        'premise': 'The chef is preparing pasta in the kitchen.',
        'hypothesis': 'Someone is cooking food.',
        'expected': 'entailment'
    },
    {
        'premise': 'The sky is clear and blue today.',
        'hypothesis': 'It is raining heavily.',
        'expected': 'contradiction'
    },
    {
        'premise': 'A person bought a new laptop.',
        'hypothesis': 'They got it from Amazon.',
        'expected': 'neutral'
    },
    {
        'premise': 'The train arrived 30 minutes late.',
        'hypothesis': 'The train was delayed.',
        'expected': 'entailment'
    },
    {
        'premise': 'The museum is open on weekends.',
        'hypothesis': 'The museum is closed every day.',
        'expected': 'contradiction'
    },
    {
        'premise': 'A woman is walking her dog.',
        'hypothesis': 'The dog is brown in color.',
        'expected': 'neutral'
    },
    {
        'premise': 'The movie theater is showing new releases.',
        'hypothesis': 'Films are being screened.',
        'expected': 'entailment'
    }
    ]
    
    # Test each example
    for i, example in enumerate(examples, 1):
        result = predictor.predict(example['premise'], example['hypothesis'])
        logger.info(f"\nExample {i}:")
        logger.info(f"Premise: {example['premise']}")
        logger.info(f"Hypothesis: {example['hypothesis']}")
        logger.info(f"Expected: {example['expected']}")
        logger.info(f"Predicted: {result['label']}")
        logger.info("Probabilities:")
        for label, prob in result['probabilities'].items():
            logger.info(f"  {label}: {prob:.3f}")

if __name__ == "__main__":
    main()
