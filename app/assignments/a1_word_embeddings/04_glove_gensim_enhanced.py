import numpy as np
import torch
import torch.nn as nn
import logging
import os
import time
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class PretrainedGloVe(nn.Module):
    """Direct wrapper for pretrained GloVe embeddings"""
    
    def __init__(self, embeddings, word2idx):
        super().__init__()
        self.word2idx = word2idx
        self.idx2word = {i: word for word, i in word2idx.items()}
        self.embedding_size = embeddings.shape[1]
        
        # Create embedding layer from pretrained vectors
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings))
    
    def embedding_center(self, indices):
        """Match the interface of our other models"""
        return self.embedding(indices)
    
    def forward(self, x):
        return self.embedding(x)

def load_pretrained_glove(path, dim=100):
    """Load pretrained GloVe embeddings directly
    
    Args:
        path: Path to GloVe embeddings file
        dim: Embedding dimension
        
    Returns:
        PretrainedGloVe: Model with pretrained embeddings
    """
    logger.info(f"\n{'='*20} Loading Configuration {'='*20}")
    logger.info(f"Model Path: {path}")
    logger.info(f"Embedding Dimension: {dim}\n")
    
    # Load GloVe vectors
    logger.info("Loading pretrained embeddings...")
    word2idx = {}
    vectors = []
    
    # First pass: collect words and create word2idx
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Building vocabulary")):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            word2idx[word] = i
    
    # Initialize embedding matrix
    embeddings = np.zeros((len(word2idx), dim))
    
    # Second pass: fill embedding matrix
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading embeddings"):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            vector = np.array([float(x) for x in tokens[1:]], dtype=np.float32)
            embeddings[word2idx[word]] = vector
    
    logger.info(f"Loaded {len(word2idx):,} words with dimension {dim}")
    
    # Create model
    model = PretrainedGloVe(embeddings, word2idx)
    
    # Load evaluation datasets
    logger.info("\nLoading evaluation datasets...")
    semantic_pairs, syntactic_pairs = load_word_analogies()
    similarities = load_similarity_dataset()
    logger.info(f"Loaded {len(semantic_pairs)} semantic pairs and {len(syntactic_pairs)} syntactic pairs")
    
    # Evaluate model
    logger.info("\n" + "="*50)
    logger.info("Starting Model Evaluation")
    logger.info("="*50)
    
    # Semantic analogies evaluation
    logger.info("\nEvaluating semantic analogies...")
    semantic_acc = evaluate_analogies(model, word2idx, model.idx2word, semantic_pairs)
    logger.info(f"Number of semantic pairs evaluated: {len(semantic_pairs)}")
    logger.info(f"Semantic accuracy: {semantic_acc:.4f}")
    
    # Syntactic analogies evaluation
    logger.info("\nEvaluating syntactic analogies...")
    syntactic_acc = evaluate_analogies(model, word2idx, model.idx2word, syntactic_pairs)
    logger.info(f"Number of syntactic pairs evaluated: {len(syntactic_pairs)}")
    logger.info(f"Syntactic accuracy: {syntactic_acc:.4f}")
    
    # Word similarity evaluation
    logger.info("\nEvaluating word similarities...")
    similarity_corr, mse, num_pairs = evaluate_similarity(model, word2idx, similarities)
    logger.info(f"Number of similarity pairs evaluated: {num_pairs}")
    logger.info(f"Spearman correlation: {similarity_corr:.4f}")
    logger.info(f"Mean squared error: {mse:.4f}")
    
    # Example analogies
    logger.info("\nExample analogies evaluation:")
    example_analogies = [
        ('king', 'man', 'queen', 'woman'),
        ('paris', 'france', 'rome', 'italy'),
        ('good', 'better', 'bad', 'worse'),
        ('small', 'smaller', 'large', 'larger')
    ]
    
    for a, b, c, d in example_analogies:
        if all(word in word2idx for word in [a, b, c, d]):
            # Get embeddings
            va = model.embedding(torch.tensor(word2idx[a]))
            vb = model.embedding(torch.tensor(word2idx[b]))
            vc = model.embedding(torch.tensor(word2idx[c]))
            vd = model.embedding(torch.tensor(word2idx[d]))
            
            # Calculate cosine similarity between analogy pairs
            cos = nn.CosineSimilarity(dim=0)
            similarity = cos(vb - va, vd - vc)
            logger.info(f"Analogy {a}:{b} :: {c}:{d} - Similarity: {similarity:.4f}")
    
    # Example similarities
    logger.info("\nExample word similarities:")
    example_pairs = [
        ('man', 'woman'),
        ('king', 'queen'),
        ('computer', 'machine'),
        ('happy', 'sad')
    ]
    
    for word1, word2 in example_pairs:
        if word1 in word2idx and word2 in word2idx:
            # Get embeddings
            v1 = model.embedding(torch.tensor(word2idx[word1]))
            v2 = model.embedding(torch.tensor(word2idx[word2]))
            
            # Calculate cosine similarity
            cos = nn.CosineSimilarity(dim=0)
            similarity = cos(v1, v2)
            logger.info(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
    
    # Print evaluation summary
    logger.info(f"\n{'='*20} Evaluation Summary {'='*20}")
    logger.info(f"Semantic Accuracy: {semantic_acc:.4f} ({len(semantic_pairs)} pairs)")
    logger.info(f"Syntactic Accuracy: {syntactic_acc:.4f} ({len(syntactic_pairs)} pairs)")
    logger.info(f"Similarity Correlation: {similarity_corr:.4f} ({num_pairs} pairs)")
    logger.info(f"Mean Squared Error: {mse:.4f}")
    logger.info("="*50)
    
    # Save model in our format
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"glove_pretrained_d{dim}.pt")
    save_model(model, word2idx, model.idx2word, model_path, model_type="glove_pretrained")
    logger.info(f"\nModel saved to {model_path}")
    
    return model, {
        'word2idx': word2idx,
        'idx2word': model.idx2word,
        'semantic_accuracy': semantic_acc,
        'syntactic_accuracy': syntactic_acc,
        'similarity_correlation': similarity_corr,
        'mse': mse,
        'num_pairs': num_pairs,
        'model_path': model_path,
        'vocab_size': len(word2idx),
        'embedding_size': dim
    }

if __name__ == "__main__":
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Configurations for pretrained models
    configs = [
        {
            'path': 'glove.6B.100d.txt',
            'dim': 100
        },
        {
            'path': 'glove.6B.300d.txt',
            'dim': 300
        }
    ]
    
    # Load and evaluate models
    for config in configs:
        logger.info(f"\nLoading GloVe with config: {config}")
        try:
            start_time = time.time()
            model, results = load_pretrained_glove(**config)
            loading_time = time.time() - start_time
            
            model_name = f"GloVe-Pretrained (d={config['dim']})"
            evaluator.evaluate_model(
                model, 
                results['word2idx'], 
                results['idx2word'],
                model_name,
                window_size=None,  # N/A for pretrained models
                training_time=loading_time,  # Use loading time instead
                final_loss=None,  # N/A for pretrained models
                semantic_accuracy=results['semantic_accuracy'],
                syntactic_accuracy=results['syntactic_accuracy'],
                similarity_correlation=results['similarity_correlation'],
                mse=results['mse']
            )
            
            logger.info(f"\nModel Statistics:")
            logger.info(f"Vocabulary Size: {results['vocab_size']:,}")
            logger.info(f"Embedding Size: {results['embedding_size']}")
            logger.info(f"Loading Time: {loading_time:.2f}s")
            
        except FileNotFoundError:
            logger.error(f"Pretrained embeddings not found at {config['path']}")
            logger.error("Please download the embeddings from https://nlp.stanford.edu/data/glove.6B.zip")
            logger.error("and extract them to the 'pretrained' directory")
            continue
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            continue
    
    # Print evaluation results
    logger.info("\nTraining Metrics:")
    evaluator.print_training_table()
    
    logger.info("\nSimilarity Metrics:")
    evaluator.print_similarity_table()