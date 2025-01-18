import numpy as np
import torch
from collections import Counter
import nltk
from nltk.corpus import brown
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import time
import logging
import os
import requests
from torch import nn
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_news_corpus():
    """Load and preprocess the Brown corpus news category"""
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        nltk.download('brown')
    
    # Get news category sentences
    news_sents = brown.sents(categories='news')
    
    # Lowercase and join sentences
    corpus = [" ".join(sent).lower() for sent in news_sents]
    return corpus

def prepare_vocab(corpus, min_count=5):
    """Create vocabulary from corpus with minimum frequency threshold"""
    # Tokenize
    tokenized = [sent.split() for sent in corpus]
    # Count words
    word_counts = Counter([word for sent in tokenized for word in sent])
    # Filter by minimum count
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    vocab.append('<UNK>')
    
    # Create mappings
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return tokenized, vocab, word2idx, idx2word

def load_word_analogies():
    """Load semantic and syntactic test sets"""
    semantic_file = "evaluation/capital-common-countries.txt"
    syntactic_file = "evaluation/past-tense.txt"
    
    semantic_pairs = []
    syntactic_pairs = []
    
    # Create evaluation directory if it doesn't exist
    os.makedirs("evaluation", exist_ok=True)
    
    # Create sample semantic analogies (capital-country)
    semantic_analogies = [
        "athens greece berlin germany",
        "athens greece moscow russia",
        "athens greece paris france",
        "berlin germany london england",
        "berlin germany madrid spain",
        "berlin germany paris france",
        "london england paris france",
        "london england rome italy",
        "madrid spain paris france",
        "madrid spain rome italy",
        "paris france rome italy",
        "rome italy tokyo japan"
    ]
    
    # Create sample syntactic analogies (verb past tense)
    syntactic_analogies = [
        "dance danced smile smiled",
        "dance danced walk walked",
        "decrease decreased increase increased",
        "describe described destroy destroyed",
        "eat ate speak spoke",
        "fall fell rise rose",
        "feed fed speak spoke",
        "find found lose lost",
        "go went speak spoke",
        "grow grew shrink shrank",
        "lose lost win won",
        "say said speak spoke",
        "sing sang write wrote",
        "sit sat speak spoke",
        "take took give gave"
    ]
    
    # Write sample files
    with open(semantic_file, 'w') as f:
        f.write('\n'.join(semantic_analogies))
    
    with open(syntactic_file, 'w') as f:
        f.write('\n'.join(syntactic_analogies))
    
    # Load and parse files
    def load_analogies(filename):
        pairs = []
        with open(filename, 'r') as f:
            for line in f:
                w1, w2, w3, w4 = line.strip().lower().split()
                pairs.append((w1, w2, w3, w4))
        return pairs
    
    semantic_pairs = load_analogies(semantic_file)
    syntactic_pairs = load_analogies(syntactic_file)
    
    return semantic_pairs, syntactic_pairs

def evaluate_analogies(model, word2idx, idx2word, pairs):
    """Evaluate word analogy accuracy"""
    correct = 0
    total = 0
    
    for w1, w2, w3, w4 in pairs:
        if w1 not in word2idx or w2 not in word2idx or w3 not in word2idx or w4 not in word2idx:
            continue
            
        # Get embeddings
        v1 = model.embedding_center(torch.LongTensor([word2idx[w1]])).detach()
        v2 = model.embedding_center(torch.LongTensor([word2idx[w2]])).detach()
        v3 = model.embedding_center(torch.LongTensor([word2idx[w3]])).detach()
        
        # v2 - v1 + v3 should be close to v4
        predicted = v2 - v1 + v3
        
        # Find closest word
        distances = []
        for idx in range(len(word2idx)):
            vec = model.embedding_center(torch.LongTensor([idx])).detach()
            dist = torch.nn.functional.cosine_similarity(predicted, vec)
            distances.append((dist.item(), idx))
        
        # Sort by similarity
        distances.sort(reverse=True)
        
        # Get top prediction
        pred_word = idx2word[distances[0][1]]
        
        if pred_word == w4:
            correct += 1
        total += 1
        
    return correct / total if total > 0 else 0

def load_similarity_dataset():
    """Load the WordSim-353 dataset for word similarity evaluation"""
    wordsim_path = "evaluation/wordsim353.txt"
    
    if not os.path.exists(wordsim_path):
        logger.error(f"WordSim-353 file not found at {wordsim_path}")
        return create_fallback_dataset()
    
    # Load and parse the dataset
    similarities = []
    try:
        with open(wordsim_path, 'r', encoding='utf-8') as f:
            # Read all lines
            lines = f.readlines()
            
            # Check if there's a header and skip if present
            start_idx = 0
            if lines and any(header in lines[0].lower() for header in ['word1', 'word2', 'score', 'human']):
                start_idx = 1
            
            # Parse each line
            for line in lines[start_idx:]:
                try:
                    # Handle both tab and space-separated formats
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        word1, word2, score = parts[0], parts[1], float(parts[-1])
                        similarities.append((word1.lower(), word2.lower(), float(score)/10))  # Normalize to 0-1
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping malformed line in similarity dataset: {line.strip()}")
                    continue
        
        if similarities:
            logger.info(f"Successfully loaded {len(similarities)} word pairs from WordSim-353")
            return similarities
        else:
            logger.error("No valid similarities found in WordSim-353 file")
            return create_fallback_dataset()
            
    except Exception as e:
        logger.error(f"Error loading WordSim-353: {e}")
        return create_fallback_dataset()

def create_fallback_dataset():
    """Create a minimal fallback dataset for when WordSim-353 is unavailable"""
    logger.warning("Using fallback similarity dataset")
    return [
        ("car", "automobile", 1.0),
        ("gem", "jewel", 0.96),
        ("journey", "voyage", 0.89),
        ("boy", "lad", 0.83),
        ("coast", "shore", 0.79),
        ("asylum", "madhouse", 0.77),
        ("magician", "wizard", 0.73),
        ("midday", "noon", 0.71),
        ("furnace", "stove", 0.69),
        ("food", "fruit", 0.65),
    ]

def evaluate_similarity(model, word2idx, similarities):
    """Evaluate model performance on word similarity task"""
    model_sims = []
    human_sims = []
    num_pairs = 0
    
    for w1, w2, score in similarities:
        if w1 not in word2idx or w2 not in word2idx:
            continue
            
        # Get word vectors
        v1 = model.embedding_center(torch.tensor([word2idx[w1]]))
        v2 = model.embedding_center(torch.tensor([word2idx[w2]]))
        
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(v1, v2).item()
        
        model_sims.append(cos_sim)
        human_sims.append(score)
        num_pairs += 1
    
    if len(model_sims) > 1:
        # Calculate correlation and MSE
        correlation = spearmanr(model_sims, human_sims)[0]  # Take only the correlation value, not p-value
        mse = mean_squared_error(human_sims, model_sims)
        return correlation, mse, num_pairs
    return 0.0, 0.0, 0

class ModelEvaluator:
    """Class to evaluate and compare different word embedding models"""
    
    def __init__(self):
        self.results = {}
        self.similarities = load_similarity_dataset()
        self.semantic_pairs, self.syntactic_pairs = load_word_analogies()
    
    def evaluate_model(self, model, word2idx, idx2word, model_name, window_size=None, training_time=None, final_loss=None):
        """Evaluate a single model and store its results"""
        # Evaluate similarities
        correlation, mse, num_pairs = evaluate_similarity(model, word2idx, self.similarities)
        
        # Evaluate analogies
        semantic_acc = evaluate_analogies(model, word2idx, idx2word, self.semantic_pairs)
        syntactic_acc = evaluate_analogies(model, word2idx, idx2word, self.syntactic_pairs)
        
        self.results[model_name] = {
            'window_size': window_size,
            'training_time': training_time,
            'final_loss': final_loss,
            'correlation': correlation,
            'mse': mse,
            'num_pairs': num_pairs,
            'semantic_acc': semantic_acc,
            'syntactic_acc': syntactic_acc
        }
    
    def print_training_table(self):
        """Print a table comparing training metrics and accuracy"""
        # Headers
        headers = ['Model', 'Window Size', 'Training Loss', 'Training Time', 'Syntactic Acc', 'Semantic Acc']
        col_widths = [max(len(str(h)), 15) for h in headers]
        
        # Update column widths based on data
        for model_name, metrics in self.results.items():
            col_widths[0] = max(col_widths[0], len(model_name))
            values = [
                metrics.get('window_size', 'N/A'),
                metrics.get('final_loss', 'N/A'),
                metrics.get('training_time', 'N/A'),
                metrics['syntactic_acc'],
                metrics['semantic_acc']
            ]
            for i, value in enumerate(values):
                col_widths[i+1] = max(col_widths[i+1], len(f'{value:.4f}' if isinstance(value, float) else str(value)))
        
        # Print header
        header_line = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
        separator = '-' * len(header_line)
        print('\nTraining and Accuracy Results:')
        print(separator)
        print(header_line)
        print(separator)
        
        # Print each model's results
        for model_name, metrics in self.results.items():
            row = [
                model_name.ljust(col_widths[0]),
                str(metrics.get('window_size', 'N/A')).ljust(col_widths[1]),
                f"{metrics.get('final_loss', 'N/A'):.4f}".ljust(col_widths[2]) if isinstance(metrics.get('final_loss'), float) else 'N/A'.ljust(col_widths[2]),
                f"{metrics.get('training_time', 'N/A'):.2f}s".ljust(col_widths[3]) if isinstance(metrics.get('training_time'), float) else 'N/A'.ljust(col_widths[3]),
                f"{metrics['syntactic_acc']:.4f}".ljust(col_widths[4]),
                f"{metrics['semantic_acc']:.4f}".ljust(col_widths[5])
            ]
            print(' | '.join(row))
        print(separator)
    
    def print_similarity_table(self):
        """Print a table comparing similarity metrics against human judgments"""
        # Get unique model types
        model_types = {
            name.split()[0]: [] for name in self.results.keys()
        }
        
        # Group results by model type
        for model_name, metrics in self.results.items():
            model_type = model_name.split()[0]
            model_types[model_type].append((model_name, metrics))
        
        # Headers
        headers = ['Metric'] + list(model_types.keys()) + ['Y true']
        col_widths = [max(len(str(h)), 15) for h in headers]
        
        # Print header
        header_line = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
        separator = '-' * len(header_line)
        print('\nSimilarity Comparison Results:')
        print(separator)
        print(header_line)
        print(separator)
        
        # Print MSE row
        mse_row = ['MSE'.ljust(col_widths[0])]
        for model_type in model_types:
            # Get best MSE for this model type
            best_mse = min((m['mse'] for _, m in model_types[model_type]), default='N/A')
            mse_row.append(f"{best_mse:.4f}".ljust(col_widths[len(mse_row)]) if isinstance(best_mse, float) else 'N/A'.ljust(col_widths[len(mse_row)]))
        mse_row.append('1.0000'.ljust(col_widths[-1]))  # Y true column
        print(' | '.join(mse_row))
        print(separator)
    
    def get_results_dict(self):
        """Return the results dictionary for external use"""
        return self.results

def save_model(model, word2idx, idx2word, model_path, model_type=None):
    """Save model and vocabulary mappings
    
    Args:
        model: The PyTorch model to save
        word2idx: Word to index mapping
        idx2word: Index to word mapping
        model_path: Base path for saving the model
        model_type: Type of model (skipgram, skipgram_neg, glove)
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Add model type to filename if provided
    if model_type:
        path_parts = os.path.splitext(model_path)
        model_path = f"{path_parts[0]}_{model_type}{path_parts[1]}"
    
    # Save the PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'word2idx': word2idx,
        'idx2word': idx2word,
        'embedding_dim': model.embedding_center.embedding_dim,
        'vocab_size': len(word2idx),
        'model_type': model_type
    }, model_path)
    logger.info(f"Model saved to {model_path}")

def load_model(model_path):
    """Load model and vocabulary mappings"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the saved state
    checkpoint = torch.load(model_path)
    
    # Create model instance
    vocab_size = checkpoint['vocab_size']
    embedding_dim = checkpoint['embedding_dim']
    
    # Determine model type from filename
    is_glove = 'glove' in model_path.lower()
    
    # Model class that handles both GloVe and Skip-gram architectures
    class EmbeddingModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, is_glove=False):
            super().__init__()
            self.embedding_center = nn.Embedding(vocab_size, embedding_dim)
            self.embedding_outside = nn.Embedding(vocab_size, embedding_dim)
            
            # Only GloVe models have bias terms
            self.is_glove = is_glove
            if is_glove:
                self.center_bias = nn.Embedding(vocab_size, 1)
                self.outside_bias = nn.Embedding(vocab_size, 1)
            
        def forward(self, center_words, context_words=None):
            if context_words is None:
                return self.embedding_center(center_words)
            
            # Get embeddings
            center_embeds = self.embedding_center(center_words)
            context_embeds = self.embedding_outside(context_words)
            
            # Calculate dot product
            dot_product = torch.sum(center_embeds * context_embeds, dim=1)
            
            # Add biases for GloVe models
            if self.is_glove:
                center_bias = self.center_bias(center_words).squeeze()
                context_bias = self.outside_bias(context_words).squeeze()
                return dot_product + center_bias + context_bias
            
            return dot_product
    
    # Create model with appropriate type
    model = EmbeddingModel(vocab_size, embedding_dim, is_glove=is_glove)
    
    # For Skip-gram models, filter out bias terms if they exist in state dict
    if not is_glove:
        state_dict = checkpoint['model_state_dict']
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                             if not any(x in k for x in ['bias', 'outside'])}
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()  # Set to evaluation mode
    
    return model, checkpoint['word2idx'], checkpoint['idx2word']

def find_similar_words(model, word2idx, idx2word, query, k=10):
    """Find top-k similar words for a query using the trained model
    
    Args:
        model: The trained PyTorch model
        word2idx: Word to index mapping
        idx2word: Index to word mapping
        query: The query word or phrase
        k: Number of similar words to return (default: 10)
    
    Returns:
        List of tuples containing the similar word and its similarity score
    """
    if isinstance(query, str):
        # Single word query
        if query not in word2idx:
            return []
        query_idx = word2idx[query]
        query_vec = model.embedding_center(torch.LongTensor([query_idx])).detach()
    else:
        # Multiple word query - average the vectors
        query_words = query.lower().split()
        vectors = []
        for word in query_words:
            if word in word2idx:
                word_idx = word2idx[word]
                vectors.append(model.embedding_center(torch.LongTensor([word_idx])).detach())
        if not vectors:
            return []
        query_vec = torch.mean(torch.stack(vectors), dim=0)

    # Calculate similarities with all words
    similarities = []
    all_indices = torch.LongTensor(range(len(word2idx)))
    all_embeddings = model.embedding_center(all_indices).detach()
    
    # Normalize vectors for cosine similarity
    query_vec = query_vec / query_vec.norm()
    all_embeddings = all_embeddings / all_embeddings.norm(dim=1, keepdim=True)
    
    # Calculate similarities in batches to avoid memory issues
    batch_size = 1000
    for i in range(0, len(word2idx), batch_size):
        batch_embeddings = all_embeddings[i:i+batch_size]
        sims = torch.matmul(query_vec, batch_embeddings.t()).squeeze()
        
        for j, sim in enumerate(sims):
            idx = i + j
            if idx < len(word2idx):  # Ensure we don't go out of bounds
                similarities.append((idx2word[idx], sim.item()))
    
    # Sort by similarity and return top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    # Filter out the query word itself if it's in the results
    similarities = [(w, s) for w, s in similarities if w != query]
    return similarities[:k]

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
