import torch
import logging
from pathlib import Path
from tabulate import tabulate
from utils import load_model, find_similar_words
from models import Skipgram

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_skipgram(model_path):
    """Load Skip-gram model"""
    try:
        model, word2idx, idx2word = load_model(Skipgram, model_path)
        return model, word2idx, idx2word
    except Exception as e:
        logger.error(f"Failed to load Skip-gram model from {model_path}: {str(e)}")
        return None

def display_similar_words(query_words, model, word2idx, idx2word, top_k=10):
    """Display similar words for the Skip-gram model
    
    Args:
        query_words (list): List of words to find similar words for
        model: The Skip-gram model
        word2idx (dict): Word to index mapping
        idx2word (dict): Index to word mapping
        top_k (int): Number of similar words to display
    """
    for query in query_words:
        print(f"\nSimilar words to '{query}':")
        print("-" * 60)
        
        if query not in word2idx:
            logger.warning(f"Word '{query}' not in vocabulary")
            continue
            
        try:
            similar_words = find_similar_words(query, model, word2idx, idx2word, k=top_k)
            
            # Prepare table data
            table_data = []
            for i, (word, sim) in enumerate(similar_words, 1):
                table_data.append([f"{i}", word, f"{sim:.4f}"])
            
            # Print table
            headers = ["Rank", "Word", "Similarity"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        except Exception as e:
            logger.error(f"Error finding similar words: {str(e)}")
            continue
        print()

def main():
    # Model path
    model_path = "NLP/A1/saved_models/w2_e100_skipgram.pt"
    
    # Load Skip-gram model
    result = load_skipgram(model_path)
    if not result:
        logger.error("Could not load Skip-gram model. Please ensure the model file exists.")
        return
    
    model, word2idx, idx2word = result
    
    # Query words to test
    query_words = [
        "king", "computer", "good", "day", 
        "python", "machine", "learning", "artificial",
        # Add some domain-specific words
        "data", "algorithm", "network", "science",
        # Add some common words
        "time", "person", "world", "work"
    ]
    
    # Display similar words
    display_similar_words(query_words, model, word2idx, idx2word)

if __name__ == "__main__":
    main()
