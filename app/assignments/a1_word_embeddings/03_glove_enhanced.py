import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import time
from collections import defaultdict
from tqdm import tqdm
from utils import (
    load_news_corpus, prepare_vocab, load_word_analogies,
    evaluate_analogies, load_similarity_dataset, evaluate_similarity,
    save_model, load_model, find_similar_words, ModelEvaluator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class GloVe(nn.Module):
    def __init__(self, voc_size, emb_size):
        super(GloVe, self).__init__()
        self.embedding_center = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
        self.center_bias = nn.Embedding(voc_size, 1)
        self.outside_bias = nn.Embedding(voc_size, 1)
        
    def forward(self, center, outside, coocs, weighting):
        center_embed = self.embedding_center(center)
        outside_embed = self.embedding_outside(outside)
        
        center_bias = self.center_bias(center).squeeze()
        outside_bias = self.outside_bias(outside).squeeze()
        
        inner_product = torch.sum(center_embed * outside_embed, dim=2).squeeze()
        
        prediction = inner_product + center_bias + outside_bias
        
        loss = weighting * torch.pow(prediction - torch.log(coocs), 2)
        return torch.mean(loss)

def build_cooccurrence_matrix(tokenized, vocab_size, word2idx, window_size=5):
    """Build word co-occurrence matrix"""
    logger.info("Building co-occurrence matrix...")
    cooccurrence = defaultdict(float)
    
    for sentence in tqdm(tokenized, desc="Processing sentences"):
        for center_pos, center_word in enumerate(sentence):
            center_idx = word2idx.get(center_word, word2idx['<UNK>'])
            
            # For each context word in window
            for context_pos in range(
                max(0, center_pos - window_size),
                min(len(sentence), center_pos + window_size + 1)
            ):
                if context_pos != center_pos:
                    context_word = sentence[context_pos]
                    context_idx = word2idx.get(context_word, word2idx['<UNK>'])
                    distance = abs(context_pos - center_pos)
                    cooccurrence[(center_idx, context_idx)] += 1.0 / distance
    
    logger.info(f"Created co-occurrence matrix with {len(cooccurrence)} non-zero entries")
    return cooccurrence

def train(corpus, window_size=5, embedding_size=100, x_max=100, alpha=0.75, batch_size=128, epochs=5):
    """Train the GloVe model"""
    logger.info(f"\n{'='*20} Training Configuration {'='*20}")
    logger.info(f"Window Size: {window_size}")
    logger.info(f"Embedding Size: {embedding_size}")
    logger.info(f"X_max: {x_max}")
    logger.info(f"Alpha: {alpha}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Epochs: {epochs}\n")
    
    # Prepare data
    logger.info("Preparing training data...")
    tokenized, vocab, word2idx, idx2word = prepare_vocab(corpus)
    logger.info(f"Vocabulary size: {len(vocab)} words")
    
    # Build co-occurrence matrix
    cooc_matrix = build_cooccurrence_matrix(tokenized, len(vocab), word2idx, window_size)
    
    # Initialize model
    model = GloVe(len(vocab), embedding_size)
    optimizer = optim.Adam(model.parameters())
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load evaluation datasets
    logger.info("Loading evaluation datasets...")
    semantic_pairs, syntactic_pairs = load_word_analogies()
    similarities = load_similarity_dataset()
    logger.info(f"Loaded {len(semantic_pairs)} semantic pairs and {len(syntactic_pairs)} syntactic pairs")
    
    # Training metrics
    best_loss = float('inf')
    losses = []
    start_time = time.time()
    
    logger.info(f"\n{'='*20} Starting Training {'='*20}")
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        # Create batches from co-occurrence matrix
        training_pairs = []
        with tqdm(total=len(cooc_matrix), desc="Creating training pairs") as pbar:
            for (i, j), xij in cooc_matrix.items():
                if xij > 0:
                    training_pairs.append((i, j, xij))
                    pbar.update(1)
        
        # Shuffle training pairs
        np.random.shuffle(training_pairs)
        
        # Progress bar for batches
        num_batches = len(training_pairs) // batch_size + (1 if len(training_pairs) % batch_size != 0 else 0)
        pbar = tqdm(range(0, len(training_pairs), batch_size), 
                   desc=f"Epoch {epoch+1}/{epochs}",
                   total=num_batches)
        
        for i in pbar:
            # Get batch
            batch = training_pairs[i:i + batch_size]
            
            # Convert to tensors
            i_batch = torch.LongTensor([x[0] for x in batch]).unsqueeze(1)
            j_batch = torch.LongTensor([x[1] for x in batch]).unsqueeze(1)
            xij_batch = torch.FloatTensor([x[2] for x in batch])
            
            # Weight function
            weights = torch.pow(xij_batch / x_max, alpha)
            weights[xij_batch > x_max] = 1
            
            # Forward pass
            loss = model(i_batch, j_batch, xij_batch, weights)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            current_loss = loss.item()
            total_loss += current_loss
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # Calculate average loss for epoch
        avg_loss = total_loss / batch_count
        losses.append(avg_loss)
        
        # Evaluate model
        logger.info(f"\nEvaluating epoch {epoch+1}...")
        semantic_acc = evaluate_analogies(model, word2idx, idx2word, semantic_pairs)
        syntactic_acc = evaluate_analogies(model, word2idx, idx2word, syntactic_pairs)
        similarity_corr, mse, num_pairs = evaluate_similarity(model, word2idx, similarities)
        
        # Print epoch summary
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        logger.info(f"Semantic Accuracy: {semantic_acc:.4f}")
        logger.info(f"Syntactic Accuracy: {syntactic_acc:.4f}")
        logger.info(f"Similarity Correlation: {similarity_corr:.4f}")
        logger.info(f"MSE: {mse:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            logger.info("New best model! Saving checkpoint...")
            model_dir = "saved_models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"w{window_size}_e{embedding_size}_glove.pt")
            save_model(model, word2idx, idx2word, model_path, model_type="glove")
    
    training_time = time.time() - start_time
    logger.info(f"\n{'='*20} Training Complete {'='*20}")
    logger.info(f"Total training time: {training_time:.2f}s")
    logger.info(f"Best loss achieved: {best_loss:.4f}")
    
    return model, {
        'word2idx': word2idx,
        'idx2word': idx2word,
        'losses': losses,
        'training_time': training_time,
        'final_loss': losses[-1] if losses else None,
        'best_loss': best_loss,
        'semantic_accuracy': semantic_acc,
        'syntactic_accuracy': syntactic_acc,
        'similarity_correlation': similarity_corr,
        'mse': mse,
        'num_pairs': num_pairs,
        'model_path': model_path
    }

if __name__ == "__main__":
    # Load corpus
    corpus = load_news_corpus()
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Training configurations
    configs = [
        {
            'window_size': 5,
            'embedding_size': 100,
            'x_max': 100,
            'alpha': 0.75,
            'batch_size': 128,
            'epochs': 5
        },
        {
            'window_size': 10,
            'embedding_size': 100,
            'x_max': 100,
            'alpha': 0.75,
            'batch_size': 128,
            'epochs': 5
        }
    ]
    
    # Train and evaluate models
    for config in configs:
        logger.info(f"\nTraining GloVe with config: {config}")
        model, results = train(corpus, **config)
        
        model_name = f"GloVe (w={config['window_size']}, α={config['alpha']})"
        evaluator.evaluate_model(
            model, 
            results['word2idx'], 
            results['idx2word'],
            model_name,
            window_size=config['window_size'],
            training_time=results['training_time'],
            final_loss=results['final_loss']
        )
    
    # Print evaluation results
    logger.info("\nTraining Metrics:")
    evaluator.print_training_table()
    
    logger.info("\nSimilarity Metrics:")
    evaluator.print_similarity_table()
