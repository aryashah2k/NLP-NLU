import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import random
import os
import time
from collections import Counter
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

class SkipgramNeg(nn.Module):
    def __init__(self, voc_size, emb_size):
        super(SkipgramNeg, self).__init__()
        self.embedding_center = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
        self.logsigmoid = nn.LogSigmoid()
    
    def forward(self, center, outside, negative):
        # Get embeddings
        center_embed = self.embedding_center(center)
        outside_embed = self.embedding_outside(outside)
        neg_embed = self.embedding_outside(negative)
        
        # Positive score
        pos_score = self.logsigmoid(torch.sum(outside_embed * center_embed, dim=2)).squeeze()
        
        # Negative score
        neg_score = self.logsigmoid(-torch.bmm(neg_embed, center_embed.transpose(1, 2)).squeeze())
        neg_score = torch.sum(neg_score, dim=1)
        
        loss = -(pos_score + neg_score).mean()
        return loss

def create_unigram_table(word_counts, vocab_size, table_size=1e6):
    pow_freq = np.array(list(word_counts.values())) ** 0.75
    power_sum = sum(pow_freq)
    ratio = pow_freq / power_sum
    count = np.round(ratio * table_size)
    
    table = []
    for idx, x in enumerate(count):
        # Ensure idx is within vocabulary range
        if idx < vocab_size:
            table.extend([idx] * int(x))
    return table

def negative_sampling(targets, unigram_table, k, vocab_size):
    batch_size = targets.shape[0]
    neg_samples = []
    
    for i in range(batch_size):
        negs = []
        target_idx = targets[i].item()
        while len(negs) < k:
            neg = random.choice(unigram_table)
            # Make sure the negative sample is within vocabulary range
            if neg != target_idx and neg < vocab_size:
                negs.append(neg)
        neg_samples.append(negs)
    
    return torch.LongTensor(neg_samples)

def create_skipgrams(sentence, window_size):
    skipgrams = []
    for i in range(len(sentence)):
        for w in range(-window_size, window_size + 1):
            context_pos = i + w
            if context_pos < 0 or context_pos >= len(sentence) or context_pos == i:
                continue
            skipgrams.append((sentence[i], sentence[context_pos]))
    return skipgrams

def prepare_batch(skipgrams, batch_size, word2idx, unigram_table, neg_samples=5):
    # Random sample from skipgrams
    indices = np.random.choice(len(skipgrams), batch_size, replace=False)
    
    centers = [[word2idx.get(skipgrams[i][0], word2idx['<UNK>'])] for i in indices]
    outsides = [[word2idx.get(skipgrams[i][1], word2idx['<UNK>'])] for i in indices]
    
    # Convert to tensors
    centers = torch.LongTensor(centers)
    outsides = torch.LongTensor(outsides)
    
    # Generate negative samples
    negative = negative_sampling(outsides.squeeze(), unigram_table, neg_samples, len(word2idx))
    
    return centers, outsides, negative

def train(corpus, window_size=2, embedding_size=100, neg_samples=5, batch_size=128, epochs=5):
    """Train the Skip-gram model with negative sampling"""
    logger.info(f"\n{'='*20} Training Configuration {'='*20}")
    logger.info(f"Window Size: {window_size}")
    logger.info(f"Embedding Size: {embedding_size}")
    logger.info(f"Negative Samples: {neg_samples}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Epochs: {epochs}\n")
    
    # Prepare data
    logger.info("Preparing training data...")
    tokenized, vocab, word2idx, idx2word = prepare_vocab(corpus)
    logger.info(f"Vocabulary size: {len(vocab)} words")
    
    # Create skipgrams
    logger.info("Creating skipgrams...")
    all_skipgrams = []
    for sentence in tqdm(tokenized, desc="Processing sentences"):
        all_skipgrams.extend(create_skipgrams(sentence, window_size))
    logger.info(f"Created {len(all_skipgrams)} skipgrams")
    
    # Create unigram table for negative sampling
    logger.info("Creating unigram table...")
    word_counts = Counter([word for sent in tokenized for word in sent])
    unigram_table = create_unigram_table(word_counts, len(vocab))
    logger.info(f"Created unigram table with {len(unigram_table)} entries")
    
    # Load evaluation datasets
    logger.info("Loading evaluation datasets...")
    semantic_pairs, syntactic_pairs = load_word_analogies()
    similarities = load_similarity_dataset()
    logger.info(f"Loaded {len(semantic_pairs)} semantic pairs and {len(syntactic_pairs)} syntactic pairs")
    
    # Initialize model
    model = SkipgramNeg(len(vocab), embedding_size)
    optimizer = optim.Adam(model.parameters())
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training metrics
    best_loss = float('inf')
    losses = []
    start_time = time.time()
    
    logger.info(f"\n{'='*20} Starting Training {'='*20}")
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        # Progress bar for batches
        num_batches = len(all_skipgrams) // batch_size + (1 if len(all_skipgrams) % batch_size != 0 else 0)
        pbar = tqdm(range(0, len(all_skipgrams), batch_size), 
                   desc=f"Epoch {epoch+1}/{epochs}",
                   total=num_batches)
        
        for i in pbar:
            # Prepare batch
            centers, outsides, negative = prepare_batch(
                all_skipgrams[i:i+batch_size],
                min(batch_size, len(all_skipgrams) - i),
                word2idx,
                unigram_table,
                neg_samples
            )
            
            # Forward pass
            loss = model(centers, outsides, negative)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            current_loss = loss.item()
            epoch_loss += current_loss
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # Calculate average loss for epoch
        avg_loss = epoch_loss / batch_count
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
            model_path = os.path.join(model_dir, f"w{window_size}_e{embedding_size}_skipgram_neg.pt")
            save_model(model, word2idx, idx2word, model_path, model_type="skipgram_neg")
    
    training_time = time.time() - start_time
    logger.info(f"Training Time: {training_time:.2f}s")
    
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
            'window_size': 2,
            'embedding_size': 100,
            'neg_samples': 5,
            'batch_size': 128,
            'epochs': 5
        },
        {
            'window_size': 5,
            'embedding_size': 100,
            'neg_samples': 10,
            'batch_size': 128,
            'epochs': 5
        }
    ]
    
    # Train and evaluate models
    for config in configs:
        logger.info(f"\nTraining Skip-gram Negative Sampling with config: {config}")
        model, results = train(corpus, **config)
        
        model_name = f"Skipgram-NEG (w={config['window_size']}, n={config['neg_samples']})"
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
