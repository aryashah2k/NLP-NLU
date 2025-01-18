import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from tqdm import tqdm
from utils import (
    load_news_corpus, prepare_vocab, load_word_analogies,
    evaluate_analogies, load_similarity_dataset, evaluate_similarity,
    Timer, save_model, load_model, find_similar_words, ModelEvaluator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class Skipgram(nn.Module):
    def __init__(self, voc_size, emb_size):
        super(Skipgram, self).__init__()
        self.embedding_center = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
    
    def forward(self, center, outside, all_vocabs):
        center_embedding = self.embedding_center(center)
        outside_embedding = self.embedding_outside(outside)
        all_vocabs_embedding = self.embedding_outside(all_vocabs)
        
        # Calculate loss
        top_term = torch.exp(outside_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2))
        lower_term = all_vocabs_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2)
        lower_term_sum = torch.sum(torch.exp(lower_term), 1)
        
        loss = -torch.mean(torch.log(top_term / lower_term_sum))
        return loss

def create_skipgrams(sentence, window_size):
    skipgrams = []
    for i in range(len(sentence)):
        for w in range(-window_size, window_size + 1):
            context_pos = i + w
            if context_pos < 0 or context_pos >= len(sentence) or context_pos == i:
                continue
            skipgrams.append((sentence[i], sentence[context_pos]))
    return skipgrams

def prepare_batch(skipgrams, batch_size, word2idx, vocab_size):
    # Random sample from skipgrams
    indices = np.random.choice(len(skipgrams), batch_size, replace=False)
    
    centers = [[word2idx.get(skipgrams[i][0], word2idx['<UNK>'])] for i in indices]
    outsides = [[word2idx.get(skipgrams[i][1], word2idx['<UNK>'])] for i in indices]
    
    # Convert to tensors
    centers = torch.LongTensor(centers)
    outsides = torch.LongTensor(outsides)
    all_vocabs = torch.arange(vocab_size).expand(batch_size, vocab_size)
    
    return centers, outsides, all_vocabs

def train(corpus, window_size=2, embedding_size=100, batch_size=128, epochs=5):
    logger.info(f"\n{'='*20} Training Configuration {'='*20}")
    logger.info(f"Window Size: {window_size}")
    logger.info(f"Embedding Size: {embedding_size}")
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
    
    # Initialize model
    model = Skipgram(len(vocab), embedding_size)
    optimizer = optim.Adam(model.parameters())
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load evaluation datasets
    logger.info("Loading evaluation datasets...")
    semantic_pairs, syntactic_pairs = load_word_analogies()
    similarities = load_similarity_dataset()
    logger.info(f"Loaded {len(semantic_pairs)} semantic pairs and {len(syntactic_pairs)} syntactic pairs")
    
    # Training metrics
    best_loss = float('inf')
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
            centers, outsides, all_vocabs = prepare_batch(
                all_skipgrams[i:i+batch_size],
                min(batch_size, len(all_skipgrams) - i),
                word2idx,
                len(vocab)
            )
            
            # Forward pass
            loss = model(centers, outsides, all_vocabs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            current_loss = loss.item()
            epoch_loss += current_loss
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg_loss': f'{epoch_loss/batch_count:.4f}'
            })
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / batch_count
        
        # Evaluate model
        logger.info(f"\nEvaluating epoch {epoch+1}...")
        with Timer() as eval_timer:
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
        logger.info(f"Evaluation Time: {eval_timer.interval:.2f}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            logger.info("New best model! Saving checkpoint...")
            model_dir = "saved_models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"w{window_size}_e{embedding_size}.pt")
            save_model(model, word2idx, idx2word, model_path, model_type="skipgram")
    
    training_time = time.time() - start_time
    logger.info(f"\n{'='*20} Training Complete {'='*20}")
    logger.info(f"Total training time: {training_time:.2f}s")
    logger.info(f"Best loss achieved: {best_loss:.4f}")
    
    return model, {
        'final_loss': avg_loss,
        'best_loss': best_loss,
        'training_time': training_time,
        'semantic_accuracy': semantic_acc,
        'syntactic_accuracy': syntactic_acc,
        'similarity_correlation': similarity_corr,
        'mse': mse,
        'num_pairs': num_pairs,
        'model_path': model_path,
        'word2idx': word2idx,
        'idx2word': idx2word
    }

if __name__ == "__main__":
    # Load corpus
    corpus = load_news_corpus()
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Train models with different configurations
    configs = [
        {'window_size': 2, 'embedding_size': 100},
        {'window_size': 5, 'embedding_size': 100}
    ]
    
    for config in configs:
        logger.info(f"\nTraining Skip-gram with config: {config}")
        model, results = train(corpus, **config)
        
        model_name = f"Skipgram (w={config['window_size']})"
        evaluator.evaluate_model(
            model, 
            results['word2idx'], 
            results['idx2word'], 
            model_name,
            window_size=config['window_size'],
            training_time=results['training_time'],
            final_loss=results['final_loss']
        )
    
    # Print both tables
    evaluator.print_training_table()
    evaluator.print_similarity_table()
