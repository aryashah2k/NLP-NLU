import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime
from .bert_scratch import BertConfig, BertModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sbert_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class SentenceBERT(nn.Module):
    def __init__(self, bert_model=None, hidden_size=256, config=None):
        super().__init__()
        if bert_model is None:
            if config is None:
                config = BertConfig()
                config.hidden_size = hidden_size
            self.bert = BertModel(config)
        else:
            self.bert = bert_model
            
        self.fc = nn.Linear(hidden_size * 3, 3)  # 3 classes: entailment, contradiction, neutral
        
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, premise_input_ids, premise_attention_mask, 
               hypothesis_input_ids, hypothesis_attention_mask):
        # Get embeddings for premise
        premise_outputs = self.bert(premise_input_ids, attention_mask=premise_attention_mask)
        if isinstance(premise_outputs, tuple):
            premise_outputs = premise_outputs[0]  # Get hidden states if tuple is returned
        premise_embeddings = self.mean_pooling(premise_outputs, premise_attention_mask)
        
        # Get embeddings for hypothesis
        hypothesis_outputs = self.bert(hypothesis_input_ids, attention_mask=hypothesis_attention_mask)
        if isinstance(hypothesis_outputs, tuple):
            hypothesis_outputs = hypothesis_outputs[0]  # Get hidden states if tuple is returned
        hypothesis_embeddings = self.mean_pooling(hypothesis_outputs, hypothesis_attention_mask)
        
        # Create feature vector [u, v, |u-v|] as described in the paper
        features = torch.cat([
            premise_embeddings,
            hypothesis_embeddings,
            torch.abs(premise_embeddings - hypothesis_embeddings)
        ], dim=1)
        
        return self.fc(features)

def load_datasets(num_samples=None):
    logger.info("Loading SNLI and MNLI datasets...")
    
    # Load SNLI dataset
    snli_dataset = load_dataset("snli")
    
    # Load MNLI dataset
    mnli_dataset = load_dataset("multi_nli")
    
    # Rename MNLI labels to match SNLI
    def rename_labels(example):
        label_map = {0: 0, 1: 1, 2: 2}  # entailment: 0, contradiction: 1, neutral: 2
        example['label'] = label_map[example['label']]
        return example
    
    # Process MNLI dataset to match SNLI format
    mnli_dataset = mnli_dataset.map(rename_labels)
    
    # Combine datasets
    train_dataset = concatenate_datasets([
        snli_dataset['train'],
        mnli_dataset['train']
    ])
    
    val_dataset = concatenate_datasets([
        snli_dataset['validation'],
        mnli_dataset['validation_matched']
    ])
    
    # Subsample if specified
    if num_samples is not None:
        train_dataset = train_dataset.shuffle(seed=42).select(range(num_samples))
        val_dataset = val_dataset.shuffle(seed=42).select(range(num_samples))
    
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    return {
        'train': train_dataset,
        'validation': val_dataset
    }

def preprocess_data(datasets, tokenizer, max_length=128):
    logger.info("Preprocessing datasets...")
    
    def preprocess_function(examples):
        # Tokenize premises
        premise_encodings = tokenizer(
            examples['premise'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        
        # Tokenize hypotheses
        hypothesis_encodings = tokenizer(
            examples['hypothesis'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        
        return {
            'premise_input_ids': premise_encodings['input_ids'],
            'premise_attention_mask': premise_encodings['attention_mask'],
            'hypothesis_input_ids': hypothesis_encodings['input_ids'],
            'hypothesis_attention_mask': hypothesis_encodings['attention_mask'],
            'labels': examples['label']
        }
    
    # Process each split
    processed_datasets = {}
    for split, dataset in datasets.items():
        processed_datasets[split] = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        processed_datasets[split].set_format('torch')
    
    return processed_datasets

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            premise_input_ids = batch['premise_input_ids'].to(device)
            premise_attention_mask = batch['premise_attention_mask'].to(device)
            hypothesis_input_ids = batch['hypothesis_input_ids'].to(device)
            hypothesis_attention_mask = batch['hypothesis_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                premise_input_ids, premise_attention_mask,
                hypothesis_input_ids, hypothesis_attention_mask
            )
            
            preds = torch.argmax(outputs, dim=1)
            
            # Only include valid labels (not -1)
            valid_mask = labels >= 0
            if valid_mask.any():
                all_preds.extend(preds[valid_mask].cpu().numpy())
                all_labels.extend(labels[valid_mask].cpu().numpy())
    
    # Convert to numpy arrays for sklearn
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Get unique labels for report
    unique_labels = sorted(set(all_labels) | set(all_preds))
    label_names = ['entailment', 'contradiction', 'neutral']
    
    # Ensure we have mapping for all labels
    if len(unique_labels) > len(label_names):
        logger.warning(f"Found {len(unique_labels)} classes but only {len(label_names)} names provided. Some predictions may be invalid.")
        
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, 
        all_preds,
        labels=range(len(label_names)),  # Only use valid labels
        target_names=label_names,
        zero_division=0
    )
    
    return accuracy, report

def save_model(model, tokenizer, config, metrics, output_dir='sbert_model'):
    """Save the model, tokenizer, configuration and metrics."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model state
    model_path = os.path.join(output_dir, 'pytorch_model.bin')
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, 'tokenizer')
    tokenizer.save_pretrained(tokenizer_path)
    
    # Convert config to serializable dictionary
    config_dict = {
        'vocab_size': config.vocab_size,
        'hidden_size': config.hidden_size,
        'num_hidden_layers': config.num_hidden_layers,
        'num_attention_heads': config.num_attention_heads,
        'intermediate_size': config.intermediate_size,
        'max_len': config.max_len,
        'batch_size': config.batch_size,
        'gradient_accumulation_steps': config.gradient_accumulation_steps
    }
    
    # Convert metrics to serializable format
    accuracy, report = metrics
    metrics_dict = {
        'accuracy': float(accuracy),
        'classification_report': report
    }
    
    # Save configuration and metrics
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'model_config': config_dict,
            'metrics': metrics_dict,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }, f, indent=4)
    
    logger.info(f"Model and configuration saved to {output_dir}")

def main():
    # Load datasets with smaller batch size
    datasets = load_datasets(num_samples=1000)
    logger.info(f"Loaded {len(datasets['train'])} training samples and {len(datasets['validation'])} validation samples")
    
    # Initialize tokenizer and get vocab size
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = len(tokenizer.vocab)
    logger.info(f"Tokenizer vocabulary size: {vocab_size}")
    
    # Load pre-trained BERT from our custom implementation
    config = BertConfig()
    config.vocab_size = vocab_size  # Set vocab size to match tokenizer
    config.batch_size = 2  # Extremely small batch size
    config.hidden_size = 64  # Minimal hidden size
    config.num_hidden_layers = 2  # Minimal number of layers
    config.num_attention_heads = 4  # Reduced attention heads
    config.gradient_accumulation_steps = 16  # Heavy gradient accumulation
    config.intermediate_size = 256  # Minimal intermediate size
    config.max_len = 32  # Minimal sequence length
    config.attention_probs_dropout_prob = 0.1
    config.hidden_dropout_prob = 0.1
    
    bert_model = BertModel(config)
    
    # Try to load pre-trained weights if available
    try:
        checkpoints = [f for f in os.listdir('model_checkpoints') if f.startswith('bert_epoch_')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2]))
            checkpoint_path = os.path.join('model_checkpoints', latest_checkpoint)
            
            # Load checkpoint with proper handling
            checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Load to CPU first
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out mismatched keys
            model_dict = bert_model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() 
                         if k in model_dict and v.shape == model_dict[k].shape}
            
            # Load filtered state dict
            bert_model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded compatible weights from {checkpoint_path}")
        else:
            logger.warning("No pre-trained BERT checkpoints found. Starting with random initialization.")
    except Exception as e:
        logger.warning(f"Failed to load pre-trained BERT weights: {e}")
    
    # Initialize Sentence-BERT
    model = SentenceBERT(bert_model, hidden_size=config.hidden_size, config=config)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.bert, 'enable_gradient_checkpointing'):
        model.bert.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing")
    
    # Move model to GPU after all initialization
    model = model.to(device)
    
    # Clear GPU memory before starting
    torch.cuda.empty_cache()
    
    # Preprocess datasets with reduced sequence length
    tokenized_datasets = preprocess_data(datasets, tokenizer, max_length=config.max_len)
    
    # Create dataloaders with minimal batch size
    train_dataloader = DataLoader(
        tokenized_datasets['train'],
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        persistent_workers=False
    )
    
    val_dataloader = DataLoader(
        tokenized_datasets['validation'],
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=0,
        persistent_workers=False
    )
    
    # Initialize optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        eps=config.adam_epsilon
    )
    
    # Training loop
    num_epochs = 3
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)  # More efficient than setting to zero
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(progress_bar):
            # Clear cache at the start of each step
            torch.cuda.empty_cache()
            
            # Process one input at a time to minimize memory usage
            premise_outputs = None
            hypothesis_outputs = None
            
            # Move batch to device with non_blocking transfer
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Mixed precision forward pass with careful memory management
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(
                    batch['premise_input_ids'],
                    batch['premise_attention_mask'],
                    batch['hypothesis_input_ids'],
                    batch['hypothesis_attention_mask']
                )
                loss = criterion(outputs, batch['labels'])
                loss = loss / config.gradient_accumulation_steps
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Force GPU memory cleanup
                torch.cuda.empty_cache()
            
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({'loss': total_loss / (step + 1)})
            
            # Aggressive memory cleanup after each step
            del outputs, loss, batch
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Evaluate
        model.eval()
        metrics = evaluate_model(model, val_dataloader)
        logger.info(f"Validation metrics: {metrics}")
        
        # Save model
        save_model(model, tokenizer, config, metrics)
        
        # Clear cache between epochs
        torch.cuda.empty_cache()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
