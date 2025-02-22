import math
import re
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm
import os
import json
from datetime import datetime
from transformers import get_linear_schedule_with_warmup, BertTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bert_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_free_gpu():
    """Get the GPU with the most available memory."""
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    # Get the number of GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return torch.device('cpu')
    
    # Find GPU with most free memory
    max_free_memory = 0
    selected_gpu = 0
    
    for gpu_id in range(n_gpus):
        try:
            # Get free memory for this GPU
            free_memory = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                selected_gpu = gpu_id
        except:
            continue
    
    device = torch.device(f'cuda:{selected_gpu}')
    logger.info(f"Selected GPU {selected_gpu} with {max_free_memory/1024/1024:.2f}MB free memory")
    return device

# Set device and seeds for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = get_free_gpu()
logger.info(f"Using device: {device}")

# Model configuration
class BertConfig:
    def __init__(self):
        # Model architecture
        self.vocab_size = None  # Will be set after data loading
        self.hidden_size = 256
        self.num_hidden_layers = 6
        self.num_attention_heads = 8
        self.intermediate_size = 1024
        
        # Dropout and normalization
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.layer_norm_eps = 1e-12
        
        # Sequence parameters
        self.max_position_embeddings = 128
        self.max_len = 128
        self.type_vocab_size = 2
        self.pad_token_id = 0
        
        # Special tokens
        self.mask_token_id = 3
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.pad_token_id = 0
        
        # Training parameters
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.gradient_accumulation_steps = 4
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.warmup_ratio = 0.1

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize with smaller values for stability
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in [self.query, self.key, self.value]:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertSelfAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm1 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm2 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = F.gelu
        
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm1(attention_output + hidden_states)
        
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.activation(intermediate_output)
        
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm2(layer_output + attention_output)
        
        return layer_output

class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        
        # MLM head
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Enable gradient checkpointing for memory efficiency
        self.gradient_checkpointing = False
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        hidden_states = embedding_output
        
        for layer in self.encoder:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    extended_attention_mask,
                )
            else:
                hidden_states = layer(hidden_states, extended_attention_mask)
            
        # MLM loss calculation
        if masked_lm_labels is not None:
            prediction_scores = self.mlm_head(hidden_states)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), 
                                    masked_lm_labels.view(-1))
            return masked_lm_loss
            
        return hidden_states
    
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

def pad_sequence(tokens, max_len, pad_token):
    """Pad or truncate a sequence to max_len."""
    if len(tokens) > max_len:
        return tokens[:max_len]
    return tokens + [pad_token] * (max_len - len(tokens))

def prepare_batch(texts, word2idx, config):
    """Prepare a batch of texts for BERT training."""
    # Convert texts to token ids and pad
    batch_input_ids = []
    for text in texts:
        tokens = text.split()
        token_ids = [word2idx.get(word, word2idx['[UNK]']) for word in tokens]
        # Add [CLS] at start and [SEP] at end
        token_ids = [word2idx['[CLS]']] + token_ids + [word2idx['[SEP]']]
        # Pad sequence
        padded_ids = pad_sequence(token_ids, config.max_len, word2idx['[PAD]'])
        batch_input_ids.append(padded_ids)
    
    # Convert to tensor
    input_ids = torch.tensor(batch_input_ids).to(device)
    attention_mask = (input_ids != word2idx['[PAD]']).float()
    
    return input_ids, attention_mask

def load_and_preprocess_data():
    logger.info("Loading BookCorpus dataset...")
    # Load only 100k samples as specified
    dataset = load_dataset('bookcorpus', split='train[:100000]')
    
    logger.info("Preprocessing text data...")
    texts = dataset['text']
    texts = [text.lower() for text in texts]
    texts = [re.sub(r'[^\w\s]', '', text) for text in texts]
    
    # Create vocabulary
    logger.info("Creating vocabulary...")
    word_set = set()
    for text in texts:
        words = text.split()
        word_set.update(words)
    
    # Add special tokens
    vocab = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]'] + list(word_set)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    return texts, word2idx, vocab

def save_model_and_config(model, config, epoch, loss, save_dir='model_checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(save_dir, f'bert_epoch_{epoch}_{timestamp}.pt')
    config_path = os.path.join(save_dir, f'config_{timestamp}.json')
    
    # Save model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, model_path)
    
    # Save config
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Config saved to {config_path}")

def main():
    logger.info("Starting BERT training from scratch")
    
    # Enable mixed precision training with proper initialization
    scaler = torch.amp.GradScaler(enabled=True)
    
    # Load and preprocess data
    texts, word2idx, vocab = load_and_preprocess_data()
    
    # Initialize config
    config = BertConfig()
    config.vocab_size = len(vocab)
    logger.info(f"Vocabulary size: {config.vocab_size}")
    
    # Initialize model with gradient checkpointing
    model = BertModel(config).to(device)
    model.enable_gradient_checkpointing()
    logger.info("Model initialized with gradient checkpointing")
    
    # Initialize optimizer with weight decay and proper learning rate
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=1e-8)
    
    # Add learning rate scheduler with warmup
    num_training_steps = len(texts) // (config.batch_size * config.gradient_accumulation_steps) * 5
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    
    try:
        for epoch in range(5):
            total_loss = 0
            valid_loss_count = 0
            optimizer.zero_grad()  # Reset gradients at start of epoch
            
            progress_bar = tqdm(range(0, len(texts), config.batch_size), 
                              desc=f"Epoch {epoch+1}")
            
            for step, batch_start in enumerate(progress_bar):
                batch_texts = texts[batch_start:batch_start + config.batch_size]
                
                # Prepare batch data with padding
                input_ids, attention_mask = prepare_batch(batch_texts, word2idx, config)
                
                # Create masked tokens
                masked_labels = input_ids.clone()
                special_tokens = {word2idx['[PAD]'], word2idx['[CLS]'], word2idx['[SEP]']}
                mask_candidates = torch.ones_like(input_ids, device=device).bool()
                for special_token in special_tokens:
                    mask_candidates &= (input_ids != special_token)
                
                # Apply masking with 15% probability to valid tokens
                mask_prob = torch.full(input_ids.shape, 0.15, device=device)
                mask = (torch.bernoulli(mask_prob).bool() & mask_candidates)
                
                masked_labels[~mask] = -1  # Only compute loss on masked tokens
                input_ids[mask] = word2idx['[MASK]']
                
                try:
                    # Mixed precision forward pass
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        loss = model(
                            input_ids,
                            attention_mask=attention_mask,
                            masked_lm_labels=masked_labels
                        )
                        
                        # Scale loss for gradient accumulation
                        loss = loss / config.gradient_accumulation_steps
                        
                        # Check if loss is valid
                        if not torch.isfinite(loss):
                            logger.warning(f"Non-finite loss detected: {loss.item()}")
                            continue
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Gradient accumulation
                    if (step + 1) % config.gradient_accumulation_steps == 0:
                        # Clip gradients
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # Optimizer step
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    # Update loss statistics
                    loss_value = loss.item() * config.gradient_accumulation_steps
                    if np.isfinite(loss_value):
                        total_loss += loss_value
                        valid_loss_count += 1
                    
                    progress_bar.set_postfix({
                        'loss': loss_value,
                        'lr': scheduler.get_last_lr()[0]
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"Out of memory in batch. Skipping batch and clearing cache.")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    raise e
                
                # Clear cache periodically
                if step % 100 == 0 and hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            
            # Calculate average loss only from valid losses
            avg_loss = total_loss / valid_loss_count if valid_loss_count > 0 else float('nan')
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f} (from {valid_loss_count} valid batches)")
            
            # Save model checkpoint
            save_model_and_config(model, config, epoch+1, avg_loss)
        
        logger.info("Training completed!")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"GPU out of memory error: {e}")
            logger.info("Try reducing batch_size or model size further if this error persists")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        raise e

if __name__ == "__main__":
    main()
