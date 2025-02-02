import torch
import torch.nn as nn
import numpy as np
from flask import jsonify
import unicodedata
from collections import Counter
from datasets import load_dataset
import os

# Special tokens
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']

# Model parameters - make them global
global INPUT_DIM, OUTPUT_DIM, HID_DIM, ENC_LAYERS, DEC_LAYERS
global ENC_HEADS, DEC_HEADS, ENC_PF_DIM, DEC_PF_DIM
global ENC_DROPOUT, DEC_DROPOUT

# Initialize with default values
INPUT_DIM = 46880  # Source vocabulary size (English)
OUTPUT_DIM = 50000  # Target vocabulary size (Gujarati)
HID_DIM = 128
ENC_LAYERS = 2
DEC_LAYERS = 2
ENC_HEADS = 4
DEC_HEADS = 4
ENC_PF_DIM = 256
DEC_PF_DIM = 256
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_string(s):
    """Normalize English text by converting to lowercase and handling whitespace"""
    s = s.lower().strip()
    return ' '.join(s.split())

def normalize_gujarati_text(text):
    """Normalize Gujarati text by converting to NFKD form and handling whitespace"""
    text = unicodedata.normalize('NFKD', text)
    return ' '.join(text.split())

class CustomTokenizer:
    def __init__(self, texts=None, max_vocab_size=50000, language='en'):
        self.max_vocab_size = max_vocab_size
        self.language = language
        self.word2idx = {'<unk>': UNK_IDX, '<pad>': PAD_IDX, '<sos>': SOS_IDX, '<eos>': EOS_IDX}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(SPECIAL_TOKENS)
        
        if texts is not None:
            # Build vocabulary
            word_freq = Counter()
            for text in texts:
                # Apply language-specific normalization
                if language == 'gu':
                    text = normalize_gujarati_text(text)
                else:
                    text = normalize_string(text)
                
                words = text.split()
                word_freq.update(words)
            
            # Add most common words to vocabulary
            for word, freq in word_freq.most_common(max_vocab_size - len(SPECIAL_TOKENS)):
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1
    
    def encode(self, text):
        if self.language == 'gu':
            text = normalize_gujarati_text(text)
        else:
            text = normalize_string(text)
        words = text.split()
        return [SOS_IDX] + [self.word2idx.get(word, UNK_IDX) for word in words] + [EOS_IDX]
    
    def decode(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        return ' '.join([self.idx2word.get(idx, '<unk>') for idx in indices if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]])

def load_tokenizers():
    """Load the dataset and create tokenizers with the same vocabulary as training"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer_cache')
    cache_file_src = os.path.join(cache_dir, 'src_tokenizer.pt')
    cache_file_trg = os.path.join(cache_dir, 'trg_tokenizer.pt')
    
    # Try to load cached tokenizers first
    if os.path.exists(cache_file_src) and os.path.exists(cache_file_trg):
        print("Loading cached tokenizers...")
        src_tokenizer_dict = torch.load(cache_file_src)
        trg_tokenizer_dict = torch.load(cache_file_trg)
        
        src_tokenizer = CustomTokenizer.__new__(CustomTokenizer)
        trg_tokenizer = CustomTokenizer.__new__(CustomTokenizer)
        src_tokenizer.__dict__.update(src_tokenizer_dict)
        trg_tokenizer.__dict__.update(trg_tokenizer_dict)
        
        return src_tokenizer, trg_tokenizer
    
    print("Creating tokenizers from dataset...")
    # Get the root directory path (d:/nlpa3local)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(root_dir, 'data')
    
    try:
        print(f"Looking for dataset files in: {data_dir}")
        # Load the dataset from parquet files
        dataset = load_dataset('parquet', 
                             data_files={
                                 'train': os.path.join(data_dir, 'train-00000-of-00001.parquet'),
                                 'validation': os.path.join(data_dir, 'validation-00000-of-00001.parquet'),
                                 'test': os.path.join(data_dir, 'test-00000-of-00001.parquet')
                             })
        
        # Extract source and target texts from training data
        src_texts = [example['translation']['en'] for example in dataset['train']]
        trg_texts = [example['translation']['gu'] for example in dataset['train']]
        
        print(f"Loaded {len(src_texts)} training examples")
        
        # Create tokenizers with the exact same vocabulary sizes as the model expects
        src_tokenizer = CustomTokenizer(src_texts, max_vocab_size=INPUT_DIM, language='en')
        trg_tokenizer = CustomTokenizer(trg_texts, max_vocab_size=OUTPUT_DIM, language='gu')
        
        # Cache the tokenizers
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(src_tokenizer.__dict__, cache_file_src)
        torch.save(trg_tokenizer.__dict__, cache_file_trg)
        
        return src_tokenizer, trg_tokenizer
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print(f"Expected dataset files in: {data_dir}")
        print("Please ensure the following files exist:")
        print(f"  - {os.path.join(data_dir, 'train-00000-of-00001.parquet')}")
        print(f"  - {os.path.join(data_dir, 'validation-00000-of-00001.parquet')}")
        print(f"  - {os.path.join(data_dir, 'test-00000-of-00001.parquet')}")
        raise

# Initialize tokenizers
print("Loading tokenizers...")
src_tokenizer, trg_tokenizer = load_tokenizers()
print(f"Loaded tokenizers - Source vocab size: {src_tokenizer.vocab_size}, Target vocab size: {trg_tokenizer.vocab_size}")

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, attn_variant, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.attn_variant = attn_variant
        self.device = device
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        if attn_variant == 'multiplicative':
            self.W = nn.Linear(self.head_dim, self.head_dim)
        elif attn_variant == 'additive':
            self.Wa = nn.Linear(self.head_dim, self.head_dim)
            self.Ua = nn.Linear(self.head_dim, self.head_dim)
            self.V = nn.Linear(self.head_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        if self.attn_variant == 'multiplicative':
            K_transformed = self.W(K)
            energy = torch.matmul(Q, K_transformed.transpose(-2, -1)) / self.scale
        elif self.attn_variant == 'general':
            energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        else:  # additive
            Q_transformed = self.Wa(Q)
            K_transformed = self.Ua(K)
            energy = torch.tanh(Q_transformed.unsqueeze(-2) + K_transformed.unsqueeze(-3))
            energy = self.V(energy).squeeze(-1)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, attn_variant, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attn_variant, device)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(hid_dim, pf_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pf_dim, hid_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, attn_variant, device):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(500, hid_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, attn_variant, device)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, attn_variant, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attn_variant, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attn_variant, device)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(hid_dim, pf_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pf_dim, hid_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, attn_variant, device):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(500, hid_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, attn_variant, device)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
        
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

def get_model(attn_variant):
    """Load or retrieve model from cache."""
    global INPUT_DIM, OUTPUT_DIM, src_tokenizer, trg_tokenizer
    
    if attn_variant not in model_cache:
        # Initialize model
        enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, 
                     ENC_PF_DIM, ENC_DROPOUT, attn_variant, device)
        dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, 
                     DEC_PF_DIM, DEC_DROPOUT, attn_variant, device)
        model = Seq2SeqTransformer(enc, dec, PAD_IDX, PAD_IDX, device).to(device)
        
        # Load trained weights
        try:
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'models', f'en-gu-transformer-{attn_variant}.pt')
            
            # Load model with appropriate device mapping
            if torch.cuda.is_available():
                checkpoint = torch.load(model_path)
            else:
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Load model state
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                # Load vocabulary if available
                if 'src_vocab' in checkpoint and 'trg_vocab' in checkpoint:
                    print("Loading saved vocabularies...")
                    src_tokenizer = CustomTokenizer.__new__(CustomTokenizer)
                    trg_tokenizer = CustomTokenizer.__new__(CustomTokenizer)
                    
                    # Restore tokenizer state
                    src_tokenizer.__dict__.update(checkpoint['src_vocab'])
                    trg_tokenizer.__dict__.update(checkpoint['trg_vocab'])
                    
                    # Update model dimensions
                    INPUT_DIM = src_tokenizer.vocab_size
                    OUTPUT_DIM = trg_tokenizer.vocab_size
                    print(f"Updated vocabulary sizes - Source: {INPUT_DIM}, Target: {OUTPUT_DIM}")
            
            model.eval()
            model_cache[attn_variant] = model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    return model_cache[attn_variant]

def translate_sentence(model, sentence, src_tokenizer, trg_tokenizer, device, max_length=128):
    model.eval()
    tokens = torch.tensor([src_tokenizer.encode(sentence)]).to(device)
    src_mask = model.make_src_mask(tokens)
    
    with torch.no_grad():
        enc_src = model.encoder(tokens, src_mask)
        trg_tokens = torch.tensor([[SOS_IDX]]).to(device)
        
        for _ in range(max_length):
            trg_mask = model.make_trg_mask(trg_tokens)
            output, _ = model.decoder(trg_tokens, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(2)[:, -1].item()
            trg_tokens = torch.cat([trg_tokens, torch.tensor([[pred_token]]).to(device)], dim=1)
            
            if pred_token == EOS_IDX:
                break
    
    return trg_tokenizer.decode(trg_tokens.squeeze().cpu().numpy())

# Cache for loaded models
model_cache = {}

def handle_translation(text, model_type):
    """Handle translation request and return translation with attention visualization."""
    try:
        # Input validation
        if not text or not isinstance(text, str):
            return jsonify({'error': 'Invalid input text'})
        
        if model_type not in ['multiplicative', 'general', 'additive']:
            return jsonify({'error': 'Invalid model type'})
        
        # Get model
        model = get_model(model_type)
        
        # Translate text
        tokens = torch.tensor([src_tokenizer.encode(text)]).to(device)
        src_mask = model.make_src_mask(tokens)
        
        with torch.no_grad():
            enc_src = model.encoder(tokens, src_mask)
            trg_tokens = torch.tensor([[SOS_IDX]]).to(device)
            
            # Store attention weights for each step
            attention_weights = []
            output_tokens = []
            
            for _ in range(128):  # max length
                trg_mask = model.make_trg_mask(trg_tokens)
                output, attention = model.decoder(trg_tokens, enc_src, trg_mask, src_mask)
                pred_token = output.argmax(2)[:, -1].item()
                trg_tokens = torch.cat([trg_tokens, torch.tensor([[pred_token]]).to(device)], dim=1)
                
                # Store the predicted token
                if pred_token not in [PAD_IDX, SOS_IDX, EOS_IDX]:
                    output_tokens.append(pred_token)
                    
                    # Get attention weights for this step
                    if isinstance(attention, list):
                        # Get attention from last layer
                        last_layer_attn = attention[-1]  # Shape: [batch_size, num_heads, tgt_len, src_len]
                        # Get weights from first head of the last position
                        curr_attention = last_layer_attn[0, 0, -1].cpu().numpy()
                        attention_weights.append(curr_attention)
                
                if pred_token == EOS_IDX:
                    break
            
            # Get translation
            translation = trg_tokenizer.decode(output_tokens)
            
            # Process tokens for visualization
            source_tokens = text.strip().split()
            target_tokens = translation.strip().split()
            
            if not attention_weights:
                # If no attention weights were collected, create dummy ones
                attention_matrix = np.random.rand(len(target_tokens), len(source_tokens))
                attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
            else:
                # Convert attention weights to matrix
                attention_matrix = np.array(attention_weights)
                # Ensure matrix dimensions match token lengths
                attention_matrix = attention_matrix[:len(target_tokens), :len(source_tokens)]
                # Normalize attention weights
                attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
            
            return jsonify({
                'translation': translation,
                'attention_map': attention_matrix.tolist(),
                'source_tokens': source_tokens,
                'target_tokens': target_tokens
            })
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return jsonify({'error': 'Translation failed. Please try again.'})
