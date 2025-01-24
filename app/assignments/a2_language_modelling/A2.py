#!/usr/bin/env python
# coding: utf-8

# #  AT82.05 Artificial Intelligence: Natural Language Understanding (NLU)
# 
# ## A2: Language Model
# 
# ### Name: Arya Shah
# ### StudentID: st125462
# 
# -----------
# 
# In this assignment, we will focus on building a language model using a text dataset of my choice- I make use of the Shakespeare Corpus provided by MIT. The objective is to train a model that can generate coherent and contextually relevant text based on a given input. Additionally, I will develop a simple web application (integrating with the existing web app storing all the assignments for this course) to demonstrate the capabilities of my language model interactively.
# 
# You can find the GitHub Repository for the assignment here: https://github.com/aryashah2k/NLP-NLU

# ## Task 1. Dataset Acquisition - Your first task is to find a suitable text dataset. (1 points) ✅
# 
# Choose your dataset and provide a brief description. Ensure to source this dataset from reputable public databases or repositories. It is imperative to give proper credit to the dataset source in your documentation. ✅
# 
# Note: The dataset can be based on any theme such as Harry Potter, Star Wars, jokes, Isaac Asimov’s works, Thai stories, etc. The key requirement is that the dataset should be text-rich and suitable for language modeling. ✅
# 
# ---------------
# 
# I make use of the following dataset: **The Complete Works of William Shakespeare- Provided by Project Gutenberg**
# 
# Link: https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt

# # Task 2. Model Training - Incorporate the chosen dataset into our existing code framework. ✅
# 
# Train a language model that can understand the context and style of the text.

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import random
import re
from collections import Counter

# Set device and random seed for reproducibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.itos = ['<unk>', '<eos>']
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
    
    def build_vocab(self, text):
        # Simple tokenization by splitting on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text.lower())
        counter = Counter(tokens)
        
        # Add tokens that appear more than min_freq times
        for token, count in counter.items():
            if count >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
    
    def tokenize(self, text):
        # Convert text to tokens
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text.lower())
        return tokens
    
    def encode(self, tokens):
        # Convert tokens to indices
        return [self.stoi.get(token, self.stoi['<unk>']) for token in tokens]
    
    def decode(self, indices):
        # Convert indices back to tokens
        return [self.itos[idx] for idx in indices]
    
    def __len__(self):
        return len(self.itos)

def load_and_preprocess_data(file_path, train_ratio=0.7, val_ratio=0.15):
    print("Loading Shakespeare dataset...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("\nBuilding vocabulary...")
    vocab = Vocabulary(min_freq=2)
    vocab.build_vocab(text)
    
    print("\nTokenizing and encoding data...")
    tokens = vocab.tokenize(text)
    indices = vocab.encode(tokens)
    
    # Split into train, validation, and test sets
    total_len = len(indices)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    
    train_data = indices[:train_len]
    val_data = indices[train_len:train_len + val_len]
    test_data = indices[train_len + val_len:]
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")
    
    return train_data, val_data, test_data, vocab

def batchify(data, batch_size):
    # Work out how many batches we can get from the data
    num_batches = len(data) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit
    data = data[:num_batches * batch_size]
    # Reshape into [batch_size, num_batches]
    data = torch.tensor(data).view(batch_size, -1)
    return data

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, 
                           dropout=dropout_rate if num_layers > 1 else 0,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_dim, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hid_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_other)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(4 * self.hid_dim,
                self.emb_dim if i == 0 else self.hid_dim).uniform_(-init_range_other, init_range_other)
            self.lstm.all_weights[i][1] = torch.FloatTensor(4 * self.hid_dim,
                self.hid_dim).uniform_(-init_range_other, init_range_other)
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hid_dim),
                weight.new_zeros(self.num_layers, batch_size, self.hid_dim))
    
    def detach_hidden(self, hidden):
        return tuple(h.detach() for h in hidden)
    
    def forward(self, src, hidden):
        # src: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, seq_len, emb_dim]
        
        output, hidden = self.lstm(embedded, hidden)
        # output: [batch_size, seq_len, hid_dim]
        # hidden: (h_n, c_n), each [num_layers, batch_size, hid_dim]
        
        output = self.dropout(output)
        prediction = self.fc(output)
        # prediction: [batch_size, seq_len, vocab_size]
        return prediction, hidden

def get_batch(data, seq_len, idx):
    seq_len = min(seq_len, data.size(1) - 1 - idx)
    src = data[:, idx:idx+seq_len].contiguous()
    tgt = data[:, idx+1:idx+1+seq_len].contiguous()
    return src, tgt

def train(model, data, optimizer, criterion, batch_size, seq_len, clip, epoch):
    model.train()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    
    num_batches = (data.size(1) - 1) // seq_len
    progress_bar = tqdm(range(0, data.size(1) - 1, seq_len),
                       desc=f'Epoch {epoch}', total=num_batches)
    
    for idx in progress_bar:
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)
        
        src, tgt = get_batch(data, seq_len, idx)
        src, tgt = src.to(device), tgt.to(device)
        
        prediction, hidden = model(src, hidden)
        
        # Flatten the prediction and target tensors for loss calculation
        prediction = prediction.reshape(-1, prediction.size(-1))
        tgt = tgt.reshape(-1)
        
        loss = criterion(prediction, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})
    
    return total_loss / num_batches

def evaluate(model, data, criterion, batch_size, seq_len):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    num_batches = (data.size(1) - 1) // seq_len
    
    with torch.no_grad():
        for idx in range(0, data.size(1) - 1, seq_len):
            src, tgt = get_batch(data, seq_len, idx)
            src, tgt = src.to(device), tgt.to(device)
            
            prediction, hidden = model(src, hidden)
            hidden = model.detach_hidden(hidden)
            
            # Flatten the prediction and target tensors for loss calculation
            prediction = prediction.reshape(-1, prediction.size(-1))
            tgt = tgt.reshape(-1)
            
            loss = criterion(prediction, tgt)
            total_loss += loss.item()
    
    return total_loss / num_batches

def plot_training_history(train_losses, val_losses, train_ppls, val_ppls):
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot perplexity
    plt.subplot(1, 2, 2)
    plt.plot(train_ppls, label='Train PPL')
    plt.plot(val_ppls, label='Validation PPL')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def generate_text(prompt, model, vocab, max_len=50, temperature=0.8, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    model.eval()
    tokens = vocab.tokenize(prompt.lower())
    input_ids = torch.tensor([vocab.encode(tokens)]).to(device)
    hidden = model.init_hidden(1)
    
    generated_indices = []
    with torch.no_grad():
        for _ in range(max_len):
            prediction, hidden = model(input_ids, hidden)
            prediction = prediction[:, -1, :] / temperature
            probs = torch.softmax(prediction, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Resample if we get <unk>
            while next_token == vocab.stoi['<unk>']:
                next_token = torch.multinomial(probs, 1).item()
            
            if next_token == vocab.stoi['<eos>']:
                break
                
            generated_indices.append(next_token)
            input_ids = torch.tensor([[next_token]]).to(device)
    
    generated_tokens = vocab.decode(generated_indices)
    return ' '.join([prompt] + generated_tokens)

def main():
    # Hyperparameters
    BATCH_SIZE = 64
    SEQ_LEN = 35
    EMB_DIM = 400
    HID_DIM = 1024
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.5
    CLIP = 0.25
    LR = 0.001
    EPOCHS = 25
    
    print(f"Using device: {device}")
    
    # Load and preprocess data
    train_data, valid_data, test_data, vocab = load_and_preprocess_data(
        't8.shakespeare.txt'
    )
    
    print("\nPreparing data batches...")
    train_data = batchify(train_data, BATCH_SIZE).to(device)
    valid_data = batchify(valid_data, BATCH_SIZE).to(device)
    test_data = batchify(test_data, BATCH_SIZE).to(device)
    
    print(f"Train shape: {train_data.shape}")
    print(f"Valid shape: {valid_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    # Initialize model, optimizer, and criterion
    model = LSTMLanguageModel(
        len(vocab), EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT_RATE
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_valid_loss = float('inf')
    train_losses = []
    val_losses = []
    train_ppls = []
    val_ppls = []
    
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_data, optimizer, criterion,
                          BATCH_SIZE, SEQ_LEN, CLIP, epoch+1)
        valid_loss = evaluate(model, valid_data, criterion,
                            BATCH_SIZE, SEQ_LEN)
        
        # Calculate perplexities
        train_ppl = math.exp(train_loss)
        valid_ppl = math.exp(valid_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        train_ppls.append(train_ppl)
        val_ppls.append(valid_ppl)
        
        # Plot training history
        plot_training_history(train_losses, val_losses, train_ppls, val_ppls)
        
        epoch_mins, epoch_secs = divmod(int(time.time() - start_time), 60)
        print(f'\nEpoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {valid_ppl:.3f}')
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_ppl': train_ppl,
                'valid_ppl': valid_ppl,
            }, 'shakespeare_model.pt')
            print(f'\tNew best model saved!')
        
        scheduler.step(valid_loss)
    
    # Load best model and evaluate on test set
    checkpoint = torch.load('shakespeare_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = evaluate(model, test_data, criterion, BATCH_SIZE, SEQ_LEN)
    test_ppl = math.exp(test_loss)
    print(f'\nTest Loss: {test_loss:.3f} | Test PPL: {test_ppl:.3f}')
    
    # Generate text with different temperatures
    print("\nGenerating text with different temperatures...")
    test_prompts = [
        "To be or not",
        "Romeo, Romeo",
        "All the world's",
        "Friends, Romans, countrymen"
    ]
    
    temperatures = [0.2, 0.5, 0.7, 0.8, 1.0]
    seed = 42  # for reproducibility
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        for temp in temperatures:
            generated = generate_text(
                prompt, model, vocab, 
                max_len=100, temperature=temp, 
                seed=seed
            )
            print(f"Temperature {temp:.2f}:")
            print(f"{generated}\n")

if __name__ == "__main__":
    main()


# In[ ]:




