import torch
import torch.nn as nn
import re
from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.itos = ['<unk>', '<eos>']
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
    
    def build_vocab(self, text):
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text.lower())
        counter = Counter(tokens)
        
        for token, count in counter.items():
            if count >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
    
    def tokenize(self, text):
        return re.findall(r'\b\w+\b|[.,!?;]', text.lower())
    
    def encode(self, tokens):
        return [self.stoi.get(token, self.stoi['<unk>']) for token in tokens]
    
    def decode(self, indices):
        return [self.itos[idx] for idx in indices]
    
    def __len__(self):
        return len(self.itos)

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
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
    
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device))
    
    def detach_hidden(self, hidden):
        return tuple(h.detach() for h in hidden)
    
    def forward(self, src, hidden):
        # src shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, seq_len, emb_dim]
        output, hidden = self.lstm(embedded, hidden)  # output: [batch_size, seq_len, hid_dim]
        output = self.dropout(output)
        prediction = self.fc(output)  # [batch_size, seq_len, vocab_size]
        return prediction, hidden

def generate_text(prompt, model, vocab, max_len=50, temperature=0.8):
    device = next(model.parameters()).device
    model.eval()
    
    # Tokenize and encode the prompt
    tokens = vocab.tokenize(prompt.lower())
    input_ids = torch.tensor([vocab.encode(tokens)]).to(device)
    hidden = model.init_hidden(1)  # batch_size = 1 for generation
    
    print(f"Input tokens: {tokens}")
    print(f"Input IDs: {input_ids.tolist()[0]}")
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_len):
            # Get predictions
            prediction, hidden = model(input_ids, hidden)
            prediction = prediction[:, -1, :] / temperature
            
            # Apply softmax and sample
            probs = torch.softmax(prediction, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Resample if we get <unk>
            while next_token == vocab.stoi['<unk>']:
                next_token = torch.multinomial(probs, 1).item()
            
            if next_token == vocab.stoi['<eos>']:
                break
                
            generated_tokens.append(next_token)
            input_ids = torch.tensor([[next_token]]).to(device)
    
    # Decode the generated tokens
    generated_words = vocab.decode(generated_tokens)
    print(f"Generated tokens: {generated_words}")
    
    # Join the words with spaces, but handle punctuation properly
    result = []
    for i, word in enumerate(generated_words):
        if word in [',', '.', '!', '?', ';', ':']:
            result.append(word)
        elif i > 0 and generated_words[i-1] not in [',', '.', '!', '?', ';', ':']:
            result.append(' ' + word)
        else:
            result.append(word)
    
    return prompt + ''.join(result)
