#!/usr/bin/env python
# coding: utf-8

"""
Data loader for the hate speech/toxic comment classification task.
This script loads the Jigsaw Toxic Comment Classification dataset and preprocesses it for training.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_toxic_comments_dataset(model_name_or_path="bert-base-uncased", max_length=128, batch_size=32):
    """
    Load the Jigsaw Toxic Comment Classification dataset and preprocess it.
    
    Args:
        model_name_or_path (str): The model name or path for the tokenizer
        max_length (int): Maximum sequence length
        batch_size (int): Batch size for data loaders
    
    Returns:
        tokenizer: The tokenizer used for preprocessing
        train_dataloader: DataLoader for training data
        eval_dataloader: DataLoader for evaluation data
        test_dataloader: DataLoader for test data
        num_labels: Number of labels in the dataset
    """
    print("Loading Jigsaw Toxic Comment Classification dataset...")
    
    # Load the Civil Comments dataset (contains toxic comments)
    dataset = load_dataset("civil_comments")
    
    # Extract the relevant columns (text and toxicity label)
    dataset = dataset.map(
        lambda example: {
            "text": example["text"], 
            "label": 1 if example["toxicity"] > 0.5 else 0
        }
    )
    
    # Create a balanced dataset (50% toxic, 50% non-toxic)
    toxic_comments = dataset["train"].filter(lambda example: example["label"] == 1)
    non_toxic_comments = dataset["train"].filter(lambda example: example["label"] == 0)
    
    # Sample to ensure balance
    max_samples = min(len(toxic_comments), len(non_toxic_comments), 25000)  # Limit to 25k per class for efficiency
    toxic_samples = toxic_comments.select(range(max_samples))
    non_toxic_samples = non_toxic_comments.select(range(max_samples))
    
    # Combine and shuffle
    from datasets import concatenate_datasets
    balanced_dataset = concatenate_datasets([toxic_samples, non_toxic_samples])
    balanced_dataset = balanced_dataset.shuffle(seed=SEED)
    
    # Split into train, validation, and test sets (80%, 10%, 10%)
    train_val_dataset, test_dataset = balanced_dataset.train_test_split(test_size=0.1, seed=SEED).values()
    train_dataset, val_dataset = train_val_dataset.train_test_split(test_size=0.11, seed=SEED).values()  # 0.11 of 90% is ~10% of total
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # Format datasets for PyTorch
    # Remove all columns except those needed by the model
    columns_to_keep = ['input_ids', 'attention_mask', 'label']
    tokenized_train = tokenized_train.remove_columns([col for col in tokenized_train.column_names if col not in columns_to_keep])
    tokenized_val = tokenized_val.remove_columns([col for col in tokenized_val.column_names if col not in columns_to_keep])
    tokenized_test = tokenized_test.remove_columns([col for col in tokenized_test.column_names if col not in columns_to_keep])
    
    # Rename 'label' to 'labels' to match model expectations
    tokenized_train = tokenized_train.rename_column('label', 'labels')
    tokenized_val = tokenized_val.rename_column('label', 'labels')
    tokenized_test = tokenized_test.rename_column('label', 'labels')
    
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")
    tokenized_test.set_format("torch")
    
    # Create data loaders
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_dataloader = DataLoader(
        tokenized_train, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_val, batch_size=batch_size, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_test, batch_size=batch_size, collate_fn=data_collator
    )
    
    # Number of labels (binary classification - toxic or not)
    num_labels = 2
    
    return tokenizer, train_dataloader, eval_dataloader, test_dataloader, num_labels, train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    # Test the function
    tokenizer, train_dataloader, eval_dataloader, test_dataloader, num_labels, _, _, _ = load_toxic_comments_dataset()
    
    print(f"Number of labels: {num_labels}")
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of evaluation batches: {len(eval_dataloader)}")
    print(f"Number of test batches: {len(test_dataloader)}")
    
    # Check a batch
    for batch in train_dataloader:
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        break
