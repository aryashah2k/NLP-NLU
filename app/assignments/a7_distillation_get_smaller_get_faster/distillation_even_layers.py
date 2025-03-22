#!/usr/bin/env python
# coding: utf-8

"""
Implement knowledge distillation from BERT model (12 layers) to a smaller model (6 layers)
using the even-numbered layers {2, 4, 6, 8, 10, 12} from the teacher model.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    get_scheduler,
    AutoTokenizer
)
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
from torch.nn import Module
import evaluate
from data_loader import load_toxic_comments_dataset

# Set random seed for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DistillKL(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    """
    def __init__(self):
        super(DistillKL, self).__init__()

    def forward(self, output_student, output_teacher, temperature=1):
        """
        Args:
            output_student: Student model output (logits)
            output_teacher: Teacher model output (logits)
            temperature: Temperature for softening probability distributions
        
        Returns:
            KL divergence loss
        """
        T = temperature
        
        KD_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(output_student/T, dim=-1),
            F.softmax(output_teacher/T, dim=-1)
        ) * T * T
        
        return KD_loss

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def distill_bert_weights_even_layers(teacher, student):
    """
    Copy weights from teacher to student, selecting even-numbered layers from the teacher.
    
    Args:
        teacher: Teacher model
        student: Student model
        
    Returns:
        Student model with initialized weights
    """
    # If the part is an entire BERT model or a BERTFor..., unpack and iterate
    if isinstance(teacher, BertModel) or type(teacher).__name__.startswith('BertFor'):
        for teacher_part, student_part in zip(teacher.children(), student.children()):
            distill_bert_weights_even_layers(teacher_part, student_part)
    # Else if the part is an encoder, copy even-numbered layers from teacher to student
    elif isinstance(teacher, BertEncoder):
        teacher_encoding_layers = [layer for layer in next(teacher.children())]  # 12 layers
        student_encoding_layers = [layer for layer in next(student.children())]  # 6 layers
        
        # Map even-numbered layers (1-indexed) from teacher to student
        # In 0-indexed arrays, this means layers 1, 3, 5, 7, 9, 11
        even_layer_indices = [1, 3, 5, 7, 9, 11]
        
        for i, teacher_idx in enumerate(even_layer_indices):
            student_encoding_layers[i].load_state_dict(teacher_encoding_layers[teacher_idx].state_dict())
            print(f"Copied teacher layer {teacher_idx} to student layer {i}")
    # Else the part is a head or something else, copy the state_dict
    else:
        student.load_state_dict(teacher.state_dict())

    return student

def train_and_evaluate_distillation(
    teacher_model_name="bert-base-uncased",
    num_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    temperature=2.0,
    batch_size=32,
    max_length=128,
    output_dir="even_layer_model"
):
    """
    Train and evaluate a distilled model with even-numbered layers.
    
    Args:
        teacher_model_name: Name of the teacher model
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        temperature: Temperature for knowledge distillation
        batch_size: Batch size for training
        max_length: Maximum sequence length
        output_dir: Directory to save model and results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    tokenizer, train_dataloader, eval_dataloader, test_dataloader, num_labels, _, _, _ = load_toxic_comments_dataset(
        model_name_or_path=teacher_model_name,
        max_length=max_length,
        batch_size=batch_size
    )
    
    # Load teacher model
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        teacher_model_name,
        num_labels=num_labels
    )
    
    print(f"Teacher model loaded: {teacher_model_name}")
    print(f"Teacher model config: {teacher_model.config}")
    
    # Create student model configuration
    configuration = teacher_model.config.to_dict()
    # Half the number of hidden layers
    configuration['num_hidden_layers'] //= 2
    configuration = BertConfig.from_dict(configuration)
    
    # Create uninitialized student model
    student_model = type(teacher_model)(configuration)
    
    # Initialize student model with even layers from teacher
    student_model = distill_bert_weights_even_layers(teacher=teacher_model, student=student_model)
    
    # Print model sizes
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)
    print(f'Teacher parameters: {teacher_params:,}')
    print(f'Student parameters: {student_params:,}')
    print(f'Parameter reduction: {(1 - student_params/teacher_params) * 100:.2f}%')
    
    # Move models to device
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # Set up loss functions
    criterion_div = DistillKL()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_cos = nn.CosineEmbeddingLoss()
    
    # Set up optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        params=student_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Calculate number of training steps
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_epochs * num_update_steps_per_epoch
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # Set up metric for evaluation
    metric = evaluate.load("accuracy")
    
    # Training and evaluation
    progress_bar = tqdm(range(num_training_steps))
    
    # Lists to store losses and metrics
    train_losses = []
    train_losses_cls = []
    train_losses_div = []
    train_losses_cos = []
    eval_losses = []
    eval_accuracies = []
    
    best_eval_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Training
        student_model.train()
        teacher_model.eval()
        
        train_loss = 0
        train_loss_cls = 0
        train_loss_div = 0
        train_loss_cos = 0
        
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Compute student output
            outputs = student_model(**batch)
            
            # Compute teacher output
            with torch.no_grad():
                output_teacher = teacher_model(**batch)
            
            # assert size
            assert outputs.logits.size() == output_teacher.logits.size()
            
            # Classification loss
            loss_cls = outputs.loss
            train_loss_cls += loss_cls.item()
            
            # Distillation loss (KL divergence)
            loss_div = criterion_div(outputs.logits, output_teacher.logits, temperature)
            train_loss_div += loss_div.item()
            
            # Cosine embedding loss
            loss_cos = criterion_cos(
                output_teacher.logits,
                outputs.logits,
                torch.ones(output_teacher.logits.size()[0]).to(device)
            )
            train_loss_cos += loss_cos.item()
            
            # Combined loss
            loss = (loss_cls + loss_div + loss_cos) / 3
            train_loss += loss.item()
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        # Calculate average losses for epoch
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_loss_cls = train_loss_cls / len(train_dataloader)
        avg_train_loss_div = train_loss_div / len(train_dataloader)
        avg_train_loss_cos = train_loss_cos / len(train_dataloader)
        
        # Store losses
        train_losses.append(avg_train_loss)
        train_losses_cls.append(avg_train_loss_cls)
        train_losses_div.append(avg_train_loss_div)
        train_losses_cos.append(avg_train_loss_cos)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train loss: {avg_train_loss:.4f}')
        print(f'  - Loss_cls: {avg_train_loss_cls:.4f}')
        print(f'  - Loss_div: {avg_train_loss_div:.4f}')
        print(f'  - Loss_cos: {avg_train_loss_cos:.4f}')
        
        # Evaluation
        student_model.eval()
        eval_loss = 0
        
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = student_model(**batch)
            
            loss_cls = outputs.loss
            eval_loss += loss_cls.item()
            
            # Get predictions
            predictions = outputs.logits.argmax(dim=-1)
            
            # Add batch to metric
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"]
            )
        
        # Calculate average evaluation loss and accuracy
        avg_eval_loss = eval_loss / len(eval_dataloader)
        eval_losses.append(avg_eval_loss)
        
        eval_metric = metric.compute()
        eval_accuracy = eval_metric['accuracy']
        eval_accuracies.append(eval_accuracy)
        
        print(f'  Eval loss: {avg_eval_loss:.4f}')
        print(f'  Eval accuracy: {eval_accuracy:.4f}')
        
        # Save the best model
        if eval_accuracy > best_eval_accuracy:
            best_eval_accuracy = eval_accuracy
            student_model.save_pretrained(os.path.join(output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
            print(f'  New best model saved! Accuracy: {best_eval_accuracy:.4f}')
    
    # Final test evaluation
    student_model.eval()
    test_loss = 0
    metric = evaluate.load("accuracy")
    
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = student_model(**batch)
        
        loss = outputs.loss
        test_loss += loss.item()
        
        # Get predictions
        predictions = outputs.logits.argmax(dim=-1)
        
        # Add batch to metric
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"]
        )
    
    # Calculate average test loss and accuracy
    avg_test_loss = test_loss / len(test_dataloader)
    test_metric = metric.compute()
    test_accuracy = test_metric['accuracy']
    
    print(f'Test Results:')
    print(f'  Test loss: {avg_test_loss:.4f}')
    print(f'  Test accuracy: {test_accuracy:.4f}')
    
    # Save test results
    test_results = {
        'test_loss': avg_test_loss,
        'test_accuracy': test_accuracy,
        'teacher_params': teacher_params,
        'student_params': student_params,
        'parameter_reduction': (1 - student_params/teacher_params) * 100
    }
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_losses_cls': train_losses_cls,
        'train_losses_div': train_losses_div,
        'train_losses_cos': train_losses_cos,
        'eval_losses': eval_losses,
        'eval_accuracies': eval_accuracies,
    }
    
    # Plot and save training curves
    epochs_list = range(1, num_epochs + 1)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_list, train_losses, label='Total Train Loss')
    plt.plot(epochs_list, train_losses_cls, label='Train Loss_cls')
    plt.plot(epochs_list, train_losses_div, label='Train Loss_div')
    plt.plot(epochs_list, train_losses_cos, label='Train Loss_cos')
    plt.plot(epochs_list, eval_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs_list, eval_accuracies, label='Validation Accuracy')
    plt.axhline(y=test_accuracy, color='r', linestyle='--', label=f'Test Accuracy: {test_accuracy:.4f}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    
    # Save final model
    student_model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    # Save results as text file
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write('Test Results:\n')
        f.write(f'  Test loss: {avg_test_loss:.4f}\n')
        f.write(f'  Test accuracy: {test_accuracy:.4f}\n\n')
        f.write(f'Model Information:\n')
        f.write(f'  Teacher parameters: {teacher_params:,}\n')
        f.write(f'  Student parameters: {student_params:,}\n')
        f.write(f'  Parameter reduction: {(1 - student_params/teacher_params) * 100:.2f}%\n')
    
    return student_model, test_results, history

if __name__ == "__main__":
    # Train and evaluate the distilled model with even layers
    student_model, test_results, history = train_and_evaluate_distillation(
        teacher_model_name="bert-base-uncased",
        num_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        temperature=2.0,
        batch_size=32,
        max_length=128,
        output_dir="even_layer_model"
    )
    
    print("Distillation with even layers completed successfully!")
    print(f"Test accuracy: {test_results['test_accuracy']:.4f}")
