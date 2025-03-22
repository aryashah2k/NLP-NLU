#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Toxic Comment Classifier

A standalone script that uses a pre-trained Hugging Face model to classify
text as toxic or non-toxic and visualize the results.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Default model to use (replace with your model)
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

class ToxicClassifier:
    def __init__(self, model_name_or_path):
        """Initialize the classifier with a pre-trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            print(f"Loading model: {model_name_or_path}")
            
            # Load model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
            
            # Set labels based on the model
            if hasattr(self.model.config, 'id2label'):
                self.id2label = self.model.config.id2label
            else:
                # Default for binary toxicity classification
                self.id2label = {0: "Non-Toxic", 1: "Toxic"}
            
            print(f"Label mapping: {self.id2label}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def classify(self, text, max_length=128):
        """Classify a text as toxic or non-toxic."""
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1).item()
        
        # Get label name
        label = self.id2label[prediction]
        
        # Return classification results
        result = {
            'text': text,
            'prediction': prediction,
            'label': label,
            'probabilities': probabilities.cpu().numpy()[0],
            'confidence': probabilities.cpu().numpy()[0][prediction],
            'id2label': self.id2label
        }
        
        return result
    
    def batch_classify(self, texts, max_length=128):
        """Classify multiple texts."""
        results = []
        for text in texts:
            result = self.classify(text, max_length)
            results.append(result)
        
        return results

def visualize_result(result, display=True, save_path=None):
    """
    Visualize classification result with a color-coded confidence bar.
    
    Args:
        result: Classification result dictionary
        display: Whether to display the plot
        save_path: Path to save the visualization image
    """
    # Create color gradient from green (non-toxic) to red (toxic)
    cmap = LinearSegmentedColormap.from_list('toxicity', ['green', 'yellow', 'red'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 3]})
    
    # Plot text and result
    truncated_text = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
    ax1.text(0.5, 0.5, f"Text: {truncated_text}\n\nClassification: {result['label']}\nConfidence: {result['confidence']:.2%}", 
             horizontalalignment='center', verticalalignment='center', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    ax1.set_axis_off()
    
    # Plot probability bars
    labels = list(result['id2label'].values())
    probs = result['probabilities']
    colors = [cmap(p) if i == 1 else cmap(1-p) for i, p in enumerate(probs)]
    
    ax2.barh(labels, probs, color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Probability')
    ax2.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Annotate bars with percentage
    for i, p in enumerate(probs):
        ax2.text(max(p + 0.01, 0.1), i, f"{p:.2%}", va='center')
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
    
    # Display if requested
    if display:
        plt.show()
    
    return fig

def visualize_batch_results(results, display=True, save_path=None):
    """
    Visualize batch classification results with a comparison chart.
    
    Args:
        results: List of classification result dictionaries
        display: Whether to display the plot
        save_path: Path to save the visualization image
    """
    # Create color gradient
    cmap = LinearSegmentedColormap.from_list('toxicity', ['green', 'yellow', 'red'])
    
    # Prepare data
    texts = [r['text'][:50] + "..." if len(r['text']) > 50 else r['text'] for r in results]
    toxic_probs = [r['probabilities'][1] for r in results]
    labels = [r['label'] for r in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(results) * 0.5)))
    
    # Plot bars
    y_pos = np.arange(len(texts))
    bars = ax.barh(y_pos, toxic_probs, color=[cmap(p) for p in toxic_probs])
    
    # Add labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(texts)
    ax.set_xlabel('Toxicity Probability')
    ax.set_title('Toxicity Classification Comparison')
    ax.set_xlim(0, 1)
    
    # Add threshold line
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Add annotations
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{toxic_probs[i]:.2%} - {labels[i]}", va='center')
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
    
    # Display if requested
    if display:
        plt.show()
    
    return fig

def interactive_mode(classifier):
    """Run an interactive classification session."""
    print("\n===== Toxic Comment Classifier =====")
    print("Enter comments to classify (or 'quit' to exit).")
    
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'quit':
            break
        
        result = classifier.classify(text)
        
        print(f"\nClassification: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        # Visualize
        visualize_result(result)

def main():
    parser = argparse.ArgumentParser(description="Toxic Comment Classifier")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        help=f"Model ID or path (default: {DEFAULT_MODEL})")
    
    # Input methods
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--file", type=str, help="File with texts (one per line)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output file for visualization")
    parser.add_argument("--no-display", action="store_true", help="Don't display visualization")
    
    args = parser.parse_args()
    
    # Create classifier
    classifier = ToxicClassifier(
        model_name_or_path=args.model
    )
    
    # Handle input mode
    if args.interactive:
        interactive_mode(classifier)
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            results = classifier.batch_classify(texts)
            
            # Print results
            print("\n----- Classification Results -----")
            for i, result in enumerate(results):
                print(f"{i+1}. Text: {result['text'][:50]}...")
                print(f"   Classification: {result['label']} (Confidence: {result['confidence']:.2%})")
            
            # Visualize
            visualize_batch_results(
                results, 
                display=not args.no_display,
                save_path=args.output or 'batch_results.png'
            )
            
        except Exception as e:
            print(f"Error processing file: {e}")
    elif args.text:
        result = classifier.classify(args.text)
        
        # Print result
        print("\n----- Classification Result -----")
        print(f"Text: {result['text']}")
        print(f"Classification: {result['label']} (Confidence: {result['confidence']:.2%})")
        
        # Visualize
        visualize_result(
            result, 
            display=not args.no_display,
            save_path=args.output or 'classification_result.png'
        )
    else:
        # Default to interactive mode
        interactive_mode(classifier)

if __name__ == "__main__":
    main()
