#!/usr/bin/env python
# coding: utf-8

"""
Inference module for the A5 Optimization Human Preference assignment.

This module provides functionality to interact with the DPO-trained model
for text generation tasks using the Hugging Face Inference API.

Author: Arya
Date: March 2025
"""

import logging
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dpo_inference.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DPOPredictor:
    """
    Class for making predictions with the DPO-trained model using Hugging Face Inference API.
    """
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize the DPO predictor.
        
        Args:
            model_id (str): Hugging Face model ID for the DPO-trained model
        """
        logger.info(f"Initializing DPO predictor with model: {model_id} using Hugging Face Inference API")
        
        self.model_id = model_id
        self.api_token = os.getenv("HF_TOKEN")
        
        if not self.api_token:
            logger.warning("HF_TOKEN not found in environment variables. API calls may be rate limited.")
        
        logger.info("DPO predictor initialized successfully")
    
    def generate_response(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
        """
        Generate a response for the given prompt using the Hugging Face Inference API.
        
        Args:
            prompt (str): Input prompt text
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
            
        Returns:
            str: Generated response text
        """
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        
        # Format the prompt according to the model's expected format
        # TinyLlama uses Llama 2 style formatting
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Prepare API request
        api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True
            }
        }
        
        try:
            # Make API request
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse response
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    generated_text = result[0]["generated_text"]
                else:
                    generated_text = result[0]
            else:
                generated_text = result
                
            # Extract just the response part (remove the prompt)
            if isinstance(generated_text, str) and "[/INST]" in generated_text:
                response_start = generated_text.find("[/INST]") + len("[/INST]")
                generated_text = generated_text[response_start:].strip()
            
            logger.info(f"Generated response of length: {len(generated_text) if isinstance(generated_text, str) else 'unknown'}")
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return f"Error generating response: {str(e)}"
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error parsing API response: {str(e)}")
            return f"Error processing model response: {str(e)}"


# Singleton instance for use in web app
dpo_predictor = None

def get_predictor(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Get or create a DPOPredictor instance.
    
    Args:
        model_name (str): Name of the model to use
        
    Returns:
        DPOPredictor: An instance of DPOPredictor
    """
    global dpo_predictor
    
    if dpo_predictor is None:
        logger.info("Creating new DPOPredictor instance")
        dpo_predictor = DPOPredictor(model_id=model_name)
        logger.info("Successfully created DPOPredictor instance")
    
    return dpo_predictor
