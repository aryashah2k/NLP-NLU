# NLP Assignments

This repository contains my assignments for the AT82.05 Artificial Intelligence: Natural Language Understanding (NLU) course at AIT.

|Table of Contents|
|-----------------|
|<a href="https://github.com/aryashah2k/NLP-NLU/tree/main?tab=readme-ov-file#a1-thats-what-i-like">A1: That's What I LIKE!</a>|
|<a href="https://github.com/aryashah2k/NLP-NLU/tree/main?tab=readme-ov-file#a2-language-modelling">A2: Language modelling</a>|
|<a href="https://github.com/aryashah2k/NLP-NLU/tree/main?tab=readme-ov-file#a3-make-your-own-machine-translation-language">A3: Make Your Own Machine Translation Language</a>|
|<a href="https://github.com/aryashah2k/NLP-NLU/tree/main?tab=readme-ov-file#a4-do-you-agree">A4: Do You AGREE!</a>|
|<a href="https://github.com/aryashah2k/NLP-NLU/tree/main?tab=readme-ov-file#a5-optimization-human-preference">A5: Optimization Human Preference</a>|
|<a href="https://github.com/aryashah2k/NLP-NLU/blob/main/app/static/coming_soon.gif">A6....Coming Soon!</a>|
|<a href="https://github.com/aryashah2k/NLP-NLU/tree/main?tab=readme-ov-file#setup-and-installation">Setup & Installation</a>|

## Project Structure

```
nlp-assignments/
├── app/                        # Main Flask application
│   ├── static/                # Static files (CSS, JS, images)
│   ├── templates/             # HTML templates
│   ├── assignments/           # Assignment-specific modules
│   │   ├── a1_word_embeddings/
│   │   │   ├── models/       # Trained models
│   │   │   └── utils.py      # Utility functions
│   │   └── a2_language_model/ # Future assignment
│   └── core/                  # Core application code
│       ├── __init__.py
│       └── routes.py         # Flask routes
├── data/                      # Dataset storage
├── notebooks/                 # Jupyter notebooks
├── reports/                 # Assignment Report PDFs
├── tests/                     # Test files
├── requirements.txt           # Python dependencies
└── run.py                    # Application entry point
```

## Assignments

### A1: That's what I LIKE!
Word embeddings implementation including:
- Skip-gram
- Skip-gram with Negative Sampling
- GloVe
- Glove (Gensim) Evaluation

| App Demo | 
|----------|
| <img src="https://github.com/aryashah2k/NLP-NLU/blob/main/assets/A1_Demo.gif"> | 

|App Tests|
|---------|
|<img src="https://github.com/aryashah2k/NLP-NLU/blob/main/assets/A1_Tests.gif"> |

**Screenshots**
|Home|
|----|
|![Home](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/home.PNG)|

|A1 Page|
|----|
|![A1](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a1.PNG)|

#### Summary of Results

|Model|Skipgram|NEG|GloVe|GloVe (gensim) 100 Dim|GloVe (gensim) 300 Dim|Y_True|
|-----|--------|---|-----|----------------------|----------------------|------|
|MSE|0.2474|0.1879|0.2666|0.0750|0.0502|1.0000|

-----------------

### A2: Language Modelling
Train a model that can generate coherent and contextually relevant text based on a given input using LSTM

| App Demo | 
|----------|
| <img src="https://github.com/aryashah2k/NLP-NLU/blob/main/assets/A2_Demo.gif"> |

**Screenshots**
|A2 Page|
|-------|
|![A2](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a2.PNG)|

#### Summary of Results

|Training History|
|----------------|
|![1](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a2_assets/training_history.png)|


```text
Generating text with different temperatures...

Prompt: To be or not
--------------------------------------------------
Temperature 0.20:
To be or not . i ll be sworn to be a man . i have rather be a fool , and i am sure to be a man . i am a fool , and i am a fool , and i will be sworn . i am a good man , and i am not to be a man . i am a good man , and i will be a good man . i am a man , and i ll tell you . i am not a good man , and i ll be a man , and i will

Temperature 0.50:
To be or not , or a great gallant man , and no more than a most king s body . the very fiend , the next of the king , and the king , a very fold fellow . i have a king , and the king of wales , or all the duke of york , with the king , the duke of wales , and the queen of gloucester , with the king , with the duke of westmoreland , and two a prince of venice ; a weaver in his forces and his train , attended of the king ,

Temperature 0.70:
To be or not , or a great gallant man , and no more than a most king s body . ah , my gentle lord , i pray you , master , i am too guilty , and to the hand of these entreaty , after the instant that can be ; i have rather be glad his will , as you are in love , a rush . charles . how goes it ? if thou dost ; i say , how you may find him . exit . scene ii . inverness . the church of leonato s house enter helena

Temperature 0.80:
To be or not ere it were so . i have rather add let me have a king s body with the great right . i pray them , sir , thou art sick ; but so , as to the hand of these entreaty , after the instant shall she be long ; let me not be nam d with my hat . i break you in a mocker . perchance that i can must be content to give ; their powers are wont d in a basket . exit . scene ii . inverness . the church of leonato s house enter

Temperature 1.00:
To be or not ere it were so near i have , and let me have employ d s body with the great right . i pray them ; you shall go with it ; but so , constant to true mrs ; i prithee , after addition to beg of speech i ; let me not be nam d with my hat . i break you in a mocker . perchance that i can must argument if london give ; their lands are wont d in a twofold air ; and never put in myself obeying old of your wrong . dost thou
```
-----------------

### A3: Make Your Own Machine Translation Language
Translating between my Native Language `Gujarati` and English. I have experimented with different types of attention mechanisms, including general attention, multiplicative attention, and additive attention, to evaluate their effectiveness in the translation process.

| App Demo | 
|----------|
| <img src="https://github.com/aryashah2k/NLP-NLU/blob/main/assets/A3_Demo.gif"> |

**Screenshots**
|A3 Page|
|-------|
|![A2](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a3.PNG)|

#### Summary of Results

| Attention Variant | Training Loss | Training PPL | Validation Loss | Validation PPL | BLEU Score | Training Time |
|------------------|---------------|--------------|-----------------|----------------|------------|---------------|
| multiplicative   | 1.763         | 5.832        | 3.473          | 32.249         | 22.61      | 83.4m        |
| general         | 1.765         | 5.841        | 3.447          | 31.403         | 27.01      | 76.9m        |
| additive        | 1.681         | 5.371        | 3.355          | 28.652         | 17.90      | 85.8m        |

|Training Curve(Additive)|Training Curve(Multiplicative)|Training Curve(General)|
|------------------------|------------------------|------------------------|
|![1](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a3_assets/loss_plot_additive.png)|![2](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a3_assets/loss_plot_multiplicative.png)|![3](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a3_assets/loss_plot_general.png)|

-----------------

### A4: Do You Agree
Training a pre-trained model like BERT from scratch, focusing on leveraging text embeddings to capture semantic similarity. Additionally, we will explore how to adapt the loss function for tasks like Natural Language Inference (NLI) to enhance the model’s ability to understand semantic relationships between texts

| App Demo | 
|----------|
| <img src="https://github.com/aryashah2k/NLP-NLU/blob/main/assets/A4_Demo.gif"> |

**Screenshots**
|A4 Page|
|-------|
|![A4](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a4.PNG)|

#### Summary of Results

##### Overall Metrics (Final Epoch)
| Metric | Value |
|--------|--------|
| Accuracy | 40.25% |
| Average Loss | 1.0340 |
| Weighted F1-Score | 0.38 |
| Macro F1-Score | 0.29 |

##### Class-wise Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|-----------|---------|
| Entailment | 0.43 | 0.38 | 0.41 | 273 |
| Contradiction | 0.41 | 0.61 | 0.49 | 273 |
| Neutral | 0.33 | 0.21 | 0.25 | 246 |

##### Cosine Similarity Analysis
| Class | Mean | Standard Deviation |
|-------|------|-------------------|
| Entailment | 0.7780 | 0.1795 |
| Contradiction | 0.8076 | 0.1534 |
| Neutral | 0.7976 | 0.1491 |

##### Final Conclusion

The model's performance on the SNLI/MNLI dataset shows several key characteristics:

1. **Overall Performance**:
   - Moderate accuracy (40.25%) on the validation set
   - Consistent but relatively high loss (1.0340)
   - Balanced performance across classes with weighted F1-score of 0.38

2. **Class-specific Performance**:
   - Best performance on Contradiction class (F1: 0.49)
   - Moderate performance on Entailment class (F1: 0.41)
   - Poorest performance on Neutral class (F1: 0.25)

3. **Model Behavior**:
   - Shows good discrimination between classes (cosine similarities differ)
   - Demonstrates learning progression (decreasing loss over epochs)
   - Maintains consistent performance across validation checks

4. **Areas for Improvement**:
   - Neutral class recognition needs significant improvement
   - Overall accuracy could be enhanced with more training data
   - Model confidence (cosine similarities) could be more distinctive between classes

The model shows promise but would benefit from further optimization and larger training datasets for production use.

------------------

### A5: Optimization Human Preference
Using Hugging Face models to optimize human preference, specifically leveraging the Direct Preference Optimization (DPO) trainer. Worked with preference datasets, train a model, and push it to the Hugging Face model hub

| App Demo | 
|----------|
| <img src="https://github.com/aryashah2k/NLP-NLU/blob/main/assets/A5_Demo.gif"> |

**Screenshots**
|A5 Page|
|-------|
|![A5](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a5.png)|



#### Summary of Results

|Inference Results|Training Time|Loss Plot|
|-----------------|-------------|---------|
|![infer](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a5_assets/a5_dpo_custom.png)|![traintime](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a5_assets/dpo_training_time.png)|![loss](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a5_assets/dpo_losses.png)|

##### Key Features

1. Dataset Processing: Uses Anthropic's HH-RLHF (Human Harmlessness - Reinforcement Learning from Human Feedback) dataset containing pairs of preferred and dispreferred responses.
2. Efficient Training with QLoRA: Implements Parameter-Efficient Fine-Tuning using QLoRA (Quantized Low-Rank Adaptation) to reduce memory requirements while maintaining performance.
3. 4-bit Quantization: Utilizes the BitsAndBytes library to enable 4-bit quantization, significantly reducing the memory footprint of the model.
4. Training Visualization: Includes a comprehensive visualization system that tracks and plots training metrics including losses and reward accuracy.
5. Hugging Face Integration: Seamlessly uploads the trained model to Hugging Face Hub for easy sharing and deployment.

##### Technical Implementation
- Base Model: TinyLlama-1.1B-Chat-v1.0
- Training Method: Direct Preference Optimization (DPO)
- Quantization: 4-bit quantization using BitsAndBytes
- Parameter-Efficient Fine-Tuning: QLoRA with rank 16, alpha 32
- Target Modules: Attention layers (q_proj, k_proj, v_proj, o_proj) and MLP layers (gate_proj, up_proj, down_proj)
- Training Dataset: 5,000 examples from Anthropic's HH-RLHF
- Evaluation Dataset: 500 examples for validation

##### Training Process
1. Loads and preprocesses the Anthropic HH-RLHF dataset
2. Initializes the TinyLlama model with 4-bit quantization
3. Applies QLoRA for parameter-efficient fine-tuning
4. Trains using the DPO algorithm to optimize for human preferences
5. Tracks and visualizes training metrics
6. Saves and uploads the trained model to Hugging Face Hub

------------------

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python run.py
   ```
5. Run tests:
   ```bash
   pytest -v
   ```
You can also run the tests individually by specifying the target file to test

## License
© 2024 Arya Shah. All rights reserved. | MIT
