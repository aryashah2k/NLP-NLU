# NLP Assignments

This repository contains my assignments for the AT82.05 Artificial Intelligence: Natural Language Understanding (NLU) course at AIT.

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

| App Demo | App Tests |
|----------|-----------|
| <img src="https://github.com/aryashah2k/NLP-NLU/blob/main/assets/A1_Demo.gif"> | <img src="https://github.com/aryashah2k/NLP-NLU/blob/main/assets/A1_Tests.gif"> |

**Screenshots**
|Home|A1 Page|
|----|-------|
|![Home](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/home.PNG)|![A1](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a1.PNG)|

### A2: Language Modelling
Train a model that can generate coherent and contextually relevant text based on a given input using LSTM

| App Demo | 
|----------|
| <img src="https://github.com/aryashah2k/NLP-NLU/blob/main/assets/A2_Demo.gif"> |

**Screenshots**
|A2 Page|
|-------|
|![A2](https://github.com/aryashah2k/NLP-NLU/blob/main/assets/a2.PNG)|

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
