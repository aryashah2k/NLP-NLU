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

### A2: Language Modelling
Coming soon!

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

## License
© 2024 Arya Shah. All rights reserved.
