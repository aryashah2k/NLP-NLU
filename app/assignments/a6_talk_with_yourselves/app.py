from flask import Flask, render_template, request, jsonify
import json
import os
from simple_rag import initialize_rag_system
from model_analysis import analyze_retriever_model, analyze_generator_model

app = Flask(__name__)

# Initialize the RAG system
print("Initializing RAG system...")
rag_system = initialize_rag_system()
print("RAG system initialized and ready to use")

# Store conversation history for the demo
qa_history = []

# Run initial model analysis with a few test questions
test_questions = [
    "How old are you?",
    "What is your highest level of education?",
    "What are your core beliefs regarding technology?",
    "What programming languages do you know?",
    "Where did you work before Google?"
]

# Store analysis results
retriever_analysis = analyze_retriever_model(rag_system, test_questions, verbose=False)
generator_analysis = analyze_generator_model(rag_system, test_questions, verbose=False)

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html', 
                          retriever_analysis=retriever_analysis,
                          generator_analysis=generator_analysis)

@app.route('/chat', methods=['POST'])
def chat():
    """Process chat messages and return responses"""
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Query the RAG system
    result = rag_system.query(user_message)
    
    # Add to history
    qa_pair = {
        "question": result["question"],
        "answer": result["answer"]
    }
    qa_history.append(qa_pair)
    
    # Save the updated history to a JSON file
    with open('qa_history.json', 'w') as f:
        json.dump(qa_history, f, indent=2)
    
    # Return the result
    return jsonify({
        "answer": result["answer"],
        "sources": result["sources"]
    })

@app.route('/qa_history', methods=['GET'])
def get_qa_history():
    """Return the question-answer history"""
    return jsonify(qa_history)

@app.route('/analyze', methods=['POST'])
def analyze_question():
    """Analyze a specific question with both models"""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Analyze with retriever model
    retriever_results = rag_system.vector_store.similarity_search(question)
    
    # Analyze with generator model (get full response)
    generation_result = rag_system.query(question)
    
    # Return analysis results
    return jsonify({
        "retriever_results": retriever_results,
        "generation_result": generation_result
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Run the Flask app
    app.run(debug=True, port=5000)
