#!/usr/bin/env python
# coding: utf-8

# Script to generate question-answer pairs in JSON format
import json
from simple_rag import initialize_rag_system

# List of required questions
questions = [
    "How old are you?",
    "What is your highest level of education?",
    "What major or field of study did you pursue during your education?",
    "How many years of work experience do you have?",
    "What type of work or industry have you been involved in?",
    "Can you describe your current role or job responsibilities?",
    "What are your core beliefs regarding the role of technology in shaping society?",
    "How do you think cultural values should influence technological advancements?",
    "As a master's student, what is the most challenging aspect of your studies so far?",
    "What specific research interests or academic goals do you hope to achieve during your time as a master's student?"
]

def generate_qa_json():
    print("Initializing RAG system...")
    rag_system = initialize_rag_system()
    print("RAG system initialized")
    
    qa_pairs = []
    
    print("\nGenerating answers for the required questions:")
    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/10: {question}")
        result = rag_system.query(question)
        
        qa_pair = {
            "question": question,
            "answer": result["answer"]
        }
        qa_pairs.append(qa_pair)
    
    # Save to JSON file
    with open('qa_responses.json', 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    
    print(f"\nGenerated answers for all {len(questions)} questions")
    print("Results saved to qa_responses.json")
    
    return qa_pairs

if __name__ == "__main__":
    generate_qa_json()
