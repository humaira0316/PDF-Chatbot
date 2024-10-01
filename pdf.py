# Suppress TensorFlow warning
import os
import time
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from PyPDF2 import PdfReader
from transformers import pipeline

# Step 1: Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file_path):
    reader = PdfReader(pdf_file_path)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text()
    return extracted_text

# Step 2: Function to summarize the document
def summarize_document(document_text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    max_chunk_size = 1000
    document_chunks = [document_text[i:i + max_chunk_size] for i in range(0, len(document_text), max_chunk_size)]
    
    summary = ""
    for chunk in document_chunks:
        input_length = len(chunk.split())
        max_length = int(input_length * 0.5)
        min_length = int(input_length * 0.2)

        if max_length <= min_length:
            max_length = min_length + 5
        
        summary += summarizer(chunk, max_length=max_length, min_length=min_length)[0]['summary_text'] + " "
    
    return summary.strip()

# Step 3: Function for asking questions using the extracted text and NLP model
def ask_question_about_pdf(question, document_text):
    nlp = pipeline("question-answering")
    result = nlp(question=question, context=document_text)
    return result['answer']

# Function to calculate efficiency metrics
def calculate_efficiency_metrics(predicted_answer, ground_truth_answer):
    exact_match = int(predicted_answer == ground_truth_answer)
    f1 = f1_score_calculation(predicted_answer, ground_truth_answer)
    bleu = bleu_score_calculation(predicted_answer, ground_truth_answer)
    return exact_match, f1, bleu

# Function to calculate F1 score
def f1_score_calculation(predicted_answer, ground_truth_answer):
    # Convert to sets for precision and recall calculation
    predicted_set = set(predicted_answer.split())
    ground_truth_set = set(ground_truth_answer.split())
    
    # Calculate precision and recall
    true_positive = len(predicted_set.intersection(ground_truth_set))
    precision = true_positive / len(predicted_set) if len(predicted_set) > 0 else 0
    recall = true_positive / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
    
    # Calculate F1 score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# Function to calculate BLEU score
def bleu_score_calculation(predicted_answer, ground_truth_answer):
    reference = ground_truth_answer.split()
    generated = predicted_answer.split()
    return sentence_bleu([reference], generated)

# Example usage
pdf_file_path = r"d:\The Turing Test.pdf"  # Path to your PDF file
document_text = extract_text_from_pdf(pdf_file_path)

# Get a summary of the document
summary = summarize_document(document_text)
print(f"Summary of the document: {summary}")

# Ask a question about the document
question = "What is Unification?"  # Replace this with any question
predicted_answer = ask_question_about_pdf(question, document_text)

# Set ground truth answer for evaluation
ground_truth_answer = "The unification refers to the process of combining separate elements into a single entity."  # Replace with the actual answer

# Calculate efficiency metrics
exact_match, f1, bleu = calculate_efficiency_metrics(predicted_answer, ground_truth_answer)

# Output the results
print(f"Predicted Answer: {predicted_answer}")
print(f"Exact Match: {exact_match}")
print(f"F1 Score: {f1:.4f}")
print(f"BLEU Score: {bleu:.4f}")
