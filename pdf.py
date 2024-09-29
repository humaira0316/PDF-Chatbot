# Suppress TensorFlow warning

from PyPDF2 import PdfReader
from transformers import pipeline
# Step 1: Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file_path):
    # Initialize the PdfReader object
    reader = PdfReader(pdf_file_path)
    
    # Initialize an empty string to hold the extracted text
    extracted_text = ""
    
    # Loop through each page of the PDF and extract text
    for page in reader.pages:
        extracted_text += page.extract_text()
    
    return extracted_text

# Step 2: Function to summarize the document
def summarize_document(document_text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Split document into manageable chunks
    max_chunk_size = 1000
    document_chunks = [document_text[i:i + max_chunk_size] for i in range(0, len(document_text), max_chunk_size)]
    
    summary = ""
    for chunk in document_chunks:
        input_length = len(chunk.split())  # Calculate the number of words in the chunk
        
        # Ensure max_length is reasonable and greater than min_length
        max_length = int(input_length * 0.5)  # Set max_length to 50% of the input length
        min_length = int(input_length * 0.2)  # Set min_length to 20% of the input length

        # Ensure max_length is greater than min_length
        if max_length <= min_length:
            max_length = min_length + 5  # Set a minimum gap to avoid the error
        
        # Summarize the chunk with dynamically set min_length and max_length
        summary += summarizer(chunk, max_length=max_length, min_length=min_length)[0]['summary_text'] + " "
    
    return summary.strip()
# Step 3: Function for asking questions using the extracted text and NLP model
def ask_question_about_pdf(question, document_text):
    # Initialize a question-answering pipeline
    nlp = pipeline("question-answering")
    
    # Perform question answering using the provided document text
    result = nlp(question=question, context=document_text)
    
    return result['answer']

# Example usage
pdf_file_path = r"d:\The Turing Test.pdf"  # Path to your PDF file

# Extract text from the PDF
document_text = extract_text_from_pdf(pdf_file_path)

# Get a summary of the document
summary = summarize_document(document_text)
print(f"Summary of the document: {summary}")

# Ask a question about the document
question = "What is Unification?"  # Replace this with any question
answer = ask_question_about_pdf(question, document_text)

# Output the answer
print(f"Question: {question}")
print(f"Answer: {answer}")
