PDF Chatbot with Summarization
This project allows users to interact with a PDF document by uploading it, extracting its content, and summarizing the key points. It also enables users to ask questions about the content of the PDF, with the model providing answers based on the extracted text.

Features
PDF Text Extraction: Extracts text from a PDF document.
Summarization: Summarizes the extracted content using state-of-the-art NLP models.
Question Answering: Allows users to ask questions about the document's content and get answers using a pre-trained question-answering model.
Evaluation Metrics: Measures the accuracy and efficiency of the model using metrics like Exact Match (EM), F1 score, and BLEU score.
Installation
To set up this project, you need to install the necessary dependencies.

Clone the repository:

bash
Copy code
git clone https://github.com/your-username/pdf-chatbot.git
cd pdf-chatbot
Install the required Python libraries:

bash
Copy code
pip install -r requirements.txt
Ensure you have the following installed:

transformers
PyPDF2
nltk
sklearn
Usage
Place your PDF file in the project directory.

Run the script with your desired PDF file:

bash
Copy code
python Pd_project.py
The script will extract text from the PDF, generate a summary, and allow you to ask questions about the content.

Example
Here's an example workflow:

Upload a PDF: The script will extract text from the file.
Summarization: A summary of the document will be generated.
Ask a Question: You can input a question, and the script will provide an answer based on the document content.
Efficiency Metrics
The project includes basic evaluation metrics for analyzing the model's performance:

Exact Match (EM): Checks if the predicted answer matches the ground truth exactly.
F1 Score: Evaluates the balance between precision and recall.
BLEU Score: Assesses the similarity between the predicted and actual answer.
Contributions
Contributions are welcome! If you have any suggestions, feel free to submit a pull request or open an issue.
