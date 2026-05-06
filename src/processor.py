import pdfplumber
import docx
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text(file_path):
    """Extracts text from PDF or DOCX files."""
    text = ""
    if file_path.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        text = " ".join([para.text for para in doc.paragraphs])
    return text

def clean_text(text):
    """Cleans text by removing special characters and extra spaces."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.lower().strip()

def rank_resumes(job_desc, resumes_list):
    """Ranks resumes against a JD using TF-IDF and Cosine Similarity."""
    documents = [clean_text(job_desc)] + [clean_text(r) for r in resumes_list]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # JD is at index 0, resumes are index 1 onwards
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return scores