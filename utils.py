import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (run once)
nltk.download('stopwords')

# Load stopwords once
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean and preprocess input text
    """
    text = str(text).lower()                 # lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)   # remove symbols
    words = text.split()                    # tokenize
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return " ".join(words)