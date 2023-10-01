import json
import numpy

def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
    # read RAW data from dataset given
    pass
    # raw_data = f.read()
    # return raw_data

# Preprocess data
def preprocess(data):
    # Tokenize data
    tokens = nltk.word_tokenize(data)
    # Lowercase all words

    tokens = [word.lower() for word in tokens]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens
