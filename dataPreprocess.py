import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
nltk.download('punkt')

def preprocess_data(file_path='spam.csv'):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df['target'])
    df = df.drop_duplicates(keep='first')

    df['num_characters'] = df['text'].apply(len)
    df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
    df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

    df['transformed_text'] = df['text'].apply(transform_text)

    spam_corpus = [word for msg in df[df['target'] == 1]['transformed_text'].tolist() for word in msg.split()]
    ham_corpus = [word for msg in df[df['target'] == 0]['transformed_text'].tolist() for word in msg.split()]
    vectorizer = None
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=3000)
        X = vectorizer.fit_transform(df['transformed_text']).toarray()
    else:
        X = vectorizer.transform(df['transformed_text']).toarray()

    y = df['target'].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, vectorizer

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]

    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    ps = PorterStemmer()
    y = [ps.stem(i) for i in y]

    return " ".join(y)

if __name__ == "__main__":
    X_train, X_valid, X_test, y_train, y_valid, y_test,vectorizer = preprocess_data()

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_valid:", X_valid.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_valid:", y_valid.shape)
    print("Shape of y_test:", y_test.shape)
    
