import pandas as pd
import numpy as np
import re
import nltk
import ssl
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from cleantext import clean
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report
from logistic_regression import label_entry
import wandb
# Initialize NLTK resources if needed
def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
            ssl._create_default_https_context = _create_unverified_https_context
        except AttributeError:
            pass
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('stopwords', quiet=True)

initialize_nltk()

# Preprocessing functions (similar to your FakeNewsCorpus pipeline)
date_patterns = re.compile(
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d+,?\s+\d+\b|'
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+,?\s+\d+\b|'
    r'(\d+)-(\d+)-(\d+) ?(\d*):?(\d*):?(\d*)(\.\d+)?'
)
stop_words = set(stopwords.words('english'))

def clean_column(text):
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = re.sub(date_patterns, ' <DATE> ', text)
    text = clean(text,
                 fix_unicode=True,
                 to_ascii=True, 
                 no_punct=True,
                 no_urls=True,                  
                 no_emails=True,                
                 no_numbers=True,  
                 replace_with_punct="",             
                 no_line_breaks=True,
                 replace_with_url=" <URL> ",
                 replace_with_email=" <EMAIL> ",
                 replace_with_number=" <NUMBER> ",
                 lower=True)
    return text

def remove_stopwords(text):
    if not isinstance(text, str):
        return text
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def stem_text(text):
    if not isinstance(text, str):
        return text
    tokens = text.split()
    stemmer = nltk.stem.PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

def process_text(text):
    text = clean_column(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text
vectorizer = TfidfVectorizer(max_features=10000)

def neural_network(train_x:pd.Series, train_y:pd.Series, labels):
    """
    Initialize an SKLearn classifier neural network.
    """
    classifier = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(50,), activation='relu', random_state=1, max_iter=500)

    vector_col = vectorizer.fit_transform(train_x)
    label_col = train_y.to_numpy()
    print("Done vectorizing data.")
    classifier.fit(vector_col, label_col)
    print("Done training model.")
    return classifier

def test_model(model:MLPClassifier, val_x:pd.Series, val_y:pd.Series, test_X=None, test_Y=None, wandb_init = False):
    label_col = val_y.to_numpy()
    vector_col = vectorizer.transform(val_x)

    # Evaluate scores:
    val_pred = model.predict(vector_col)
    val_f1 = f1_score(label_col, val_pred, average='weighted')
    class_report = classification_report(label_col, val_pred)
    print("CLASS REPORT VALIDATION DATA:")
    print(class_report)

    if wandb_init:
        wandb.init(project="gds-project-test")
        wandb.log({"val_f1": val_f1})
        wandb.finish()

    if test_X and test_Y:
        vector_col_test = vectorizer.transform(test_X)
        label_col_test = test_Y.to_numpy()
        test_pred = model.predict(vector_col_test)
        test_f1 = f1_score(label_col_test, test_pred, average='weighted')
        class_report_test = classification_report(label_col_test, test_pred) 
        print("CLASS REPORT TEST DATA:")
        print(class_report_test)
        return test_f1
    else:
        return val_f1

if __name__ == "__main__":
    """
    Show test-case.
    """
    labels = (["fake", "satire", "bias", "conspiracy", "junksci", "state"],
        ["clickbait", "political", "reliable"])
    unwanted_labels = ["unreliable", "rumor", "unknown", "hate","2018-02-10 13:43:39.521661"]

    #split corpus data
    data = pd.read_csv("cleaned_corpus.csv").dropna()
    data.drop(data.index[(data["type"].isin(unwanted_labels))],axis=0,inplace=True)
    data_X = data['content']
    data_Y = data['type'].apply(label_entry, args=(labels,))
    print("Done getting data ready.")

    X_train, X_val_test, Y_train, Y_val_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=16)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5, random_state=16)
    print("Done splitting data.")

    model = neural_network(X_train, Y_train, labels)
    print("Done training model.")

    fake_news_val_f1 = test_model(model, X_val, Y_val)
    print(f"VALIDATION DATA F1: {fake_news_val_f1}")

    # TASK 2: Cross-Domain Evaluation on LIAR dataset
    def map_liar_label(label):
        if label in ["pants-fire", "false", "barely-true", "half-true"]:
            return 0
        elif label in ["mostly-true", "true"]:
            return 1
        else:
            return None
        
    liar_data = pd.read_csv("LIAR_dataset/train.tsv", sep='\t', header=None)
    liar_X = liar_data[2] 
    liar_Y = liar_data[1].apply(map_liar_label)
    liar_valid = liar_Y.notnull()
    liar_X = liar_X[liar_valid]
    liar_X = liar_X.apply(process_text)
    liar_Y = liar_Y[liar_valid]

    print("Evaluating FakeNewsCorpus-trained model on LIAR dataset")
    liar_f1 = test_model(model, liar_X, liar_Y)
    print(f"LIAR Dataset F1 (Cross-domain): {liar_f1}")

    # TASK 3: Comparison
    print("Comparison of Results:")
    print(f"FakeNewsCorpus Validation F1: {fake_news_val_f1}")
    print(f"LIAR Dataset F1 (Cross-domain): {liar_f1}")

