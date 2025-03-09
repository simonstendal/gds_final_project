"""
This module helps prepare corpus data for logistic regression by labeling as reliable/unreliable, 
and vectorizing its word-content for LogisticRegression.

Once prepared the remaining functions can be used to complete simple logistic regression making it
able to predict whether a vectorized document is false or reliable.
"""
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

def label_entry(type, labels):
    """
    Label one document as fake or reliable, defaults to None (which can be removed with .dropna()).
    """
    if type in labels[0]:
        return "fake"
    elif type in labels[1]:
        return "reliable"
    else:
        return None

def vectorize_document(document:str, unique_words):
    """
    Turns a list of words into a vector containing a 'Bag of Words', each entry showing the amount of
    one popular word from unique_words.
    """
    return np.array([document.count(u_word) for u_word in unique_words])

def vectorize_and_label_data(cleaned_corpus:pd.DataFrame, type_column:str, content_column:str,
                             labels, unique_words):
    """
    Creates a new dataframe from a corpus, saving the document as a vector with an associated label.
    """
    new_df = pd.DataFrame(columns=["label", "vector"])

    new_df["label"] = cleaned_corpus[type_column].apply(label_entry, args=(labels,))
    new_df["vector"] = cleaned_corpus[content_column].apply(vectorize_document, args=(unique_words,))
    return new_df

def logistic_regression(x_vectors, y_label):
    """
    Initialize an SKLearn Logistic regression model.
    """

    model = LogisticRegression()
    model.fit(x_vectors, y_label)
    return model

def setup_regression(corpus_path, type_column, content_column, labels, unique_words):
    """
    Turns a Corpus into a Logistic regression model, by vectorizing its contents.
    """
    corpus = pd.read_csv(corpus_path)
    df = vectorize_and_label_data(corpus, type_column, content_column, labels, unique_words).dropna()
    
    x = np.vstack(df["vector"].values)
    y = df["label"]
    return logistic_regression(x, y)

if __name__ == "__main__":
    """
    Show test-case.
    """
    labels = (["fake", "satire", "bias", "conspiracy", "junksci", "hate", "unreliable"],
        ["state", "clickbait", "political", "reliable"])
    
    unique = ["state", "us", "column"] #Placeholder for top 10k words
    filepath = "news_sample.csv" #Placeholder for cleaned corpus

    model = setup_regression(filepath, "type", "content", labels, unique)
    test = np.array([[0,0,0]])
    print(model.predict(test))