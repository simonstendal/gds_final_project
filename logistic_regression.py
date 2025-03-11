"""
This module helps prepare corpus data for logistic regression by labeling as reliable/unreliable, 
and vectorizing its word-content for LogisticRegression.

Once prepared the remaining functions can be used to complete simple logistic regression making it
able to predict whether a vectorized document is false or reliable.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import pandas as pd
import numpy as np
import wandb
import wandb.integration

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

    model = LogisticRegression(solver="liblinear")
    model.fit(x_vectors, y_label)
    return model

def setup_regression(test_corpus_path, type_column, content_column, labels, unique_words, wandb_init=False):
    """
    Turns a (test-data)corpus into a Logistic regression model, by vectorizing its contents.
    """
    corpus = pd.read_csv(test_corpus_path)
    df = vectorize_and_label_data(corpus, type_column, content_column, labels, unique_words).dropna()
    
    x = np.vstack(df["vector"].values)
    y = df["label"]
    model = logistic_regression(x, y)

    if wandb_init:
        run = wandb.init(project="gds-project-test")
        accuracy = cross_val_score(model, x, y, scoring="accuracy").mean()
        f1_macro = cross_val_score(model, x, y, scoring="f1_macro").mean()
        neg_log_loss = cross_val_score(model, x, y, scoring="neg_log_loss").mean()

        wandb.log({"accuracy": accuracy, "f1_macro": f1_macro, "neg_log_loss": neg_log_loss})
        wandb.finish()
    return model

if __name__ == "__main__":
    """
    Show test-case.
    """
    labels = (["fake", "satire", "bias", "conspiracy", "junksci", "hate", "unreliable"],
        ["state", "clickbait", "political", "reliable"])
    
    unique = ["state", "us", "column"] #Placeholder for top 10k words
    filepath = "news_sample.csv" #Placeholder for cleaned corpus

    model = setup_regression(filepath, "type", "content", labels, unique, wandb_init=True)
    test = np.array([[0,0,0]])
    print(model.predict(test))