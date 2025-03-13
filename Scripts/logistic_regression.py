"""
This module helps prepare corpus data for logistic regression by labeling as reliable/unreliable, 
and vectorizing its word-content for LogisticRegression.

Once prepared the remaining functions can be used to complete simple logistic regression making it
able to predict whether a vectorized document is false or reliable.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
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
        print(type)
        return None


def logistic_regression(cleaned_corpus:pd.DataFrame, labels):
    """
    Initialize an SKLearn Logistic regression model.
    """
    label_col = cleaned_corpus['type'].apply(label_entry, args=(labels,))
    vectorizer = CountVectorizer(max_features=1000)
    vector_col = vectorizer.fit_transform(cleaned_corpus['content'])
    model = LogisticRegression(solver="liblinear")
    model.fit(vector_col, label_col)
    return model

def test_model(model:LogisticRegression, validation_corpus:pd.DataFrame, test_corpos=None, wandb_init = False):
    label_col = validation_corpus["type"].apply(label_entry, args=(labels,))
    vectorizer = CountVectorizer(max_features=1000)
    vector_col = vectorizer.fit_transform(validation_corpus["content"])
    if test_corpos:
        label_col_test = validation_corpus["type"].apply(label_entry, args=(labels,))
        vector_col_test = vectorizer.fit_transform(validation_corpus["content"])
    # test_score = model.score(vector_col, label_col_test)

    # test score left out while training:
    # test_data = pd.read_csv(test_corpus_path)
    # test_df = vectorize_and_label_data(test_data, type_column, content_column, labels, unique_words)
    # test_x = np.vstack(test_df["vector"].values)
    # test_y = test_df["label"]

    # Evaluate scores:
    val_score = model.score(vector_col, label_col)
    # test_score = model.score(test_x, test_y)


    if wandb_init:
        wandb.init(project="gds-project-test")
        wandb.log({ "val_accuracy": val_score,})
        wandb.finish()
    return val_score

if __name__ == "__main__":
    """
    Show test-case.
    """
    labels = (["fake", "satire", "bias", "conspiracy", "junksci", "hate", "unreliable", "state","unknown"],
        ["clickbait", "political", "reliable"])
    

    #split corpus data
    traindata = pd.read_csv("train.csv").dropna()
    valdata = pd.read_csv("validation.csv").dropna()
    # testdata = "test.csv"

    model = logistic_regression(traindata, labels)
    model_score=test_model(model,valdata)
    print(model_score)
