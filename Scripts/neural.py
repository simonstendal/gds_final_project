"""
This module helps prepare corpus data for logistic regression by labeling as reliable/unreliable, 
and vectorizing its word-content for LogisticRegression.

Once prepared the remaining functions can be used to complete simple logistic regression making it
able to predict whether a vectorized document is false or reliable.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import wandb
import wandb.integration

vectorizer = TfidfVectorizer(max_features=10000)

def label_entry(type, labels):
    """
    Label one document as fake or reliable, defaults to None (which can be removed with .dropna()).
    """
    if type in labels[0]:
        return "fake"
    elif type in labels[1]:
        return "reliable"
    else:
        print(f"MISSED TYPE:{type}")
        return None


def logistic_regression(train_x:pd.Series, train_y:pd.Series, labels):
    """
    Initialize an SKLearn Logistic regression model.
    """
    vector_col = vectorizer.fit_transform(train_x).astype(np.float32)
    label_col = train_y.to_numpy()
    print("Done vectorizing data.")
    model = MLPClassifier(random_state=1, max_iter=300).fit(vector_col, label_col)
    print("Done iterating data.")
    
    print("Done training model.")
    return model

def test_model(model:LogisticRegression, val_x:pd.Series, val_y:pd.Series, test_X=None, test_Y=None, wandb_init = False):
    label_col = val_y.to_numpy()
    vector_col = vectorizer.transform(val_x).astype(np.float32)

    # Evaluate scores:
    model.predict_proba(vector_col[:1])
    model.predict(vector_col[:5, :])
    val_score = model.score(vector_col, label_col)

    if wandb_init:
        wandb.init(project="gds-project-test")
        wandb.log({ "val_accuracy": val_score,})
        wandb.finish()

    if test_X is not None and test_Y is not None:
        vector_col_test = vectorizer.transform(test_X).astype(np.float32)
        label_col_test = test_Y.to_numpy()
        model.predict_proba(vector_col_test[:1])
        model.predict(vector_col_test[:5, :])
        test_score = model.score(vector_col_test, label_col_test)
        return val_score, test_score
    else:
        return val_score

if __name__ == "__main__":
    """
    Show test-case.
    """
    labels = (["fake", "satire", "bias", "conspiracy", "junksci", "hate", "state"],
        ["clickbait", "political", "reliable"])
    unwanted_labels = ["unreliable", "rumor", "unknown", "2018-02-10 13:43:39.521661"]

    #split corpus data
    data = pd.read_csv("../news_sample_processed.csv").dropna()
    data.drop(data.index[(data["type"].isin(unwanted_labels))],axis=0,inplace=True)
    data_X = data['content']
    data_Y = data['type'].apply(label_entry, args=(labels,))
    print("Done getting data ready.")

    X_train, X_val_test, Y_train, Y_val_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=16)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5, random_state=16)
    print("Done splitting data.")

    model = logistic_regression(X_train, Y_train, labels)
    print("Done training model.")

    val_score, test_score = test_model(model, X_val, Y_val, X_test, Y_test)

    print(f"Validation score: {val_score}")
    print(f"Test score: {test_score}")