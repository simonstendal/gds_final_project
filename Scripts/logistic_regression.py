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
    model = LogisticRegression(class_weight="balanced", random_state=16, solver="saga", max_iter=1000, n_jobs=-1)
    print("Done iterating data.")
    model.fit(vector_col, label_col)
    print("Done training model.")
    return model

def test_model(model:LogisticRegression, val_x:pd.Series, val_y:pd.Series, test_X=None, test_Y=None, wandb_init = False):
    label_col = val_y.to_numpy()
    vector_col = vectorizer.transform(val_x).astype(np.float32)

    # Evaluate scores:
    val_score = model.score(vector_col, label_col)

    if wandb_init:
        wandb.init(project="gds-project-test")
        wandb.log({ "val_accuracy": val_score,})
        wandb.finish()

    if test_X and test_Y:
        vector_col_test = vectorizer.transform(test_X).astype(np.float32)
        label_col_test = test_Y.to_numpy()
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
    data = pd.read_csv("Processed.csv").dropna()
    data.drop(data.index[(data["type"].isin(unwanted_labels))],axis=0,inplace=True)
    data_X = data['content']
    data_Y = data['type'].apply(label_entry, args=(labels,))
    print("Done getting data ready.")

    bbc_data = pd.read_csv("../Data_in_csv/bbc_processed.csv").dropna()
    bbc_data_X = bbc_data['content']
    bbc_data_Y = bbc_data['type']

    X_train, X_val_test, Y_train, Y_val_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=16)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5, random_state=16)
    print("Done splitting data.")

    X_val_bbc = pd.concat([X_val, bbc_data_X])
    Y_val_bbc = pd.concat([Y_val, bbc_data_Y])

    model = logistic_regression(X_train, Y_train, labels)
    print("Done training model.")
    val_score = test_model(model, X_val, Y_val)
    bbc_val_score = test_model(model, X_val_bbc, Y_val_bbc)

    print(f"f1 score without bbc data: {val_score}")
    print(f"f1 score with bbc data: {bbc_val_score}")