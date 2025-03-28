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
        return 0
    elif type in labels[1]:
        return 1
    else:
        print(f"MISSED TYPE:{type}")
        return None

def logistic_regression(train_x:pd.Series, train_y:pd.Series, labels):
    """
    Initialize an SKLearn Logistic regression model.
    """
    vector_col = vectorizer.fit_transform(train_x).astype(np.float32)
    label_col = train_y
    print("Done vectorizing data.")
    model = LogisticRegression(class_weight="balanced", random_state=16, solver="saga", max_iter=1000, n_jobs=-1)
    print("Done iterating data.")
    model.fit(vector_col, label_col)
    print("Done training model.")
    return model

def test_model(model:LogisticRegression, val_x:pd.Series, val_y:pd.Series, test_X=None, test_Y=None, wandb_init = False):
    label_col = val_y
    vector_col = vectorizer.transform(val_x).astype(np.float32)

    # Evaluate scores:
    val_predict = model.predict(vector_col)
    val_score = f1_score(label_col, val_predict)

    if wandb_init:
        wandb.init(project="gds-project-test")
        wandb.log({ "val_accuracy": val_score,})
        wandb.finish()

    if test_X is not None and test_Y is not None:
        vector_col_test = vectorizer.transform(test_X).astype(np.float32)
        label_col_test = test_Y
        test_predict = model.predict(vector_col_test)
        test_score = f1_score(label_col_test, test_predict)
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
    train = pd.read_csv("Data_in_csv/train.csv").dropna()
    train.drop(train.index[(train["type"].isin(unwanted_labels))],axis=0,inplace=True)
    train_X = train['content']
    train_Y = train['type'].apply(label_entry, args=(labels,)).to_numpy().astype(np.float32)

    validation = pd.read_csv("Data_in_csv/validation.csv").dropna()
    validation.drop(validation.index[(validation["type"].isin(unwanted_labels))],axis=0,inplace=True)
    validation_X = validation['content']
    validation_Y = validation['type'].apply(label_entry, args=(labels,)).to_numpy().astype(np.float32)

    test = pd.read_csv("Data_in_csv/test.csv").dropna()
    test.drop(test.index[(test["type"].isin(unwanted_labels))],axis=0,inplace=True)
    test_X = test['content']
    test_Y = test['type'].apply(label_entry, args=(labels,)).to_numpy().astype(np.float32)

    bbc_data = pd.read_csv("Data_in_csv/bbc_processed.csv").dropna()
    bbc_data_X = bbc_data['content']
    bbc_data_Y = bbc_data['type'].apply(label_entry, args=(labels,)).to_numpy().astype(np.float32)

    X_val_bbc = pd.concat([validation_X, bbc_data_X])
    Y_val_bbc = np.concat([validation_Y, bbc_data_Y])
    print("Done getting data ready.")

    model = logistic_regression(train_X, train_Y, labels)
    print("Done training model.")
    val_score = test_model(model, validation_X, validation_Y)
    bbc_val_score = test_model(model, X_val_bbc, Y_val_bbc)
    test_score = test_model(model, test_X, test_Y)

    print(f"f1 validation score without bbc data: {val_score}")
    print(f"f1 validation score with bbc data: {bbc_val_score}")
    print(f"f1 test score: {test_score}")

