import pandas as pd
import numpy as np
import wandb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report
from logistic_regression import label_entry
import wandb.integration


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
        class_report_test = classification_report(label_col, val_pred)
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

    test = test_model(model, X_val, Y_val)
    print(f"VALIDATION DATA F1: {test}")

