"""
This module helps prepare corpus data for logistic regression by labeling as reliable/unreliable, 
and vectorizing its word-content for LogisticRegression.

Once prepared the remaining functions can be used to complete simple logistic regression making it
able to predict whether a vectorized document is false or reliable.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from collections import Counter
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
    return np.array([Counter(document.split()).get(u_word, 0) for u_word in unique_words])

def vectorize_and_label_data(cleaned_corpus:pd.DataFrame, type_column:str, content_column:str,
                             labels, unique_words):
    """
    Creates a new dataframe from a corpus, saving the document as a vector with an associated label.
    """
    label_col = cleaned_corpus[type_column].apply(label_entry, args=(labels,))
    vector_col = [vectorize_document(doc, unique_words) for doc in cleaned_corpus[content_column]]
    return pd.DataFrame({"label": label_col, "vector": vector_col}).dropna()

def logistic_regression(x_vectors, y_label):
    """
    Initialize an SKLearn Logistic regression model.
    """
    model = LogisticRegression(solver="liblinear")
    model.fit(x_vectors, y_label)
    return model

def setup_regression(train_corpus_path, val_corpus_path, type_column, content_column, labels,
                     unique_words, test_corpus_path = None, wandb_init=False):
    """
    Turns a (test-data)corpus into a Logistic regression model, by vectorizing its contents.
    """
    corpus = pd.read_csv(train_corpus_path)
    df = vectorize_and_label_data(corpus, type_column, content_column, labels, unique_words)
    
    x = np.vstack(df["vector"].values)
    y = df["label"]
    model = logistic_regression(x, y)


    validation_data = pd.read_csv(val_corpus_path)
    val_df = vectorize_and_label_data(validation_data, type_column, content_column, labels, unique_words)
    val_x = np.vstack(val_df["vector"].values)
    val_y = val_df["label"]

    # test score left out while training:
    # test_data = pd.read_csv(test_corpus_path)
    # test_df = vectorize_and_label_data(test_data, type_column, content_column, labels, unique_words)
    # test_x = np.vstack(test_df["vector"].values)
    # test_y = test_df["label"]

    # Evaluate scores:
    train_score = model.score(x, y)
    val_score = model.score(val_x, val_y)
    # test_score = model.score(test_x, test_y)


    if wandb_init:
        run = wandb.init(project="gds-project-test")
        wandb.log({"train_accuracy": train_score, "val_accuracy": val_score,})
        wandb.finish()
    return val_score

if __name__ == "__main__":
    """
    Show test-case.
    """
    labels = (["fake", "satire", "bias", "conspiracy", "junksci", "hate", "unreliable", "state"],
        ["clickbait", "political", "reliable"])
    
    #set-up unique_words
    file = open("stemmed_freq.txt").readlines()
    unique_words = []
    for line in file:
        unique_words.append(line.split(" :")[0]) 

    #split corpus data
    traindata = "train.csv"
    valdata = "train.csv"
    # testdata = "train.csv"

    model_score = setup_regression(traindata, valdata, "type", "content", labels, unique_words, wandb_init=True)
    print(model_score)

    # Load and process the data for saving
    print("Creating combined CSV with vectorized and original data...")
    df = pd.read_csv(traindata)
    
    # Vectorize the documents
    vectorized_data = np.array([vectorize_document(doc, unique_words) for doc in df['content']])
    
    # Create column names for the vectorized features
    vector_columns = [f'word_freq_{word}' for word in unique_words]
    
    # Create a new dataframe with the vectorized data
    vector_df = pd.DataFrame(vectorized_data, columns=vector_columns)
    
    # Combine with original data
    combined_df = pd.concat([df, vector_df], axis=1)
    
    # Save to CSV
    output_file = 'combined_vectorized_data.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined data to {output_file}")