import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import processtext as pt
import threading
import matplotlib.pyplot as plt
import math

num_workers = os.cpu_count()
mutex = threading.Lock()

df = pd.read_csv('../Data_in_csv/processed_with_word_freq.csv')
column_headers = list(df.columns.values)

for col in column_headers:
    print(f"Column name: {col}")
    print(f"Number of missing values: {df[col].isnull().sum()}")
    print(f"Number of unique values: {df[col].nunique()}")
    print(f"Number of unique non-missing values: {df[col].dropna().nunique()}")
    print()
word_map = {}

def graph_frequencyCleaned(data):
    items, frequencies = zip(*data[:10000])
    items = [item.replace('$', '\\$') for item in items]
    frequencies = [math.log(frequency) for frequency in frequencies]

    fig, ax = plt.subplots()
    ax.plot(items, frequencies)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Word Frequency Data')

    ax.set_xticklabels([]) 

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()