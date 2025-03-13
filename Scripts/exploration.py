import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import processtext as pt
import threading
import matplotlib.pyplot as plt
import math

num_workers = os.cpu_count()
mutex = threading.Lock()

df = pd.read_csv('995,000_rows.csv')
column_headers = list(df.columns.values)
'''
for col in column_headers:
    print(f"Column name: {col}")
    print(f"Number of missing values: {df[col].isnull().sum()}")
    print(f"Number of unique values: {df[col].nunique()}")
    print(f"Number of unique non-missing values: {df[col].dropna().nunique()}")
    print()
'''
word_map = {}

def most_frequent_words(row):
    global word_map
    if not isinstance(row, str):
        return {}
    words = row.split()
    local_map = {}  

    for word in words:
        if word in local_map:
            local_map[word] += 1
        else:
            local_map[word] = 1

    with mutex:
        for word, count in local_map.items():
            if word in word_map:
                word_map[word] += count
            else:
                word_map[word] = count

def threaded_word_freq(df):
    global word_map
    word_map.clear()
    rows = df['content'].dropna().tolist()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(most_frequent_words, rows)
    return sorted(word_map.items(), key=lambda x: x[1], reverse=True)


wordFreq = threaded_word_freq(df)
#print(wordFreq[:100])

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

graph_frequencyCleaned(wordFreq)
