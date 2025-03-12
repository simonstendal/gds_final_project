import pandas as pd
import nltk
import numpy as np

from collections import Counter

def txt_to_dict(file_path):
    freq_dict = {}
    with open(file_path) as f:
        for line in f:
            word, freq = line.strip().split(': ')
            freq_dict[word] = int(freq)
    return freq_dict

stemmed_freq = txt_to_dict('../stemmed_freq.txt')

word_freq = Counter()

def word_freq_text(text):
    if not isinstance(text, str):
        return [0] * len(stemmed_freq)
    text_counter = Counter(text.split())
    val = [text_counter.get(word, 0) for word in stemmed_freq.keys()]
    print(val)
    return val

def word_freq_df(df):
    word_freq_vectors = np.array([word_freq_text(text) for text in df['content']])
    return word_freq_vectors

output_file = '../Data_in_csv/processed_with_word_freq.csv'

df = pd.read_csv('../Data_in_csv/995,000_rows_processed.csv')
df['word frequency'] = list(word_freq_df(df))
df.to_csv(output_file, mode='a', index=False)

print(f"Completed! Combined data saved to {output_file}")