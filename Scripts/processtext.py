import re
import pandas as pd
import sys
import nltk
import ssl
import os
from concurrent.futures import ProcessPoolExecutor
import threading

from nltk.corpus import stopwords
from cleantext import clean
from collections import Counter

def initialize_nltk():
    """Initialize NLTK resources if they don't exist."""
    try:
        # Check if resources are already downloaded
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        # Only download if resources are missing
        try:
            _create_unverified_https_context = ssl._create_unverified_context
            ssl._create_default_https_context = _create_unverified_https_context
        except AttributeError:
            pass
        
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('stopwords', quiet=True)

wordPattern = re.compile(r'[!"#€%&/()=?*+´¨^~\[\]{}<>|;:,.-_\[\]`´\s\ ]')
allWords = 0

num_workers = os.cpu_count()
mutex = threading.Lock()

stop_words = set(stopwords.words('english'))
stemmer = nltk.stem.PorterStemmer()

stopword_freq = Counter()
stemmed_freq = Counter()

def update_word_arrays(text):
    global uniqueWords
    if not isinstance(text, str):
        return
    
    tokens = text.split()
    with mutex:
        allWords += len(tokens)

date_patterns = re.compile(
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d+,?\s+\d+\b|'
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+,?\s+\d+\b|'
    r'(\d+)-(\d+)-(\d+) ?(\d*):?(\d*):?(\d*)(\.\d+)?'
)

'''
Here to clean the text data, we are using the cleantext library.
'''
def clean_column(text):
    if not isinstance(text, str):
        return text
    
    text = text.lower()
    text = re.sub(date_patterns, ' <DATE> ', text)

    text = clean(text,
            fix_unicode=True,
            to_ascii=True, 
            no_punct=True,
            no_urls=True,                  
            no_emails=True,                
            no_numbers=True,  
            replace_with_punct= "",             
            no_line_breaks=True,
            replace_with_url=" <URL> ",
            replace_with_email=" <EMAIL> ",
            replace_with_number=" <NUMBER> ",
            lower=True
            )
    return text
    
'''
This function will remove stopwords from the text
'''
def stem_and_stopwords(text):
    if not isinstance(text, str):
        return text
    
    tokens = nltk.word_tokenize(text)
    processed_tokens = []
    for word in tokens:
        if word.lower() not in stop_words:
            if word in ['<', '>']:
                continue
            if word in ['date', 'url', 'number', 'email']:
                continue
            if len(word) == 1:
                continue

            # Count word frequency after stopword removal
            stopword_freq[word.lower()] += 1

            # Stem and count frequency after stemming
            stemmed_word = stemmer.stem(word)
            stemmed_freq[stemmed_word.lower()] += 1
            processed_tokens.append(stemmed_word)

    return ' '.join(processed_tokens)


"""
Process a single text entry (cleaning, stopwords removal, stemming).
"""
def process_text(text):
    text = clean_column(text)
    text = stem_and_stopwords(text)
    return text

def process_df(df):
    """Process the entire DataFrame."""
    
    # Clean, remove stopwords, and stem the text
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        df['content'] = list(executor.map(process_text, df['content']))
    
    # Update word arrays
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(update_word_arrays, df['content'])
    
    return df

# Initialize NLTK resources
initialize_nltk()

'''
This function takes the input csv file and output csv file as arguments.
It reads the input csv file, cleans the text data and saves the cleaned data to the output csv file.
'''
def main():
    if len(sys.argv) != 3:
        print("Usage: python cleantext.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    df_chunks = pd.read_csv(input_csv, usecols=['type','content'], chunksize=10000)

    with open(output_csv, 'w', encoding='utf-8') as f:
        for chunk in df_chunks:
            processed_chunk = process_df(chunk)
            processed_chunk.to_csv(f, index=False, header=f.tell()==0)
    print(f"Cleaned data saved to {output_csv}")

    with open('stopword_freq.txt', 'w', encoding='utf-8') as f:
        for word, freq in stopword_freq.most_common(10000):  # Top 10 most frequent words after stopword removal
            f.write(f"{word}: {freq}\n")

    with open('stemmed_freq.txt', 'w', encoding='utf-8') as f:
        for word, freq in stemmed_freq.most_common(10000):  # Top 10 most frequent stemmed words
            f.write(f"{word}: {freq}\n")

if __name__ == "__main__":
    main()