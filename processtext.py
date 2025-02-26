import re
import pandas as pd
import sys
from cleantext import clean
import nltk
from nltk.corpus import stopwords
import ssl
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

try:
    stopwords.words('english')
except LookupError:
    print("NLTK stopwords corpus not found. Downloading...")
    nltk.download('stopwords')

wordPattern = re.compile(r'[!"#€%&/()=?*+´¨^~\[\]{}<>|;:,.-_\[\]`´\s\ ]')
allWords = []  # List to store all words
uniqueWords = set()  # Set to store unique words

num_workers = os.cpu_count()
mutex = threading.Lock()

def update_word_arrays(text):
    global allWords, uniqueWords
    if not isinstance(text, str):
        return
    
    tokens = text.split()
    with mutex:
        allWords.extend(tokens)
        uniqueWords.update(tokens)

datePattern1 = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+[0-9]+,?\s+[0-9]+\b'
datePattern2 = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+[0-9]+,?\s+[0-9]+\b'
datePattern3 = r'([0-9]+)-([0-9]+)-([0-9]+) ?([0-9]*):?([0-9]*):?([0-9]*)(\.[0-9]+)?'

'''
Here to clean the text data, we are using the cleantext library.
'''
def clean_column(text):
    if not isinstance(text, str):
        return text
    
    text = text.lower()
    text = re.sub(datePattern3, ' <DATE> ', text, flags=re.IGNORECASE)
    text = re.sub(datePattern2, ' <DATE> ', text, flags=re.IGNORECASE)
    text = re.sub(datePattern1, ' <DATE> ', text, flags=re.IGNORECASE)

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
This function will clear stop words, assuming the text has been cleaned.
'''
def remove_stopwords(text):
    if not isinstance(text, str):
        return text
    
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(filtered_tokens)

'''
This function will perform stemming, assuming the text has been cleaned.
'''
def stem_text(text):
    if not isinstance(text, str):
        return text

    tokens = text.split(' ')
    stemmer = nltk.stem.PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

"""
Process a single text entry (cleaning, stopwords removal, stemming).
"""
def process_text(text):
    text = clean_column(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text

def process_df(df):
    """Process the entire DataFrame."""
    rows = df['content'].dropna().tolist()
    
    # Clean, remove stopwords, and stem the text
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        df['content'] = list(executor.map(process_text, rows))
    
    # Update word arrays
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(update_word_arrays, df['content'])
    
    print(f"Words after processing: {len(allWords)}")
    print(f"Unique words after processing: {len(uniqueWords)}")
    return df
        
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
    
    df = pd.read_csv(input_csv)
    cleaned_df = process_df(df)
    cleaned_df.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv}")

if __name__ == "__main__":
    main()