import re
import pandas as pd
import sys
from cleantext import clean
import nltk
from nltk.corpus import stopwords
import ssl

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

datePattern1 = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+[0-9]+,?\s+[0-9]+\b'
datePattern2 = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+[0-9]+,?\s+[0-9]+\b'
datePattern3 = r'([0-9]+)-([0-9]+)-([0-9]+) ?([0-9]*):?([0-9]*):?([0-9]*)(\.[0-9]+)?'

'''
This function will clear stop words, assuming the text has been cleaned.
'''
def remove_stopwords(text):
    if not isinstance(text, str):
        return text
    
    tokens = nltk.word_tokenize(text)
    for word in tokens:
        if word in stopwords.words('english'):
            tokens.remove(word)
    return " ".join(tokens)

'''
This function will perform stemming, assuming the text has been cleaned.
'''
def stem_text(text):
    if not isinstance(text, str):
        return text

    list = text.split(' ')
    stemmer = nltk.stem.PorterStemmer()
    singles = [stemmer.stem(plural) for plural in list]
    return " ".join(singles)

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

def process_df(df):
    object_columns = df.select_dtypes(include=['object']).columns
    int64_columns = df.select_dtypes(include=['int64']).columns
    float64_columns = df.select_dtypes(include=['float64']).columns

    for col in object_columns:
        df[col] = df[col].apply(clean_column)

    print(f"Words after cleaning: {len(all_words(df))}")
    
    for col in object_columns:
        df[col] = df[col].apply(remove_stopwords)

    print(f"Words after removing stopwords: {len(all_words(df))}")
    print(f"Unique words after removing stopwords: {len(unique_words(df))}")

    for col in object_columns:
        df[col] = df[col].apply(stem_text)

    print(f"Words after stemming the data: {len(all_words(df))}")
    print(f"Unique words after stemming the data: {len(unique_words(df))}")

    '''
    for col in int64_columns:
        df[col] = df[col].apply(lambda x: clean_columns(str(x)))
        df[col] = df[col].apply(lambda x: tokenize_columns(str(x)))
    
    for col in float64_columns:
        df[col] = df[col].apply(lambda x: clean_columns(str(x)))
        df[col] = df[col].apply(lambda x: tokenize_columns(str(x)))
    '''
    return df

wordPattern = re.compile(r'[!"#€%&/()=?*+´¨^~\[\]{}<>|;:,.-_\[\]`´\s\ ]')
def all_words(df):
    allWords = []  # List to store all words
    object_columns = df.select_dtypes(include=['object']).columns
    
    for col in object_columns:
        for text in df[col]:  # Iterate over each row in the column
            if not isinstance(text, str):
                continue  # Skip non-string values
            words = text.split()  # Split text into words
            for word in words:
                # Skip words that are entirely non-word symbols (e.g., punctuation)
                if not wordPattern.match(word):
                    allWords.append(word)  # Add valid words to the list
    return allWords

def unique_words(df):
    uniqueWords = set()  # Set to store unique words
    object_columns = df.select_dtypes(include=['object']).columns
    
    for col in object_columns:
        for text in df[col]:  # Iterate over each row in the column
            if not isinstance(text, str):
                continue  # Skip non-string values
            words = text.split()  # Split text into words
            for word in words:
                # Skip words that are entirely non-word symbols (e.g., punctuation)
                if not wordPattern.match(word):
                    uniqueWords.add(word)  # Add valid words to the set
    return uniqueWords

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