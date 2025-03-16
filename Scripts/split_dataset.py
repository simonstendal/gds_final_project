import sys
import pandas as pd
from sklearn.model_selection import train_test_split

if len(sys.argv) != 2:
    print("Put the name of the input data like this: python split_dataset.py <input_csv>")
    sys.exit(1) 

input_csv = sys.argv[1]

try:
    chunks = []
    for chunk in pd.read_csv(input_csv, chunksize=10000):
        chunks.append(chunk)
    df = pd.concat(chunks)
except FileNotFoundError:
    print(f"Error: File '{input_csv}' not found.")
    sys.exit(1)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

train_df.to_csv('../Data_in_csv/train.csv', index = False)
val_df.to_csv('../Data_in_csv/validation.csv', index=False)
test_df.to_csv('../Data_in_csv/test.csv', index = False)

print('Processed_data.csv has succesfully been split into train.csv, validation.csv, test.csv')