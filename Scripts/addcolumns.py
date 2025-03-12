import pandas as pd

df = pd.read_csv('../Data_in_csv/995,000_rows.csv')
processed_df = pd.read_csv('../Data_in_csv/Processed_995,000.csv')

df['content'] = processed_df['content']

f = open('../Data_in_csv/995,000_rows_processed.csv', 'w')
df.to_csv(f, index=False)