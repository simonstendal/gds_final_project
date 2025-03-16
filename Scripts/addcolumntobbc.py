import pandas as pd

df = pd.read_csv("../Data_in_csv/bbc_news_contents.csv")

for index, row in df.iterrows():
    df.at[index, "type"] = "reliable"

df.to_csv("../Data_in_csv/bbc_news_train_reliable.csv", index=False)