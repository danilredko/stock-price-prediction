import pandas as pd
import numpy as np
import tensorflow_hub as hub
from tqdm import tqdm

reddit = pd.read_csv("data/reddit.csv").set_index("Date")
yahoo = pd.read_csv("data/yahoo.csv").set_index("Date")
combined_df = reddit.join(yahoo).reset_index()
combined_df.fillna("0.0", inplace=True)
combined_df.pop("Date")

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
top = combined_df.filter(regex="Top.*").to_numpy().tolist()

result = []
for i, row in tqdm(enumerate(top), total=len(top)):

    embedding = embed(row).numpy().flatten().tolist()
    embedding.extend(combined_df.iloc[i][yahoo.columns].to_numpy().tolist())
    embedding.append(combined_df["Label"][i])
    result.append(embedding)

result = np.array(result)

np.save("converted_data.npy", result)
