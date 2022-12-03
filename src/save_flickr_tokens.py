import os
import pandas as pd
from tokenizer import Tokenizer
from tqdm import tqdm

tokenizer = Tokenizer('../tokens/flickr')

df = pd.read_csv(os.path.join('../data/flickr30k/results.csv'), sep='|')

df.dropna(axis=0, inplace=True)

comments = df['comment']

for idx in tqdm(range(len(comments))):
  tokenizer.update_tokens(comments.iloc[idx])

tokenizer.save()

df.to_csv('../data/flickr30k/results.csv') # this will save the tsv as a csv (delimeter -> ",")