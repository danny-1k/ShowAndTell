# the flicker dataset from kaggle has some issues


# Fix the sep

f = open('../data/flickr30k/results.csv', 'r').read()
f = f.replace('| ', '|')
open('../data/flickr30k/results.csv', 'w').write(f)

import os
import pandas as pd

df = pd.read_csv(os.path.join('../data/flickr30k/results.csv'), sep='|')
df.dropna(axis=0, inplace=True)
df.to_csv('../data/flickr30k/results.csv') # this will save the tsv as a csv (delimeter -> ",")
