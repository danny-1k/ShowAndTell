# the flicker dataset from kaggle has some issues


# Fix the sep

f = open('../data/flickr30k/results.csv', 'r').read()
f = f.replace('| ', '|')
open('../data/flickr30k/results.csv', 'w').write(f)