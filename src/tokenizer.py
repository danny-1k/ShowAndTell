import os

class Tokenizer:
    def __init__(self, save_dir, freq_limit=5) -> None:
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.freq_limit = freq_limit


        
        self.tokens = ['START', 'END', 'PAD']

        self.idx_token = {
            0:self.tokens[0],
            1:self.tokens[1],
            2:self.tokens[2],
        }

        self.token_idx = {
            self.tokens[0]: 0,
            self.tokens[1]: 1,
            self.tokens[2]: 2,
        }

        self.counts = {}

    def update_tokens(self, sample) -> None:
        sample = ''.join([s for s in sample.lower() if s.isalpha() or s==' '])
        words = sample.split()

        for word in words:
            if not self.counts.get(word):
                self.counts[word] = 1

            else:
                self.counts[word] += 1

                if self.counts[word] >= self.freq_limit:
                    if word not in self.tokens:
                        self.tokens.append(word)
                        self.idx_token[len(self.tokens)-1] = word
                        self.token_idx[word] = len(self.tokens) - 1


    def save(self) -> None:
        open(os.path.join(self.save_dir, 'tokens'), 'w').write(str(self.tokens))
        open(os.path.join(self.save_dir, 'token_idx'), 'w').write(str(self.token_idx))
        open(os.path.join(self.save_dir, 'idx_token'), 'w').write(str(self.idx_token))
        open(os.path.join(self.save_dir, 'counts'), 'w').write(str(self.counts))


    def load(self) -> None:
        try:
            self.tokens = eval(open(os.path.join(self.save_dir, 'tokens'), 'r').read())
            self.token_idx = eval(open(os.path.join(self.save_dir, 'token_idx'), 'r').read())
            self.idx_token = eval(open(os.path.join(self.save_dir, 'idx_token'), 'r').read())
            self.counts = eval(open(os.path.join(self.save_dir, 'counts'), 'r').read())


        except:
            print(f'Could not load from {self.save_dir}')