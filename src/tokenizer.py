import json
import string


class Tokenizer:
    """Utitlity class for building Vocab.
    Saves the vocab, token mappings and token counts as json.
    """

    def __init__(self, freq_threshold: int = 0) -> None:
        """Initialize tokenizer

        Args:
            freq_threshold (int, optional): The minimum number of occurences each word in the vocab must have. Defaults to 0.
        """
        self.freq_threshold = freq_threshold
        self.vocab = ["SOS", "EOS", "UNK"]
        self.ix_to_token = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.token_to_ix = {"SOS": 0, "EOS": 1, "UNK": 2}
        self.counts = {}

    def add_words(self, sentence: str) -> None:
        """Method to split sentence to words and add the words to the vocab.

        Args:
            sentence (str): sentence to add words from
        """

        sentence = self.remove_punctation(sentence)
        sentence = sentence.replace("\n", "")

        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word: str):
        """Method to add a word to the vocab

        Args:
            word (str): word
        """
        word = word.lower()


        if not self.counts.get(word):
            self.counts[word] = 0

        self.counts[word] += 1

        if word not in self.vocab:
            if self.counts[word] >= self.freq_threshold:
                ix = len(self.vocab)
                self.ix_to_token[ix] = word
                self.token_to_ix[word] = ix
                self.vocab.append(word)


    def save(self, f: str) -> None:
        """Save vocab and converter dicts to a json file.

        Args:
            f (str): file to save as
        """

        json.dump({
            "vocab": self.vocab,
            "freq_treshold": self.freq_threshold,
            "ix_to_token": self.ix_to_token,
            "token_to_ix": self.token_to_ix,
            "counts": self.counts,
        }, open(f, "w"))

    def load(self, f: str) -> None:
        """Load saved meta data from json file

        Args:
            f (str): file to load from
        """

        data = json.load(open(f, "r"))

        try:

            self.vocab = data["vocab"]
            self.freq_threshold = data["freq_treshold"]
            self.ix_to_token = data["ix_to_token"]
            self.token_to_ix = data["token_to_ix"]
            self.counts = data["counts"]

        except:
            raise ValueError(f"Could not load from {f}")

    def decode(self, tokens: list) -> str:
        """Decode list of token indices

        Args:
            tokens (list): Token indices

        Returns:
            str: Decoded text
        """
        decoded = [self.ix_to_token[ix] for ix in tokens]
        return " ".join(decoded)

    def encode(self, text: str) -> list:
        """Encode string into list of token indices

        Args:
            text (str): Text to encode

        Returns:
            list: List of token indices
        """
        encoded = []

        text = self.remove_punctation(text)

        for word in ["SOS", *text.split(" "), "EOS"]:
            if word == "":
                continue
            if word not in self.vocab:
                word = "UNK"
            encoded.append(self.token_to_ix[word])

        return encoded

    def remove_punctation(self, text: str) -> str:
        """Method for removing punction from text

        Args:
            text (str): Text

        Returns:
            str: Clean text
        """
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text
