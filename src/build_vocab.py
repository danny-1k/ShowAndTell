import json
from tokenizer import Tokenizer
from tqdm import tqdm


def build_vocab(freq_threshold:int, save:str):
    """Function for building vocabulary from downloaded COCO annotations

    Args:
        freq_threshold (int): The minimum number of times a token must occur in the annotations.
        save (str): Where to save the vocabulary metadata.
    """
    tokenizer = Tokenizer(freq_threshold=freq_threshold)
    dir = "../data/coco/annotations/"

    for f in tqdm(["captions_train2014.json", "captions_val2014.json"]):
        f = dir+f

        captions = json.load(open(f, "r"))["annotations"]

        for caption in tqdm(captions):
            caption = caption["caption"]

            tokenizer.add_words(caption)

    tokenizer.save(save)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--freq", type=int,
                        help="Frequency Threshold", default=0)

    parser.add_argument("--save", type=str, help="File to save tokens as", default="../tokens/tokens.json")

    args = parser.parse_args()
    build_vocab(args.freq, args.save)