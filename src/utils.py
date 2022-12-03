import torch
from PIL import Image

def read_img(f):
    img = Image.open(f)
    return img

def process_text(tokenizer, text):
    output = tokenizer.encode(text)
    output = torch.Tensor(output)

    return output
