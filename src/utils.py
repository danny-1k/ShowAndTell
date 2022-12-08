import torch
from PIL import Image

def read_img(f):
    img = Image.open(f)
    return img

def process_text(tokenizer, text):
    output = tokenizer.encode(text)
    output = torch.Tensor(output)

    return output

def save_checkpoint(checkpoint, checkpoint_f):
    torch.save(checkpoint, checkpoint_f)

def load_checkpoint(checkpoint_f, device='cpu'):
    return torch.load(checkpoint_f, map_location=device)