import torch
import torch.nn as nn
from torchvision import transforms
from models import ShowAndTellNet
from PIL import Image

from tokenizer import Tokenizer


def predict(image_path: str, encoder: str, decoder: str, embed_size: int, hidden_size: int, num_layers: int, checkpoint: str, tokenizer: Tokenizer, device: str) -> str:

    img = Image.open(image_path).convert("RGB")

    test_transform = transforms.Compose(
       [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
    )

    net = ShowAndTellNet(
        encoder=encoder,
        decoder=decoder,
        vocab_size=len(tokenizer.vocab),
        embed_size=embed_size,
        hidden_size=hidden_size,
    )

    net.load_state_dict(torch.load(
        f"../checkpoints/{encoder}_{decoder}/{checkpoint}/checkpoint.pt", map_location=device))

    net.to(device)

    img = test_transform(img).unsqueeze(0)
    img = img.to(device)

    output = net.predict(img, tokenizer)

    print(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--encoder", default="resnet", type=str)
    parser.add_argument("--decoder", default="gru", type=str)
    parser.add_argument("--embed_size", default=128, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--checkpoint", default="001", type=str)
    parser.add_argument("--tokens", default="../tokens/tokens.json")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    tokenizer = Tokenizer()
    tokenizer.load(args.tokens)

    predict(args.image, args.encoder, args.decoder, args.embed_size,
            args.hidden_size, args.num_layers, args.checkpoint, tokenizer, args.device)
