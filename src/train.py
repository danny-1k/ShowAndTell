import os
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from data import CocoDataset
from models import ShowAndTellNet
from tokenizer import Tokenizer
from utils import dataloader_collate_fn
from tqdm import tqdm
from time import sleep


def setup(run_name: str, encoder: str, decoder: str):
    """Creates all the necessary folders before training

    Args:
        run_name (str): run_name. Used for seperating different trials.
        encoder (str): Encoder string must be one of "vgg16", "vgg16_bn", "resnet", "inception".
        decoder (str): Decoder string must be one of "lstm", "gru".

    Raises:
        ValueError: Each run must be unique
    """
    run_name = f"{encoder}_{decoder}/{run_name}"
    if not os.path.exists(f"../checkpoints/{run_name}"):
        os.makedirs(f"../checkpoints/{run_name}")
        
    dir_name = os.path.join(f"../runs/{run_name}")
    if os.path.exists(dir_name):
        if len(os.listdir(dir_name)) > 0:
            raise ValueError(f"{dir_name} already exists. Cannot overwrite.")
        else:
            return run_name
    os.makedirs(dir_name)
    return run_name


def run(epochs:int, lr:float, weight_decay:float, num_workers:int, batch_size:int, encoder:str, decoder:str, run_name:str, tokenizer:Tokenizer, embed_size:int, hidden_size:int):
    """_summary_

    Args:
        epochs (int): Number of epochs to train model for.
        lr (float): Learning rate.
        weight_decay (float): Weight decay
        num_workers (int): Num workers for dataloader.
        batch_size (int): Number of mini-batches
        encoder (str): Encoder string
        decoder (str): Decoder string
        run_name (str): run identifier
        tokenizer (Tokenizer): Tokenizer
        embed_size (int): size of embeddings
        hidden_size (int): size of hidden state
    """

    writer = SummaryWriter(
        f"../runs/{setup(run_name, encoder, decoder)}"
    )

    train_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomAffine((-60, 60)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
    )

    train = CocoDataset(train=True, transforms=train_transforms, tokenizer=tokenizer)
    test = CocoDataset(train=False, transforms=test_transforms, tokenizer=tokenizer)

    train = DataLoader(train, batch_size=batch_size,
                       shuffle=True, num_workers=num_workers, collate_fn=dataloader_collate_fn)
    test = DataLoader(test, batch_size=batch_size, num_workers=num_workers, collate_fn=dataloader_collate_fn)

    net = ShowAndTellNet(
        encoder=encoder,
        decoder=decoder,
        vocab_size=len(tokenizer.vocab),
        embed_size=embed_size,
        hidden_size=hidden_size,
    )

    net.to("cuda")

    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    lossfn = nn.CrossEntropyLoss()

    best_loss = float("inf")

    for epoch in range(epochs):
        net.train()

        train_loss_epoch = []
        test_loss_epoch = []

        with tqdm(train, unit="batch") as tepoch:
            for img, caption, lengths in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")

                net.zero_grad()

                img = img.to("cuda")
                caption = caption.to("cuda")
                # target = torch.nn.utils.rnn.pack_padded_sequence(caption, lengths=lengths, batch_first=True).to("cuda")[0].long()

                teacher_forcing = caption[:, :-1]
                target = caption[:, 1:]

                p = net(img, teacher_forcing)

                loss = lossfn(p.view(-1, p.shape[-1]), target.contiguous().view(-1).long())

                # print(loss.item())

                train_loss_epoch.append(loss.item())

                tepoch.set_postfix(loss=loss.item())

                loss.backward()
                optimizer.step()

        with torch.no_grad():
            net.eval()
            with tqdm(test, unit="batch") as tepoch:
                for img, caption, lengths in tepoch:
                    tepoch.set_description(f"Epoch {epoch + 1}")

                    img = img.to("cuda")
                    caption = caption.to("cuda")
                    # target = torch.nn.utils.rnn.pack_padded_sequence(caption, lengths=lengths, batch_first=True).to("cuda")[0].long()

                    teacher_forcing = caption[:, :-1]
                    target = caption[:, 1:]

                    p = net(img, teacher_forcing)

                    loss = lossfn(p.view(-1, p.shape[-1]), target.contiguous().view(-1).long())

                    test_loss_epoch.append(loss.item())
                    tepoch.set_postfix(loss=loss.item())


        train_loss_epoch = sum(train_loss_epoch)/len(train_loss_epoch)
        test_loss_epoch = sum(test_loss_epoch)/len(test_loss_epoch)

        if test_loss_epoch < best_loss:
            torch.save(net.state_dict(), f"../checkpoints/{encoder}_{decoder}/{run_name}/checkpoint.pt")

        writer.add_scalar("Loss/train", train_loss_epoch, global_step=epoch+1)
        writer.add_scalar("Loss/test", test_loss_epoch, global_step=epoch+1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)

    parser.add_argument("--encoder", type=str, default="resnet")
    parser.add_argument("--decoder", type=str, default="gru")

    parser.add_argument("--run_name", type=str, default="001")
    parser.add_argument("--seed", type=int, default=2023)

    parser.add_argument("--tokens", type=str, default="../tokens/tokens.json")

    args = parser.parse_args()

    import random
    import torch
    import numpy as np

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = Tokenizer()
    tokenizer.load(args.tokens)

    run(
        epochs=args.epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        encoder=args.encoder,
        decoder=args.decoder,
        run_name=args.run_name,
        tokenizer=tokenizer,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_workers=args.num_workers
    )
