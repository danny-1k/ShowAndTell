if __name__ == '__main__':
    from argparse import ArgumentParser


    parser = ArgumentParser()
    
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate. Default 1e-4')
    parser.add_argument('--encoder', default='vgg16', help='Visual Encoder. Can be \'vgg16\' or \'vgg19\'. Default vgg16')
    parser.add_argument('--epochs', default=100, type=int, help='Maximum number of epochs to train for. Default 100')
    parser.add_argument('--seq_len', default=100, type=int, help="Maximum number of tokens per sequence. Truncation may occur for some sequences in the training dataset. Default 100")
    parser.add_argument('--save_dir', required=True, help="Where should I save the checkpoints?")
    parser.add_argument('--resume', action='store_true', help='Whether or not to resume training from a checkpoint')
    parser.add_argument('--checkpoint_name', default='best_checkpoint.pth', help='If --resume is true, supply this flag with the name of the checkpoint to load')


    args = parser.parse_args()

    import os
    import torch
    import torch.nn as nn
    from tokenizer import Tokenizer
    from torch.optim import Adam, SGD
    from torch.utils.data import DataLoader
    from torchvision.models.vgg import vgg19, vgg16
    from data import Flickr
    from models import ShowAndTellNet
    from tqdm import tqdm
    from utils import save_checkpoint, load_checkpoint


    tokenizer = Tokenizer('../tokens/flickr', max_len=args)
    tokenizer.load()

    vocab_size = len(tokenizer.tokens)
    encoder = {'vgg16': vgg16, 'vgg19': vgg19}[args.encoder](pretrained=True).eval().cuda()

    for param in encoder.parameters():
        param.requires_grad = False

    net = ShowAndTellNet(vocab_size=vocab_size, encoder=encoder).cuda()

    epochs = args.epochs
    lossfn = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=args.lr)
    
    train = DataLoader(
        Flickr(tokenizer=tokenizer, train=True),
        batch_size=64,
        shuffle=True,
    )

    test = DataLoader(
        Flickr(tokenizer=tokenizer, train=False),
        batch_size=64,
        shuffle=True,
    )

    lowest_loss = float('inf')

    train_loss = 0
    test_loss = 0

    if args.resume:
        if os.path.exists(os.path.join(args.save_dir, args.checkpoint_name)):
            checkpoint = load_checkpoint(os.path.join(args.save_dir, args.checkpoint_name))
            print(f'Loaded Checkpoint at Epoch {checkpoint["epoch"]} & Best Loss {checkpoint["best_loss"]:.3f}')
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_loss = checkpoint['train_loss']
            test_loss = checkpoint['test_loss']
            lowest_loss = checkpoint['best_loss']
            start_epoch = checkpoint['epoch']-1
        else:
            start_epoch = 0
            print('Failed to load checkpoint... Training from scratch')
    else:
        start_epoch = 0


    for epoch in range(start_epoch, epochs):
        net.train()
        for x,y in tqdm(train):
            x = x.cuda()
            y = y.cuda()

            optimizer.zero_grad()
            p = net(x, max_len=args.seq_len) # (N, seq_len, 7689), Y -> (N, seq_len)

            loss = lossfn(
                p.view(p.shape[0]*p.shape[1], -1),
                y.long().view(-1)
            )


            train_loss = (1-.7)*train_loss + .7*loss.item()

            loss.backward()
            optimizer.step()


        with torch.no_grad():
            net.eval()

            for x,y in tqdm(test):
                x = x.cuda()
                y = y.cuda()

                p = net(x, max_len=args.seq_len) # (N, seq_len, 7689), Y -> (N, seq_len)

                loss = lossfn(
                    p.view(p.shape[0]*p.shape[1], -1),
                    y.long().view(-1)
                )


                test_loss = (1-.7)*test_loss + .7*loss.item()


        if test_loss < lowest_loss:
            lowest_loss = test_loss

        checkpoint = {
            'epoch': epoch+1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'best_loss': lowest_loss,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if test_loss == lowest_loss:
            save_checkpoint(checkpoint, os.path.join(args.save_dir, 'best_checkpoint.pth'))
        else:
            save_checkpoint(checkpoint,  os.path.join(args.save_dir, 'checkpoint.pth'))
            
        print(f'EPOCH : {epoch+1} Train Loss : {train_loss:.3f} Test Loss: {test_loss:3f}')
