import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16


class ShowAndTellNet(nn.Module):
    def __init__(self, vocab_size, num_rnn_layers=1, encoder=vgg16(), hidden_size=512, embed_dim=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder.features

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        self.img2h = nn.LSTM(
            input_size=encoder.classifier[0].in_features,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True
        ).to(self.device)

        self.decoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True
        ).to(self.device)

        self.p = nn.Linear(hidden_size, vocab_size).to(self.device)


    def forward(self, x, max_len):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1).unsqueeze(1)

        _, (h, c)  = self.img2h(x)

        sentence = torch.zeros((x.shape[0], max_len, self.vocab_size)).to(self.device)

        S_t = torch.zeros((x.shape[0], 1)).long().to(self.device)

        for t in range(max_len):

            out = self.embed(S_t) # (N, embed_size)
            out, (h, c) = self.decoder(out, (h, c))

            # out of shape (N, 1, hidden_size)

            out = self.p(out[:, -1, :])

            sentence[:, t] = out

            S_t = out.argmax(axis=1, keepdim=True) 


        return sentence


    
if __name__ == '__main__':

    vgg = vgg16(pretrained=True).requires_grad_(False).eval()

    x = torch.zeros((1, 3, 224, 224))

    net = ShowAndTellNet(vocab_size=3, encoder=vgg, hidden_size=5, embed_dim=4)

    print(net(x, 10).shape)