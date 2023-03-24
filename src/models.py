import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg16_bn, resnet152, inception_v3
from typing import Union
from tokenizer import Tokenizer


class ShowAndTellNet(nn.Module):
    def __init__(self, encoder: str, decoder: str, vocab_size: int, embed_size: int, hidden_size: int) -> None:
        """This network consists of an encoder (Image Encoder) and Decoder

        The encoder gets the image and compresses it into a meaningful representation used to `prime` the decoder
        to generate the caption.

        The encoder is a pretrained CNN and the Decoder can be an LSTM or GRU.

        Args:
            encoder (str): Decoder name. (resnet, vgg16, vgg16_bn, inception).
            decoder (str): Decoder name. (lstm, gru).
            vocab_size (int): Size of vocabulary.
            embed_size (int): Size of embedding (For both images and words).
            hidden_size (int): Size of hidden state of decoder.
        """
        super().__init__()
        self.encoder_str  = encoder
        self.decoder_str = decoder
        self.vocab_size = vocab_size

        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.encoder, out_dim = self.fetch_encoder(encoder)
        self.decoder = self.fetch_decoder(decoder)(
            embed_size, hidden_size)

        self.encoder_out = nn.Linear(out_dim, embed_size)
        self.decoder_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Feed forward of Caption generator.
        We encode the images with the pretrained encoder and project it into a lower dimensional embedding which
        we feed into the decoder at the first time step. After the first timestep, we feed the output of the decoder back.

        Note: The implementation of the decoding step using teacher forcing. Which means instead of feeding in the output of the decoder,
        we feed in the target. Teacher forcing has been shown to produce faster convergence.

        Args:
            x (torch.Tensor): Image
            captions (torch.Tensor): Padded Target captions.

        Returns:
            torch.Tensor: Predicted
        """
        
        with torch.no_grad():
            x = self.encoder(x)
            x = x.view(x.shape[0], -1)
            
        x = self.encoder_out(x)  # (batch_size, embed_size)

        batch_size = x.shape[0]

        timesteps = captions.shape[1]

        hidden = self.get_states(batch_size)
        if isinstance(hidden, tuple):
            hidden = (hidden[0].to(x.device), hidden[1].to(x.device))
        else:
            hidden = hidden.to(x.device)

        caption_embed = self.embed(captions.long())

        outputs = torch.zeros((x.shape[0], timesteps, self.vocab_size)).to(x.device)

        for t in range(timesteps):
            if t == 0: # At first timestep, feed in image embedding
                hidden = self.decoder(x, hidden)
            else:
                hidden = self.decoder(caption_embed[:, t, :], hidden)

            out_t = self.decoder_out(hidden[0] if self.decoder_str == "lstm" else hidden)

            outputs[:, t, :] = out_t

        return outputs

    def predict(self, x: torch.Tensor, tokenizer:Tokenizer, max_len:int = 20):

        with torch.no_grad():
            output = []

            # x of shape (1, 224, 224)

            hidden = self.get_states(1)
            if isinstance(hidden, tuple):
                hidden = (hidden[0].to(x.device), hidden[1].to(x.device))
            else:
                hidden = hidden.to(x.device)

            current_token = tokenizer.encode("") # this gives SOS, EOS
            current_token = current_token[:1] #[SOS]
            current_token = torch.Tensor(current_token).unsqueeze(0).long().to(x.device)

            x = self.encoder(x)
            x = x.view(x.shape[0], -1)
            x = self.encoder_out(x)

            # print(x.mean(), x.std())

            for t in range(max_len):
                if t == 0:
                    hidden = self.decoder(x, hidden)
                else:
                    embedded = self.embed(current_token)
                    # print(embedded.shape)
                    hidden = self.decoder(embedded[:, 0, :], hidden)

                # else:
                # current_token_embed = self.embed(current_token)
                # hidden = self.decoder(x, hidden)

                out_t = self.decoder_out(hidden[0] if self.decoder_str == "lstm" else hidden)

                # print(out_t.max())


                output.append(out_t.argmax(1)[0].item())

                current_token = torch.Tensor([output[-1]]).unsqueeze(0).long().to(x.device)

                if output[-1] == tokenizer.token_to_ix["EOS"]:
                    break

        output = tokenizer.decode([str(i) for i in output])

        return output


    def get_states(self, batch_size:int) -> Union[tuple, torch.Tensor]:
        """Method to get hidden states of current decoder
        If the decoder used is an lstm, it returns `(hidden, cell)`
        if the decoder used is a gry, it returns `hidden`
        Args:
            batch_size (int): Current batch size

        Returns:
            Union[tuple, torch.Tensor]: Hidden state
        """

        if self.decoder_str == "lstm":
            h, c = torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
            return (h, c)
        
        else:
            h = torch.zeros(batch_size, self.hidden_size)
            return h

    def fetch_encoder(self, encoder: str) -> nn.Module:
        """Method to fetch pretrained CNN feature extractor.

        Args:
            encoder (str): Encoder string must be one of "vgg16", "vgg16_bn", "resnet", "inception".

        Raises:
            ValueError: Encoder string must be one of "vgg16", "vgg16_bn", "resnet", "inception".

        Returns:
            nn.Module: Decoder class
        """
        if encoder == "vgg16":
            encoder = vgg16(pretrained=True)

            for param in encoder.parameters():
                param.requires_grad_(False)

            return encoder.features, encoder.classifier[0].in_features

        if encoder == "vgg16_bn":
            encoder = vgg16_bn(pretrained=True)

            for param in encoder.parameters():
                param.requires_grad_(False)

            return encoder.features, encoder.classifier[0].in_features

        if encoder == "resnet":
            encoder = resnet152(pretrained=True)

            for param in encoder.parameters():
                param.requires_grad_(False)

            features = nn.Sequential(*list(encoder.children())[:-1])

            for param in features.parameters():
                param.requires_grad_(False)

            return features, encoder.fc.in_features

        if encoder == "inception":

            encoder = inception_v3(pretrained=True)

            for param in encoder.parameters():
                param.requires_grad_(False)

            features = nn.Sequential(*list(encoder.children())[:-1])

            for param in features.parameters():
                param.requires_grad_(False)

            return features, encoder.fc.in_features
        else:
            raise ValueError(
                f"Encoder with encoder name {encoder} does not exist.")

    def fetch_decoder(self, decoder: str) -> nn.Module:
        """Method to fetch a recurrent decoder.
        Because of the implementation of the decoding step in the model, we make use of LSTMCell's and GRUCell's

        Args:
            decoder (str): Decoder string. Must be one of "lstm", "gru".

        Raises:
            ValueError: Decoder string must be one of "lstm", "gru".

        Returns:
            nn.Module: Decoder class
        """
        if decoder == "gru":
            return nn.GRUCell

        if decoder == "lstm":
            return nn.LSTMCell

        else:
            raise ValueError(
                f"Encoder with encoder name {decoder} does not exist.")
