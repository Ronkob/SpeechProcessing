import os
import numpy as np
import torch
from torch import nn
import torchaudio
import wandb
import random
import librosa
import PreProcessing, Evaluating


class PhaseThreeModel(torch.nn.Module):
    """
    this model will consist of a more complicated acoustic model, but still with no language model,
    only ctc loss
    the architecture will be:
    1. CNN layers
    2. ResNet CNN - for a flatter loss surface
    3. Bidirectional GRU (gated recurrent units - cheaper alternative than LSTM)
    4. MLP head
    """
    def __init__(self, config,
                 n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1,
                 *args, **kwargs):

        super(PhaseThreeModel, self).__init__()
        self.features_extractor = PreProcessing.FeatureExtractor()
        self.ctc_loss = nn.CTCLoss()
        self.cnn = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # n residual cnn layers with filter size of 32

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])

        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)

        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),  # better activation function such as RELU for a flattened surface
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

    def predict(self, x):
        output = self.forward(x) # (batch, time, classes)
        output = torch.nn.functional.softmax(output, dim=2)
        # turn output to the max probability for each time step
        output = torch.argmax(output, dim=2).squeeze(1)  # (batch, time)
        return output


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = nn.functional.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = nn.functional.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = nn.functional.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


def train_model_phase_three(model, train_dataloader, device='cpu', test_dataloader=None, config=None):
    # Define optimizer (you may want to adjust parameters according to your needs)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    num_epochs = config.epochs

    first_label = train_dataloader.dataset[0][1][0]
    first_label_length = train_dataloader.dataset[0][1][1]
    first_wav = train_dataloader.dataset[0][0][0].to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0

        print("Epoch: ", epoch, "/", num_epochs, " (", epoch / num_epochs * 100, "%)")
        print("First label: ", PreProcessing.labels_to_text(first_label))
        model_output = model(first_wav.unsqueeze(0))
        print("Model Output: ", PreProcessing.labels_to_text(model.predict(first_wav.unsqueeze(0))[0]))
        print("Model Prediction: ",
              Evaluating.GreedyDecoder(model_output, [first_label], [first_label_length],
                                       blank_label=28, collapse_repeated=True))

        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            (inputs, input_lengths), (labels, labels_lengths) = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs_logsoft = torch.nn.functional.log_softmax(outputs, dim=2)

            # Calculate input and target lengths
            input_lengths = torch.tensor(input_lengths, dtype=torch.long)
            target_lengths = torch.tensor(labels_lengths, dtype=torch.long)

            # calculate loss
            outputs_to_ctc = outputs_logsoft.transpose(0, 1)  # (time, batch, num_classes)
            loss = model.ctc_loss(outputs_to_ctc, labels, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

            if config.wandb_init:
                wandb.log({"epoch": epoch, "batch": i,
                           "step": epoch * len(train_dataloader) + i,
                           "loss": loss.item()})

        if test_dataloader is not None:
            wer, cer = Evaluating.evaluate_model(model, test_dataloader)
            print(f'WER: {wer}, CER: {cer}')
            wandb.log({"epoch": epoch,
                       "running_loss": running_loss,
                       "WER": wer,
                       "CER": cer,
                       })

    print('Finished Training')
