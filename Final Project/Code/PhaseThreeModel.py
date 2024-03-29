import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchaudio
import wandb
import random
import librosa
import PreProcessing, Evaluating, PhaseTwoModel, CTCdecoder
import tqdm

BLANK_IDX = PreProcessing.BLANK_IDX


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
        n_feats = n_feats // 2
        self.features_extractor = PreProcessing.FeatureExtractorV2()
        # self.net_phase2 = PhaseTwoModel.NeuralNetAudioPhaseTwo()
        # self.ctc_loss = torch.nn.CTCLoss()
        self.cnn = torch.nn.Conv2d(1, 32, kernel_size=3, stride=stride, padding=1)
        self.relu = torch.nn.ReLU()

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
        x = nn.functional.gelu(self.cnn(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

    def predict(self, x):
        # self.eval()
        output = self.forward(x)  # (batch, time, classes)
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
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
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
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


def create_eval_func_beam_search(test_dataloader, kind, config, beam_search):
    def eval_func(model):
        model.eval()
        wer, cer = Evaluating.evaluate_beam_search(model, test_dataloader, beam_search)
        print(f'WER: {wer}, CER: {cer}')
        if config.wandb_init:
            wandb.log({
                       "Evaluation Method: ": "Beam Search" if beam_search is not None else "Greedy",
                       "WER": wer,
                       "CER": cer,
                       "data": kind,
                       })
        else:
            print("WER: ", wer, "CER: ", cer, "data: ", kind, "Evaluation Method: ", "Beam Search" if beam_search is not None else "Greedy")

    return eval_func


def train_model_phase_three(model, train_dataloader, criterion, device='cpu', test_dataloader=None,
                            config=None):
    # Define optimizer (you may want to adjust parameters according to your needs)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = None
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate,
    #                                           steps_per_epoch=int(len(train_dataloader)),
    #                                           epochs=config.epochs + 50,
    #                                           anneal_strategy='linear')
    num_epochs = config.epochs

    beam_search_decoder = CTCdecoder.create_beam_search_decoder(CTCdecoder.Files())
    eval_func_greedy = create_eval_func_beam_search(test_dataloader, 'test', config, beam_search=None)
    eval_func_beam_test = create_eval_func_beam_search(test_dataloader, 'test', config,
                                                      beam_search=beam_search_decoder)
    eval_func_beam_train = create_eval_func_beam_search(train_dataloader, 'train', config,
                                                      beam_search=beam_search_decoder)

    (inputs, input_lengths), (labels, labels_lengths) = next(iter(train_dataloader))
    first_input, first_label, first_label_length = inputs[0].to(device), labels[0].to(device), \
        labels_lengths[0]

    print("First Input: ", first_input.shape, "First Label: ", first_label, "First Label Length: ",
          first_label_length)
    print("Input txt: " + PreProcessing.labels_to_text(first_label))

    # beam_search_decoder = CTCdecoder.create_beam_search_decoder(CTCdecoder.Files())
    for epoch in tqdm.tqdm(range(num_epochs)):
        print(' \n ')
        model_preds = model(first_input.unsqueeze(0).to(device))
        print("model preds: ", model_preds.shape)
        print("Model Output: ", PreProcessing.labels_to_text(torch.argmax(model_preds, dim=2)[0]))

        # print("Model Prediction Greedy: ",
        #       Evaluating.GreedyDecoder(model_preds, [first_label], [first_label_length],
        #                                blank_label=BLANK_IDX, collapse_repeated=True))
        # beam_search_pred = beam_search_decoder(model_preds.to('cpu'))
        # print("Beam Search Prediction: ", beam_search_pred)

        print("Epoch: ", epoch, "/", num_epochs, " (", epoch / num_epochs * 100, "%)")
        run_single_epoch(config, model, optimizer, scheduler, criterion, train_dataloader, device,
                         epoch, eval_function=None)

    eval_func_beam_test(model)
    eval_func_beam_train(model)
    eval_func_greedy(model)
    # for epoch in tqdm.tqdm(range(num_epochs, num_epochs + 50)):
    #     print("Epoch: ", epoch, "/", num_epochs, " (", epoch / num_epochs * 100, "%)")
    #     run_single_epoch(config, model, optimizer, scheduler, criterion, train_dataloader, device,
    #                      epoch, eval_function=eval_func_beam)

    print('Finished Training')


def run_single_epoch(config, model, optimizer, scheduler, criterion, train_dataloader, device,
                     epoch, eval_function=None):
    model.train()
    running_loss = 0.0

    # model_preds = model(first_input.unsqueeze(0))
    # print("model preds: ", model_preds.shape)
    # print("Model Output: ", PreProcessing.labels_to_text(torch.argmax(model_preds, dim=2)[0]))

    # print("Model Prediction: ",
    #       Evaluating.GreedyDecoder(model_preds, [first_label], [first_label_length],
    #                                blank_label=BLANK_IDX, collapse_repeated=True))
    # beam_search_pred = beam_search_decoder(model_preds)
    # print("Beam Search Prediction: ", beam_search_pred)

    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        (inputs, input_lengths), (labels, labels_lengths) = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        outputs_logsoft = torch.nn.functional.log_softmax(outputs, dim=2)
        outputs_to_ctc = outputs_logsoft.transpose(0, 1)  # (time, batch, num_classes)

        # calculate loss
        loss = criterion(outputs_to_ctc, labels, input_lengths, labels_lengths)
        loss.backward()

        optimizer.step()
        # scheduler.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

        if config.wandb_init:
            wandb.log({"epoch": epoch, "batch": i,
                       "step": epoch * len(train_dataloader) + i,
                       "loss": loss.item(),
                       "running_loss": running_loss})

    if eval_function is not None:
        eval_function(model, epoch, running_loss)
