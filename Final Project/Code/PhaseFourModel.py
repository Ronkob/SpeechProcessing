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


class PhaseFourModel(torch.nn.Module):
    """
    this model will consist of a more complicated acoustic model, but still with no language model,
    only ctc loss
    the architecture will be:
    1. CNN layers
    2. ResNet CNN - for a flatter loss surface
    3. Bidirectional GRU (gated recurrent units - cheaper alternative than LSTM)
    4. MLP head
    """

    def __init__(self, config, input_channels=1, num_classes=PreProcessing.NUM_CLASSES+1):
        super(PhaseFourModel, self).__init__()
        # 2D Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers (you may need to adjust the input size)
        self.fc1 = nn.Linear(128 * 16, 64)  # Adjust based on the output size of the last Conv layer
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        sizes = x.size()
        # Flatten
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, features, time)
        x = x.transpose(1, 2)  # (batch, time, features)

        # Fully connected layers with ReLU and dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # Output layer
        x = self.fc3(x)
        return x


def create_eval_func_beam_search(test_dataloader, config, beam_search):
    def eval_func(model, epoch, running_loss):
        model.eval()
        wer, cer = Evaluating.evaluate_beam_search(model, test_dataloader, beam_search)
        print(f'WER: {wer}, CER: {cer}')
        if config.wandb_init:
            wandb.log({"epoch": epoch,
                       "running_loss": running_loss,
                       "WER": wer,
                       "CER": cer,
                       })
        else:
            print("Epoch: ", epoch, "Running Loss: ", running_loss, "WER: ", wer, "CER: ", cer)

    return eval_func


def train_model_phase_three(model, train_dataloader, criterion, device='cpu', test_dataloader=None,
                            config=None):
    # Define optimizer (you may want to adjust parameters according to your needs)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate,
                                              steps_per_epoch=int(len(train_dataloader)),
                                              epochs=config.epochs + 50,
                                              anneal_strategy='linear')
    num_epochs = config.epochs

    beam_search_decoder = CTCdecoder.create_beam_search_decoder(CTCdecoder.Files())
    eval_func_greedy = create_eval_func_beam_search(test_dataloader, config, beam_search=None)
    eval_func_beam = create_eval_func_beam_search(test_dataloader, config, beam_search=beam_search_decoder)

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

    for epoch in tqdm.tqdm(range(num_epochs, num_epochs + 50)):
        print("Epoch: ", epoch, "/", num_epochs, " (", epoch / num_epochs * 100, "%)")
        run_single_epoch(config, model, optimizer, scheduler, criterion, train_dataloader, device,
                         epoch, eval_function=eval_func_beam)

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

        input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
        target_lengths = torch.as_tensor(labels_lengths, dtype=torch.long)

        # calculate loss
        outputs_to_ctc = outputs_logsoft.transpose(0, 1)  # (time, batch, num_classes)

        loss = criterion(outputs_to_ctc, labels, input_lengths, target_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()

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

