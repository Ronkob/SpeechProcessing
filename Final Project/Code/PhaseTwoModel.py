import os
import numpy as np
import torch
import torchaudio
import wandb
import random
import librosa
import PreProcessing


class PhaseTwoModel(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(PhaseTwoModel, self).__init__()
        self.features_extractor = PreProcessing.FeatureExtractor()
        self.neural_net = NeuralNetAudioPhaseTwo()
        self.ctc_loss = torch.nn.CTCLoss()

    def forward(self, x):
        x = self.features_extractor(x)
        # reshape the tensor to have the batch dimension first
        x = self.neural_net(x)
        return x

    def predict(self, x):
        output = self.forward(x)
        output = torch.nn.functional.softmax(output, dim=2)
        # turn output to the max probability for each time step
        output = torch.argmax(output, dim=2).squeeze(1)
        return output


class NeuralNetAudioPhaseTwo(torch.nn.Module):
    def __init__(self):
        super(NeuralNetAudioPhaseTwo, self).__init__()

        # Define your layers here
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 1), stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1, padding=1)
        self.fc = torch.nn.Linear(192, PreProcessing.NUM_CLASSES + 1)  # adjust this according to your needs

    def forward(self, x):
        # Define your forward pass here
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, (2, 1))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, (2, 1))
        x = torch.permute(x, (3, 0, 1, 2))  # swap the time and channel dimensions
        x = x.view(x.size(0), x.size(1), -1)  # flatten the tensor, keep the time dimension
        x = self.fc(x)
        return x


class AudioDatasetPhaseTwo(torch.utils.data.Dataset):
    def __init__(self, wavs, txts):
        self.wavs = wavs
        self.wav_lengths = [wav.shape[1] for wav in wavs]  # Store original lengths before padding
        # pad the wavs with zeros to make them all the same length
        max_length = max(self.wav_lengths)
        for i, wav in enumerate(wavs):
            wavs[i] = torch.nn.functional.pad(wav, (0, max_length - wav.shape[1]), 'constant', 0)

        self.txts = txts
        self.txt_lengths = [len(txt) for txt in txts]  # Store original lengths before padding
        # pad the txts with spaces to make them all the same length
        max_length = max(self.txt_lengths)
        for i, txt in enumerate(txts):
            txts[i] = txt + 'a' * (max_length - len(txt))

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        wav = self.wavs[idx]
        wav_length = PreProcessing.calculate_num_frames(self.wav_lengths[idx])

        txt = self.txts[idx]
        txt_length = self.txt_lengths[idx]

        # Convert the label to a sequence of integers
        txt_lower = txt.lower()
        label = torch.Tensor(PreProcessing.text_to_labels(txt_lower, PreProcessing.VOCABULARY)).long()

        return (wav, wav_length,), (label, txt_length)


def train_model_phase_two(model, dataloader, config=None):
    # Define optimizer (you may want to adjust parameters according to your needs)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    num_epochs = config.epochs

    for epoch in range(num_epochs):
        running_loss = 0.0
        print("Epoch: ", epoch, "/", num_epochs, " (", epoch / num_epochs * 100, "%)")
        print("First label: ", dataloader.dataset[0][1][0])
        print("Prediction of the first sample in the dataset: ", model(dataloader.dataset[0][0][0].unsqueeze(
            0)))
        print("Model Predict: ", model.predict(dataloader.dataset[0][0][0].unsqueeze(0)))
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            (inputs, input_lengths), (labels, labels_lengths) = data  # todo padding & loading

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)  # todo should be (N, T, Letters) ?
            outputs_logsoft = torch.nn.functional.log_softmax(outputs, dim=1)  # todo log_softmax,
            # todo all are negative

            # Calculate input and target lengths
            input_lengths = torch.tensor(input_lengths, dtype=torch.long)
            target_lengths = torch.tensor(labels_lengths, dtype=torch.long)

            # calculate loss
            loss = model.ctc_loss(outputs_logsoft, labels, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

            if config.wandb_init:
                wandb.log({"running_loss": running_loss, "epoch": epoch, "batch": i,
                       "step": epoch * len(dataloader) + i,
                       "loss": loss.item()})

        print("First label: ", dataloader.dataset[0][1][0])
        print("Prediction of the first sample in the dataset: ", model(dataloader.dataset[0][0][0].unsqueeze(
            0)))

    print('Finished Training')
