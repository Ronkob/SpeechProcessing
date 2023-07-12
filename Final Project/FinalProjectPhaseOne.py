import os
import numpy as np
import torch
import torchaudio

import librosa

DATA_PATH = 'an4/an4/'


def get_file_paths(data_path):
    """
    loads the data from the wav and txt files into
    """

    wav_files = []
    txt_files = []

    for wav_file in os.listdir(data_path + 'wav/'):
        wav_files.append(wav_file) if wav_file.endswith('.wav') else None

    for txt_file in os.listdir(data_path + 'txt/'):
        txt_files.append(txt_file) if txt_file.endswith('.txt') else None

    # sort the lists to make sure they are alligned with same file names
    wav_files.sort()
    txt_files.sort()

    # check that the lists are the same length
    if len(wav_files) != len(txt_files):
        raise ValueError('wav and txt files are not the same length')

    # # print the first 3 files to make sure they are alligned
    # for i in range(3):
    #     print(wav_files[i], txt_files[i])

    return wav_files, txt_files


def load_data(mode, data_path):
    mode = mode.lower()
    if mode == 'train':
        data_path += 'train/an4/'
    elif mode == 'test':
        data_path += 'test/an4/'
    elif mode == 'validate':
        data_path += 'val/an4/'
    else:
        raise ValueError('mode must be train, test, or validate')

    wav_files, txt_files = get_file_paths(data_path)

    wavs = []
    txts = []

    # load the wav files
    for wav_file in wav_files:
        wav = librosa.load(data_path + 'wav/' + wav_file)[0]  # only load the data, not the sample rate
        wav = torch.from_numpy(wav).float().unsqueeze(0)  # convert to tensor and add channel dimension
        wavs.append(wav)

    # load the txt files
    for txt_file in txt_files:
        txt = open(data_path + 'txt/' + txt_file, 'r')
        label = txt.readline()
        txts.append(label)

    return wavs, txts


class NeuralNetAudio(torch.nn.Module):
    def __init__(self):
        super(NeuralNetAudio, self).__init__()

        # Define your layers here
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = torch.nn.Linear(10560, 10)  # adjust this according to your needs
        self.softmax = torch.nn.Softmax(dim=1)  # adjust this according to your needs

    def forward(self, x):
        # x = x.unsqueeze(1)  # add channel dimension
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = self.fc(x)
        x = self.softmax(x)
        return x


def train_model(model, dataloader, num_epochs=10):
    # Define optimizer (you may want to adjust parameters according to your needs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            # Calculate input and target lengths
            input_lengths = torch.full(size=(1,), fill_value=outputs.shape[0], dtype=torch.long)
            target_lengths = torch.full(size=(1,), fill_value=len(labels), dtype=torch.long)

            # print for debugging
            print("input_lengths: ", input_lengths)
            print("target_lengths: ", target_lengths)
            print("outputs shape: ", outputs.size())
            print("labels shape: ", len(labels))
            print("outputs logsoft size: ", outputs.size())

            # Calculate loss
            loss = model.ctc_loss(outputs, labels, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


class FeatureExtractor(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(FeatureExtractor, self).__init__()

    def forward(self, x):
        mfccs = torchaudio.transforms.MFCC(n_mfcc=13)(x)
        return mfccs


class PhaseOneModel(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(PhaseOneModel, self).__init__()
        self.features_extractor = FeatureExtractor()
        self.neural_net = NeuralNetAudio()
        self.ctc_loss = torch.nn.CTCLoss()

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.neural_net(x)
        return x


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, wavs, txts):
        self.wavs = wavs
        self.txts = txts

        # Create a mapping of characters to unique integers
        self.char_to_int = {char: i + 1 for i, char in enumerate(
            set(''.join(self.txts)))}  # +1 because 0 is reserved for the blank label in CTC

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        wav = self.wavs[idx]
        txt = self.txts[idx]

        # Convert the label to a sequence of integers
        label = [self.char_to_int[char] for char in txt]

        return wav, label


if __name__ == '__main__':
    wavs, txts = load_data(mode='test', data_path=DATA_PATH)

    # Now you can create a Dataset and DataLoader for your data
    dataset = AudioDataset(wavs, txts)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)  # adjust the batch size as needed

    model = PhaseOneModel()
    print(model.forward(wavs[0].unsqueeze(0)))  # unsqueeze to add batch dimension

    train_model(model, dataloader)
