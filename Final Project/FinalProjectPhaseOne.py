import os
import numpy as np
import torch
import torchaudio

import librosa

DATA_PATH = 'an4/an4/'
NUM_CLASSES = 27  # adjust this according to your needs
VOCABULARY = 'abcdefghijklmnopqrstuvwxyz '

# Constants for the feature extraction
N_MFCC = 13
N_FFT = 400
WIN_LENGTH = 400  # number of samples in each frame
HOP_LENGTH = WIN_LENGTH // 2  # number of samples between successive frames
N_MELS = 120  # number of Mel bands to generate


def text_to_labels(text, vocabulary):
    """
    Converts a string of text into a list of integer labels.

    Arguments:
    text -- the string of text to convert
    vocabulary -- a string of characters that are considered valid labels. The position of each character
    in the string
                  is its label.

    Returns:
    A list of integer labels.
    """
    return [vocabulary.index(char) + 1 for char in text if char in vocabulary]


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
        self.fc = torch.nn.Linear(192, NUM_CLASSES + 1)  # adjust this according to your needs

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


def train_model_phase_one(model, dataloader, num_epochs=10):
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
            outputs_logsoft = torch.nn.functional.log_softmax(outputs, dim=1)

            # Calculate input and target lengths
            input_lengths = torch.full(size=(outputs.shape[1],), fill_value=outputs.shape[0],
                                       dtype=torch.long)
            target_lengths = torch.full(size=(labels.shape[0],), fill_value=labels.shape[1], dtype=torch.long)

            # # print for debugging
            # print("input_lengths: ", input_lengths)
            # print("target_lengths: ", target_lengths)
            # print("outputs shape: ", outputs.size())
            # print("labels shape: ", len(labels))
            # print("outputs logsoft size: ", outputs.size())
            #
            # # Convert labels to numerical values
            # # one_hot_labels = labels_to_one_hot(labels, NUM_CLASSES)
            # print("one hot labels: ", labels)
            #
            # # Calculate loss
            # print("labels: ", labels)
            loss = model.ctc_loss(outputs_logsoft, labels, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')


def train_model_phase_two(model, dataloader, num_epochs=10):
    # Define optimizer (you may want to adjust parameters according to your needs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            (inputs, input_lengths), (labels, labels_lengths) = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs_logsoft = torch.nn.functional.log_softmax(outputs, dim=1)

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

    print('Finished Training')


class FeatureExtractor(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(FeatureExtractor, self).__init__()

    def forward(self, x):
        mfccs = torchaudio.transforms.MFCC(n_mfcc=13,
                                           melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH,
                                                      "n_mels": N_MELS, "center": False,
                                                      "win_length": WIN_LENGTH},
                                           )(x)
        return mfccs


class PhaseOneModel(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(PhaseOneModel, self).__init__()
        self.features_extractor = FeatureExtractor()
        self.neural_net = NeuralNetAudio()
        self.ctc_loss = torch.nn.CTCLoss()

    def forward(self, x):
        x = self.features_extractor(x)
        # reshape the tensor to have the batch dimension first
        x = self.neural_net(x)
        return x


class AudioDatasetPhaseOne(torch.utils.data.Dataset):
    def __init__(self, wavs, txts):
        self.wavs = wavs
        # pad the wavs with zeros to make them all the same length
        max_length = max([wav.shape[1] for wav in wavs])
        for i, wav in enumerate(wavs):
            wavs[i] = torch.nn.functional.pad(wav, (0, max_length - wav.shape[1]), 'constant', 0)

        self.txts = txts
        # pad the txts with spaces to make them all the same length
        max_length = max([len(txt) for txt in txts])
        for i, txt in enumerate(txts):
            txts[i] = txt + ' ' * (max_length - len(txt))

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        wav = self.wavs[idx]
        txt = self.txts[idx]

        # Convert the label to a sequence of integers
        txt_lower = txt.lower()
        label = torch.Tensor(text_to_labels(txt_lower, VOCABULARY)).long()
        # print("converting txt to labels .. ", txt, " -> ", label)

        return wav, label


def calculate_num_frames(signal_length, win_length=WIN_LENGTH, hop_length=HOP_LENGTH):
    return 1 + (signal_length - win_length) // hop_length


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
            txts[i] = txt + ' ' * (max_length - len(txt))

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        wav = self.wavs[idx]
        wav_length = calculate_num_frames(self.wav_lengths[idx])

        txt = self.txts[idx]
        txt_length = self.txt_lengths[idx]

        # Convert the label to a sequence of integers
        txt_lower = txt.lower()
        label = torch.Tensor(text_to_labels(txt_lower, VOCABULARY)).long()

        return (wav, wav_length), (label, txt_length)


if __name__ == '__main__':
    wavs, txts = load_data(mode='test', data_path=DATA_PATH)

    # Now you can create a Dataset and DataLoader for your data
    dataset = AudioDatasetPhaseTwo(wavs, txts)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3)  # adjust the batch size as needed

    model = PhaseOneModel()
    # print(model.forward(wavs[0].unsqueeze(0)))  # unsqueeze to add batch dimension

    train_model_phase_two(model, dataloader)

    output = torch.nn.functional.log_softmax(model.forward(wavs[0].unsqueeze(0)), dim=1)  # unsqueeze to add
    # print the index of the highest probability, which is the predicted label. the output is of shape (
    # time, batch, num_classes+1)
    print("predicted label: ", output.max())
    print("predicted label: ", output.argmax(dim=2))
    print("predicted label: ", output.argmax(dim=2).squeeze(1))  # remove the batch dimension
    print("predicted label: ", output.argmax(dim=2).squeeze(1).tolist())  # convert to list

    # batch dimension
