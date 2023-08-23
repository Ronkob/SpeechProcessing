import os
import numpy as np
import torch
import torchaudio
from torch import nn

import wandb
import random
import librosa

DATA_PATH = 'an4/'
# DATA_PATH = "/content/drive/MyDrive/Year3/Speach/final/an4/"
NUM_CLASSES = 27  # adjust this according to your needs
BLANK_IDX = NUM_CLASSES  # blank is ?, the last
VOCABULARY = "abcdefghijklmnopqrstuvwxyz ?"  # ? is blank, | is silence

# Constants for the feature extraction
N_MFCC = 20
N_FFT = 400  # number of samples in each fourier transform
WIN_LENGTH = 400  # number of samples in each frame
HOP_LENGTH = WIN_LENGTH // 2  # number of samples between successive frames
N_MELS = 40  # number of Mel bands to generate
N_FEATURES = N_MFCC  # number of features to use


class FeatureExtractor(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(FeatureExtractor, self).__init__()

    def forward(self, x):
        mfccs = torchaudio.transforms.MFCC(n_mfcc=N_MFCC,
                                           melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH,
                                                      "n_mels": N_MELS, "center": False,
                                                      "win_length": WIN_LENGTH})(x)
        return mfccs


class FeatureExtractorV2(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(FeatureExtractorV2, self).__init__()

    def forward(self, x):
        mrfcc_dict = {"n_fft": N_FFT, "hop_length": HOP_LENGTH,
                      "n_mels": N_MELS,
                      "win_length": WIN_LENGTH}

        device = x.device
        mfccs = torchaudio.transforms.MFCC(n_mfcc=N_MFCC,
                                           melkwargs=mrfcc_dict).to(device)(x)
        # add some more features, like the delta and delta-delta
        mfccs_delta = torchaudio.functional.compute_deltas(mfccs)
        mfccs_delta_delta = torchaudio.functional.compute_deltas(mfccs_delta)


        # add even more features to detect syllables, such as: the energy, the pitch, the formants, etc.
        energy = torchaudio.transforms.MFCC(n_mfcc=1, melkwargs=mrfcc_dict).to(device)(x)
        energy_delta = torchaudio.functional.compute_deltas(energy)
        energy_delta_delta = torchaudio.functional.compute_deltas(energy_delta)

        # concatenate the features
        features = torch.cat((mfccs, mfccs_delta, mfccs_delta_delta,
                              energy, energy_delta, energy_delta_delta), dim=2).to(device)
        return features


def calculate_num_frames(signal_length, win_length=WIN_LENGTH, hop_length=HOP_LENGTH):
    return 1 + (signal_length - win_length) // hop_length


def text_to_labels(text, vocabulary=VOCABULARY):
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
    return [vocabulary.index(char) for char in text if char in vocabulary]


def labels_to_text(labels, vocabulary=VOCABULARY):
    """
    Converts a list of integer labels into a string.

    Arguments:
    labels -- the list of integer labels to convert
    vocabulary -- a string of characters that are considered valid labels. The position of each character
    in the string
                  is its label.

    Returns:
    A string of the labels.
    """
    return ''.join([vocabulary[label] for label in labels])


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

    return wavs[:2], txts[:2]


class AudioDatasetV2(torch.utils.data.Dataset):
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

        return (wav, wav_length,), (label, txt_length)


class AudioDatasetV3(torch.utils.data.Dataset):
    def __init__(self, wavs, txts):
        self.wavs = wavs
        self.txts = txts

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        return self.wavs[idx], self.txts[idx]


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=44000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100),
)


def process_data(data):
    spectrograms = []
    labels = []
    labels_lengths = []
    input_lengths = []
    for wav, txt in data:
        # spectogram = torchaudio.transforms.MFCC(n_mfcc=N_FEATURES,
        #                                         melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH,
        #                                                    "n_mels": N_MELS, "center": False,
        #                                                    "win_length": WIN_LENGTH})(wav).squeeze(0)
        spectogram = train_audio_transforms(wav).squeeze(0).transpose(0, 1)
        # spectogram = spectogram.transpose(0, 1)  # (time, channel, feature)
        spectrograms.append(spectogram)
        labels.append(torch.tensor(text_to_labels(txt.lower())))
        labels_lengths.append(len(txt))  # Store original lengths before padding
        input_lengths.append(spectogram.shape[1] // 2)  # Store original lengths before padding

    spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(
        2, 3)  # (# batch, channel, feature, time)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return (spectrograms, input_lengths), (labels, labels_lengths)
