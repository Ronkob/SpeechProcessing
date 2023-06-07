import os
from abc import abstractmethod

import librosa
import numpy as np
import torch
import typing as tp
from dataclasses import dataclass


@dataclass
class ClassifierArgs:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with 
    default values (so run won't break when we test this).
    """
    # we will use this to give an absolute path to the data, make sure you read the data using this argument. 
    # you may assume the train data is the same
    path_to_training_data_dir: str = "./Resources/train_files"
    # path_to_training_data_dir: str = "./train_files" TODO: change this to the correct path

    # you may add other args here


class DigitClassifier():
    """
    You should Implement your classifier object here
    """

    def __init__(self, args: ClassifierArgs):
        self.path_to_training_data = args.path_to_training_data_dir
        self.data, self.labels = self.load_data()
        self.features = self.get_features(self.data)  # shape (batch, channels, time)

    def get_features(self, audio_batch):
        """
        function to calculate the mfcc for each audio file
        audio_batch: a batch of audio files
        return: Tensor of mfcc features of shape [Batch, Channels, Time]
        """
        # calculate the mfcc for each audio file
        mfccs = []
        for i in range(len(audio_batch)):
            mfccs.append(librosa.feature.mfcc(y=audio_batch[i].numpy()))

        # mfccs_standardized = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1,
        #                                                                               keepdims=True)

        mfcc = torch.Tensor(np.asarray(mfccs))

        return mfcc

    def load_data(self):
        """
        function to load the training data
        """
        # load the data
        data = []
        classes = ['one', 'two', 'three', 'four', 'five']
        for i, class_ in enumerate(classes):
            path = self.path_to_training_data + '/' + class_
            # only .wav files
            files = [file for file in os.listdir(path) if file.endswith('.wav')]
            # files = os.listdir(path)
            for file in files:
                data.append((librosa.util.normalize(librosa.load(path + '/' + file)[0]), i + 1))

        # load the data into a tensor of size (batch, channels, time)
        audio_data = torch.Tensor(np.asarray([data[i][0] for i in range(len(data))]))
        labels = torch.Tensor(np.asarray([data[i][1] for i in range(len(data))]))

        return audio_data, labels

    @abstractmethod
    def classify_using_eucledian_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[
        int]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        # convert the input to a list of tensors
        if isinstance(audio_files, list):
            audio_files = [librosa.util.normalize(librosa.load(file)[0]) for file in audio_files]
            audio_files = [torch.from_numpy(file) for file in audio_files]
            audio_files = torch.stack(audio_files)

        features = self.get_features(audio_files)

        # calculate the distance between each file and the training data
        distances = []
        for mfcc in features:
            vector_distances = []
            for train_example in self.features:
                vector_distances.append(torch.dist(mfcc, train_example))
            distances.append(torch.Tensor(vector_distances))

        distances = torch.stack(distances)

        # find the minimum distance for each file
        arg_min_distances = torch.argmin(distances, dim=1)
        predictions = [self.labels[i] for i in arg_min_distances]

        return predictions

    @abstractmethod
    def classify_using_DTW_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using DTW distance
        audio_files: list of audio file paths or a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        # convert the input to a list of tensors
        if isinstance(audio_files, list):
            audio_files = [librosa.util.normalize(librosa.load(file)[0]) for file in audio_files]
            audio_files = [torch.from_numpy(file) for file in audio_files]
            audio_files = torch.stack(audio_files)

        mfcc = self.get_features(audio_files)

        # calculate the distance between each file and the training data
        distances = []
        for audio_file in mfcc:
            vector_distances = []
            for train_example in self.features:
                vector_distances.append(self.dtw(audio_file, train_example))
            distances.append(torch.Tensor(vector_distances))

        distances = torch.stack(distances)

        # find the minimum distance for each file
        arg_min_distances = torch.argmin(distances, dim=1)
        predictions = [self.labels[i] for i in arg_min_distances]

        return predictions

    @abstractmethod
    def classify(self, audio_files: tp.List[str]) -> tp.List[str]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of ABSOLUTE audio file paths
        return: a list of strings of the following format: '{filename} - {predict using euclidean distance}
        - {predict using DTW distance}'
        Note: filename should not include parent path, but only the file name itself.
        """
        euclidean_predictions = self.classify_using_eucledian_distance(audio_files)
        print("euclidean_predictions: ")
        dtw_predictions = self.classify_using_DTW_distance(audio_files)
        print("dtw_predictions: ")

        output = []
        for i in range(len(audio_files)):
            output.append(f'{audio_files[i]} - {euclidean_predictions[i]} - {dtw_predictions[i]}')

        print("Eucledian accuracy: "
              , self.get_accuracy(euclidean_predictions, torch.ones(len(euclidean_predictions)).long()))
        print("DTW accuracy: "
                , self.get_accuracy(dtw_predictions, torch.ones(len(dtw_predictions)).long()))
        return output

    # a function that gets a list of predictions, and a list of labels and returns the accuracy
    @staticmethod
    def get_accuracy(predictions: tp.List[int], labels: tp.List[int]) -> float:
        """
        function to calculate the accuracy of the model
        predictions: list of predictions
        labels: list of labels
        return: accuracy
        """
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == labels[i]:
                correct += 1

        return correct / len(predictions)
    def dtw(self, seq1, seq2):
        """
        This function should return the DTW distance between the two sequences
        x: torch.tensor [13, 520]. this is the mfcc
        y: torch.tensor [13, 520]. this is the mfcc
        """
        # fill
        N = len(seq1[0])
        cost = np.zeros((N, N))
        for row in range(N):
            for j in range(N):
                x_frame = seq1[:, row]
                y_frame = seq2[:, row]
                cost[row, j] = torch.norm(x_frame - y_frame)
        # extract
        accumulated = np.zeros((N, N))
        accumulated[0, 0] = cost[0, 0]
        # first row
        for col in range(1, N):
            accumulated[0, col] = cost[0, col] + accumulated[0, col - 1]
        # first col
        for row in range(1, N):
            accumulated[row, 0] = cost[row, 0] + accumulated[row - 1, 0]
        # rest
        for row in range(1, N):
            for col in range(1, N):
                accumulated[row, col] = cost[row, col] + min(accumulated[row - 1, col],
                                                             accumulated[row, col - 1],
                                                             accumulated[row - 1, col - 1])
        return accumulated[-1, -1]


class ClassifierHandler:

    @staticmethod
    def get_pretrained_model() -> DigitClassifier:
        """
        This function should load a pretrained / tuned 'DigitClassifier' object.
        We will use this object to evaluate your classifications
        """
        raise NotImplementedError("function is not implemented")


# test dtw on small matrix:
def test_dtw():
    N = 4
    cost = torch.Tensor([[1, 20, 1, 1], [20, 1, 500, 1], [500, 500, 500, 1], [500, 500, 500, 1]])
    # extract
    accumulated = np.zeros((N, N))
    accumulated[0, 0] = cost[0, 0]
    # first row
    for col in range(1, N):
        accumulated[0, col] = cost[0, col] + accumulated[0, col - 1]
    # first col
    for row in range(1, N):
        accumulated[row, 0] = cost[row, 0] + accumulated[row - 1, 0]
    # rest
    for row in range(1, N):
        for col in range(1, N):
            accumulated[row, col] = cost[row, col] + min(accumulated[row - 1, col],
                                                         accumulated[row, col - 1],
                                                         accumulated[row - 1, col - 1])
    print(accumulated[-1, -1])
    assert accumulated[-1, -1] == 25


if __name__ == '__main__':
    # # load 2 files of the same digit
    # x, sr = librosa.load('Resources/train_files/five/6ceeb9aa_nohash_0.wav')
    # y, sr = librosa.load('Resources/train_files/five/9ff1b8b6_nohash_0.wav')
    # # normalize to same volume
    # x = librosa.util.normalize(x)
    # y = librosa.util.normalize(y)
    #
    # # mfcc
    # x_mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
    # y_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # # dtw
    # same_digits_dtw_res = dtw(torch.Tensor(x_mfcc), torch.Tensor(y_mfcc))
    # # eucledian
    # same_digits_euclid_res = torch.norm(torch.Tensor(x_mfcc) - torch.Tensor(y_mfcc))
    # # load 2 files of different digits
    # z, sr = librosa.load('Resources/train_files/four/ad63d93c_nohash_0.wav')
    # z = librosa.util.normalize(z)
    #
    # # mfcc
    # z_mfcc = librosa.feature.mfcc(y=z, sr=sr, n_mfcc=13)
    # # dtw
    # different_digits_dtw_res = dtw(torch.Tensor(x_mfcc), torch.Tensor(z_mfcc))
    # # eucledian
    # different_digits_euclid_res = torch.norm(torch.Tensor(x_mfcc) - torch.Tensor(z_mfcc))
    # # compare
    # assert same_digits_dtw_res < different_digits_dtw_res

    digit_classifier = DigitClassifier(ClassifierArgs())
    # get a list of all filenames in the test directory
    # test_files = [os.path.join('Resources', 'test_files', f) for f in os.listdir('Resources/test_files')[
    # :10]
    #               if f.endswith('.wav')]
    test_files = [os.path.join('Resources', 'tests_labeled', 'tests', 'one', f) for f in os.listdir(
        'Resources/tests_labeled/tests/one')
                  if f.endswith('.wav')]

    # classify
    predictions = digit_classifier.classify(test_files)

    # print predictions
    print('\n'.join(predictions))
