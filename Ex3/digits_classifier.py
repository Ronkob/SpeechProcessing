import os
from abc import abstractmethod
from typing import List, Tuple
import time
import librosa
import numpy as np
import torch
import typing as tp
from dataclasses import dataclass

from librosa.sequence import dtw
from matplotlib import pyplot as plt


# a decorator to measure the time of a function
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        # prints the time in minutes and seconds and to the 3rd digit after the dot
        print("Execution time: ", round((end_time - start_time) / 60, 0), " minutes and ",
              round((end_time - start_time) % 60, 3), " seconds")

        return ret

    return wrapper  # returns the decorated function


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
    path_to_training_data_dir: str = os.path.join("Resources", "train_files")
    pruning_window_size: int = 40
    # you may add other args here


class DigitClassifier:
    """
    You should Implement your classifier object here
    """

    def __init__(self, args: ClassifierArgs):
        self.path_to_training_data = args.path_to_training_data_dir
        self.data, self.labels = self.load_data_repo(self.path_to_training_data)
        self.features = self.get_features(self.data)  # shape (batch, channels, time)
        self.pruning_window_size = args.pruning_window_size

    @staticmethod
    def get_features(audio_batch: torch.Tensor) -> torch.Tensor:
        """
        function to calculate the mfcc for each audio file
        audio_batch: a batch of audio files
        return: Tensor of mfcc features of shape [Batch, Channels, Time]
        """
        # calculate the mfcc for each audio file
        mfccs = []
        for i in range(len(audio_batch)):
            mfccs.append(librosa.feature.mfcc(y=audio_batch[i].numpy()))

        # convert the list to a tensor
        mfcc = torch.Tensor(np.asarray(mfccs))
        return mfcc

    @staticmethod
    def load_audio_from_list(audio_files: List[str]) -> torch.Tensor:
        """
        function to load a list of audio files
        audio_files: list of audio file paths
        return: Tensor of audio files of shape [Batch, Channels, Time]
        """
        audio_files = [librosa.load(file, mono=True)[0] for file in audio_files]
        audio_files = [torch.from_numpy(file) for file in audio_files]
        audio_files = torch.stack(audio_files)
        return audio_files.unsqueeze(1)

    def load_data_repo(self, path_to_repo: str = None, classes: List[str] = None):
        if path_to_repo is None:
            path_to_repo = self.path_to_training_data
        if classes is None:
            classes = ['one', 'two', 'three', 'four', 'five']
        # load the data
        data = []
        labels = []
        for i, class_ in enumerate(classes):
            path = os.path.join(path_to_repo, class_)
            files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.wav')]
            data += self.load_audio_from_list(files)
            labels += [i + 1] * len(files)

        # load the data into a tensor of size (batch, channels, time), from a list of tensors
        audio_data = torch.stack(data)
        # add a dimension for the channels
        return audio_data, labels

    @abstractmethod
    def classify_using_euclidean_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[
        int]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of audio file paths or a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        predictions = self.classify_1_nearest_neighbor(audio_files, torch.dist)
        return predictions

    @abstractmethod
    def classify_using_DTW_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using DTW distance
        audio_files: list of audio file paths or a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        distance_function = lambda mfcc, train_example: \
            self.dtw_v2(mfcc, train_example, window_size=self.pruning_window_size)[-1, -1]
        predictions = self.classify_1_nearest_neighbor(audio_files, distance_function)

        return predictions

    @abstractmethod
    def classify_using_librosa_DTW_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> \
            tp.List[int]:
        distance_function = lambda mfcc, train_example: dtw(mfcc, train_example, backtrack=False)[-1, -1]
        predictions = self.classify_1_nearest_neighbor(audio_files, distance_function)
        return predictions

    def classify_1_nearest_neighbor(self, audio_files: tp.Union[tp.List[str], torch.Tensor],
                                    distance_function: callable) -> tp.List[int]:
        # convert the input to a list of tensors

        if isinstance(audio_files, list):
            audio_files = self.load_audio_from_list(audio_files)

        assert len(audio_files.shape) == 3, \
            "The input should be a batch of audio files of shape [Batch, Channels, Time]"

        mfcc = self.get_features(audio_files)

        # calculate the distance between each file and the training data
        distances = []
        for mfcc in mfcc:
            vector_distances = []
            for train_example in self.features:
                vector_distances.append(distance_function(mfcc, train_example))
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
        euclidean_predictions = self.classify_using_euclidean_distance(audio_files)
        dtw_predictions = self.classify_using_DTW_distance(audio_files)

        output = []
        for i in range(len(audio_files)):
            output.append(f''
                          f'{os.path.split(audio_files[i])[1]} - {euclidean_predictions[i]} -'
                          f' {dtw_predictions[i]}')

        # save the output to a file
        with open('output.txt', 'w') as f:
            for line in output:
                f.write(line + '\n')

        return output

    @abstractmethod
    @measure_time
    def evaluate(self, audio_files: tp.List[str], labels) -> dict:
        print('Evaluating the model...')
        print('evaluating using DTW distance...')
        dtw_predictions = self.classify_using_DTW_distance(audio_files)
        print('evaluating using librosa DTW distance...')
        librosa_dtw_predictions = self.classify_using_librosa_DTW_distance(audio_files)
        print('evaluating using euclidean distance...')
        euclidean_predictions = self.classify_using_euclidean_distance(audio_files)

        # calculate the accuracy of the model
        dtw_accuracy = self.get_accuracy(dtw_predictions, labels)
        euclidean_accuracy = self.get_accuracy(euclidean_predictions, labels)
        librosa_dtw_accuracy = self.get_accuracy(librosa_dtw_predictions, labels)

        return {'dtw_accuracy': dtw_accuracy, 'euclidean_accuracy': euclidean_accuracy,
                'librosa_dtw_accuracy': librosa_dtw_accuracy, }

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

    @staticmethod
    def dtw_v2(t1, t2, dist=torch.dist, window_size=20):
        """
        :param t1: tensor of shape (channels, features, time1)
        :param t2: tensor of shape (channels, features, time2)
        :param window_size: the size of the window to use in the DTW algorithm
        :param dist: a function that calculates distance between two tensors
        :return: the Dynamic Time Warping distance between t1 and t2
        """

        # Get tensors shapes
        channels, features, time1 = t1.size()
        _, _, time2 = t2.size()

        # Initialize the cost matrix
        dtw_matrix = torch.full((time1 + 1, time2 + 1), float('inf'))

        # Set the starting point to 0
        dtw_matrix[0, 0] = 0

        # Calculate the cost for each possible alignment
        for i in range(1, time1 + 1):
            for j in range(max(1, i - window_size), min(time2 + 1, i + window_size)):
                cost = dist(t1[:, :, i - 1], t2[:, :, j - 1])
                # Update the cost matrix
                dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1],
                                              dtw_matrix[i - 1, j - 1])

        # Return the DTW distance
        return dtw_matrix

    @staticmethod
    def dtw(seq1: torch.Tensor, seq2: torch.Tensor) -> torch.Tensor:
        """
        function to calculate the DTW distance between two sequences
        seq1: first sequence of shape [Channels, Features, Time]
        seq2: second sequence of shape [Channels, Features, Time]
        """

        # unsqueeze the channels dimension
        seq1 = seq1.squeeze(0)
        seq2 = seq2.squeeze(0)

        N, M = len(seq1[0])+1, len(seq2[0])+1

        # Calculate cost matrix
        cost = torch.zeros((N, M))
        for row in range(N):
            for col in range(M):
                x_frame = seq1[:, row]
                y_frame = seq2[:, col]
                cost[row, col] = torch.norm(x_frame - y_frame)

        # Initialize accumulated cost matrix
        accumulated = torch.full((N, M), float('inf'))
        accumulated[0, 0] = cost[0, 0]

        # Calculate accumulated cost matrix
        for row in range(N):
            for col in range(M):
                if row > 0:
                    accumulated[row, col] = min(accumulated[row, col],
                                                accumulated[row - 1, col] + cost[row, col])
                if col > 0:
                    accumulated[row, col] = min(accumulated[row, col],
                                                accumulated[row, col - 1] + cost[row, col])
                if row > 0 and col > 0:
                    accumulated[row, col] = min(accumulated[row, col],
                                                accumulated[row - 1, col - 1] + cost[row, col])

        return accumulated  # Return the final accumulated cost (DTW distance)


class ClassifierHandler:

    @staticmethod
    def get_pretrained_model() -> DigitClassifier:
        """
        This function should load a pretrained / tuned 'DigitClassifier' object.
        We will use this object to evaluate your classifications
        """
        # fill
        return DigitClassifier(ClassifierArgs())


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


def test_dtw2():
    digit_classifier = ClassifierHandler.get_pretrained_model()
    test_files, labels = digit_classifier.load_data_repo(
        path_to_repo=os.path.join('Resources', 'tests_labeled', 'tests'),
        classes=['one', 'two', 'three', 'four', 'five'])

    # plot the result of the DTW distance for the first file and the first example of each class
    for i in range(5):
        cost = digit_classifier.dtw_v2(digit_classifier.get_features(test_files[0]),
                                digit_classifier.get_features(test_files[i]))
        plt.imshow(cost, origin='lower')
        plt.title(f'cost matrix between {labels[0]} and {labels[i]}')
        plt.show()


def evaluate_digit_classifier():
    digit_classifier = ClassifierHandler.get_pretrained_model()
    test_files, labels = digit_classifier.load_data_repo(
        path_to_repo=os.path.join('Resources', 'tests_labeled', 'tests'),
        classes=['one', 'two', 'three', 'four', 'five'])

    output = digit_classifier.evaluate(test_files, labels)
    print(output)


def test_digit_classifier():
    digit_classifier = ClassifierHandler.get_pretrained_model()
    test_files = [os.path.join('Resources', 'test_files', f) for f in os.listdir('Resources/test_files') if
                  f.endswith('.wav')]
    output = digit_classifier.classify(test_files)
    print(output)


def old_tries():
    # load 2 files of the same digit
    x, sr = librosa.load('Resources/train_files/five/6ceeb9aa_nohash_0.wav')
    y, sr = librosa.load('Resources/train_files/five/9ff1b8b6_nohash_0.wav')
    # normalize to same volume
    x = librosa.util.normalize(x)
    y = librosa.util.normalize(y)

    # mfcc
    x_mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
    y_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # dtw
    same_digits_dtw_res = DigitClassifier.dtw(torch.Tensor(x_mfcc), torch.Tensor(y_mfcc))[-1, -1]
    # eucledian
    same_digits_euclid_res = torch.norm(torch.Tensor(x_mfcc) - torch.Tensor(y_mfcc))
    # load 2 files of different digits
    z, sr = librosa.load('Resources/train_files/four/ad63d93c_nohash_0.wav')
    z = librosa.util.normalize(z)

    # mfcc
    z_mfcc = librosa.feature.mfcc(y=z, sr=sr, n_mfcc=13)
    # dtw
    different_digits_dtw_res = DigitClassifier.dtw(torch.Tensor(x_mfcc), torch.Tensor(z_mfcc))[-1, -1]
    # eucledian
    different_digits_euclid_res = torch.norm(torch.Tensor(x_mfcc) - torch.Tensor(z_mfcc))
    # compare
    assert same_digits_dtw_res < different_digits_dtw_res

    digit_classifier = DigitClassifier(ClassifierArgs())
    # get a list of all filenames in the test directory
    test_files = [os.path.join('Resources', 'test_files', f) for f in os.listdir('Resources/test_files')[
                                                                      :10]
                  if f.endswith('.wav')]
    test_files = [os.path.join('Resources', 'tests_labeled', 'tests', 'one', f) for f in os.listdir(
        'Resources/tests_labeled/tests/one')
                  if f.endswith('.wav')]

    # classify
    predictions = digit_classifier.classify(test_files)

    # print predictions
    print('\n'.join(predictions))


if __name__ == '__main__':
    # evaluate_digit_classifier()
    test_digit_classifier()
    # test_dtw2()
