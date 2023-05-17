import json
from abc import abstractmethod

import librosa as librosa
import pandas as pd
import torch
from enum import Enum
import typing as tp
from dataclasses import dataclass
import numpy as np


class Genre(Enum):
    """
    This enum class is optional and defined for your convinience, you are not required to use it.
    Please use the int labels this enum defines for the corresponding genras in your predictions.
    """
    CLASSICAL: int = 0
    HEAVY_ROCK: int = 1
    REGGAE: int = 2


@dataclass
class TrainingParameters:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with 
    default values (so run won't break when we test this).
    """
    batch_size: int = 32
    num_epochs: int = 100
    train_json_path: str = "jsons/train.json"  # you should use this file path to load your train data
    test_json_path: str = "jsons/test.json"  # you should use this file path to load your test data
    # other training hyper parameters


@dataclass
class OptimizationParameters:
    """
    This dataclass defines optimization related hyper-parameters to be passed to the model.
    feel free to add/change it as you see fit.
    """
    learning_rate: float = 0.001


class MusicClassifier:
    """
    You should Implement your classifier object here
    """

    def __init__(self, opt_params: OptimizationParameters, **kwargs):
        """
        This defines the classifier object.
        - You should defiend your weights and biases as class components here.
        - You could use kwargs (dictionary) for any other variables you wish to pass in here.
        - You should use `opt_params` for your optimization and you are welcome to experiment
        """
        self.weights = torch.zeros(size=(kwargs["num_features"], len(Genre)))
        self.biases = torch.zeros(size=(1, len(Genre)))
        self.opt_params = opt_params

    def exctract_feats(self, wavs: torch.Tensor):
        """
        this function extract features from a given audio.
        we will not be observing this method.
        """
        # this function extract features from a given audio.
        # the first manual feature that will be extracted from the wav to detect it's genre is the zero crossing rate.
        zero_crossing_rate = librosa.feature.zero_crossing_rate(wavs)
        # the second manual feature that will be extracted from the wav to detect it's genre is the spectral centroid.
        spectral_centroid = librosa.feature.spectral_centroid(wavs)
        # the third manual feature that will be extracted from the wav to detect it's genre is the spectral rolloff.
        # the fourth manual feature that will be extracted from the wav to detect it's genre is the spectral bandwidth.
        # the fifth manual feature that will be extracted from the wav to detect it's genre is the spectral contrast.
        # the sixth manual feature that will be extracted from the wav to detect it's genre is the spectral flatness.
        # the seventh manual feature that will be extracted from the wav to detect it's genre is the chroma frequency.
        # the eighth manual feature that will be extracted from the wav to detect it's genre is the chroma deviation.
        # the ninth manual feature that will be extracted from the wav to detect it's genre is the tonnetz.

        return torch.cat((zero_crossing_rate, spectral_centroid), dim=0)
    # softmax function for numerical stability
    @staticmethod
    def softmax(z):
        exp_scores = torch.exp(z)
        return exp_scores / torch.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, feats: torch.Tensor) -> tp.Any:
        """
        this function performs a forward pass throuh the model, outputting scores for every class.
        feats: batch of extracted faetures
        """
        # a function that implements a forward pass through the model, outputting scores for every class.
        # feats: batch of extracted faetures
        # return: scores for every class
        x = feats @ self.weights + self.biases
        return self.softmax(x)

    def backward(self, feats: torch.Tensor, output_scores: torch.Tensor, labels: torch.Tensor):
        """
        this function should perform a backward pass through the model.
        - calculate loss
        - calculate gradients
        - update gradients using SGD

        Note: in practice - the optimization process is usually external to the model.
        We thought it may result in less coding needed if you are to apply it here, hence 
        OptimizationParameters are passed to the initialization function
        """

        num_examples = feats.shape[0]

        # Compute gradients
        d_scores = output_scores
        d_scores[range(num_examples), labels] -= 1
        d_scores /= num_examples

        d_weights = np.dot(feats.T,d_scores)
        d_biases = torch.sum(d_scores, axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= self.opt_params.learning_rate * d_weights
        self.biases -= self.opt_params.learning_rate * d_biases

        # Calculate the loss
        loss = self.calculate_loss(output_scores, labels)

        return loss

    def calculate_loss(self, output_scores, labels):
        num_examples = output_scores.shape[0]
        correct_logprobs = -torch.log(output_scores[range(num_examples), labels])
        data_loss = torch.sum(correct_logprobs) / num_examples
        return data_loss

    def get_weights_and_biases(self):
        """
        This function returns the weights and biases associated with this model object, 
        should return a tuple: (weights, biases)
        """
        # This function returns the weights and biases associated with this model object,
        # should return a tuple: (weights, biases)

    def classify(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        this method should recieve a torch.Tensor of shape [batch, channels, time] (float tensor) 
        and a output batch of corresponding labels [B, 1] (integer tensor)
        """
        # classify : this function performs inference, i.e. returns a integer label corresponding to a given waveform,
        # see class Genre(Enum) inside the code for a label to integer mapping. This function should support batch
        # inference, i.e. classify a torch.Tensor comprising of stacked waveforms of shape [B, 1, T ] denoting B mono
        # (single channel) waveforms of length T . Assume the input is always of this form (shape [B, 1, T ]) where
        # B = 1 in the single example case.

    def train(self, training_parameters: TrainingParameters):

        num_epochs = training_parameters.num_epochs
        batch_size = training_parameters.batch_size
        train_data = ClassifierHandler.load_wav_files(training_parameters.train_json_path)
        test_data = ClassifierHandler.load_wav_files(training_parameters.test_json_path)
        for epoch in range(num_epochs):
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                data = batch['audio']
                labels = batch['label']
                feats = self.exctract_feats(data)
                output_scores = self.forward(feats)
                loss = self.backward(feats, output_scores, labels)
                print('Epoch: {}, Loss: {}'.format(epoch + 1, loss))


class ClassifierHandler:

    @staticmethod
    def train_new_model(training_parameters: TrainingParameters) -> MusicClassifier:
        """
        This function should create a new 'MusicClassifier' object and train it from scratch.
        You could program your training loop / training manager as you see fit.
        """
        # should initialize a complete training (loading data, init model, start training/fitting) and
        # to save weights/other to model files directory. This function should recieve a TrainingParameters dataclass
        # object and perform training accordingly, see code documentation for further details.
        # load the data from the json files

        # load data
        train_data = ClassifierHandler.load_wav_files("jsons/test.json")
        test_data = ClassifierHandler.load_wav_files("jsons/test.json")

        opti_params = OptimizationParameters()

        music_classifier = MusicClassifier(opti_params, **{'num_features': 1})

        music_classifier.train(training_parameters)

        # save the model
        torch.save(music_classifier, 'model.pth')

    @staticmethod
    def get_pretrained_model() -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's trained weights / 
        hyperparameters and return the loaded model
        """
        # should load a model from model files directory. This function should return a MusicClassifier object
        # with loaded weights/other.
        model = torch.load('model.pth')
        return model

    @staticmethod
    def load_wav_files(json_file_path):
        # Read the JSON file
        with open(json_file_path) as json_file:
            data = json.load(json_file)

        # Create an empty DataFrame
        df = pd.DataFrame(columns=['label', 'audio', 'sr'])

        # Iterate over each item in the JSON data
        for item in data:
            path = item['path']
            label = item['label']
            label = str.replace(label, '-', '_')
            label = Genre[label.upper()].value
            # Load the audio file using librosa
            audio, sr = librosa.load(path, sr=None)

            # Append the path, label, and audio to the DataFrame
            df = df.append({'label': label, 'audio': audio, 'sr': sr}, ignore_index=True)

        return df


if __name__ == '__main__':
    file_name = "ex2/Ex2/parsed_data/classical/train/1.mp3"
    ClassifierHandler.train_new_model(TrainingParameters())
    model = ClassifierHandler.get_pretrained_model()
    wav = torch.load(file_name)
    outputs = model.classify(wav)
    assert outputs[0] == Genre.CLASSICAL
