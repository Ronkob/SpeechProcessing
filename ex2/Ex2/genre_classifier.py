from abc import abstractmethod
import torch
from enum import Enum
import typing as tp
from dataclasses import dataclass


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
        raise NotImplementedError("function is not implemented")

    def exctract_feats(self, wavs: torch.Tensor):
        """
        this function extract features from a given audio.
        we will not be observing this method.
        """
        raise NotImplementedError("optional, function is not implemented")

    def forward(self, feats: torch.Tensor) -> tp.Any:
        """
        this function performs a forward pass throuh the model, outputting scores for every class.
        feats: batch of extracted faetures
        """
        # raise NotImplementedError("optional, function is not implemented")

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
        raise NotImplementedError("optional, function is not implemented")

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



    @staticmethod
    def get_pretrained_model() -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's trained weights / 
        hyperparameters and return the loaded model
        """
        raise NotImplementedError("function is not implemented")
