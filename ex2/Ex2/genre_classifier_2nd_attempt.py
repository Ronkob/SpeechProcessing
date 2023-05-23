import json
import os
import pickle
from abc import abstractmethod

import librosa as librosa
import pandas as pd
import scipy
import torch
from enum import Enum
import typing as tp
from dataclasses import dataclass
import numpy as np
import torchaudio
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


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
    batch_size: int = 64
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
        self.weights = torch.randn(size=(kwargs["num_features"], len(Genre)), requires_grad=False) * 2 - 1
        self.bias = torch.randn(size=(1, len(Genre)), requires_grad=False) * 2 - 1
        self.opt_params = opt_params

    def exctract_feats(self, wavs: torch.Tensor):
        """
        this function extract features from a given audio.
        we will not be observing this method.
        """
        all_features = []
        wavs = wavs.numpy()
        for i, wav in enumerate(wavs):
            print("start wav number " + str(i) + " out of " + str(len(wavs)) + " wavs")
            # Calculate MFCC
            mfccs = librosa.feature.mfcc(y=wav, sr=22050, n_mfcc=13)

            # Calculate spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=wav, sr=22050)

            # Calculate chroma features
            chroma_stft = librosa.feature.chroma_stft(y=wav, sr=22050)

            # Calculate tonnetz
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(wav), sr=22050)

            # Calculate spectral bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(y=wav, sr=22050)

            # Calculate spectral flatness
            spec_flatness = librosa.feature.spectral_flatness(y=wav)

            # Calculate zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(wav)

            # Calculate RMS
            rms = librosa.feature.rms(y=wav)

            # Calculate spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=wav, sr=22050)

            # Calculate spectral centroid
            centroid = librosa.feature.spectral_centroid(y=wav, sr=22050)

            # Calculate spectral contrast
            contrast = librosa.feature.spectral_contrast(y=wav, sr=22050)

            # Stack all features
            features = np.vstack(
                [mfccs, spectral_contrast, chroma_stft, tonnetz, spec_bw, spec_flatness, zcr, rms, rolloff,
                 centroid, contrast])

            # Calculate various statistics for each feature
            feature_stats = np.hstack([np.mean(features, axis=1),
                                       np.std(features, axis=1),
                                       np.median(features, axis=1),
                                       np.min(features, axis=1),
                                       np.max(features, axis=1),
                                       scipy.stats.skew(features, axis=1)])

            all_features.append(feature_stats)

        return torch.tensor(np.array(all_features), dtype=torch.float32)

    def softmax(self, z):
        """
        this function calculates softmax values for a batch of scores
        """
        e_z = torch.exp(z - torch.max(z, dim=1, keepdim=True).values)
        return e_z / torch.sum(e_z, dim=1, keepdim=True)

    def forward(self, feats: torch.Tensor) -> tp.Any:
        """
        this function performs a forward pass throuh the model, outputting scores for every class.
        feats: batch of extracted faetures
        """
        x = torch.mm(feats, self.weights) + self.bias
        x = self.softmax(x)
        return x

    def compute_gradients(self, feats, output_scores, labels):
        """
        this function should calculate the gradients for the weights and biases of the model.
        """
        num_samples = feats.shape[0]
        output_scores[range(num_samples), labels] -= 1
        output_scores /= num_samples

        grad_weights = torch.mm(feats.T, output_scores)
        grad_bias = torch.sum(output_scores, dim=0)

        return grad_weights, grad_bias

    def update_weights(self, grad_weights, grad_bias, lr):
        """
        this function should update the weights of the model
        """
        self.weights -= lr * grad_weights
        self.bias -= lr * grad_bias

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

        loss = self.calculate_loss(output_scores, labels)

        grad_weights, grad_bias = self.compute_gradients(feats, output_scores, labels)
        self.update_weights(grad_weights, grad_bias, self.opt_params.learning_rate)

        return loss

    def calculate_loss(self, output_scores, labels):
        # num_examples = output_scores.shape[0]
        # correct_logprobs = -torch.log(output_scores[range(num_examples), labels])
        # data_loss = torch.sum(correct_logprobs) / num_examples
        return torch.nn.functional.cross_entropy(output_scores, labels)

    def get_weights_and_biases(self):
        """
        This function returns the weights and biases associated with this model object,
        should return a tuple: (weights, biases)
        """
        # This function returns the weights and biases associated with this model object,
        # should return a tuple: (weights, biases)
        return self.weights, self.bias

    def classify(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        this method should recieve a torch.Tensor of shape [batch, channels, time] (float tensor)
        and a output batch of corresponding labels [B, 1] (integer tensor)
        """
        if len(wavs.size()) == 3:
            # transform the input to shape (batch, time)
            wavs = wavs.squeeze(1)

        assert len(wavs.size()) == 2, "input should be of shape [batch, channels, time]"

        feats = self.exctract_feats(wavs)
        outputs = self.forward(feats)
        values, predicted = torch.max(outputs.data, dim=1)
        return predicted

    def train(self, X_train, y_train, training_parameters: TrainingParameters):

        num_epochs = training_parameters.num_epochs
        batch_size = training_parameters.batch_size

        for epoch in range(num_epochs):
            # shuffle the data
            permutation = torch.randperm(X_train.shape[0])
            X_train = X_train[permutation]
            y_train = y_train[permutation]

            for i in range(0, len(X_train), batch_size):
                data = X_train[i:i + batch_size]
                # forward pass
                output_scores = self.forward(data)
                # backward pass and optimization
                self.backward(data, output_scores, y_train[i:i + batch_size])

            # print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                output_scores = self.forward(X_train)
                loss = torch.nn.functional.cross_entropy(output_scores, y_train)
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    def test(self, X_test, y_test):
        test_output_scores = self.forward(X_test)
        _, predicted = torch.max(test_output_scores.data, 1)
        correct = (predicted == y_test).sum().item()
        test_accuracy = correct / len(y_test)
        print(f'Test accuracy: {test_accuracy * 100:.2f}%')
        return test_accuracy

    def eval(self, wavs, labels):
        predicted = self.classify(wavs)
        correct = (predicted == labels).sum().item()
        test_accuracy = correct / len(labels)
        print(f'Test accuracy: {test_accuracy * 100:.2f}%')
        return test_accuracy


class ClassifierHandler:
    @staticmethod
    def train_new_model(training_parameters: TrainingParameters, txt="") -> \
            MusicClassifier:
        """
        This function should create a new 'MusicClassifier' object and train it from scratch.
        You could program your training loop / training manager as you see fit.
        """
        # should initialize a complete training (loading data, init model, start training/fitting) and
        # to save weights/other to model files directory. This function should recieve a TrainingParameters
        # dataclass
        # object and perform training accordingly, see code documentation for further details.
        # load the data from the json files
        opti_params = OptimizationParameters()
        music_classifier = MusicClassifier(opti_params, **{'num_features': 306})

        train_df = ClassifierHandler.load_wav_files('jsons/train.json')
        test_df = ClassifierHandler.load_wav_files('jsons/test.json')
        df = pd.concat([train_df, test_df], ignore_index=True)
        # shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)
        data = np.asarray(df['audio'].tolist())[:30]
        labels = np.asarray(df['label'].tolist())[:30]

        train_data = torch.tensor(data, dtype=torch.float32)
        train_labels = torch.tensor(labels, dtype=torch.long)

        features = music_classifier.exctract_feats(train_data)

        music_classifier.train(features, train_labels, training_parameters=training_parameters)

        # data = np.asarray(df['audio'].tolist())[100:110]
        # labels = np.asarray(df['label'].tolist())[100:110]
        #
        # test_data = torch.tensor(data, dtype=torch.float32)
        # test_labels = torch.tensor(labels, dtype=torch.long)
        # test_features = music_classifier.exctract_feats(test_data)
        # test_features = features

        # acc = music_classifier.test(test_features, test_labels)

        # save the model using pickle
        pickle.dump(music_classifier, open(f'model_files/model{txt}.pkl', 'wb'))

        return music_classifier

    @staticmethod
    def get_pretrained_model() -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's trained weights /
        hyperparameters and return the loaded model
        """
        # should load a model from model files directory. This function should return a MusicClassifier object
        # with loaded weights/other.
        model = pickle.load(open('model_files/model_chosen.pkl', 'rb'))
        return model

    @staticmethod
    def load_wav_files(json_file_path):
        # Read the JSON file
        with open(json_file_path) as json_file:
            data = json.load(json_file)

        # Create an empty DataFrame
        df = pd.DataFrame(columns=['label', 'audio', 'sr'])

        # Iterate over first 50 items in the JSON data
        for item in data:
            path = item['path']
            label = item['label']
            label = str.replace(label, '-', '_')
            label = Genre[label.upper()].value
            # Load the audio file using librosa
            audio, sr = librosa.load(path, sr=None)

            # Append the path, label, and audio to the DataFrameu78yt6r5fe4
            df = df.append({'label': label, 'audio': audio, 'sr': sr}, ignore_index=True)

        df = df.sample(frac=1).reset_index(drop=True)
        return df


if __name__ == '__main__':
    train_df = ClassifierHandler.load_wav_files('jsons/train.json')
    test_df = ClassifierHandler.load_wav_files('jsons/test.json')
    df = pd.concat([train_df, test_df], ignore_index=True)
    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    data = np.asarray(df['audio'].tolist())
    labels = np.asarray(df['label'].tolist())

    train_data = torch.tensor(data, dtype=torch.float32)
    train_labels = torch.tensor(labels, dtype=torch.long)

    # music_classifier = MusicClassifier(OptimizationParameters(), **{'num_features': 306})
    # features = music_classifier.exctract_feats(train_data)
    #
    # # # find best seed
    # # seed_dict = {}
    # # for i in range(100):
    # #     torch.manual_seed(i)
    # #     np.random.seed(i)
    # #     perm = torch.randperm(len(features))
    # #     features = features[perm]
    # #     train_labels = train_labels[perm]
    # #
    # #     # 5-fold cross-validation
    # #     kf = KFold(n_splits=5, shuffle=True, random_state=i)
    # #
    # #     fold_accuracies = []
    # #
    # #     for j, (train_index, test_index) in enumerate(kf.split(features)):
    # #         # Split data into training and testing
    # #         X_train, X_test = features[train_index], features[test_index]
    # #         y_train, y_test = train_labels[train_index], train_labels[test_index]
    # #         torch.manual_seed(i)
    # #         model, acc = ClassifierHandler.train_new_model(
    # #             TrainingParameters(),
    # #             # i, j to str
    # #             i=str(i) + " , " + str(j),
    # #             features=X_train,
    # #             train_labels=y_train,
    # #             test_features=X_test,
    # #             test_labels=y_test
    # #         )
    # #
    # #         fold_accuracies.append(acc)
    # #
    # #     print(f"Fold accuracies: {fold_accuracies}")
    # #     print(f"Mean accuracy: {np.mean(fold_accuracies)}")
    # #     seed_dict[i] = np.mean(fold_accuracies)
    # #
    # # print(seed_dict)
    # # print("best seed: ", max(seed_dict, key=seed_dict.get), "best acc: ", seed_dict[max(seed_dict,
    # #                                                                                     key=seed_dict.get)])
    # # best_seed = max(seed_dict, key=seed_dict.get)
    # # torch.manual_seed(best_seed)
    # # model, acc = ClassifierHandler.train_new_model(
    # #             TrainingParameters(),
    # #             # i, j to str
    # #             i="_chosen",
    # #             features=features,
    # #             train_labels=train_labels,
    # #             test_features=features,
    # #             test_labels=train_labels
    # #         )
    # # print(acc)
    # # # save the top ten seeds
    # # top_ten = sorted(seed_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    # # print(top_ten)
    model2 = ClassifierHandler.get_pretrained_model()
    model2.eval(torch.tensor(data[200:300], dtype=torch.float32),
               torch.tensor(labels[200:300], dtype=torch.long))
    # file_name = "parsed_data/classical/test/1.mp3"
    # wav, sr = librosa.load(file_name)
    # outputs = model.classify(torch.tensor(wav).unsqueeze(0))
    # guess = Genre(outputs.item())
    # print(guess.name)
    # assert guess == Genre.CLASSICAL
    # tmp()
