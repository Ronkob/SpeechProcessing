# import the LSTMModel class from the directory above
import pickle
from collections import Counter
import PreProcessing

import numpy as np

import LSTMCorpus
import torch
import os


def load_model(embedding_dim=120, hidden_dim=128):
    model = LSTMCorpus.LSTMModel(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    print(os.listdir())
    model.load_state_dict(torch.load("lstm_model.pth"))
    return model


def indices_to_words(indices, vocab):
    reverse_vocab = {v: k for k, v in vocab.items()}
    return [reverse_vocab[idx] for idx in indices]


def predict_next_chars(model, input_text, vocab, num_words=10):
    # Tokenize the input text and convert to tensor
    input_indices = [vocab[char] for char in input_text]
    input_tensor = torch.tensor(input_indices).unsqueeze(0)

    # Pass through the model to get predictions
    with torch.no_grad():
        output = model(input_tensor)

    output = model.softmax(output)

    # Get the predicted word indices
    predicted_indices = output.argmax(dim=2).squeeze().tolist()

    print(predicted_indices)

    # If the result is a single integer, wrap it in a list
    if isinstance(predicted_indices, int):
        predicted_indices = [predicted_indices]

    # Convert indices to words
    predicted_words = indices_to_words(predicted_indices, vocab)

    return "".join(predicted_words[:num_words])


def fix_sequence_to_pred(model, input_seq, vocab):
    """
    this function will take a gibberish sequence and try to complete it, using all different lengths of
    prefixes to predict the postfixes.
    using all of the predicted postfixes, it will return the most likely sentence that the gibberish was
    meant to be.
    """
    # create a list of all possible prefixes
    prefixes = [input_seq[:i] for i in range(1, len(input_seq) + 1)]
    # create a list of all possible postfixes
    postfixes = [input_seq[i:] for i in range(len(input_seq))]

    # create a list of all possible postfixes, predicted by the model
    predicted_postfixes = [model(prefix) for prefix in prefixes]

if __name__ == '__main__':
    # load the vocabulary from vocab.pkl
    model = load_model()

    score = model.score_sequence(PreProcessing.text_to_labels("one two three"))

    print(score)