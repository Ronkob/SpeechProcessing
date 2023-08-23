import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

import PreProcessing


def preprocess_data(phrases, vocab):
    # Convert phrases to integer sequences
    input_sequences = [[vocab[char] for char in phrase] for phrase in phrases]
    # input_sequences = [[vocab[word] for word in phrase.split()] for phrase in phrases]
    # Find the maximum sequence length
    max_length = max(len(seq) for seq in input_sequences)

    # Pad sequences to the same length
    input_sequences = [seq + [0] * (max_length - len(seq)) for seq in input_sequences]

    # Prepare input and target sequences
    input_sequences = [torch.tensor(seq[:-1]) for seq in input_sequences]
    target_sequences = [torch.tensor(seq[1:]) for seq in input_sequences]

    # Make sure all sequences are of the same length
    max_length = max(len(seq) for seq in input_sequences)
    input_sequences = [torch.nn.functional.pad(seq, (0, max_length - len(seq))) for seq in input_sequences]
    target_sequences = [torch.nn.functional.pad(seq, (0, max_length - len(seq))) for seq in target_sequences]

    return input_sequences, target_sequences


class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.vocab_size = 27
        self.vocab = {}
        self.set_vocab()
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.embedding(x)
        x, hidden = self.lstm(x)
        x = self.fc(x)
        return x, hidden

    def score_sequence(self, sequence):
        # Convert the sequence to the appropriate format (e.g., tensor of character indices).
        input_sequence = torch.tensor(sequence)

        # Forward pass through your model.
        output, _ = self.forward(input_sequence)

        # Apply softmax to get probabilities.
        probabilities = nn.functional.softmax(output, dim=-1)  # (1, seq_len, vocab_size)

        # Extract the probabilities corresponding to the actual characters in the sequence.
        char_probs = probabilities.gather(1, input_sequence[1:].unsqueeze(-1))

        # Calculate the log probability of the sequence.
        log_prob = torch.log(char_probs).sum().item()

        return log_prob

    def set_vocab(self):
        # Tokenize and flatten
        tokens_string = PreProcessing.VOCABULARY
        tokens = [token for token in tokens_string]
        # tokens = [word for phrase in phrases for word in phrase.split()]
        print(tokens)
        # Create vocabulary
        self.vocab = {word: i for i, (word, _) in enumerate(Counter(tokens).items())}
        # self.vocab_size = len(self.vocab)
        # save the vocab as a pickle file
        with open("vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)


def train_model(model, train_loader, epochs=200):
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss(ignore_index=27)  # ignore index=27? index 27 is blanks

    p_bar = tqdm(range(epochs))
    for epoch in p_bar:
        for inputs, targets in train_loader:
            outputs, _ = model(inputs)
            loss = loss_function(outputs.view(-1, outputs.size(2)), targets.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            p_bar.set_description("Loss: {}".format(loss.item()))

    # Save model if needed
    torch.save(model.state_dict(), "lstm_model.pth")


def build_model_from_phrases(phrases):
    model = LSTMModel(embedding_dim=100, hidden_dim=128)
    input_sequences, target_sequences = preprocess_data(phrases, model.vocab)
    dataset = TensorDataset(torch.stack(input_sequences), torch.stack(target_sequences))
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    train_model(model, train_loader)


def collect_corpus(txt_files, data_path):
    """
    collects the corpus from the txt files
    """
    txts = []
    for txt_file in txt_files:
        txt = open(data_path + 'txt/' + txt_file, 'r')
        label = txt.readline()
        # words = label.split()
        # txts.extend(words)
        txts.append(label.lower())

    return txts


def get_corpus_paths(data_path):
    """
    loads the data from the wav and txt files into
    """
    txt_files = []
    for txt_file in os.listdir(data_path + 'txt/'):
        txt_files.append(txt_file) if txt_file.endswith('.txt') else None
    # sort the lists to make sure they are alligned with same file names
    txt_files.sort()
    return txt_files


def main():
    data_path = 'an4/train/an4/'
    txt_files = get_corpus_paths(data_path)
    corpus = collect_corpus(txt_files, data_path)
    print(corpus)
    # save the corpus as a numpy array
    np.save("corpus.npy", corpus)

    build_model_from_phrases(corpus)


if __name__ == "__main__":
    main()
