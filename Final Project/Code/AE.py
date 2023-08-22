import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from torch.nn.utils.rnn import pad_sequence
import AE_utils

# Check if GPU is available, and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your list of words (training data)
# words = ["because", "example", "corruption", "autoencoder", "implementation", "python", "colab", "notebook", "minor",
#          "reconstruction"]
words = AE_utils.words_file_to_list("words.txt")


# Define a function to introduce minor corruptions to words
def corrupt_word(word):
    return word

    # Randomly shuffle characters within the word
    word_list = list(word)
    random.shuffle(word_list)
    return ''.join(word_list)


# Create a dataset of corrupted and original words
data = [(word, corrupt_word(word)) for word in words]


# Define a function to convert words to tensors of character indices
def word_to_tensor(word, vocab):
    return torch.tensor([vocab[char] for char in [*word]])


# Create a vocabulary mapping characters to indices
vocab = {char: idx for idx, char in enumerate(set(' '.join(words)))}  # todo probably should be all the english letters

# Convert words to tensors and pad them to the maximum word length
max_word_length = max(len(word) for word, _ in data)
data = [(word_to_tensor(original, vocab), word_to_tensor(corrupted, vocab)) for original, corrupted in data]

# Pad sequences to the same length
data = [(original, corrupted, len(original), len(corrupted)) for original, corrupted in data]

# Create DataLoader for training
batch_size = 4
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)


# Define the autoencoder model
class WordAutoencoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(WordAutoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, embedding_dim, batch_first=True)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, x_lengths):
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, x_lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed_embedded)
        padded_encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)
        decoded, _ = self.decoder(padded_encoded)
        output = self.output(decoded)
        return output


# Instantiate the autoencoder model
embedding_dim = 16
hidden_dim = 32
vocab_size = len(vocab)
model = WordAutoencoder(vocab_size, embedding_dim, hidden_dim).to(device)

# Define the loss function (CrossEntropyLoss)
loss_fn = nn.CrossEntropyLoss()  # todo what about using CER

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# todo

word = "FOURTEEN"
word_vec = word_to_tensor(word, vocab)
out = model(word_vec, 8)


# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        original, corrupted, original_lengths, _ = batch
        original, corrupted = original.to(device), corrupted.to(device)

        optimizer.zero_grad()

        # Forward pass
        reconstructed = model(corrupted, original_lengths)

        # Compute the loss
        loss = loss_fn(reconstructed.view(-1, vocab_size), original.view(-1))

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {total_loss / len(dataloader):.4f}")


# Inference example
def reconstruct_word(word):
    word_tensor = word_to_tensor(word, vocab).unsqueeze(0).to(device)
    word_length = torch.tensor([len(word.split())], dtype=torch.int64).to(device)
    reconstructed = model(word_tensor, word_length)
    _, indices = torch.max(reconstructed, dim=2)
    decoded_word = ''.join([list(vocab.keys())[list(vocab.values()).index(idx)] for idx in indices.squeeze().tolist()])
    return decoded_word


original_word = "because"
corrupted_word = corrupt_word(original_word)
reconstructed_word = reconstruct_word(corrupted_word)

print(f"Original Word: {original_word}")
print(f"Corrupted Word: {corrupted_word}")
print(f"Reconstructed Word: {reconstructed_word}")
