import torch
import PreProcessing
from jiwer import wer, cer


def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    """
    Greedy Decoder
    :param output: (torch.tensor) The output from the neural network
    :param labels: (torch.tensor) The labels from the dataset
    :param label_lengths: (torch.tensor) The lengths of the labels
    :param blank_label: (int) The index of the blank label
    :param collapse_repeated: (bool) Whether to collapse repeated characters
    :return: (list, list) The decoded output and the targets
    @brief: This function takes the output from the neural network and decodes it into a list of characters
    """
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(PreProcessing.labels_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(PreProcessing.labels_to_text(decode))
    return decodes, targets


def evaluate_model(model, dataloader):
    """
    Evaluate Model
    :param model: (torch.nn.Module) The model to evaluate
    :param dataloader: (torch.utils.data.DataLoader) The dataloader to use
    :return: (float, float) The word error rate and the character error rate
    @brief: This function evaluates the model using the dataloader and returns the word error rate and the character
            error rate
    """
    outputs = []
    labels = []
    all_labels_lengths = []
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        (inputs, input_lengths), (labels_batch, labels_lengths) = data

        # forward + backward + optimize
        outputs_batch = model(inputs)
        outputs.append(outputs_batch)
        labels.append(labels_batch)
        all_labels_lengths.append(labels_lengths)

    outputs = torch.cat(outputs)
    labels = torch.cat(labels)
    all_labels_lengths = torch.cat(all_labels_lengths)

    outputs, labels = GreedyDecoder(outputs, labels, all_labels_lengths, blank_label=28,
                                    collapse_repeated=True)

    w_er = wer(outputs, labels)
    c_er = cer(outputs, labels)
    print('Word Error Rate: ' + str(w_er))
    print('Character Error Rate: ' + str(c_er))
    return w_er, c_er


# a function that calculates the word error rate
def calculate_wer(outputs, labels):
    wer_sum = 0
    for i in range(len(outputs)):
        wer_sum += wer(outputs[i], labels[i])
    return wer_sum / len(outputs)


# a function that calculates the character error rate
def calculate_cer(outputs, labels):
    cer_sum = 0
    for i in range(len(outputs)):
        cer_sum += cer(outputs[i], labels[i])
    return cer_sum / len(outputs)

