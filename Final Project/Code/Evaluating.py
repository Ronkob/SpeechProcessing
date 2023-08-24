import torch
import PreProcessing
from jiwer import wer, cer


def GreedyDecoder(output, labels, label_lengths, blank_label=27, collapse_repeated=True):
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
        # print("Decode: " + str(decode))
        decodes.append(PreProcessing.labels_to_text(decode))
    return decodes, targets


def evaluate_model(model, dataloader, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """
    Evaluate Model
    :param model: (torch.nn.Module) The model to evaluate
    :param dataloader: (torch.utils.data.DataLoader) The dataloader to use
    :return: (float, float) The word error rate and the character error rate
    @brief: This function evaluates the model using the dataloader and returns the word error rate and the
    character
            error rate
    """
    model.to(device)
    model.eval()
    outputs = []
    labels = []
    all_labels_lengths = []
    score_failure_count = 0

    predictions = []
    targets = []

    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        (inputs, input_lengths), (labels_batch, labels_lengths) = data
        inputs, labels_batch = inputs.to(device), labels_batch.to(device)
        labels_lengths = torch.as_tensor(labels_lengths, dtype=torch.long)

        # forward + backward + optimize
        outputs_batch = model(inputs)
        outputs.append(outputs_batch)
        labels.append(labels_batch)
        all_labels_lengths.append(labels_lengths)

        outputs = torch.cat(outputs)
        labels = torch.cat(labels)
        all_labels_lengths = torch.cat(all_labels_lengths)

        outputs, labels = GreedyDecoder(outputs, labels, all_labels_lengths, blank_label=27,
                                        collapse_repeated=True)
        predictions.extend(outputs)
        targets.extend(labels)
        outputs = []
        labels = []
        all_labels_lengths = []

    outputs = predictions
    labels = targets
    # print("Outputs: " + str(outputs) + "\nLabels: " + str(labels))
    try:
        outputs = [output if output.strip() != '' else '<placeholder>' for output in outputs]
        wer_score = wer(outputs, labels)
        cer_score = cer(outputs, labels)
    except Exception as e:
        print("Error calculating wer and cer. Message: {}".format(e))
        print("score failure count: {}".format(score_failure_count))
        score_failure_count += 1
        wer_score = 0
        cer_score = 0
    print('Word Error Rate: ' + str(wer_score))
    print('Character Error Rate: ' + str(cer_score))
    return wer_score, cer_score


def evaluate_beam_search(model, dataloader, beam_search_decoder, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model.to(device)
    model.eval()
    outputs = []
    labels = []
    all_labels_lengths = []
    score_failure_count = 0

    predictions = []
    targets = []

    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        (inputs, input_lengths), (labels_batch, labels_lengths) = data
        inputs, labels_batch = inputs.to(device), labels_batch.to(device)

        # forward + backward + optimize
        outputs_batch = model(inputs)

        # outputs.append(outputs_batch)
        # labels.append(labels_batch)
        # all_labels_lengths.append(labels_lengths)
        #
        # outputs = torch.cat(outputs)
        labels = labels_batch
        # all_labels_lengths = torch.cat(all_labels_lengths)
        if beam_search_decoder:
            hypos = beam_search_decoder(outputs_batch.to('cpu'))
            # outputs = [hypo[0].tokens.tolist() for hypo in hypos]
            # print("Output tokens: " + str(outputs))
            outputs_words = [hypo[0].words for hypo in hypos]
            print("Output words: " + str(outputs_words))
            outputs = [' '.join(words) for words in outputs_words]
            print("Output words: " + str(outputs))
        else:
            outputs, labels = GreedyDecoder(outputs_batch.to(device), labels.to(device),
                                            labels_lengths.to(device),
                                            blank_label=27,
                                            collapse_repeated=True)
        predictions.extend(outputs)
        targets.extend(labels)

    outputs = predictions
    labels = targets
    # print("Outputs: " + str(outputs) + "\nLabels: " + str(labels))
    try:
        outputs = [output if output.strip() != '' else '<placeholder>' for output in predictions]
        wer_score = wer(outputs, labels)
        cer_score = cer(outputs, labels)
    except Exception as e:
        try:
            # outputs = [PreProcessing.labels_to_text(output) for output in predictions]
            outputs = [output.strip() if output.strip() != '' else '<placeholder>' for output in predictions]
            labels = [PreProcessing.labels_to_text(label.tolist()).strip() for label in targets]
            print("Outputs: " + str(outputs) + "\nLabels: " + str(labels))
            wer_score = wer(outputs, labels)
            cer_score = cer(outputs, labels)
        except Exception as e:
            print("Error calculating wer and cer. Message: {}".format(e))
            print("score failure count: {}".format(score_failure_count))
            score_failure_count += 1
            wer_score = 0
            cer_score = 0
    print('Word Error Rate: ' + str(wer_score))
    print('Character Error Rate: ' + str(cer_score))
    return wer_score, cer_score

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
