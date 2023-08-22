# from dataclasses import dataclass
#
# import PreProcessing, PhaseTwoModel, tmp
# from tmp import PhaseTwoModelTmp
# import torch
# import wandb
# import random
#
# def run_phase(PhaseNumber, config=None):
#     architecture = PhaseNumber + 'Model'
#     # turn on and off wandb logging
#     # start a new wandb run to track this script
#     if config.wandb_init:
#         wandb.init(
#             # set the wandb project where this run will be logged
#             project="speechRecProj",
#
#             # track hyperparameters and run metadata
#             config={
#                 "learning_rate": config.learning_rate,
#                 "architecture": architecture,
#                 "epochs": config.epochs,
#             }
#         )
#
#     wavs, txts = PreProcessing.load_data(mode='train', data_path=PreProcessing.DATA_PATH)
#
#     # Now you can create a Dataset and DataLoader for your data
#     dataset = tmp.AudioDatasetPhaseTwo(wavs, txts)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)  # adjust the batch size as needed
#
#     model = tmp.PhaseTwoModelTmp(config)
#     tmp.train_model_phase_two(model, dataloader, config)
#
#
# @dataclass
# class Config:
#     learning_rate: float = 0.01
#     epochs: int = 50
#     wandb_init: bool = True

import os


def read_files_in_directory(directory_path, words):
    try:
        # Check if the provided path is a directory
        if not os.path.isdir(directory_path):
            print(f"{directory_path} is not a valid directory.")
            return

        # List all files in the directory
        files = os.listdir(directory_path)

        if not files:
            print(f"No files found in {directory_path}")
            return

        # Read and print the content of each file
        print(f"Contents of files in {directory_path}:")
        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()
                    content_words = content.split(' ')
                    for word in content_words:
                        words.add(word)
                    print(f"File: {file_name}\nContent:\n{content}\n")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def write_words_to_file(words):
    try:
        # Check if the provided path is a directory
        if not os.path.isdir(directory_path):
            print(f"{directory_path} is not a valid directory.")
            return

        # Create a file in the directory to write the words
        file_path = "words.txt"
        with open(file_path, 'w') as file:
            # Write each word to the file with a newline separator
            file.write('\n'.join(words))

        print(f"Words written to {file_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    # config = Config()
    # run_phase("PhaseTwo", config=config)

    directory_path = "an4/an4/train/an4/txt"
    words = set()

    read_files_in_directory(directory_path, words)

    write_words_to_file(words)


