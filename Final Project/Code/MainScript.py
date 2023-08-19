from dataclasses import dataclass

import PreProcessing, PhaseTwoModel
import torch
import wandb
import random


def run_phase(PhaseNumber, config=None):
    architecture = PhaseNumber + 'Model'
    # turn on and off wandb logging
    # start a new wandb run to track this script
    if config.wandb_init:
        wandb.init(
            # set the wandb project where this run will be logged
            project="speechRecProj",

            # track hyperparameters and run metadata
            config={
                "learning_rate": config.learning_rate,
                "architecture": architecture,
                "epochs": config.epochs,
            }
        )

    wavs, txts = PreProcessing.load_data(mode='train', data_path=PreProcessing.DATA_PATH)

    # Now you can create a Dataset and DataLoader for your data
    dataset = PhaseTwoModel.AudioDatasetPhaseTwo(wavs, txts)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)  # adjust the batch size as needed

    model = PhaseTwoModel.PhaseTwoModel(config)
    PhaseTwoModel.train_model_phase_two(model, dataloader, config)


@dataclass
class Config:
    learning_rate: float = 0.01
    epochs: int = 50
    wandb_init: bool = True


if __name__ == '__main__':
    config = Config()
    run_phase("PhaseTwo", config=config)
