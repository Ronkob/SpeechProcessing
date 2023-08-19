from dataclasses import dataclass

import PreProcessing, PhaseTwoModel, Evaluating
import torch
import wandb


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
                "batch_size": config.batch_size,
            }
        )

    wavs, txts = PreProcessing.load_data(mode='train', data_path=PreProcessing.DATA_PATH)

    # Now you can create a Dataset and DataLoader for your data
    dataset = PhaseTwoModel.AudioDatasetPhaseTwo(wavs, txts)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)  # adjust the batch size

    wavs, txts = PreProcessing.load_data(mode='test', data_path=PreProcessing.DATA_PATH)

    test_dataset = PhaseTwoModel.AudioDatasetPhaseTwo(wavs, txts)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)

    model = PhaseTwoModel.PhaseTwoModel(config)
    PhaseTwoModel.train_model_phase_two(model, train_dataloader, test_dataloader, config)
    Evaluating.evaluate_model(model, test_dataloader)


@dataclass
class Config:
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    wandb_init: bool = False


if __name__ == '__main__':
    config = Config()
    run_phase("PhaseTwo", config=config)
