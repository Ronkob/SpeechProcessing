from dataclasses import dataclass

import PreProcessing, PhaseOneModel, PhaseTwoModel, PhaseThreeModel, Evaluating
import torch
import wandb


def run_phase(PhaseNumber, phase_model_class, config=None):
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
    dataset = PreProcessing.AudioDatasetV3(wavs, txts)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                                   collate_fn=lambda x:
                                                   PreProcessing.process_data(x))
    # batch size

    # wavs, txts = PreProcessing.load_data(mode='test', data_path=PreProcessing.DATA_PATH)
    # test_dataset = PreProcessing.AudioDataset(wavs, txts)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)

    model = PhaseThreeModel.PhaseThreeModel(config,
                                            n_cnn_layers=config.hyperparams['n_cnn_layers'],
                                            n_rnn_layers= config.hyperparams['n_rnn_layers'],
                                            rnn_dim= config.hyperparams['rnn_dim'],
                                            n_class= config.hyperparams['n_class'],
                                            n_feats= config.hyperparams['n_feats'],
                                            stride= config.hyperparams['stride'],
                                            dropout= config.hyperparams['dropout'],
                                            )
    PhaseThreeModel.train_model_phase_three(model, train_dataloader, test_dataloader=None, config=config)
    # Evaluating.evaluate_model(model, test_dataloader)


@dataclass
class Config:
    learning_rate: float = 0.01
    epochs: int = 1
    batch_size: int = 4
    wandb_init: bool = False

    hyperparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 28,
        "n_feats": 13,
        "stride": 2,
        "dropout": 0.1,
    }


if __name__ == '__main__':
    config = Config()
    run_phase("PhaseThree", PhaseThreeModel, config=config)
