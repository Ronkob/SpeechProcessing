from dataclasses import dataclass

import PreProcessing, PhaseOneModel, PhaseTwoModel, PhaseThreeModel, Evaluating, PhaseFourModel
import torch
import wandb


# %%
@dataclass
class Config:
    learning_rate: float = 0.001
    epochs: int = 600
    batch_size: int = 32
    wandb_init: bool = False

    hyperparams = {
        "n_cnn_layers": 2,
        "n_rnn_layers": 1,
        "rnn_dim": 32,
        "n_class": PreProcessing.NUM_CLASSES+1,
        "n_feats": 128,
        "stride": 1,
        "dropout": 0.1,
    }

def create_model(PhaseNumber, phase_model_class, config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
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
                "hyperparams": config.hyperparams,
            }
        )

    test_dataloader = None

    # Now you can create a Dataset and DataLoader for your data
    wavs, txts = PreProcessing.load_data(mode='train', data_path=PreProcessing.DATA_PATH)

    dataset = PreProcessing.AudioDatasetV3(wavs, txts)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                                   shuffle=True,
                                                   collate_fn=lambda x:
                                                   PreProcessing.process_data(x))
    wavs, txts = PreProcessing.load_data(mode='test', data_path=PreProcessing.DATA_PATH)
    test_dataset = PreProcessing.AudioDatasetV3(wavs, txts)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size,
                                                  collate_fn=lambda x:
                                                  PreProcessing.process_data(x)
                                                  )

    # model = PhaseThreeModel.PhaseThreeModel(config,
    #                                         n_cnn_layers=config.hyperparams['n_cnn_layers'],
    #                                         n_rnn_layers=config.hyperparams['n_rnn_layers'],
    #                                         rnn_dim=config.hyperparams['rnn_dim'],
    #                                         n_class=config.hyperparams['n_class'],
    #                                         n_feats=config.hyperparams['n_feats'],
    #                                         stride=config.hyperparams['stride'],
    #                                         dropout=config.hyperparams['dropout'],
    #                                         )
    model = PhaseFourModel.PhaseFourModel(config)
    # wavs, txts = PreProcessing.load_data(mode='train', data_path=PreProcessing.DATA_PATH)
    # dataset = PreProcessing.AudioDatasetV2(wavs, txts)
    # train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
    #                                                shuffle=False)
    #
    # wavs, txts = PreProcessing.load_data(mode='test', data_path=PreProcessing.DATA_PATH)
    # test_dataset = PreProcessing.AudioDatasetV2(wavs, txts)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size,
    #                                               shuffle=False)
    # model = PhaseTwoModel.PhaseTwoModel()

    return model, train_dataloader, train_dataloader, device


if __name__ == '__main__':
    torch.manual_seed(42)
    model, train_dataloader, test_dataloader, device = create_model("PhaseThree",
                                                                    PhaseThreeModel.PhaseThreeModel,
                                                                    config=Config(wandb_init=False))
    criterion = torch.nn.CTCLoss(blank=PreProcessing.BLANK_IDX).to(device)
    PhaseThreeModel.train_model_phase_three(model, train_dataloader, criterion, device,
                                            test_dataloader=test_dataloader, config=Config(wandb_init=False))

