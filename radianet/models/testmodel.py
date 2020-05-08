import torch
import torch.nn as nn
import torch.functional as F
import optuna

class Simple3DCNN(nn.Module):

    def __init__(self, trial, config):
        super().__init__()

        self.hparams = self._get_hparams(trial)
        self.config = config
        self.conv_block = self.get_3dconv_block(1, 10, (3,3,3), batchnorm=True, maxpool=True, dropout=True)

    def calculate_linear_input(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        linear_input = x.shape[1]

        self.feedforward_block = self.get_feedforward_block( linear_input)

    @staticmethod
    def _get_hparams( trial):

        n_fw_layers = trial.suggest_int("n_fw_layers", 1, 3)

        output_fw_dim = []
        for i in range(n_fw_layers):
            output_fw_dim.append(int(trial.suggest_loguniform("output_fw_dim{}".format(i), 4, 128)) )

        dropout = trial.suggest_uniform("dropout", 0.2, 0.5)

        hparams = {'n_fw_layers':n_fw_layers,
                    'output_fw_dim':output_fw_dim,
                    'dropout':dropout}

        return hparams

    def get_feedforward_block(self, linear_input, dropout = True):

        layers = []

        n_layers = self.hparams['n_fw_layers']
        input_dim = linear_input


        for i in range(n_layers):

            output_dim = self.hparams['output_fw_dim'][i]

            layers.append(nn.Linear(input_dim, output_dim ))
            layers.append(nn.ReLU())

            if dropout == True:
                layers.append(nn.Dropout(self.hparams['dropout']))

            input_dim = output_dim


        layers.append(nn.Linear(input_dim, self.config.N_CLASSES ))
        layers.append(nn.Sigmoid())

        fw_block = nn.Sequential(*layers)
        return fw_block


    def get_3dconv_block(self, in_channels, out_channels, kernel_size, 
                         batchnorm = False, maxpool = False, dropout = False):
        modules = []
        modules.append(nn.Conv3d(in_channels, out_channels, kernel_size))

        if batchnorm == True:
            modules.append(nn.BatchNorm3d(num_features=out_channels))

        modules.append(nn.ReLU())

        if maxpool == True:
            modules.append(nn.MaxPool3d(2) )

        if dropout == True:
            modules.append(nn.Dropout())

        conv_block = nn.Sequential(*modules)

        return conv_block

    def conv(self, x):
        x = self.conv_block(x)
        return x

    def feedforward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.feedforward_block(x)
        return x


    def forward(self, x):
        x = self.conv(x)
        x = self.feedforward(x)

        return x

if __name__ == "__main__":
    from ..datasets import SampleDataset3D
    from .. import config

    config = config.Config
    datasets = SampleDataset3D()

    train_loader = torch.utils.data.DataLoader(
            datasets.data,
            batch_size= config.BATCHSIZE,
            shuffle=False,
        )
    
    for i in train_loader:
        test_data = i
        print(test_data.shape)
        break

    def objective(trial):

        model = Simple3DCNN(trial, config)
        model.calculate_linear_input(test_data)
        
        print(model(test_data))
        return 5

    study = optuna.create_study()
    study.optimize(objective, n_trials= config.N_TRIALS)

