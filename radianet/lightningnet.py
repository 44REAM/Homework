import pytorch_lightning as pl
from pytorch_lightning import Callback
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import optuna
import numpy as np

class LightningNet(pl.LightningModule):
    def __init__(self, trial, config, model, datasets):
        super().__init__()
        self.model = model(trial, config)
        self.hparams = self._get_hparams(trial)
        self.config = config
        self.datasets = datasets

    @staticmethod
    def _get_hparams(trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        hparams = {
            'lr':lr
        }

        return hparams

    def prepare_data(self):

        self.train_dataset = DataLoader(self.datasets['train'], batch_size=self.config.BATCHSIZE)
        self.val_dataset = DataLoader(self.datasets['val'], batch_size=self.config.BATCHSIZE)

        try:
            for batch, _ in self.train_dataset:
                test_data = batch
                break
            self.model.calculate_linear_input(test_data)
        except:
            print('linear input not have been calculate')

    def forward(self, data):

        return self.model(data)

    def training_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)

        return {"loss": F.binary_cross_entropy(output, target)}

    def validation_step(self, batch, batch_nb):
        data, target = batch


        output = self.forward(data)
        output = output.reshape(-1)

        return {"val_loss": F.binary_cross_entropy(output, target)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        # Pass the accuracy to the `DictLogger` via the `'log'` key.
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr = self.hparams['lr'])

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

if __name__ == '__main__':
    from .datasets import SampleDataset3D, Dataset
    from .config import Config
    from .models import Simple3DCNN
    from .callbacks import MetricsCallback

    config = Config

    def objective(trial):

        metrics_callback = MetricsCallback()

        trainer = pl.Trainer(
            logger=False,
            max_epochs=config.EPOCHS,
            callbacks=[metrics_callback],
            gpus=0 if torch.cuda.is_available() else None
        )

        n_sample = 5
        datasets = SampleDataset3D(n_sample = n_sample)
        datasets = datasets.get_dataset()

        lightning_model = LightningNet(trial, config, Simple3DCNN, datasets)
        trainer.fit(lightning_model)

        return metrics_callback.metrics[-1]["val_loss"]

    study = optuna.create_study()
    study.optimize(objective, n_trials= config.N_TRIALS)