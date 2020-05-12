# pylint: disable=redefined-outer-name
# pylint: disable=W0221

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
import optuna


class LightningNet(pl.LightningModule):
    def __init__(self, trial, config, model, dataloader):
        super().__init__()
        self.model = model(trial, config)
        self.config = config

        self.dataloader = dataloader
        self.hparams = self._get_hparams(trial)
        self.train_dataset = self.dataloader['train']
        self.val_dataset = self.dataloader['val']

    def _get_hparams(self, trial):
        learning_rate = trial.suggest_loguniform('lr', self.config.MIN_LR, self.config.MAX_LR)
        hparams = {
            'lr': learning_rate
        }

        return hparams

    def prepare_data(self):

        try:
            for batch, _ in self.train_dataset:
                test_data = batch
                break
            self.model.calculate_linear_input(test_data)
        except AttributeError:
            print('linear input not have been calculate')

    def forward(self, data):

        return self.model(data)

    def training_step(self, batch, _):
        data, target = batch
        output = self.forward(data)

        return {"loss": F.binary_cross_entropy(output, target)}

    def validation_step(self, batch, _):
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
        return Adam(self.model.parameters(), lr=self.hparams['lr'])

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset


if __name__ == '__main__':
    from .datasets import SampleDataset3D, Transforms, SampleDataset2D
    from .config import Config
    from .models import Simple3DCNN, MyEfficientNet
    from .callbacks import MetricsCallback
    from .utils import get_dataloader

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
        transforms = Transforms()
        datasets = SampleDataset2D(transforms, n_sample=n_sample)
        dataloader = get_dataloader(datasets, config.BATCHSIZE)

        lightning_model = LightningNet(trial, config, MyEfficientNet, dataloader)
        trainer.fit(lightning_model)

        return metrics_callback.metrics[-1]["val_loss"]

    study = optuna.create_study()
    study.optimize(objective, n_trials=config.N_TRIALS)
