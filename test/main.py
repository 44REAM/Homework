

import pytorch_lightning as pl
import optuna

def objective(trial):

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number), "{epoch}"), monitor="val_acc"
    )


    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        logger=False,
        val_percent_check=PERCENT_VALID_EXAMPLES,
        checkpoint_callback=checkpoint_callback,
        max_epochs=EPOCHS,
        gpus=0 if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_acc"),
    )

    model = LightningNet(trial)
    trainer.fit(model)

    return metrics_callback.metrics[-1]["val_acc"]



if __name__ == "__main__":



    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    shutil.rmtree(MODEL_DIR)
