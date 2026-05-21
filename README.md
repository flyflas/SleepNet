# Sleep PyTorch Training

## Epoch Context Mode

Training and evaluation now use offline bidirectional epoch-context samples instead of independent single-epoch samples. Each sample is a fixed window centered on one epoch, and the target label is always the label of that center epoch only.

Default context settings are defined in `args.py`:

- `context_left = 2`
- `context_right = 2`
- `context_length = 5`
- `context_center_index = 2`
- `split_group_policy = "subject"`

Frequency-only context inputs have shape `[N, 5, 3, 29, 128]`. Frequency+time context inputs have shape `[N, 5, 3, 2, 29, 128]`. Context windows are built inside a single PSG record, never across record boundaries. Epochs at the start and end of each record that cannot form a full context window are dropped; the training logs report the dropped boundary epoch count.

Fold and validation splits are assigned by subject group before context windows are constructed, preventing the same PSG record or subject group from appearing in multiple splits. Evaluation reconstructs the same group-safe context-window contract and reports metrics against center-epoch labels.

Context runs use the `transformer_epoch_context` model name and a context-specific `Kfold_models` output root. Old single-epoch checkpoints are not expected to load in this architecture unless checkpoint compatibility is explicitly implemented.

Run commands:

```bash
python Kfold_trainer.py
python result_evaluate_flexible_new.py
```

Useful verification commands:

```bash
python -m py_compile args.py data_loader.py model_transformer_cross_c.py Kfold_trainer.py result_evaluate_flexible_new.py
python data_loader.py
```

`python data_loader.py`, full training, and full evaluation require the external Sleep-EDF arrays configured under `/openbayes/input/input0`.

## MLflow Monitoring

The training script can report metrics, parameters, tags, and checkpoints to a remote MLflow Tracking Server. The client reads configuration from `.env` and environment variables. Secrets are never hard-coded in the training code.

Install the optional client packages:

```bash
pip install mlflow python-dotenv
```

Configure local environment variables:

```bash
export MLFLOW_TRACKING_URI="http://YOUR_SERVER_IP:5000"
export MLFLOW_TRACKING_USERNAME="admin"
export MLFLOW_TRACKING_PASSWORD="YOUR_PASSWORD"
export MLFLOW_EXPERIMENT_NAME="pytorch-training"
export MLFLOW_LOG_EVERY_N_STEPS=20
export MLFLOW_STRICT=false
export MLFLOW_ENABLE_ARTIFACTS=true

python Kfold_trainer.py
```

You can also place the same values in a local `.env` file. The repository includes `.env.example` as a template. The `.env` file is ignored by Git because it can contain credentials.

MLflow logging is best-effort by default. If MLflow is not installed, the tracking URI is missing, or the remote server is unavailable, training continues and prints a short warning. Set `MLFLOW_STRICT=true` to make MLflow errors fail training.
