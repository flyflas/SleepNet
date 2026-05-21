# Sleep PyTorch Training

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
