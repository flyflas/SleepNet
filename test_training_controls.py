import os
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

import Kfold_trainer as trainer
from early_stop_tool import EarlyStopping


class SyntheticContextDataset(Dataset):
    def __init__(self, config, labels):
        self.config = config
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.records = [
            type(
                'Record',
                (),
                {
                    'record_id': 'synthetic-record',
                    'subject_id': 'synthetic-subject',
                    'epoch_count': len(self.labels),
                    'features': torch.zeros(1, config.num_channels, config.pad_size, config.dim_model),
                },
            )()
        ]
        generator = torch.Generator().manual_seed(1234 + len(labels))
        self.data = torch.randn(
            len(self.labels),
            config.context_length,
            config.num_channels,
            config.pad_size,
            config.dim_model,
            generator=generator,
        )
        self.dropped_boundary_epochs = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class SmokeConfig:
    def __init__(self):
        self.device = torch.device('cpu')
        self.num_fold = 1
        self.num_classes = 5
        self.num_epochs = 2
        self.batch_size = 4
        self.pad_size = 29
        self.learning_rate = 0.1

        self.dropout = 0.2
        self.dim_model = 128
        self.forward_hidden = 512
        self.fc_hidden = 512
        self.num_head = 8
        self.num_encoder = 8
        self.num_encoder_multi = 2
        self.model_name = 'training_controls_smoke'

        self.context_num_head = 8
        self.context_num_encoder = 2
        self.context_forward_hidden = 1024
        self.context_dropout = 0.2
        self.context_use_local_center_concat = True

        self.use_positional_encoding = False
        self.mamba_d_state = 16
        self.mamba_d_conv = 4
        self.mamba_expand = 2

        self.label_smoothing = 0.1
        self.weight_decay = 0.03
        self.grad_clip = 0.25

        self.use_amp = False
        self.amp_dtype = 'bf16'
        self.allow_tf32 = False
        self.compile_model = False
        self.compile_mode = 'max-autotune'
        self.deterministic = True
        self.train_log_every_n_steps = 1
        self.progress_every_n_steps = 1000
        self.progress_min_interval = 999

        self.num_workers = 0
        self.prefetch_factor = 2
        self.pin_memory = False
        self.drop_last_train_batch = False
        self.use_time = False

        self.context_left = 2
        self.context_right = 2
        self.context_length = 5
        self.context_center_index = 2
        self.split_group_policy = 'subject'
        self.subject_id_length = 6
        self.validation_group_fraction = 0.5
        self.num_channels = 3

        self.scheduler_factor = 0.5
        self.scheduler_patience = 0
        self.scheduler_min_lr = 1e-6

        self.early_stop_patience = 10
        self.early_stop_delta = 0.01


class TinyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Linear(1, config.num_classes)

    def forward(self, x):
        reduced = x.float().mean(dim=tuple(range(1, x.dim()))).unsqueeze(1)
        return self.classifier(reduced)


class NoopMlflowLogger:
    validation_records = []
    artifact_paths = []
    end_statuses = []

    def __init__(self, *args, **kwargs):
        pass

    def start(self, params=None, tags=None):
        self.params = params or {}
        self.tags = tags or {}

    def log_train_step(self, *args, **kwargs):
        pass

    def log_validation(self, step, epoch, val_loss, val_acc=None, extra_metrics=None):
        self.validation_records.append(
            {
                'step': step,
                'epoch': epoch,
                'val_loss': float(val_loss),
                'val_acc': None if val_acc is None else float(val_acc),
                'extra_metrics': extra_metrics or {},
            }
        )

    def log_artifact(self, local_path, artifact_path=None):
        if not os.path.exists(local_path):
            raise AssertionError(f'checkpoint artifact was not created: {local_path}')
        self.artifact_paths.append((local_path, artifact_path))

    def end(self, status='FINISHED'):
        self.end_statuses.append(status)


class QuietLoop:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **kwargs):
        pass


def test_early_stopping_min_delta():
    model = nn.Linear(1, 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        early_stopping = EarlyStopping(
            patience=2,
            delta=0.05,
            mode='min',
            monitor_name='val_loss',
        )

        first_checkpoint = early_stopping(1.00, model, os.path.join(tmpdir, 'epoch0.pkl'))
        if first_checkpoint is None or early_stopping.best_metric != 1.00:
            raise AssertionError('initial val_loss should save the first checkpoint')

        no_checkpoint = early_stopping(1.03, model, os.path.join(tmpdir, 'epoch1.pkl'))
        if no_checkpoint is not None or early_stopping.counter != 1:
            raise AssertionError('worse val_loss should increment patience without saving')

        small_improvement = early_stopping(0.97, model, os.path.join(tmpdir, 'epoch2.pkl'))
        if small_improvement is not None or early_stopping.counter != 2:
            raise AssertionError('improvement smaller than delta should not reset patience')

        clear_improvement = early_stopping(0.94, model, os.path.join(tmpdir, 'epoch3.pkl'))
        if clear_improvement is None or early_stopping.best_metric != 0.94 or early_stopping.counter != 0:
            raise AssertionError('val_loss improvement larger than delta should save and reset patience')


def test_training_controls_smoke():
    torch.manual_seed(0)
    config = SmokeConfig()
    output_root = Path(tempfile.mkdtemp(prefix='sleep-training-controls-'))
    NoopMlflowLogger.validation_records = []
    NoopMlflowLogger.artifact_paths = []
    NoopMlflowLogger.end_statuses = []

    fake_records = [
        type('Record', (), {'labels': torch.tensor([0, 1, 2, 3, 4, 0], dtype=torch.long)})()
    ]
    scheduler_metrics = []
    early_stopping_metrics = []
    early_stopping_init = {}
    val_evaluations = []
    clip_calls = []
    loss_kwargs = []
    optimizer_kwargs = []

    real_evaluate = trainer.evaluate
    real_scheduler = trainer.optim.lr_scheduler.ReduceLROnPlateau
    real_early_stopping = trainer.EarlyStopping
    real_clip_grad_norm = trainer.nn.utils.clip_grad_norm_
    real_cross_entropy = trainer.nn.CrossEntropyLoss
    real_adamw = trainer.optim.AdamW
    real_config = trainer.Config
    real_path = trainer.Path
    real_transformer = trainer.Transformer
    real_mlflow_logger = trainer.MlflowTrainingLogger
    real_load_mlflow_env = trainer.load_mlflow_env
    real_send_notification = trainer.send_all_tasks_finished_notification
    real_load_record_sequences = trainer.load_record_sequences
    real_build_record_folds = trainer.build_record_folds
    real_build_context_datasets_for_fold = trainer.build_context_datasets_for_fold
    real_build_context_output_root = trainer.build_context_output_root
    real_tqdm = trainer.tqdm

    def tracking_evaluate(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        if kwargs.get('split_name') == 'val':
            val_evaluations.append(float(result[1]))
        return result

    class TrackingScheduler:
        def __init__(self, optimizer, *args, **kwargs):
            self.inner = real_scheduler(optimizer, *args, **kwargs)

        def step(self, metric):
            metric = float(metric)
            if not val_evaluations or metric != val_evaluations[-1]:
                raise AssertionError('scheduler.step must receive the latest evaluated val_loss')
            scheduler_metrics.append(metric)
            return self.inner.step(metric)

    class TrackingEarlyStopping(real_early_stopping):
        def __init__(self, *args, **kwargs):
            early_stopping_init.update(kwargs)
            super().__init__(*args, **kwargs)

        def __call__(self, metric_value, model, path):
            metric = float(metric_value)
            if not val_evaluations or metric != val_evaluations[-1]:
                raise AssertionError('early stopping must receive the latest evaluated val_loss')
            early_stopping_metrics.append(metric)
            return super().__call__(metric_value, model, path)

    def tracking_clip_grad_norm(parameters, max_norm, *args, **kwargs):
        clip_calls.append(float(max_norm))
        return real_clip_grad_norm(parameters, max_norm, *args, **kwargs)

    def tracking_cross_entropy(*args, **kwargs):
        loss_kwargs.append(kwargs.copy())
        return real_cross_entropy(*args, **kwargs)

    def tracking_adamw(*args, **kwargs):
        optimizer_kwargs.append(kwargs.copy())
        return real_adamw(*args, **kwargs)

    try:
        trainer.Config = lambda: config
        trainer.Path = lambda: type('SmokePath', (), {'path_labels': 'unused', 'path_TF': 'unused'})()
        trainer.Transformer = TinyTransformer
        trainer.MlflowTrainingLogger = NoopMlflowLogger
        trainer.load_mlflow_env = lambda: None
        trainer.send_all_tasks_finished_notification = lambda: None
        trainer.load_record_sequences = lambda **kwargs: fake_records
        trainer.build_record_folds = lambda records, config, random_state=0: [(records, records)]
        trainer.build_context_datasets_for_fold = lambda **kwargs: (
            SyntheticContextDataset(config, [0, 1, 2, 3, 4, 0, 1, 2]),
            SyntheticContextDataset(config, [0, 1, 2, 3]),
            SyntheticContextDataset(config, [1, 2, 3, 4]),
        )
        trainer.build_context_output_root = lambda config, base_root='./Kfold_models': str(output_root)
        trainer.tqdm = QuietLoop
        trainer.evaluate = tracking_evaluate
        trainer.optim.lr_scheduler.ReduceLROnPlateau = TrackingScheduler
        trainer.EarlyStopping = TrackingEarlyStopping
        trainer.nn.utils.clip_grad_norm_ = tracking_clip_grad_norm
        trainer.nn.CrossEntropyLoss = tracking_cross_entropy
        trainer.optim.AdamW = tracking_adamw

        trainer.train(save_all_checkpoint=False, start_fold=0)
    finally:
        trainer.Config = real_config
        trainer.Path = real_path
        trainer.Transformer = real_transformer
        trainer.MlflowTrainingLogger = real_mlflow_logger
        trainer.load_mlflow_env = real_load_mlflow_env
        trainer.send_all_tasks_finished_notification = real_send_notification
        trainer.load_record_sequences = real_load_record_sequences
        trainer.build_record_folds = real_build_record_folds
        trainer.build_context_datasets_for_fold = real_build_context_datasets_for_fold
        trainer.build_context_output_root = real_build_context_output_root
        trainer.tqdm = real_tqdm
        trainer.evaluate = real_evaluate
        trainer.optim.lr_scheduler.ReduceLROnPlateau = real_scheduler
        trainer.EarlyStopping = real_early_stopping
        trainer.nn.utils.clip_grad_norm_ = real_clip_grad_norm
        trainer.nn.CrossEntropyLoss = real_cross_entropy
        trainer.optim.AdamW = real_adamw

    expected_epochs = config.num_epochs
    if len(NoopMlflowLogger.validation_records) != expected_epochs:
        raise AssertionError('smoke run should complete the configured two validation epochs')
    if scheduler_metrics != val_evaluations:
        raise AssertionError('scheduler should step exactly once per val_loss')
    if early_stopping_metrics != val_evaluations:
        raise AssertionError('early stopping should monitor val_loss exactly once per epoch')
    if early_stopping_init.get('mode') != 'min':
        raise AssertionError('early stopping should be initialized with mode="min"')
    if early_stopping_init.get('delta') != config.early_stop_delta:
        raise AssertionError('early stopping should use config.early_stop_delta')
    if not clip_calls or any(max_norm != config.grad_clip for max_norm in clip_calls):
        raise AssertionError('gradient clipping should use config.grad_clip on training steps')
    if not loss_kwargs or loss_kwargs[0].get('label_smoothing') != config.label_smoothing:
        raise AssertionError('CrossEntropyLoss should use config.label_smoothing')
    if not optimizer_kwargs or optimizer_kwargs[0].get('weight_decay') != config.weight_decay:
        raise AssertionError('AdamW should use config.weight_decay')
    if not NoopMlflowLogger.artifact_paths:
        raise AssertionError('best checkpoint should be saved and logged during smoke run')
    if NoopMlflowLogger.end_statuses[-1] != 'FINISHED':
        raise AssertionError('smoke training run should finish successfully')


if __name__ == '__main__':
    test_early_stopping_min_delta()
    test_training_controls_smoke()
    print('training control smoke tests passed')
