import torch
import os

from env_utils import load_env


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if normalized in ('0', 'false', 'no', 'n', 'off'):
        return False
    return default


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _env_str(name, default):
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


class Config(object):
    """args in model and trainer"""
    def __init__(self):
        load_env()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # basic training settings
        self.num_fold = 10
        self.num_classes = 5
        self.num_epochs = 45
        self.batch_size = _env_int('SLEEP_BATCH_SIZE', 96)
        self.pad_size = 29
        self.learning_rate = 5e-5

        # model settings
        self.dropout = 0.2
        self.dim_model = 128
        self.forward_hidden = 512
        self.fc_hidden = 512
        self.num_head = 8
        self.num_encoder = 8
        self.num_encoder_multi = 2
        self.model_name = 'transformer_epoch_context'

        # epoch-level context encoder settings
        self.context_num_head = 8
        self.context_num_encoder = 2
        self.context_forward_hidden = 1024
        self.context_dropout = 0.2
        self.context_use_local_center_concat = True

        # mamba settings
        self.use_positional_encoding = False
        self.mamba_d_state = 16
        self.mamba_d_conv = 4
        self.mamba_expand = 2

        # optimization
        self.label_smoothing = 0.0
        self.weight_decay = 0.01
        self.grad_clip = 1.0

        # throughput / runtime
        self.use_amp = _env_bool('SLEEP_USE_AMP', torch.cuda.is_available())
        self.amp_dtype = _env_str('SLEEP_AMP_DTYPE', 'bf16')
        self.allow_tf32 = _env_bool('SLEEP_ALLOW_TF32', torch.cuda.is_available())
        self.compile_model = _env_bool('SLEEP_COMPILE_MODEL', False)
        self.compile_mode = _env_str('SLEEP_COMPILE_MODE', 'max-autotune')
        self.deterministic = _env_bool('SLEEP_DETERMINISTIC', False)
        self.train_log_every_n_steps = _env_int(
            'SLEEP_TRAIN_LOG_EVERY_N_STEPS',
            _env_int('MLFLOW_LOG_EVERY_N_STEPS', 100)
        )
        self.progress_every_n_steps = _env_int('SLEEP_PROGRESS_EVERY_N_STEPS', 25)
        self.progress_min_interval = _env_int('SLEEP_PROGRESS_MIN_INTERVAL', 5)

        # dataloader
        self.num_workers = _env_int('SLEEP_NUM_WORKERS', 12)
        self.prefetch_factor = _env_int('SLEEP_PREFETCH_FACTOR', 4)
        self.pin_memory = _env_bool('SLEEP_PIN_MEMORY', True)
        self.drop_last_train_batch = _env_bool('SLEEP_DROP_LAST_TRAIN_BATCH', False)
        self.use_time = False

        # context window settings
        self.context_left = 2
        self.context_right = 2
        self.context_length = self.context_left + 1 + self.context_right
        self.context_center_index = self.context_left
        self.split_group_policy = 'subject'
        self.subject_id_length = 6
        self.validation_group_fraction = 1 / (self.num_fold + 1)

        # scheduler / early stop
        self.scheduler_factor = 0.5
        self.scheduler_patience = 3
        self.scheduler_min_lr = 1e-6

        self.early_stop_patience = 12
        self.early_stop_delta = 0.0

        # logging
        self.print_distribution_first_n_epochs = 3
        self.print_distribution_every = 10

        self._validate_context_config()

    def _validate_context_config(self):
        if self.context_left < 0 or self.context_right < 0:
            raise ValueError('[ERROR] context_left/context_right must be non-negative')
        if self.context_left != self.context_right:
            raise ValueError('[ERROR] Phase 1 requires symmetric bidirectional context windows')
        expected_length = self.context_left + 1 + self.context_right
        if self.context_length != expected_length:
            raise ValueError(
                f'[ERROR] context_length={self.context_length} must equal '
                f'context_left + 1 + context_right = {expected_length}'
            )
        if self.context_center_index != self.context_left:
            raise ValueError('[ERROR] context_center_index must equal context_left')
        if self.split_group_policy not in ('subject', 'record'):
            raise ValueError('[ERROR] split_group_policy must be "subject" or "record"')
        if not 0 < self.validation_group_fraction < 1:
            raise ValueError('[ERROR] validation_group_fraction must be between 0 and 1')


class Path(object):
    """path of files in this project"""
    def __init__(self):
        load_env()
        data_root = os.getenv('DATA_ROOT')
        if not data_root or not data_root.strip():
            raise RuntimeError('[ERROR] DATA_ROOT must be set in .env or environment variables')
        data_root = data_root.strip()

        self.path_PSG = os.path.join(data_root, 'dataset/Sleep-EDF-78/sleep-edfx/sleep-cassette')
        self.path_hypnogram = os.path.join(data_root, 'dataset/Sleep-EDF-78/sleep-edfx/Hypnogram')
        self.path_raw_data = os.path.join(data_root, 'data/sleepEDF-78/data_array/raw_data')
        self.path_labels = os.path.join(data_root, 'data/sleepEDF-78/data_array/raw_data/labels')
        self.path_TF = os.path.join(data_root, 'data/sleepEDF-78/data_array/TF_data')
