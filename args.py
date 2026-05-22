import torch
import os

from env_utils import load_env


class Config(object):
    """args in model and trainer"""
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # basic training settings
        self.num_fold = 10
        self.num_classes = 5
        self.num_epochs = 45
        self.batch_size = 512
        self.pad_size = 29
        self.learning_rate = 5e-5

        # model settings
        self.dropout = 0.1
        self.dim_model = 128
        self.forward_hidden = 1024
        self.fc_hidden = 1024
        self.num_head = 8
        self.num_encoder = 16
        self.num_encoder_multi = 4
        self.model_name = 'transformer_epoch_context'

        # epoch-level context encoder settings
        self.context_num_head = 8
        self.context_num_encoder = 2
        self.context_forward_hidden = 1024
        self.context_dropout = 0.1
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

        # dataloader
        self.num_workers = 12
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
