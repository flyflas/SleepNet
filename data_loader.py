import os
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

from args import Config


_DEFAULT_USE_TIME = object()
CHANNELS = ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG']


class SleepRecord(object):
    """One PSG record with aligned per-epoch features and labels."""
    def __init__(self, features, labels, record_id, subject_id):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.record_id = record_id
        self.subject_id = subject_id
        self.epoch_count = int(self.labels.shape[0])
        self._validate()

    def _validate(self):
        if self.features.shape[0] != self.epoch_count:
            raise ValueError(
                f'[ERROR] record {self.record_id} features length {self.features.shape[0]} '
                f'!= labels length {self.epoch_count}'
            )
        if self.features.ndim == 4:
            expected_tail = (3, 29, 128)
        elif self.features.ndim == 5:
            expected_tail = (3, 2, 29, 128)
        else:
            raise ValueError(
                f'[ERROR] record {self.record_id} features must be rank 4 or 5, got {self.features.shape}'
            )
        if tuple(self.features.shape[1:]) != expected_tail:
            raise ValueError(
                f'[ERROR] record {self.record_id} feature shape tail {self.features.shape[1:]} '
                f'!= {expected_tail}'
            )
        if self.labels.ndim != 1:
            raise ValueError(f'[ERROR] record {self.record_id} labels must be rank 1, got {self.labels.shape}')


class ContextWindowDataset(Dataset):
    """Deterministic bidirectional context windows within PSG record boundaries."""
    def __init__(self, records, context_left=None, context_right=None, config=None, return_metadata=False):
        self.config = config if config is not None else Config()
        self.context_left = self.config.context_left if context_left is None else context_left
        self.context_right = self.config.context_right if context_right is None else context_right
        self.context_length = self.context_left + 1 + self.context_right
        self.context_center_index = self.context_left
        self.return_metadata = return_metadata
        self.records = list(records)
        self.indices = []
        self.dropped_boundary_epochs = 0
        self._validate_context()
        self._build_indices()

    def _validate_context(self):
        if self.context_left < 0 or self.context_right < 0:
            raise ValueError('[ERROR] context_left/context_right must be non-negative')
        if self.context_left != self.context_right:
            raise ValueError('[ERROR] Phase 1 requires symmetric bidirectional context windows')
        if self.context_length != self.context_left + 1 + self.context_right:
            raise ValueError('[ERROR] Invalid context_length')

    def _build_indices(self):
        for record_idx, record in enumerate(self.records):
            valid_count = max(0, record.epoch_count - self.context_left - self.context_right)
            self.dropped_boundary_epochs += record.epoch_count - valid_count
            for center_idx in range(self.context_left, record.epoch_count - self.context_right):
                self.indices.append((record_idx, center_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        record_idx, center_idx = self.indices[index]
        record = self.records[record_idx]
        start = center_idx - self.context_left
        end = center_idx + self.context_right + 1
        features = record.features[start:end]
        label = record.labels[center_idx]

        if features.shape[0] != self.context_length:
            raise ValueError(
                f'[ERROR] record {record.record_id} context window length {features.shape[0]} '
                f'!= {self.context_length}'
            )

        if self.return_metadata:
            metadata = {
                'record_id': record.record_id,
                'subject_id': record.subject_id,
                'center_epoch': int(center_idx)
            }
            return features, label, metadata
        return features, label

    def sample_metadata(self, index):
        record_idx, center_idx = self.indices[index]
        record = self.records[record_idx]
        return {
            'record_id': record.record_id,
            'subject_id': record.subject_id,
            'center_epoch': int(center_idx)
        }


def print_label_distribution(labels, split_name='labels'):
    """Print label distribution."""
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = labels

    unique, counts = np.unique(labels_np, return_counts=True)
    total = len(labels_np)

    print(f'[{split_name}] label distribution:')
    for u, c in zip(unique, counts):
        print(f'  class {u}: {c} ({c / total:.6f})')


def _record_group_id(record, config):
    if config.split_group_policy == 'subject':
        return record.subject_id
    if config.split_group_policy == 'record':
        return record.record_id
    raise ValueError(f'[ERROR] Unsupported split_group_policy={config.split_group_policy}')


def _record_majority_label(record):
    labels = record.labels.cpu().numpy()
    counts = np.bincount(labels.astype(np.int64))
    return int(np.argmax(counts))


def _records_by_indices(records, indices):
    return [records[int(i)] for i in indices]


def _split_record_ids(records):
    return set(record.record_id for record in records)


def _split_group_ids(records, config):
    return set(_record_group_id(record, config) for record in records)


def _assert_pairwise_disjoint(name_to_values, value_name):
    names = list(name_to_values.keys())
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            overlap = name_to_values[left_name] & name_to_values[right_name]
            if overlap:
                raise ValueError(
                    f'[ERROR] {value_name} leakage between {left_name} and {right_name}: '
                    f'{sorted(overlap)[:10]}'
                )


def assert_disjoint_record_splits(train_records, val_records, test_records, config=None):
    config = config if config is not None else Config()
    split_records = {
        'train': list(train_records),
        'val': list(val_records),
        'test': list(test_records),
    }
    _assert_pairwise_disjoint(
        {name: _split_record_ids(records) for name, records in split_records.items()},
        'record_id'
    )
    if config.split_group_policy == 'subject':
        _assert_pairwise_disjoint(
            {name: _split_group_ids(records, config) for name, records in split_records.items()},
            'subject_id'
        )


def _context_sample_count(records, config):
    return sum(max(0, record.epoch_count - config.context_left - config.context_right) for record in records)


def _label_counts_for_records(records, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for record in records:
        labels = record.labels.cpu().numpy().astype(np.int64)
        counts += np.bincount(labels, minlength=num_classes)
    return counts


def log_record_split_summary(split_name, records, config=None):
    config = config if config is not None else Config()
    groups = _split_group_ids(records, config)
    record_ids = _split_record_ids(records)
    context_samples = _context_sample_count(records, config)
    epoch_count = sum(record.epoch_count for record in records)
    label_counts = _label_counts_for_records(records, config.num_classes)
    print(
        f'[INFO][split {split_name}] groups={len(groups)}, records={len(record_ids)}, '
        f'epochs={epoch_count}, context_samples={context_samples}, '
        f'label_counts={label_counts.tolist()}'
    )


def build_record_folds(records, config=None, random_state=0):
    """Build outer folds from PSG record metadata before any context windows exist."""
    config = config if config is not None else Config()
    records = list(records)
    groups = np.array([_record_group_id(record, config) for record in records])
    y = np.array([_record_majority_label(record) for record in records], dtype=np.int64)
    x = np.arange(len(records))
    unique_groups = np.unique(groups)

    if len(unique_groups) < config.num_fold:
        raise ValueError(
            f'[ERROR] Cannot create {config.num_fold} folds from {len(unique_groups)} '
            f'unique {config.split_group_policy} groups'
        )

    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=config.num_fold,
            shuffle=True,
            random_state=random_state
        )
        split_iter = splitter.split(x, y, groups)
        strategy = 'StratifiedGroupKFold'
    else:
        splitter = GroupKFold(n_splits=config.num_fold)
        split_iter = splitter.split(x, y, groups)
        strategy = 'GroupKFold'
        print('[WARNING] StratifiedGroupKFold unavailable; using GroupKFold without class stratification.')

    print(
        f'[INFO] outer split strategy={strategy}, group_policy={config.split_group_policy}, '
        f'unique_groups={len(unique_groups)}, records={len(records)}'
    )
    return [(_records_by_indices(records, train_idx), _records_by_indices(records, test_idx))
            for train_idx, test_idx in split_iter]


def split_validation_records(train_records, config=None, random_state=0):
    """Create a deterministic validation split from training groups only."""
    config = config if config is not None else Config()
    train_records = list(train_records)
    groups = np.array([_record_group_id(record, config) for record in train_records])
    unique_groups = np.unique(groups)

    if len(unique_groups) < 2:
        raise ValueError(
            f'[ERROR] Need at least 2 training {config.split_group_policy} groups to create validation split'
        )

    target_val_groups = int(round(len(unique_groups) * config.validation_group_fraction))
    val_group_count = min(max(1, target_val_groups), len(unique_groups) - 1)
    rng = np.random.RandomState(random_state)
    shuffled_groups = np.array(unique_groups, copy=True)
    rng.shuffle(shuffled_groups)
    val_groups = set(shuffled_groups[:val_group_count])

    final_train_records = []
    val_records = []
    for record in train_records:
        if _record_group_id(record, config) in val_groups:
            val_records.append(record)
        else:
            final_train_records.append(record)

    if not final_train_records or not val_records:
        raise ValueError('[ERROR] Validation split produced an empty train or validation record set')
    return final_train_records, val_records


def build_context_datasets_for_fold(train_records, test_records, config=None, fold=0):
    config = config if config is not None else Config()
    train_records, val_records = split_validation_records(
        train_records=train_records,
        config=config,
        random_state=fold
    )
    assert_disjoint_record_splits(train_records, val_records, test_records, config=config)

    print(f'[INFO][fold {fold}] record/group split summary before context windows:')
    log_record_split_summary('train', train_records, config=config)
    log_record_split_summary('val', val_records, config=config)
    log_record_split_summary('test', test_records, config=config)

    train_set = build_context_dataset(train_records, config=config)
    val_set = build_context_dataset(val_records, config=config)
    test_set = build_context_dataset(test_records, config=config)
    return train_set, val_set, test_set


def _load_channel_arrays(path_dataset, prefix):
    arrays = []
    for channel in CHANNELS:
        file_path = os.path.join(path_dataset, f'{prefix}_{channel}_mean_std.npy')
        arrays.append(np.load(file_path).astype('float32'))
    return arrays


def _check_channel_shapes(arrays, prefix):
    expected_shape = arrays[0].shape
    if expected_shape[1:] != (29, 128):
        raise ValueError(
            f'[ERROR] {prefix} channel shape must be [N, 29, 128], got {expected_shape}'
        )

    for idx, array in enumerate(arrays[1:], start=1):
        if array.shape != expected_shape:
            raise ValueError(
                f'[ERROR] {prefix} channel {idx} shape {array.shape} != first channel shape {expected_shape}'
            )


def _time_file_status(path_dataset):
    expected_files = [
        os.path.join(path_dataset, f'TIME_{channel}_mean_std.npy')
        for channel in CHANNELS
    ]
    existing_files = [file_path for file_path in expected_files if os.path.exists(file_path)]
    missing_files = [file_path for file_path in expected_files if not os.path.exists(file_path)]
    return existing_files, missing_files


def _resolve_use_time(path_dataset, use_time, require_global_time=True):
    existing_time_files, missing_time_files = _time_file_status(path_dataset)
    has_any_time = bool(existing_time_files)
    has_all_time = not missing_time_files
    use_time_was_omitted = use_time is _DEFAULT_USE_TIME

    if use_time_was_omitted:
        if has_any_time and not has_all_time:
            raise FileNotFoundError(
                '[ERROR] use_time was omitted and partial TIME files were found. '
                'Pass use_time=False for frequency-only data or provide all TIME files. '
                f'missing={missing_time_files}'
            )
        return False
    if use_time is None:
        if has_any_time and not has_all_time:
            raise FileNotFoundError(
                '[ERROR] use_time=None auto mode found partial TIME files. '
                f'missing={missing_time_files}'
            )
        return has_all_time
    if use_time and require_global_time and not has_all_time:
        raise FileNotFoundError(
            '[ERROR] use_time=True but one or more TIME_*_mean_std.npy files are missing. '
            f'missing={missing_time_files}'
        )
    return bool(use_time)


def _record_id_from_label_file(file_name):
    if file_name.endswith('_label.npy'):
        return file_name[:-len('_label.npy')]
    if file_name.endswith('.npy'):
        return file_name[:-len('.npy')]
    return file_name


def _record_id_from_channel_file(file_name, channel):
    suffix = f'_{channel}.npy'
    if file_name.endswith(suffix):
        return file_name[:-len(suffix)]
    if file_name.endswith('.npy'):
        return file_name[:-len('.npy')]
    return file_name


def _subject_id_from_record_id(record_id, config):
    if config.split_group_policy == 'record':
        return record_id
    if len(record_id) < config.subject_id_length:
        raise ValueError(
            f'[ERROR] Cannot derive subject_id from record_id={record_id}; '
            f'expected at least {config.subject_id_length} characters'
        )
    return record_id[:config.subject_id_length]


def enumerate_label_records(path_labels, config=None):
    """Enumerate records from sorted label files before feature concat/windowing."""
    config = config if config is not None else Config()
    label_files = sorted([f for f in os.listdir(path_labels) if f.endswith('.npy')])
    if not label_files:
        raise FileNotFoundError(f'[ERROR] No .npy label files found in {path_labels}')

    records = []
    for f in label_files:
        record_id = _record_id_from_label_file(f)
        labels = np.load(os.path.join(path_labels, f)).astype(np.int64)
        if labels.ndim != 1:
            raise ValueError(f'[ERROR] Labels for {record_id} must be rank 1, got {labels.shape}')
        records.append({
            'record_id': record_id,
            'subject_id': _subject_id_from_record_id(record_id, config),
            'labels': labels,
            'epoch_count': int(labels.shape[0]),
            'label_file': f
        })
    return records


def _per_record_dir(path_dataset, prefix, channel):
    return os.path.join(path_dataset, 'records', prefix, channel)


def _per_record_files_available(path_dataset, prefix, record_ids):
    dirs = [_per_record_dir(path_dataset, prefix, channel) for channel in CHANNELS]
    existing_dirs = [d for d in dirs if os.path.isdir(d)]
    if not existing_dirs:
        return False
    if len(existing_dirs) != len(dirs):
        missing_dirs = [d for d in dirs if not os.path.isdir(d)]
        raise FileNotFoundError(f'[ERROR] Partial per-record {prefix} directories found: missing={missing_dirs}')

    missing_files = []
    for channel in CHANNELS:
        record_dir = _per_record_dir(path_dataset, prefix, channel)
        for record_id in record_ids:
            file_path = os.path.join(record_dir, f'{record_id}.npy')
            if not os.path.exists(file_path):
                missing_files.append(file_path)
    if missing_files:
        raise FileNotFoundError(
            f'[ERROR] Partial per-record {prefix} files found. First missing files: {missing_files[:5]}'
        )
    return True


def _load_per_record_feature_arrays(path_dataset, prefix, label_records):
    features_by_record = {}
    for record in label_records:
        channel_arrays = []
        for channel in CHANNELS:
            file_path = os.path.join(_per_record_dir(path_dataset, prefix, channel), f'{record["record_id"]}.npy')
            array = np.load(file_path).astype('float32')
            if array.shape != (record['epoch_count'], 29, 128):
                raise ValueError(
                    f'[ERROR] {prefix} {channel} record {record["record_id"]} shape {array.shape} '
                    f'!= {(record["epoch_count"], 29, 128)}'
                )
            channel_arrays.append(array)
        features_by_record[record['record_id']] = np.stack(channel_arrays, axis=1)
    return features_by_record


def _validate_raw_channel_order(path_labels, label_records):
    raw_data_dir = os.path.dirname(path_labels)
    label_ids = [record['record_id'] for record in label_records]

    for channel in CHANNELS:
        channel_dir = os.path.join(raw_data_dir, channel)
        if not os.path.isdir(channel_dir):
            raise FileNotFoundError(
                '[ERROR] Cannot safely reconstruct record boundaries from global arrays: '
                f'missing raw channel directory {channel_dir}'
            )
        channel_files = sorted([f for f in os.listdir(channel_dir) if f.endswith('.npy')])
        channel_ids = [_record_id_from_channel_file(f, channel) for f in channel_files]
        if channel_ids != label_ids:
            raise ValueError(
                '[ERROR] Cannot safely reconstruct record boundaries from global arrays: '
                f'{channel} file order does not match label order'
            )


def _split_global_arrays_by_records(path_labels, path_dataset, use_time, label_records):
    _validate_raw_channel_order(path_labels, label_records)

    freq_arrays = _load_channel_arrays(path_dataset, prefix='TF')
    _check_channel_shapes(freq_arrays, prefix='TF')
    expected_total = sum(record['epoch_count'] for record in label_records)
    if freq_arrays[0].shape[0] != expected_total:
        raise ValueError(
            f'[ERROR] Global TF length {freq_arrays[0].shape[0]} != total label length {expected_total}'
        )

    time_arrays = None
    if use_time:
        time_arrays = _load_channel_arrays(path_dataset, prefix='TIME')
        _check_channel_shapes(time_arrays, prefix='TIME')
        if time_arrays[0].shape[0] != expected_total:
            raise ValueError(
                f'[ERROR] Global TIME length {time_arrays[0].shape[0]} != total label length {expected_total}'
            )

    records = []
    offset = 0
    for record in label_records:
        next_offset = offset + record['epoch_count']
        freq_feature = np.stack([array[offset:next_offset] for array in freq_arrays], axis=1)
        if use_time:
            time_feature = np.stack([array[offset:next_offset] for array in time_arrays], axis=1)
            if time_feature.shape != freq_feature.shape:
                raise ValueError(
                    f'[ERROR] record {record["record_id"]} TIME shape {time_feature.shape} '
                    f'!= TF shape {freq_feature.shape}'
                )
            features = np.stack((freq_feature, time_feature), axis=2)
        else:
            features = freq_feature
        records.append(SleepRecord(
            features=features,
            labels=record['labels'],
            record_id=record['record_id'],
            subject_id=record['subject_id']
        ))
        offset = next_offset

    if offset != expected_total:
        raise ValueError(f'[ERROR] Global split offset {offset} != expected total {expected_total}')
    return records


def load_record_sequences(path_labels, path_dataset, use_time=_DEFAULT_USE_TIME, config=None):
    """Load aligned per-record sequences without constructing cross-record windows."""
    config = config if config is not None else Config()
    label_records = enumerate_label_records(path_labels, config=config)
    record_ids = [record['record_id'] for record in label_records]
    use_time = _resolve_use_time(path_dataset, use_time, require_global_time=False)

    has_per_record_tf = _per_record_files_available(path_dataset, 'TF', record_ids)
    has_per_record_time = False
    if use_time:
        has_per_record_time = _per_record_files_available(path_dataset, 'TIME', record_ids)

    if has_per_record_tf and ((not use_time) or has_per_record_time):
        freq_features = _load_per_record_feature_arrays(path_dataset, 'TF', label_records)
        time_features = _load_per_record_feature_arrays(path_dataset, 'TIME', label_records) if use_time else None
        records = []
        for record in label_records:
            record_id = record['record_id']
            if use_time:
                if time_features[record_id].shape != freq_features[record_id].shape:
                    raise ValueError(
                        f'[ERROR] record {record_id} TIME shape {time_features[record_id].shape} '
                        f'!= TF shape {freq_features[record_id].shape}'
                    )
                features = np.stack((freq_features[record_id], time_features[record_id]), axis=2)
            else:
                features = freq_features[record_id]
            records.append(SleepRecord(
                features=features,
                labels=record['labels'],
                record_id=record_id,
                subject_id=record['subject_id']
            ))
        source = 'per-record'
    else:
        if use_time:
            _resolve_use_time(path_dataset, use_time=True, require_global_time=True)
        records = _split_global_arrays_by_records(
            path_labels=path_labels,
            path_dataset=path_dataset,
            use_time=use_time,
            label_records=label_records
        )
        source = 'global-with-asserted-boundaries'

    print_record_sequence_summary(records, source=source, use_time=use_time)
    return records


def print_record_sequence_summary(records, source, use_time):
    total_epochs = sum(record.epoch_count for record in records)
    subjects = sorted(set(record.subject_id for record in records))
    print(
        f'[INFO] loaded {len(records)} records, {len(subjects)} groups, {total_epochs} epochs '
        f'from {source}, use_time={use_time}'
    )
    for record in records[:5]:
        print(
            f'  record={record.record_id}, subject={record.subject_id}, '
            f'epochs={record.epoch_count}, feature_shape={tuple(record.features.shape)}'
        )


def build_context_dataset(records, config=None, return_metadata=False):
    config = config if config is not None else Config()
    dataset = ContextWindowDataset(
        records=records,
        context_left=config.context_left,
        context_right=config.context_right,
        config=config,
        return_metadata=return_metadata
    )
    print(
        f'[INFO] context records={len(records)}, context_length={dataset.context_length}, '
        f'center_index={dataset.context_center_index}, dropped_boundary_epochs={dataset.dropped_boundary_epochs}, '
        f'context_samples={len(dataset)}'
    )
    return dataset


def materialize_context_windows(records, config=None):
    """Materialize context windows for small smoke checks or offline inspection."""
    dataset = build_context_dataset(records, config=config, return_metadata=True)
    features = []
    labels = []
    metadata = []
    for i in range(len(dataset)):
        x, y, meta = dataset[i]
        features.append(x)
        labels.append(y)
        metadata.append(meta)
    if features:
        return torch.stack(features, dim=0), torch.stack(labels, dim=0), metadata

    if dataset.records:
        empty_feature_shape = (0, dataset.context_length) + tuple(dataset.records[0].features.shape[1:])
    else:
        empty_feature_shape = (0, dataset.context_length)
    return torch.empty(empty_feature_shape), torch.empty((0,), dtype=torch.long), metadata


def data_generator(path_labels, path_dataset, use_time=_DEFAULT_USE_TIME):
    """
    Build dataset tensors.

    Omitted use_time preserves the frequency-only baseline [N, 3, 29, 128].
    use_time=False explicitly preserves the frequency-only baseline [N, 3, 29, 128].
    use_time=True requires TIME files and returns [N, 3, 2, 29, 128].
    use_time=None auto-enables freq+time data when all TIME files exist.
    """
    config = Config()

    # 1. 一定要排序，保证和 data_preprocess_TF.py 的拼接顺序一致
    label_files = sorted([f for f in os.listdir(path_labels) if f.endswith('.npy')])

    print('[INFO] label files (first 10):')
    for f in label_files[:10]:
        print(f'  {f}')

    # 2. 按排序后的文件顺序拼接 labels
    label_list = []
    for f in label_files:
        y = np.load(os.path.join(path_labels, f))
        label_list.append(y)

    labels = np.concatenate(label_list, axis=0).astype(np.int64)
    labels = torch.from_numpy(labels)

    # 3. 读取三个通道的 TF 数据
    freq_arrays = _load_channel_arrays(path_dataset, prefix='TF')
    _check_channel_shapes(freq_arrays, prefix='TF')
    freq_dataset = np.stack(freq_arrays, axis=1)

    use_time = _resolve_use_time(path_dataset, use_time)

    # 4. 堆叠成 [N, 3, 29, 128] 或 [N, 3, 2, 29, 128]
    if use_time:
        time_arrays = _load_channel_arrays(path_dataset, prefix='TIME')
        _check_channel_shapes(time_arrays, prefix='TIME')
        time_dataset = np.stack(time_arrays, axis=1)

        if time_dataset.shape != freq_dataset.shape:
            raise ValueError(
                f'[ERROR] time dataset shape {time_dataset.shape} != freq dataset shape {freq_dataset.shape}'
            )

        dataset = np.stack((freq_dataset, time_dataset), axis=2)
    else:
        dataset = freq_dataset

    dataset = torch.from_numpy(dataset)

    print(f'[INFO] dataset shape: {dataset.shape}')
    print(f'[INFO] labels shape: {labels.shape}')

    # 5. 安全检查：样本数必须一致
    if len(dataset) != len(labels):
        raise ValueError(
            f'[ERROR] dataset size ({len(dataset)}) != labels size ({len(labels)})'
        )

    print_label_distribution(labels, split_name='full dataset')

    print('[INFO] legacy data_generator returns full flattened tensors and no epoch-level validation loader.')
    print('[INFO] Use load_record_sequences() and build_context_datasets_for_fold() for leakage-safe splits.')
    return dataset, labels, None


if __name__ == '__main__':
    # 方便单独测试
    from args import Path

    path = Path()
    config = Config()
    records = load_record_sequences(
        path_labels=path.path_labels,
        path_dataset=path.path_TF,
        use_time=config.use_time,
        config=config
    )
    context_dataset = build_context_dataset(records, config=config, return_metadata=True)
    if len(context_dataset) > 0:
        x, y, metadata = context_dataset[0]
        print(f'[INFO] first context sample shape: {tuple(x.shape)}')
        print(f'[INFO] first center label: {int(y)}')
        print(f'[INFO] first metadata: {metadata}')
