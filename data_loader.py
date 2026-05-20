import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from args import Config


_DEFAULT_USE_TIME = object()


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


def _load_channel_arrays(path_dataset, prefix):
    arrays = []
    for channel in ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG']:
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
        for channel in ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG']
    ]
    existing_files = [file_path for file_path in expected_files if os.path.exists(file_path)]
    missing_files = [file_path for file_path in expected_files if not os.path.exists(file_path)]
    return existing_files, missing_files


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
        use_time = False
    elif use_time is None:
        if has_any_time and not has_all_time:
            raise FileNotFoundError(
                '[ERROR] use_time=None auto mode found partial TIME files. '
                f'missing={missing_time_files}'
            )
        use_time = has_all_time
    elif use_time and not has_all_time:
        raise FileNotFoundError(
            '[ERROR] use_time=True but one or more TIME_*_mean_std.npy files are missing. '
            f'missing={missing_time_files}'
        )

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

    # 6. 固定留出验证集
    X_train_test, X_val, y_train_test, y_val = train_test_split(
        dataset,
        labels,
        test_size=1 / (config.num_fold + 1),
        random_state=0,
        stratify=labels
    )

    print(f'[INFO] train_test size: {len(X_train_test)}')
    print(f'[INFO] val size: {len(X_val)}')

    print_label_distribution(y_train_test, split_name='train_test')
    print_label_distribution(y_val, split_name='val')

    val_set = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return X_train_test, y_train_test, val_loader


if __name__ == '__main__':
    # 方便单独测试
    from args import Path

    path = Path()
    data_generator(path_labels=path.path_labels, path_dataset=path.path_TF)
