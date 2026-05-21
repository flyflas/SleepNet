import os
import numpy as np

from scipy.fftpack import fft
from scipy import signal
from tqdm import tqdm

from args import Path


def get_npy_file_list(path_array):
    """Get sorted .npy file list."""
    file_list = sorted([f for f in os.listdir(path_array) if f.endswith('.npy')])
    return file_list


def strip_suffix(file_name, channel):
    """Remove channel-specific suffix and keep sample basename."""
    if channel == 'EEG_Fpz-Cz':
        return file_name.replace('_EEG_Fpz-Cz.npy', '')
    elif channel == 'EEG_Pz-Oz':
        return file_name.replace('_EEG_Pz-Oz.npy', '')
    elif channel == 'EOG':
        return file_name.replace('_EOG.npy', '')
    elif channel == 'labels':
        return file_name.replace('_label.npy', '')
    else:
        return file_name


def check_raw_data_alignment(path_raw_data):
    """Check whether EEG/EOG/labels files are aligned before TF transform."""
    channels = ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG', 'labels']
    file_dict = {}

    print('=' * 80)
    print('[CHECK] Checking raw_data file alignment...')
    for channel in channels:
        channel_path = os.path.join(path_raw_data, channel)
        file_list = get_npy_file_list(channel_path)
        file_dict[channel] = file_list

        print(f'\n[{channel}] num_files = {len(file_list)}')
        for f in file_list[:10]:
            print(f'  {f}')

    lengths = [len(file_dict[ch]) for ch in channels]
    if len(set(lengths)) != 1:
        raise ValueError(f'[ERROR] File counts are inconsistent across channels: {dict(zip(channels, lengths))}')

    ref_keys = [strip_suffix(f, 'EEG_Fpz-Cz') for f in file_dict['EEG_Fpz-Cz']]
    pz_keys = [strip_suffix(f, 'EEG_Pz-Oz') for f in file_dict['EEG_Pz-Oz']]
    eog_keys = [strip_suffix(f, 'EOG') for f in file_dict['EOG']]
    label_keys = [strip_suffix(f, 'labels') for f in file_dict['labels']]

    mismatch_found = False
    for i, (k1, k2, k3, k4) in enumerate(zip(ref_keys, pz_keys, eog_keys, label_keys)):
        if not (k1 == k2 == k3 == k4):
            print(f'[ERROR] Mismatch at index {i}:')
            print(f'  EEG_Fpz-Cz : {file_dict["EEG_Fpz-Cz"][i]}')
            print(f'  EEG_Pz-Oz  : {file_dict["EEG_Pz-Oz"][i]}')
            print(f'  EOG        : {file_dict["EOG"][i]}')
            print(f'  labels     : {file_dict["labels"][i]}')
            mismatch_found = True
            break

    if mismatch_found:
        raise ValueError('[ERROR] Raw data files are NOT aligned. Please fix filenames/order first.')
    else:
        print('\n[OK] Raw data files are aligned across EEG_Fpz-Cz / EEG_Pz-Oz / EOG / labels.')
    print('=' * 80)


def check_label_distribution(path_labels):
    """Check overall label distribution."""
    label_files = get_npy_file_list(path_labels)
    all_labels = []

    print('\n' + '=' * 80)
    print('[CHECK] Checking overall label distribution...')
    for f in tqdm(label_files, desc='Loading label files'):
        y = np.load(os.path.join(path_labels, f))
        all_labels.append(y)

    all_labels = np.concatenate(all_labels, axis=0)
    unique, counts = np.unique(all_labels, return_counts=True)

    print('[INFO] Overall label distribution:')
    total = len(all_labels)
    for u, c in zip(unique, counts):
        print(f'  class {u}: {c} ({c / total:.6f})')

    print(f'[INFO] Total labels: {total}')
    print('=' * 80)


def data_array_concat(path_array):
    """Concat data from each subject."""
    file_list = get_npy_file_list(path_array)
    data_list = []

    print(f'Preparing dataset from: {path_array}')
    for f in tqdm(file_list):
        data = np.load(os.path.join(path_array, f)).astype('float32')
        data_list.append(data)

    data_channel = np.concatenate(data_list, axis=0)
    data_channel = np.squeeze(data_channel, axis=1)

    print(f'[INFO] Concatenated shape from {path_array}: {data_channel.shape}')
    return data_channel


def load_record_raw_arrays(path_array, channel):
    """Load raw channel arrays as ordered per-record sequences."""
    file_list = get_npy_file_list(path_array)
    record_arrays = []

    print(f'Preparing per-record dataset from: {path_array}')
    for f in tqdm(file_list):
        data = np.load(os.path.join(path_array, f)).astype('float32')
        if data.ndim == 3 and data.shape[1] == 1:
            data = np.squeeze(data, axis=1)
        if data.ndim != 2:
            raise ValueError(f'[ERROR] Expected raw record shape [T, 3000], got {data.shape} for {f}')
        record_id = strip_suffix(f, channel)
        record_arrays.append((record_id, data))

    return record_arrays


def transform_record_arrays(record_arrays, fs, win_size, overlap, nfft):
    """Create TF and STFT-aligned time windows while preserving record boundaries."""
    tf_records = []
    time_records = []

    for record_id, data in tqdm(record_arrays, desc='Transform records'):
        X = np.zeros([data.shape[0], 29, int(nfft / 2)], dtype=np.float32)
        X_time = np.zeros([data.shape[0], 29, 128], dtype=np.float32)

        for i in range(data.shape[0]):
            Xi = spectrogram(data[i, :], win_size * fs, overlap * fs, nfft)
            Xi = 20 * np.log10(np.abs(Xi) + 1e-8)
            X[i, :, :] = Xi[:, 1:129]
            X_time[i] = interpolate_time_windows(
                data[i, :],
                window=win_size * fs,
                step=(win_size - overlap) * fs,
                target_len=128,
                num_windows=29
            )

        tf_records.append((record_id, X))
        time_records.append((record_id, X_time))

    return tf_records, time_records


def save_record_arrays(record_arrays, channel, save_dir, prefix):
    """Save normalized per-record features using the same record IDs as labels."""
    record_dir = os.path.join(save_dir, 'records', prefix, channel)
    os.makedirs(record_dir, exist_ok=True)

    metadata = []
    for record_id, data in record_arrays:
        save_path = os.path.join(record_dir, f'{record_id}.npy')
        np.save(save_path, data)
        metadata.append((record_id, int(data.shape[0])))

    print(f'[INFO] Saved {len(record_arrays)} per-record {prefix} arrays for {channel} to {record_dir}')
    return metadata


def spectrogram(x, window, n_overlap, nfft):
    """
    Transform to time-frequency images.
    This function imitates Matlab spectrogram.
    """
    len_x = len(x)
    step = window - n_overlap
    nn = nfft // 2 + 1
    num_win = int(np.floor((len_x - n_overlap) / step))
    spectrogram_data = []

    win = signal.windows.hamming(window)
    for i in range(num_win):
        subdata = x[i * step: i * step + window]
        F = fft(subdata * win, n=nfft)
        spectrogram_data.append(F[:nn])

    spectrogram_data = np.array(spectrogram_data)
    return spectrogram_data


def data_normalize(dataset, channel, save_dir, prefix='TF'):
    """Normalize datasets of each channel to zero mean and unit variance."""
    print(f'[INFO] Checking inf values before normalization for channel={channel} ...')
    for i in tqdm(range(dataset.shape[0])):
        if np.any(np.isinf(dataset[i])):
            for j in range(29):
                if np.any(np.isinf(dataset[i][j])):
                    for k in range(128):
                        if np.isinf(dataset[i][j][k]):
                            print('location of inf: ', i, ',', j, ',', k)
                            if k == 0:
                                if j == 0:
                                    dataset[i][j][k] = dataset[i][j + 1][k]
                                else:
                                    dataset[i][j][k] = dataset[i][j - 1][k]
                            else:
                                dataset[i][j][k] = dataset[i][j][k - 1]

    mean_val = np.mean(dataset)
    std_val = np.std(dataset)

    print(f'[INFO] Before normalization: mean={mean_val:.6f}, std={std_val:.6f}')
    dataset = (dataset - mean_val) / std_val

    has_inf = np.any(np.isinf(dataset))
    has_nan = np.any(np.isnan(dataset))

    save_path = os.path.join(save_dir, f'{prefix}_{channel}_mean_std.npy')
    if (not has_inf) and (not has_nan):
        np.save(save_path, dataset)
        print(f'[INFO] Saved {save_path}, shape={dataset.shape}')
        print(f'[INFO] After normalization: mean={np.mean(dataset):.6f}, std={np.std(dataset):.6f}')
    else:
        print(f'[ERROR] {channel} still contains inf or nan, not saved.')

    return dataset


def normalize_record_arrays(record_arrays, channel, save_dir, prefix='TF'):
    """Normalize per-record arrays with a global channel mean/std and save old/new layouts."""
    if not record_arrays:
        raise ValueError(f'[ERROR] No {prefix} records found for channel={channel}')

    dataset = np.concatenate([data for _, data in record_arrays], axis=0)
    normalized = data_normalize(dataset=dataset, channel=channel, save_dir=save_dir, prefix=prefix)
    if np.any(np.isinf(normalized)) or np.any(np.isnan(normalized)):
        raise ValueError(f'[ERROR] Normalized {prefix} data for {channel} contains inf or nan')

    offset = 0
    normalized_records = []
    for record_id, data in record_arrays:
        next_offset = offset + data.shape[0]
        normalized_records.append((record_id, normalized[offset:next_offset]))
        offset = next_offset

    if offset != normalized.shape[0]:
        raise ValueError(
            f'[ERROR] Per-record {prefix} normalization offset {offset} != global length {normalized.shape[0]}'
        )

    save_record_arrays(normalized_records, channel=channel, save_dir=save_dir, prefix=prefix)
    return normalized_records


def interpolate_time_windows(epoch, window=200, step=100, target_len=128, num_windows=29):
    """Slice one raw epoch with STFT-aligned windows and resample each window."""
    if epoch.shape[0] != 3000:
        raise ValueError(f'[ERROR] Expected raw epoch length 3000, got {epoch.shape[0]}')

    x_old = np.linspace(0, window - 1, window)
    x_new = np.linspace(0, window - 1, target_len)
    windows = np.zeros((num_windows, target_len), dtype=np.float32)

    for i in range(num_windows):
        start = i * step
        subdata = epoch[start:start + window]
        if subdata.shape[0] != window:
            raise ValueError(
                f'[ERROR] Window {i} has length {subdata.shape[0]}, expected {window}'
            )
        windows[i] = np.interp(x_new, x_old, subdata).astype(np.float32)

    return windows


if __name__ == '__main__':
    path = Path()

    fs = 100
    overlap = 1
    nfft = 256
    win_size = 2

    # 1. 先检查原始 raw_data 的四个目录是否严格对齐
    check_raw_data_alignment(path.path_raw_data)

    # 2. 检查 labels 总体分布
    check_label_distribution(os.path.join(path.path_raw_data, 'labels'))

    # 3. 处理三个通道的 TF 图和对齐的时域插值窗口
    for channel in ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG']:
        print('\n' + '-' * 15, f'Processing channel: {channel}', '-' * 15)

        record_raw_arrays = load_record_raw_arrays(
            path_array=os.path.join(path.path_raw_data, channel),
            channel=channel
        )

        print('Transform to TF images and time-domain interpolation windows:')
        tf_records, time_records = transform_record_arrays(
            record_raw_arrays,
            fs=fs,
            win_size=win_size,
            overlap=overlap,
            nfft=nfft
        )

        X = np.concatenate([data for _, data in tf_records], axis=0)
        X_time = np.concatenate([data for _, data in time_records], axis=0)

        print(f'[INFO] TF image shape for {channel}: {X.shape}')
        print('Normalize TF:')
        normalize_record_arrays(record_arrays=tf_records, channel=channel, save_dir=path.path_TF, prefix='TF')

        print(f'[INFO] TIME image shape for {channel}: {X_time.shape}')
        print('Normalize TIME:')
        normalize_record_arrays(record_arrays=time_records, channel=channel, save_dir=path.path_TF, prefix='TIME')
