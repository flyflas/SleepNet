import os
import numpy as np
from tqdm import tqdm

from sklearn.metrics import (
    recall_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    balanced_accuracy_score,
)

import torch
from torch.utils.data import DataLoader

from model_transformer_cross_c import Transformer
from data_loader import (
    build_context_dataset,
    build_record_folds,
    load_record_sequences,
    log_record_split_summary,
)
from args import Config, Path


CLASS_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']


class CheckpointIncompatibleError(RuntimeError):
    pass


class FoldSkippedError(RuntimeError):
    pass


def _safe_path_component(value):
    value = str(value)
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in value)


def build_context_output_root(config, base_root='./Kfold_models'):
    suffix = (
        f'{config.model_name}_ctxL{config.context_length}_'
        f'L{config.context_left}_R{config.context_right}_'
        f'{config.split_group_policy}'
    )
    return f'{base_root}_{_safe_path_component(suffix)}'


def specificity(y_true, y_pred, n=5):
    spec = []
    con_mat = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spec1 = tn / (tn + fp + 1e-12)
        spec.append(spec1)
    return np.mean(spec)


def class_wise_specificity(con_mat):
    spec_list = []
    total = np.sum(con_mat)

    for i in range(con_mat.shape[0]):
        tp = con_mat[i, i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = total - tp - fn - fp
        spec = tn / (tn + fp + 1e-12)
        spec_list.append(spec)

    return np.array(spec_list, dtype=np.float64)


def class_wise_evaluate(con_mat):
    """
    columns: precision, recall, f1, specificity, support
    """
    num_classes = con_mat.shape[0]
    class_wise_mat = np.empty((num_classes, 5), dtype=np.float64)

    spec_list = class_wise_specificity(con_mat)

    for i in range(num_classes):
        precision = con_mat[i, i] / (np.sum(con_mat[:, i]) + 1e-12)
        recall = con_mat[i, i] / (np.sum(con_mat[i, :]) + 1e-12)
        f1 = (2 * precision * recall) / (precision + recall + 1e-12)
        support = np.sum(con_mat[i, :])

        class_wise_mat[i, 0] = precision
        class_wise_mat[i, 1] = recall
        class_wise_mat[i, 2] = f1
        class_wise_mat[i, 3] = spec_list[i]
        class_wise_mat[i, 4] = support

    return class_wise_mat


def print_class_wise_result(class_wise_result, class_names=CLASS_NAMES):
    print('\n===== CLASS-WISE METRICS =====')
    print(f'{"Class":<8} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Spec":>10} {"Support":>10}')
    for i, name in enumerate(class_names):
        precision, recall, f1, spec, support = class_wise_result[i]
        print(f'{name:<8} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {spec:>10.4f} {int(support):>10}')


def test(model, test_loader, config):
    model.eval()

    pred = []
    label = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing', leave=False):
            data = data.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True)

            output = model(data)
            pred.extend(torch.argmax(output, dim=1).cpu().numpy())
            label.extend(target.cpu().numpy())

    accuracy = accuracy_score(label, pred)
    cohens_kappa = cohen_kappa_score(label, pred)
    macro_f1 = f1_score(label, pred, average='macro')
    weighted_f1 = f1_score(label, pred, average='weighted')
    average_sensitivity = recall_score(label, pred, average='macro')
    average_specificity = specificity(label, pred, n=5)
    balanced_acc = balanced_accuracy_score(label, pred)

    con_mat = confusion_matrix(label, pred, labels=[0, 1, 2, 3, 4])
    class_wise_result = class_wise_evaluate(con_mat)

    print(
        'ACC: %.4f | Kappa: %.4f | Macro-F1: %.4f | Weighted-F1: %.4f | Sens: %.4f | Spec: %.4f | Bal_ACC: %.4f'
        % (accuracy, cohens_kappa, macro_f1, weighted_f1, average_sensitivity, average_specificity, balanced_acc)
    )

    print_class_wise_result(class_wise_result)

    return (
        accuracy,
        cohens_kappa,
        macro_f1,
        weighted_f1,
        average_sensitivity,
        average_specificity,
        balanced_acc,
        con_mat,
        class_wise_result,
    )


def _load_context_checkpoint(model, path_model, config):
    try:
        state_dict = torch.load(path_model, map_location=config.device)
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise CheckpointIncompatibleError(
            f'[SKIP] checkpoint is incompatible with context model architecture: {path_model}. '
            f'Detail: {exc}'
        )


def evaluate_single_fold(config, test_records, fold, output_root):
    path_model = os.path.join(output_root, f'fold{fold}', 'model.pkl')
    if not os.path.exists(path_model):
        raise FileNotFoundError(f'[SKIP] fold {fold} context checkpoint not found: {path_model}')

    log_record_split_summary(f'test fold {fold}', test_records, config=config)
    test_set = build_context_dataset(test_records, config=config)
    if len(test_set) == 0:
        raise FoldSkippedError(f'[SKIP] fold {fold} has no valid center epochs after context boundary dropping')

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = Transformer(config).to(config.device)
    _load_context_checkpoint(model, path_model, config)

    result = test(model, test_loader, config)

    del model
    torch.cuda.empty_cache()

    return result


def evaluate(config, path):
    records = load_record_sequences(
        path_labels=path.path_labels,
        path_dataset=path.path_TF,
        use_time=config.use_time,
        config=config
    )
    folds = build_record_folds(records, config=config, random_state=0)
    output_root = build_context_output_root(config)
    print(f'[INFO] context output root = {output_root}')

    ACC = 0.0
    Kappa = 0.0
    MF1 = 0.0
    WF1 = 0.0
    Sens = 0.0
    Spec = 0.0
    Bal_ACC = 0.0
    Confusion_mat = np.zeros([5, 5], dtype=np.float64)

    valid_folds = []
    skipped_folds = []

    for fold, (_, test_records) in enumerate(folds):
        print('\n' + '-' * 15, '>', f'Fold {fold}', '<', '-' * 15)

        try:
            (
                accuracy,
                cohens_kappa,
                macro_f1,
                weighted_f1,
                average_sensitivity,
                average_specificity,
                balanced_acc,
                con_mat,
                _
            ) = evaluate_single_fold(config, test_records, fold, output_root)
        except (FileNotFoundError, FoldSkippedError, CheckpointIncompatibleError) as exc:
            print(exc)
            skipped_folds.append((fold, str(exc)))
            continue

        ACC += accuracy
        Kappa += cohens_kappa
        MF1 += macro_f1
        WF1 += weighted_f1
        Sens += average_sensitivity
        Spec += average_specificity
        Bal_ACC += balanced_acc
        Confusion_mat += con_mat

        valid_folds.append(fold)

    if len(valid_folds) == 0:
        raise RuntimeError(
            '[ERROR] No valid context fold evaluations completed. '
            f'skipped_folds={skipped_folds}'
        )

    num_valid = len(valid_folds)
    ACC /= num_valid
    Kappa /= num_valid
    MF1 /= num_valid
    WF1 /= num_valid
    Sens /= num_valid
    Spec /= num_valid
    Bal_ACC /= num_valid

    class_wise_result = class_wise_evaluate(Confusion_mat)

    return ACC, Kappa, MF1, WF1, Sens, Spec, Bal_ACC, Confusion_mat, class_wise_result, valid_folds, skipped_folds


if __name__ == '__main__':
    config = Config()
    path = Path()
    print(f'[INFO] use_time = {config.use_time}')
    print(
        f'[INFO] context_left={config.context_left}, context_right={config.context_right}, '
        f'context_length={config.context_length}, center_index={config.context_center_index}, '
        f'split_group_policy={config.split_group_policy}'
    )

    (
        ACC,
        Kappa,
        MF1,
        WF1,
        Sens,
        Spec,
        Bal_ACC,
        Confusion_mat,
        class_wise_result,
        valid_folds,
        skipped_folds,
    ) = evaluate(config, path)

    print('\n===== FINAL RESULT =====')
    print('valid_folds: ', valid_folds)
    print('skipped_folds: ', skipped_folds)
    print('ACC: ', ACC)
    print("Cohen's Kappa: ", Kappa)
    print('Macro-F1: ', MF1)
    print('Weighted-F1: ', WF1)
    print('Sensitivity (Macro Recall): ', Sens)
    print('Specificity (Macro): ', Spec)
    print('Balanced Accuracy: ', Bal_ACC)

    print('\nconfusion_mat:')
    print(Confusion_mat)

    print_class_wise_result(class_wise_result)
