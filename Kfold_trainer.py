import json
import os
import urllib.request
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from model_transformer_cross_c import Transformer
from early_stop_tool import EarlyStopping
from data_loader import (
    build_context_datasets_for_fold,
    build_record_folds,
    load_record_sequences,
)
from args import Config, Path
from mlflow_logger import MlflowTrainingLogger, load_mlflow_env


NOTIFICATION_URL = 'http://8.209.229.122:8181/send'
NOTIFICATION_HEADERS = {
    'Authentication': 'Bearer 927efb8b-2fbf-4903-9c07-8b1798d57d98',
    'Content-Type': 'application/json',
}
NOTIFICATION_PAYLOAD = {
    'subject': 'sleep 训练任务已全部完成',
    'body': 'Kfold_trainer.py 的所有 fold 训练任务已全部完成。',
}


def send_all_tasks_finished_notification():
    try:
        data = json.dumps(NOTIFICATION_PAYLOAD, ensure_ascii=False).encode('utf-8')
        request = urllib.request.Request(
            NOTIFICATION_URL,
            data=data,
            headers=NOTIFICATION_HEADERS,
            method='POST'
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except Exception as exc:
        print(f'[WARNING] Failed to send all-tasks-finished notification: {exc}')


def configure_torch_runtime(config):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
        try:
            torch.set_float32_matmul_precision('high' if config.allow_tf32 else 'highest')
        except Exception:
            pass

    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cudnn.benchmark = not config.deterministic


def amp_dtype_from_config(config):
    dtype = str(config.amp_dtype).lower()
    if dtype in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if dtype in ('fp16', 'float16', 'half'):
        return torch.float16
    raise ValueError(f'[ERROR] Unsupported SLEEP_AMP_DTYPE={config.amp_dtype}. Use bf16 or fp16.')


def autocast_context(config):
    if not config.use_amp or config.device.type != 'cuda':
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=amp_dtype_from_config(config))


def build_grad_scaler(config):
    use_fp16 = config.use_amp and config.device.type == 'cuda' and amp_dtype_from_config(config) is torch.float16
    return torch.amp.GradScaler('cuda', enabled=use_fp16)


def set_random_seed(seed=0, deterministic=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def print_label_distribution(labels, split_name='dataset'):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    print(f'\n[{split_name}] label distribution:')
    for u, c in zip(unique, counts):
        print(f'  class {u}: {c} ({c / total:.6f})')


def build_dataloader(dataset, batch_size, shuffle, config, is_train=False):
    if len(dataset) == 0:
        raise ValueError('[ERROR] Cannot build DataLoader for an empty context dataset')

    num_workers = config.num_workers
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=config.prefetch_factor if num_workers > 0 else None,
        drop_last=is_train and config.drop_last_train_batch
    )


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


def _dataset_group_count(dataset):
    return len(set(record.subject_id if dataset.config.split_group_policy == 'subject' else record.record_id
                   for record in dataset.records))


def log_context_dataset_summary(split_name, dataset):
    record_count = len(set(record.record_id for record in dataset.records))
    group_count = _dataset_group_count(dataset)
    epoch_count = sum(record.epoch_count for record in dataset.records)
    sample_count = len(dataset)
    print(
        f'[INFO][context {split_name}] input_shape={_context_input_shape(dataset)}, '
        f'groups={group_count}, records={record_count}, epochs={epoch_count}, '
        f'samples={sample_count}, dropped_boundary_epochs={dataset.dropped_boundary_epochs}'
    )


def _context_input_shape(dataset):
    if len(dataset) > 0:
        x, _ = dataset[0]
        return tuple(x.shape)
    if dataset.records:
        return (dataset.context_length,) + tuple(dataset.records[0].features.shape[1:])
    return (dataset.context_length,)


def evaluate(model, loader, criterion, config, split_name='eval', print_distribution=False):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    pred_counts = torch.zeros(config.num_classes, dtype=torch.long)
    label_counts = torch.zeros(config.num_classes, dtype=torch.long)

    with torch.inference_mode():
        for data, target in loader:
            data = data.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True).long()

            with autocast_context(config):
                output = model(data)
                loss = criterion(output, target)

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            pred = torch.argmax(output, dim=1)
            total_correct += (pred == target).sum().item()

            if print_distribution:
                pred_counts += torch.bincount(
                    pred.cpu(),
                    minlength=config.num_classes
                )[:config.num_classes]
                label_counts += torch.bincount(
                    target.cpu(),
                    minlength=config.num_classes
                )[:config.num_classes]

    if total_samples == 0:
        raise ValueError(f'[ERROR] Cannot evaluate empty {split_name} context loader')

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    if print_distribution:
        print(f'\n[{split_name}] prediction distribution:')
        for cls_idx, count in enumerate(pred_counts.tolist()):
            if count:
                print(f'  pred class {cls_idx}: {count} ({count / total_samples:.6f})')

        print(f'[{split_name}] true label distribution:')
        for cls_idx, count in enumerate(label_counts.tolist()):
            if count:
                print(f'  true class {cls_idx}: {count} ({count / total_samples:.6f})')

    return accuracy, avg_loss


def is_fold_finished(fold_dir: str) -> bool:
    """
    判定某个 fold 是否已经完整跑完。
    你当前 trainer 在 fold 结束后会保存这些 npy 文件，
    所以这些文件都存在时，就认为这个 fold 已完成。
    """
    required_files = [
        'train_LOSS.npy',
        'train_ACC.npy',
        'test_LOSS.npy',
        'test_ACC.npy',
        'val_LOSS.npy',
        'val_ACC.npy',
    ]
    return all(os.path.exists(os.path.join(fold_dir, f)) for f in required_files)


def find_first_unfinished_fold(num_fold: int, root='./Kfold_models') -> int:
    """
    自动找到第一个未完成的 fold。
    如果都完成了，返回 num_fold。
    """
    for fold in range(num_fold):
        fold_dir = os.path.join(root, f'fold{fold}')
        if not is_fold_finished(fold_dir):
            return fold
    return num_fold


def build_mlflow_params(config, fold, save_all_checkpoint):
    return {
        'model_name': config.model_name,
        'dataset_name': 'sleepEDF-78',
        'fold': fold,
        'num_fold': config.num_fold,
        'epochs': config.num_epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'optimizer_name': 'AdamW',
        'scheduler_name': None,
        'seed': 0,
        'device': str(config.device),
        'cuda_available': torch.cuda.is_available(),
        'num_classes': config.num_classes,
        'pad_size': config.pad_size,
        'dropout': config.dropout,
        'dim_model': config.dim_model,
        'forward_hidden': config.forward_hidden,
        'fc_hidden': config.fc_hidden,
        'num_head': config.num_head,
        'num_encoder': config.num_encoder,
        'num_encoder_multi': config.num_encoder_multi,
        'context_num_head': config.context_num_head,
        'context_num_encoder': config.context_num_encoder,
        'context_forward_hidden': config.context_forward_hidden,
        'context_dropout': config.context_dropout,
        'context_use_local_center_concat': config.context_use_local_center_concat,
        'context_left': config.context_left,
        'context_right': config.context_right,
        'context_length': config.context_length,
        'context_center_index': config.context_center_index,
        'split_group_policy': config.split_group_policy,
        'subject_id_length': config.subject_id_length,
        'validation_group_fraction': config.validation_group_fraction,
        'use_positional_encoding': config.use_positional_encoding,
        'weight_decay': config.weight_decay,
        'use_amp': config.use_amp,
        'amp_dtype': config.amp_dtype,
        'allow_tf32': config.allow_tf32,
        'compile_model': config.compile_model,
        'compile_mode': config.compile_mode,
        'deterministic': config.deterministic,
        'num_workers': config.num_workers,
        'prefetch_factor': config.prefetch_factor,
        'pin_memory': config.pin_memory,
        'drop_last_train_batch': config.drop_last_train_batch,
        'train_log_every_n_steps': config.train_log_every_n_steps,
        'progress_every_n_steps': config.progress_every_n_steps,
        'early_stop_patience': config.early_stop_patience,
        'save_all_checkpoint': save_all_checkpoint,
    }


def build_mlflow_tags(fold):
    return {
        'entrypoint': 'Kfold_trainer.py',
        'fold': fold,
        'training_mode': 'epoch_context',
    }


def train(save_all_checkpoint=False, start_fold=None):
    load_mlflow_env()
    config = Config()
    configure_torch_runtime(config)
    path = Path()

    print(f'[INFO] device = {config.device}')
    print(f'[INFO] batch_size = {config.batch_size}')
    print(f'[INFO] learning_rate = {config.learning_rate}')
    print(f'[INFO] num_epochs = {config.num_epochs}')
    print(
        f'[INFO] amp={config.use_amp}, amp_dtype={config.amp_dtype}, '
        f'allow_tf32={config.allow_tf32}, compile_model={config.compile_model}'
    )
    print(
        f'[INFO] num_workers={config.num_workers}, prefetch_factor={config.prefetch_factor}, '
        f'pin_memory={config.pin_memory}, drop_last_train_batch={config.drop_last_train_batch}'
    )
    print(
        f'[INFO] train_log_every_n_steps={config.train_log_every_n_steps}, '
        f'progress_every_n_steps={config.progress_every_n_steps}'
    )
    print(f'[INFO] use_time = {config.use_time}')
    print(
        f'[INFO] context_left={config.context_left}, context_right={config.context_right}, '
        f'context_length={config.context_length}, center_index={config.context_center_index}'
    )
    print(
        f'[INFO] split_group_policy={config.split_group_policy}, '
        f'validation_group_fraction={config.validation_group_fraction}'
    )

    records = load_record_sequences(
        path_labels=path.path_labels,
        path_dataset=path.path_TF,
        use_time=config.use_time,
        config=config
    )
    labels = torch.cat([record.labels for record in records], dim=0)
    print_label_distribution(labels, split_name='full record dataset')
    folds = build_record_folds(records, config=config, random_state=0)
    output_root = build_context_output_root(config)
    print(f'[INFO] context output root = {output_root}')

    # 自动找未完成 fold
    auto_start_fold = find_first_unfinished_fold(config.num_fold, root=output_root)

    if start_fold is None:
        start_fold = auto_start_fold

    print(f'[INFO] resume start fold = {start_fold}')

    if start_fold >= config.num_fold:
        print('[INFO] All folds are already finished. Nothing to do.')
        send_all_tasks_finished_notification()
        return

    for fold, (train_records, test_records) in enumerate(folds):
        fold_dir = os.path.join(output_root, f'fold{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        # 1) 小于 start_fold 的一律跳过
        if fold < start_fold:
            print(f'[INFO] Skip fold {fold} (before start_fold={start_fold}).')
            continue

        # 2) 如果该 fold 已完整完成，也跳过
        if is_fold_finished(fold_dir):
            print(f'[INFO] Skip fold {fold} (already finished).')
            continue

        print('\n' + '-' * 15 + f' > Fold {fold} < ' + '-' * 15)
        mlflow_run_name = os.getenv('MLFLOW_RUN_NAME')
        if mlflow_run_name:
            mlflow_run_name = f'{mlflow_run_name}-fold-{fold}'
        else:
            mlflow_run_name = f'fold-{fold}'

        mlflow_logger = MlflowTrainingLogger(
            run_name=mlflow_run_name,
            log_every_n_steps=config.train_log_every_n_steps
        )
        mlflow_logger.start(
            params=build_mlflow_params(config, fold, save_all_checkpoint),
            tags=build_mlflow_tags(fold)
        )

        model = None
        try:
            train_set, val_set, test_set = build_context_datasets_for_fold(
                train_records=train_records,
                test_records=test_records,
                config=config,
                fold=fold
            )
            log_context_dataset_summary('train', train_set)
            log_context_dataset_summary('val', val_set)
            log_context_dataset_summary('test', test_set)

            train_loader = build_dataloader(
                dataset=train_set,
                batch_size=config.batch_size,
                shuffle=True,
                config=config,
                is_train=True
            )
            val_loader = build_dataloader(
                dataset=val_set,
                batch_size=config.batch_size,
                shuffle=False,
                config=config
            )
            test_loader = build_dataloader(
                dataset=test_set,
                batch_size=config.batch_size,
                shuffle=False,
                config=config
            )

            model = Transformer(config).to(config.device)
            model_for_checkpoint = model
            if config.compile_model:
                model = torch.compile(model, mode=config.compile_mode)
            criterion = nn.CrossEntropyLoss()
            grad_scaler = build_grad_scaler(config)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=0.01
            )

            early_stopping = EarlyStopping(
                patience=config.early_stop_patience,
                verbose=True,
                save_all_checkpoint=save_all_checkpoint
            )

            train_ACC = []
            train_LOSS = []
            test_ACC = []
            test_LOSS = []
            val_ACC = []
            val_LOSS = []
            global_step = 0
            global_samples_seen = 0

            for epoch in range(config.num_epochs):
                model.train()

                total_train_loss = torch.zeros((), device=config.device)
                total_train_correct = torch.zeros((), device=config.device)
                total_train_samples = 0

                loop = tqdm(
                    train_loader,
                    total=len(train_loader),
                    desc=f'Fold {fold} Epoch {epoch}',
                    mininterval=config.progress_min_interval
                )

                for data, target in loop:
                    data = data.to(config.device, non_blocking=True)
                    target = target.to(config.device, non_blocking=True).long()

                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(config):
                        output = model(data)
                        loss = criterion(output, target)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pred = torch.argmax(output.detach(), dim=1)

                    batch_size = target.size(0)
                    correct_batch = (pred == target).sum()
                    total_train_loss += loss.detach() * batch_size
                    total_train_correct += correct_batch
                    total_train_samples += batch_size
                    global_step += 1
                    global_samples_seen += batch_size

                    if global_step % config.train_log_every_n_steps == 0:
                        loss_value = loss.detach().item()
                        train_acc_batch = (correct_batch.float() / batch_size).item()
                        current_lr = optimizer.param_groups[0]['lr']
                        mlflow_logger.log_train_step(
                            step=global_step,
                            epoch=epoch,
                            loss=loss_value,
                            acc=train_acc_batch,
                            lr=current_lr,
                            samples_seen=global_samples_seen,
                            extra_metrics={'fold': fold}
                        )

                    if global_step % config.progress_every_n_steps == 0:
                        loss_value = loss.detach().item()
                        train_acc_batch = (correct_batch.float() / batch_size).item()
                        loop.set_postfix(
                            loss=f'{loss_value:.4f}',
                            train_acc=f'{train_acc_batch:.4f}'
                        )

                train_loss = total_train_loss.item() / total_train_samples
                train_acc = total_train_correct.item() / total_train_samples

                need_print_dist = (epoch < 3) or (epoch % 10 == 0)

                test_acc, test_loss = evaluate(
                    model=model,
                    loader=test_loader,
                    criterion=criterion,
                    config=config,
                    split_name='test',
                    print_distribution=need_print_dist
                )
                val_acc, val_loss = evaluate(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    config=config,
                    split_name='val',
                    print_distribution=need_print_dist
                )
                mlflow_logger.log_validation(
                    step=global_step,
                    epoch=epoch,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    extra_metrics={
                        'fold': fold,
                        'test/loss': test_loss,
                        'test/acc': test_acc,
                        'train/loss_epoch': train_loss,
                        'train/acc_epoch': train_acc,
                    }
                )

                print(
                    f'Epoch: {epoch:3d} | '
                    f'train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | '
                    f'val acc: {val_acc:.4f} | val loss: {val_loss:.4f} | '
                    f'test acc: {test_acc:.4f} | test loss: {test_loss:.4f}'
                )

                train_ACC.append(train_acc)
                train_LOSS.append(train_loss)
                test_ACC.append(test_acc)
                test_LOSS.append(test_loss)
                val_ACC.append(val_acc)
                val_LOSS.append(val_loss)

                model_path = os.path.join(fold_dir, f'model_{fold}_epoch{epoch}.pkl')
                checkpoint_path = early_stopping(val_acc, model_for_checkpoint, path=model_path)
                if checkpoint_path:
                    mlflow_logger.log_artifact(checkpoint_path, artifact_path='checkpoints')

                if early_stopping.early_stop:
                    print(f'[INFO] Early stopping at epoch {epoch}')
                    break

            np.save(os.path.join(fold_dir, 'train_LOSS.npy'), np.array(train_LOSS))
            np.save(os.path.join(fold_dir, 'train_ACC.npy'), np.array(train_ACC))
            np.save(os.path.join(fold_dir, 'test_LOSS.npy'), np.array(test_LOSS))
            np.save(os.path.join(fold_dir, 'test_ACC.npy'), np.array(test_ACC))
            np.save(os.path.join(fold_dir, 'val_LOSS.npy'), np.array(val_LOSS))
            np.save(os.path.join(fold_dir, 'val_ACC.npy'), np.array(val_ACC))

            mlflow_logger.end('FINISHED')
        except Exception:
            try:
                mlflow_logger.end('FAILED')
            except Exception as exc:
                print(f'[MLflow warning] end failed while handling training exception: {exc}')
            raise
        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache()

    send_all_tasks_finished_notification()


if __name__ == '__main__':
    set_random_seed(0, deterministic=Config().deterministic)
    train(save_all_checkpoint=False, start_fold=None)
