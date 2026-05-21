from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable


def load_mlflow_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        _load_dotenv_fallback()
        return

    load_dotenv()


def _load_dotenv_fallback(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default

    return parsed if parsed > 0 else default


class MlflowTrainingLogger:
    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        log_every_n_steps: int | None = None,
        enabled: bool = True,
        strict: bool | None = None,
        enable_artifacts: bool | None = None,
    ) -> None:
        load_mlflow_env()

        self.strict = (
            strict
            if strict is not None
            else _parse_bool(os.getenv("MLFLOW_STRICT"), default=False)
        )
        self.enable_artifacts = (
            enable_artifacts
            if enable_artifacts is not None
            else _parse_bool(os.getenv("MLFLOW_ENABLE_ARTIFACTS"), default=True)
        )
        self.experiment_name = (
            experiment_name
            or os.getenv("MLFLOW_EXPERIMENT_NAME")
            or "pytorch-training"
        )
        self.run_name = run_name or os.getenv("MLFLOW_RUN_NAME")
        self.log_every_n_steps = (
            log_every_n_steps
            if log_every_n_steps is not None and log_every_n_steps > 0
            else _parse_positive_int(os.getenv("MLFLOW_LOG_EVERY_N_STEPS"), default=20)
        )
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.enabled = enabled and self._is_rank_zero()
        self._run_started = False
        self._mlflow = None

        if not self.enabled:
            return

        try:
            import mlflow
        except Exception as exc:
            self._handle_error("import mlflow", exc)
            self.enabled = False
            return

        self._mlflow = mlflow

        if not self.tracking_uri or self._is_placeholder_tracking_uri(self.tracking_uri):
            exc = RuntimeError("MLFLOW_TRACKING_URI is required for MLflow tracking")
            self._handle_error("read MLFLOW_TRACKING_URI", exc)
            self.enabled = False

    def start(self, params: dict | None = None, tags: dict | None = None) -> None:
        if not self.enabled:
            return

        if not self._startup_safe("set tracking uri", self._mlflow.set_tracking_uri, self.tracking_uri):
            return
        if not self._startup_safe("set experiment", self._mlflow.set_experiment, self.experiment_name):
            return
        if not self._startup_safe("start run", self._start_run):
            return

        if params:
            clean_params = {key: self._clean_param(value) for key, value in params.items()}
            if not self._startup_safe("log params", self._mlflow.log_params, clean_params):
                return

        merged_tags = {
            "framework": "pytorch",
            "tracking_client": "mlflow",
            "runtime": "third-party-gpu",
        }
        if tags:
            merged_tags.update(tags)

        for key, value in merged_tags.items():
            if not self._startup_safe("set tag", self._mlflow.set_tag, key, self._clean_param(value)):
                return

    def log_train_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        acc: float | None = None,
        lr: float | None = None,
        samples_seen: int | None = None,
        extra_metrics: dict | None = None,
    ) -> None:
        if not self.enabled or step % self.log_every_n_steps != 0:
            return

        metrics = {
            "train/loss": loss,
            "train/epoch": epoch,
        }
        if acc is not None:
            metrics["train/acc"] = acc
        if lr is not None:
            metrics["train/lr"] = lr
        if samples_seen is not None:
            metrics["train/samples_seen"] = samples_seen

        try:
            import torch

            if torch.cuda.is_available():
                metrics.update(
                    {
                        "gpu/memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                        "gpu/memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                        "gpu/max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                    }
                )
        except Exception as exc:
            self._handle_error("read CUDA memory metrics", exc)

        metrics.update(self._prefixed_metrics(extra_metrics, prefix="train/"))
        self._safe("log train metrics", self._mlflow.log_metrics, self._clean_metrics(metrics), step=step)

    def log_validation(
        self,
        step: int,
        epoch: int,
        val_loss: float,
        val_acc: float | None = None,
        extra_metrics: dict | None = None,
    ) -> None:
        if not self.enabled:
            return

        metrics = {
            "val/loss": val_loss,
            "val/epoch": epoch,
        }
        if val_acc is not None:
            metrics["val/acc"] = val_acc

        metrics.update(self._prefixed_metrics(extra_metrics, prefix="val/"))
        self._safe("log validation metrics", self._mlflow.log_metrics, self._clean_metrics(metrics), step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        if not self.enabled or not self.enable_artifacts:
            return

        if not os.path.exists(local_path):
            print(f"[MLflow warning] artifact does not exist: {local_path}")
            return

        self._safe(
            "log artifact",
            self._mlflow.log_artifact,
            local_path,
            artifact_path=artifact_path,
        )

    def end(self, status: str = "FINISHED") -> None:
        if not self.enabled or not self._run_started:
            return

        status = status if status in {"FINISHED", "FAILED", "KILLED"} else "FINISHED"
        self._safe("end run", self._mlflow.end_run, status=status)
        self._run_started = False

    def _start_run(self) -> None:
        self._mlflow.start_run(run_name=self.run_name)
        self._run_started = True

    def _safe(self, action: str, func: Callable, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            return self._handle_error(action, exc)

    def _startup_safe(self, action: str, func: Callable, *args: Any, **kwargs: Any) -> bool:
        try:
            func(*args, **kwargs)
            return True
        except Exception as exc:
            self._cleanup_failed_startup()
            self._handle_error(action, exc)
            self.enabled = False
            return False

    def _cleanup_failed_startup(self) -> None:
        if not self._run_started:
            return

        try:
            self._mlflow.end_run(status="FAILED")
        except Exception as exc:
            if not self.strict:
                print(f"[MLflow warning] cleanup failed startup run failed: {exc}")
        finally:
            self._run_started = False

    def _handle_error(self, action: str, exc: Exception) -> None:
        if self.strict:
            raise exc
        print(f"[MLflow warning] {action} failed: {exc}")
        return None

    @staticmethod
    def _is_rank_zero() -> bool:
        rank = os.getenv("RANK")
        if rank is not None:
            return rank == "0"

        local_rank = os.getenv("LOCAL_RANK")
        if local_rank is not None:
            return local_rank == "0"

        return True

    @staticmethod
    def _clean_param(value: Any) -> Any:
        if value is None:
            return "None"
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def _is_placeholder_tracking_uri(value: str) -> bool:
        normalized = value.strip().strip('"').strip("'")
        return normalized in {"", "http://YOUR_SERVER_IP:5000", "https://YOUR_SERVER_IP:5000"}

    def _clean_metrics(self, metrics: dict) -> dict:
        clean = {}
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                self._handle_error(f"convert metric {key}", exc)
                continue
            if math.isnan(numeric) or math.isinf(numeric):
                self._handle_error(
                    f"convert metric {key}",
                    ValueError(f"metric value is not finite: {value}"),
                )
                continue
            clean[key] = numeric
        return clean

    @staticmethod
    def _prefixed_metrics(metrics: dict | None, prefix: str) -> dict:
        if not metrics:
            return {}
        return {
            key if "/" in str(key) else f"{prefix}{key}": value
            for key, value in metrics.items()
        }
