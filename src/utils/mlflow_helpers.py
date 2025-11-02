"""MLflow integration helpers for tracking and artifact management.

Provides:
    - Experiment setup with naming conventions
    - Nested runs for HPO (study → trials)
    - Bulk parameter/metric logging
    - Artifact logging with checksums
    - Run resume/continue logic

Used by:
    - scripts/train.py: Log hyperparams, metrics, checkpoints
    - HPO objective: Nested trial runs with validation metrics
    - CI: Archive golden test results

Tracked entities:
    Parameters:
        - Config (flattened YAML)
        - Resolutions (render_px, obs_px, reward_px)
        - LUT hashes (provenance, via src.utils.hashing)
        - Reproducibility (seeds, cudnn flags)
    Metrics:
        - Training: mean_reward, mean_lpips, gpu_mem_hwm
        - HPO: avg_final_lpips_validation (minimize directly, no negation)
    Artifacts:
        - Checkpoints (.pth)
        - Validation paintings (trial_<N>/*.png)
        - Configs (train.yaml, env.yaml, etc.)

Convention: experiment names use "airbrush_<task>_v<N>" format.

Usage:
    from src.utils import mlflow_helpers, hashing
    
    # Setup experiment
    mlflow_helpers.setup_experiment("airbrush_train_v2")
    
    # Start run
    with mlflow_helpers.start_run("train_001") as run:
        # Log config
        mlflow_helpers.log_params_flat(config_dict)
        
        # Log LUT hashes for provenance
        mlflow_helpers.log_lut_hashes(lut_paths)
        
        # Training loop
        for epoch in range(100):
            mlflow_helpers.log_metrics({
                'mean_reward': reward,
                'mean_lpips': lpips_val,
            }, step=epoch)
        
        # Log checkpoint
        mlflow_helpers.log_artifact_with_hash(checkpoint_path)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import os

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available, tracking will be disabled")

from . import hashing, validators, fs

logger = logging.getLogger(__name__)


# ============================================================================
# MULTI-PROCESS SAFETY
# ============================================================================

def is_rank0() -> bool:
    """Check if current process is rank 0 (main process).
    
    Returns
    -------
    bool
        True if rank 0 or not distributed, False otherwise
    
    Notes
    -----
    Checks RANK, LOCAL_RANK environment variables.
    Only rank 0 should log to MLflow in distributed training.
    """
    rank = os.getenv("RANK")
    local_rank = os.getenv("LOCAL_RANK")
    
    # If neither is set, assume single process (rank 0)
    if rank is None and local_rank is None:
        return True
    
    # If either is set, check if it's "0"
    return rank == "0" or (rank is None and local_rank == "0")


# ============================================================================
# EXPERIMENT SETUP
# ============================================================================

def setup_experiment(
    experiment_name: str,
    artifact_location: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> str:
    """Setup or get existing MLflow experiment.
    
    Parameters
    ----------
    experiment_name : str
        Experiment name (convention: "airbrush_<task>_v<N>")
    artifact_location : Optional[str]
        Artifact root directory, None for default
    tags : Optional[Dict[str, str]]
        Experiment-level tags
    
    Returns
    -------
    str
        Experiment ID
    
    Notes
    -----
    If experiment exists, returns existing ID.
    Otherwise creates new experiment.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, setup_experiment no-op")
        return "mlflow_disabled"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
            return experiment.experiment_id
        else:
            exp_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
                tags=tags or {}
            )
            logger.info(f"Created experiment: {experiment_name} (ID: {exp_id})")
            return exp_id
    except Exception as e:
        logger.error(f"Failed to setup experiment {experiment_name}: {e}")
        raise


def start_run(
    run_name: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, str]] = None,
    experiment_id: Optional[str] = None
):
    """Start MLflow run (context manager).
    
    Parameters
    ----------
    run_name : Optional[str]
        Run name, None for auto-generated
    nested : bool
        True for nested runs (HPO trials), default False
    tags : Optional[Dict[str, str]]
        Run-level tags
    experiment_id : Optional[str]
        Experiment ID, None for active experiment
    
    Returns
    -------
    mlflow.ActiveRun
        Active run context manager
    
    Notes
    -----
    Use as context manager: `with start_run(...) as run:`
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, start_run returns dummy context")
        from contextlib import nullcontext
        return nullcontext()
    
    return mlflow.start_run(
        run_name=run_name,
        nested=nested,
        tags=tags or {},
        experiment_id=experiment_id
    )


def get_active_run_id() -> Optional[str]:
    """Get active MLflow run ID.
    
    Returns
    -------
    Optional[str]
        Run ID, None if no active run
    """
    if not MLFLOW_AVAILABLE:
        return None
    
    active_run = mlflow.active_run()
    return active_run.info.run_id if active_run else None


# ============================================================================
# PARAMETER LOGGING
# ============================================================================

def log_params_flat(params: Dict[str, Any]) -> None:
    """Log flattened parameters to MLflow with sanitation.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Parameter dictionary (can be nested)
    
    Notes
    -----
    Nested dicts are flattened with dot-separated keys.
    Uses validators.flatten_config_for_mlflow().
    Only logs from rank 0 in distributed training.
    Sanitizes keys/values to avoid MLflow limits.
    """
    if not MLFLOW_AVAILABLE or not is_rank0():
        return
    
    flat_params = validators.flatten_config_for_mlflow(params)
    
    # Sanitize parameters
    MAX_KEY_LEN = 120
    MAX_VAL_LEN = 250
    MAX_PARAMS = 100  # MLflow has limits on param count per call
    
    sanitized = {}
    for k, v in list(flat_params.items())[:MAX_PARAMS]:
        # Truncate key
        safe_key = str(k)[:MAX_KEY_LEN]
        # Truncate value, handle None
        if v is None:
            safe_val = "None"
        else:
            safe_val = str(v)[:MAX_VAL_LEN]
        sanitized[safe_key] = safe_val
    
    try:
        mlflow.log_params(sanitized)
        logger.debug(f"Logged {len(sanitized)} parameters (rank 0)")
    except Exception as e:
        logger.warning(f"Failed to log params: {e}")


def log_resolutions(
    render_px: int,
    obs_px: int,
    reward_px: int
) -> None:
    """Log resolution trio for traceability.
    
    Parameters
    ----------
    render_px : int
        Physics grid resolution
    obs_px : int
        Policy observation resolution
    reward_px : int
        LPIPS reward resolution
    
    Notes
    -----
    Critical for reproducing experiments with multi-resolution architecture.
    Only logs from rank 0 in distributed training.
    """
    if not MLFLOW_AVAILABLE or not is_rank0():
        return
    
    try:
        mlflow.log_params({
            'render_px': render_px,
            'obs_px': obs_px,
            'reward_px': reward_px,
        })
        logger.debug(f"Logged resolutions: render={render_px}, obs={obs_px}, reward={reward_px}")
    except Exception as e:
        logger.warning(f"Failed to log resolutions: {e}")


def log_lut_hashes(lut_paths: Dict[str, Union[str, Path]]) -> None:
    """Log LUT file hashes for provenance.
    
    Parameters
    ----------
    lut_paths : Dict[str, Union[str, Path]]
        Mapping of LUT names to file paths
        Example: {'color_lut': 'luts/color.pt', 'alpha_lut': 'luts/alpha.pt'}
    
    Notes
    -----
    Uses hashing.sha256_file() for deterministic hashes.
    Logged as parameters with prefix 'lut_hash.'
    Only logs from rank 0 in distributed training.
    """
    if not MLFLOW_AVAILABLE or not is_rank0():
        return
    
    hash_params = {}
    for lut_name, lut_path in lut_paths.items():
        try:
            lut_hash = hashing.sha256_file(lut_path)
            hash_params[f'lut_hash.{lut_name}'] = lut_hash[:16]  # First 16 chars
        except Exception as e:
            logger.warning(f"Failed to hash {lut_name} at {lut_path}: {e}")
    
    if hash_params:
        try:
            mlflow.log_params(hash_params)
            logger.debug(f"Logged {len(hash_params)} LUT hashes")
        except Exception as e:
            logger.warning(f"Failed to log LUT hashes: {e}")


def log_reproducibility_info(
    seed: int,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False
) -> None:
    """Log reproducibility settings.
    
    Parameters
    ----------
    seed : int
        Random seed
    cudnn_deterministic : bool
        CuDNN deterministic mode, default True
    cudnn_benchmark : bool
        CuDNN benchmark mode, default False
    """
    if not MLFLOW_AVAILABLE or not is_rank0():
        return
    
    try:
        mlflow.log_params({
            'seed': seed,
            'cudnn_deterministic': cudnn_deterministic,
            'cudnn_benchmark': cudnn_benchmark,
        })
        logger.debug(f"Logged reproducibility info: seed={seed}")
    except Exception as e:
        logger.warning(f"Failed to log reproducibility info: {e}")


# ============================================================================
# METRIC LOGGING
# ============================================================================

def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None
) -> None:
    """Log metrics to MLflow.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Metric dictionary (flat, no nesting)
    step : Optional[int]
        Step/epoch number, None for auto-increment
    
    Notes
    -----
    Metrics are time-series data indexed by step.
    Only logs from rank 0 in distributed training.
    """
    if not MLFLOW_AVAILABLE or not is_rank0():
        return
    
    try:
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics at step {step}")
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")


def log_metric(
    key: str,
    value: float,
    step: Optional[int] = None
) -> None:
    """Log single metric to MLflow.
    
    Parameters
    ----------
    key : str
        Metric name
    value : float
        Metric value
    step : Optional[int]
        Step/epoch number, None for auto-increment
    
    Notes
    -----
    Only logs from rank 0 in distributed training.
    """
    if not MLFLOW_AVAILABLE or not is_rank0():
        return
    
    try:
        mlflow.log_metric(key, value, step=step)
    except Exception as e:
        logger.warning(f"Failed to log metric {key}: {e}")


# ============================================================================
# ARTIFACT LOGGING
# ============================================================================

def log_artifact(
    local_path: Union[str, Path],
    artifact_path: Optional[str] = None
) -> None:
    """Log artifact to MLflow.
    
    Parameters
    ----------
    local_path : Union[str, Path]
        Path to local file
    artifact_path : Optional[str]
        Relative path within artifact store, None for root
    
    Notes
    -----
    Artifacts are immutable files stored with run.
    """
    if not MLFLOW_AVAILABLE:
        return
    
    local_path = Path(local_path)
    if not local_path.exists():
        logger.warning(f"Artifact not found: {local_path}, skipping")
        return
    
    try:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        logger.debug(f"Logged artifact: {local_path}")
    except Exception as e:
        logger.warning(f"Failed to log artifact {local_path}: {e}")


def log_artifact_with_hash(
    local_path: Union[str, Path],
    artifact_path: Optional[str] = None,
    log_hash_as_param: bool = True
) -> None:
    """Log artifact with SHA-256 hash for provenance.
    
    Parameters
    ----------
    local_path : Union[str, Path]
        Path to local file
    artifact_path : Optional[str]
        Relative path within artifact store, None for root
    log_hash_as_param : bool
        Also log hash as parameter, default True
    
    Notes
    -----
    Hash is logged as parameter 'artifact_hash.<filename>'.
    Useful for checkpoint provenance.
    """
    if not MLFLOW_AVAILABLE:
        return
    
    local_path = Path(local_path)
    
    # Log artifact
    log_artifact(local_path, artifact_path)
    
    # Compute and log hash
    if log_hash_as_param:
        try:
            file_hash = hashing.sha256_file(local_path)
            param_name = f'artifact_hash.{local_path.name}'
            mlflow.log_param(param_name, file_hash[:16])
            logger.debug(f"Logged hash for {local_path.name}: {file_hash[:16]}")
        except Exception as e:
            logger.warning(f"Failed to hash artifact {local_path}: {e}")


def log_artifacts_dir(
    local_dir: Union[str, Path],
    artifact_path: Optional[str] = None
) -> None:
    """Log entire directory as artifacts.
    
    Parameters
    ----------
    local_dir : Union[str, Path]
        Path to local directory
    artifact_path : Optional[str]
        Relative path within artifact store, None for root
    
    Notes
    -----
    Recursively logs all files in directory.
    """
    if not MLFLOW_AVAILABLE:
        return
    
    local_dir = Path(local_dir)
    if not local_dir.exists():
        logger.warning(f"Artifact directory not found: {local_dir}, skipping")
        return
    
    try:
        mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)
        logger.debug(f"Logged artifact directory: {local_dir}")
    except Exception as e:
        logger.warning(f"Failed to log artifact directory {local_dir}: {e}")


# ============================================================================
# HPO HELPERS
# ============================================================================

def log_hpo_trial(
    trial_number: int,
    hyperparams: Dict[str, Any],
    validation_scores: Dict[str, float],
    checkpoint_path: Union[str, Path],
    validation_image_dir: Optional[Union[str, Path]] = None
) -> None:
    """Log complete HPO trial results.
    
    Parameters
    ----------
    trial_number : int
        Trial number (0-indexed)
    hyperparams : Dict[str, Any]
        Sampled hyperparameters
    validation_scores : Dict[str, float]
        Validation metrics (e.g., {'avg_final_lpips_validation': 0.15})
    checkpoint_path : Union[str, Path]
        Path to trained checkpoint
    validation_image_dir : Optional[Union[str, Path]]
        Directory with validation paintings, None to skip
    
    Notes
    -----
    Logs hyperparams, validation scores, checkpoint, and images.
    Used by HPO objective function.
    """
    if not MLFLOW_AVAILABLE:
        return
    
    try:
        # Log hyperparameters
        log_params_flat(hyperparams)
        
        # Log validation scores as metrics
        log_metrics(validation_scores, step=trial_number)
        
        # Log checkpoint with hash
        log_artifact_with_hash(checkpoint_path, artifact_path="checkpoints")
        
        # Log validation images
        if validation_image_dir:
            validation_image_dir = Path(validation_image_dir)
            if validation_image_dir.exists():
                log_artifacts_dir(validation_image_dir, artifact_path="validation_paintings")
        
        logger.info(f"Logged HPO trial {trial_number} results")
    except Exception as e:
        logger.error(f"Failed to log HPO trial {trial_number}: {e}")


def get_best_run_from_study(
    experiment_name: str,
    metric: str = "avg_final_lpips_validation",
    ascending: bool = True
) -> Optional[Dict[str, Any]]:
    """Get best run from HPO study.
    
    Parameters
    ----------
    experiment_name : str
        Experiment name
    metric : str
        Metric to optimize, default 'avg_final_lpips_validation'
    ascending : bool
        True for minimize (lower is better), False for maximize
    
    Returns
    -------
    Optional[Dict[str, Any]]
        Best run info with keys: 'run_id', 'params', 'metrics', 'artifact_uri'
        None if no runs found
    
    Notes
    -----
    Queries MLflow tracking server for best run.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, cannot query best run")
        return None
    
    try:
        client = MlflowClient()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.warning(f"Experiment not found: {experiment_name}")
            return None
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if not runs:
            logger.warning(f"No runs found in experiment: {experiment_name}")
            return None
        
        best_run = runs[0]
        
        return {
            'run_id': best_run.info.run_id,
            'params': best_run.data.params,
            'metrics': best_run.data.metrics,
            'artifact_uri': best_run.info.artifact_uri,
        }
    except Exception as e:
        logger.error(f"Failed to get best run: {e}")
        return None


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================

def save_checkpoint_and_log(
    checkpoint: Dict[str, Any],
    checkpoint_path: Union[str, Path],
    artifact_path: str = "checkpoints"
) -> None:
    """Save checkpoint to disk and log to MLflow.
    
    Parameters
    ----------
    checkpoint : Dict[str, Any]
        Checkpoint dictionary (usually contains 'model_state_dict', 'optimizer_state_dict', etc.)
    checkpoint_path : Union[str, Path]
        Local path to save checkpoint
    artifact_path : str
        Relative path within artifact store, default 'checkpoints'
    
    Notes
    -----
    Saves checkpoint atomically, then logs with hash.
    """
    import torch
    
    checkpoint_path = Path(checkpoint_path)
    
    # Ensure parent directory exists
    fs.ensure_dir(checkpoint_path.parent)
    
    # Save checkpoint atomically (via temp file)
    tmp_path = checkpoint_path.with_suffix('.tmp' + checkpoint_path.suffix)
    try:
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    
    # Log to MLflow
    log_artifact_with_hash(checkpoint_path, artifact_path=artifact_path)


def load_checkpoint_from_run(
    run_id: str,
    checkpoint_name: str = "checkpoint.pth",
    download_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Download checkpoint from MLflow run.
    
    Parameters
    ----------
    run_id : str
        MLflow run ID
    checkpoint_name : str
        Checkpoint filename, default 'checkpoint.pth'
    download_dir : Optional[Union[str, Path]]
        Local download directory, None for temp directory
    
    Returns
    -------
    Path
        Path to downloaded checkpoint
    
    Raises
    ------
    FileNotFoundError
        If checkpoint not found in run
    """
    if not MLFLOW_AVAILABLE:
        raise RuntimeError("MLflow not available, cannot download checkpoint")
    
    try:
        client = MlflowClient()
        
        if download_dir is None:
            import tempfile
            download_dir = Path(tempfile.mkdtemp())
        else:
            download_dir = Path(download_dir)
            fs.ensure_dir(download_dir)
        
        # Download artifact
        artifact_path = f"checkpoints/{checkpoint_name}"
        local_path = client.download_artifacts(run_id, artifact_path, dst_path=str(download_dir))
        
        logger.info(f"Downloaded checkpoint from run {run_id} to {local_path}")
        return Path(local_path)
    except Exception as e:
        logger.error(f"Failed to download checkpoint from run {run_id}: {e}")
        raise FileNotFoundError(f"Checkpoint not found in run {run_id}: {e}") from e
