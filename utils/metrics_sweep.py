from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.timesfm_metrics_from_results import get_model_config, process_run_dir


def write_threshold_sweeps(
    run_dir: str | Path,
    model: str,
    *,
    overwrite: bool = True,
    thresholds: Optional[str] = None,
) -> None:
    cfg = get_model_config(model, results_root=None, thresholds=thresholds)
    process_run_dir(Path(run_dir), cfg, overwrite=overwrite)
