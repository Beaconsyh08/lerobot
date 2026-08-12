from pathlib import Path

try:
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401

    _AVAILABLE = True
    _ERROR = None
except Exception as exc:  # pragma: no cover - depends on optional runtime env
    _AVAILABLE = False
    _ERROR = str(exc)

from lerobot.data_platform.precompute.preprocess.action_dim import run_convert_action
from lerobot.data_platform.precompute.preprocess.common import PreprocessResult, ProgressCallback
from lerobot.data_platform.precompute.preprocess.dataset_merge import run_merge
from lerobot.data_platform.precompute.preprocess.dataset_split import run_split
from lerobot.data_platform.precompute.preprocess.dataset_subtract import run_subtract
from lerobot.data_platform.precompute.preprocess.dataset_version import run_convert_v3
from lerobot.data_platform.precompute.preprocess.field_ops import run_drop_field
from lerobot.data_platform.precompute.preprocess.flag_clear import clear_all_flags
from lerobot.data_platform.precompute.preprocess.flag_fixes import run_flag_fix
from lerobot.data_platform.precompute.preprocess.quality_flags import run_quality_flag_detection
from lerobot.data_platform.precompute.preprocess.smooth_action import run_smooth_action
from lerobot.data_platform.precompute.preprocess.standardize import run_standardize_dataset


def get_capabilities() -> dict:
    return {
        "available": _AVAILABLE,
        "error": _ERROR,
        "ops": [
            {
                "name": "convert_action",
                "available": _AVAILABLE,
                "description": "Trim action/state vectors to a target dimension.",
            },
            {
                "name": "convert_v3",
                "available": _AVAILABLE,
                "description": "Convert v2.1 datasets to the LeRobot 0.4.4 v3.0 layout; report v3.0 inputs as already converted.",
            },
            {
                "name": "drop_field",
                "available": _AVAILABLE,
                "description": "Remove one parquet field and metadata feature.",
            },
            {
                "name": "smooth_action",
                "available": _AVAILABLE,
                "description": "Smooth action vectors with a centered moving average.",
            },
            {
                "name": "standardize",
                "available": _AVAILABLE,
                "description": "Normalize DVT data to 16D action/state with stage/subtask writeback.",
            },
            {
                "name": "quality_flags",
                "available": _AVAILABLE,
                "description": "Re-scan parquet action/state for abnormal episode flags.",
            },
            {
                "name": "clear_flags",
                "available": _AVAILABLE,
                "description": "Clear manual and automatic viewer flags.",
            },
            {
                "name": "flag_fixes",
                "available": _AVAILABLE,
                "description": "Apply batch fixes for quality-flagged episodes.",
            },
            {"name": "split", "available": _AVAILABLE, "description": "Create a re-indexed subset dataset."},
            {
                "name": "merge",
                "available": _AVAILABLE,
                "description": "Merge two or more compatible LeRobot datasets.",
            },
            {
                "name": "subtract",
                "available": _AVAILABLE,
                "description": "Subtract matching episodes from a base dataset.",
            },
        ],
    }


def run_preprocess_op(
    op: str,
    src_root: Path | None = None,
    out_root: Path | None = None,
    progress_callback: ProgressCallback = None,
    **kwargs,
) -> PreprocessResult:
    if not _AVAILABLE:
        raise RuntimeError(f"Preprocess dependencies are unavailable: {_ERROR}")
    if op == "convert_action":
        return run_convert_action(src_root, out_root=out_root, progress_callback=progress_callback, **kwargs)
    if op == "convert_v3":
        return run_convert_v3(src_root, out_root=out_root, progress_callback=progress_callback, **kwargs)
    if op == "drop_field":
        return run_drop_field(src_root, out_root=out_root, progress_callback=progress_callback, **kwargs)
    if op == "smooth_action":
        return run_smooth_action(src_root, out_root=out_root, progress_callback=progress_callback, **kwargs)
    if op == "standardize":
        return run_standardize_dataset(
            src_root, out_root=out_root, progress_callback=progress_callback, **kwargs
        )
    if op == "quality_flags":
        return run_quality_flag_detection(src_root, progress_callback=progress_callback, **kwargs)
    if op == "clear_flags":
        return clear_all_flags(src_root, progress_callback=progress_callback, **kwargs)
    if op == "flag_fixes":
        return run_flag_fix(src_root, progress_callback=progress_callback, **kwargs)
    if op == "split":
        return run_split(src_root, out_root=out_root, progress_callback=progress_callback, **kwargs)
    if op == "merge":
        return run_merge(out_root=out_root, progress_callback=progress_callback, **kwargs)
    if op == "subtract":
        return run_subtract(src_root, out_root=out_root, progress_callback=progress_callback, **kwargs)
    raise ValueError(f"Unknown preprocess op: {op}")
