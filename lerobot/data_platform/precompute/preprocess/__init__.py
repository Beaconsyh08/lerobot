from lerobot.data_platform.precompute.preprocess.action_dim import run_convert_action
from lerobot.data_platform.precompute.preprocess.common import (
    PreprocessResult,
    default_preprocess_path,
)
from lerobot.data_platform.precompute.preprocess.dataset_merge import run_merge
from lerobot.data_platform.precompute.preprocess.dataset_split import parse_episode_range, run_split
from lerobot.data_platform.precompute.preprocess.dataset_subtract import run_subtract
from lerobot.data_platform.precompute.preprocess.dataset_version import (
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_V3_CONVERT_WORKERS,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    IMAGE_VIDEO_MODE_LEROBOT_OFFICIAL,
    IMAGE_VIDEO_MODE_RGB_LOSSLESS,
    IMAGE_VIDEO_MODES,
    V3_BASELINE,
    default_v3_path,
    detect_dataset_version,
    normalize_image_video_mode,
    repair_v3_video_timestamps,
    run_convert_v3,
    validate_v3_dataset,
)
from lerobot.data_platform.precompute.preprocess.delete_episodes import delete_episodes_inplace
from lerobot.data_platform.precompute.preprocess.field_ops import run_drop_field
from lerobot.data_platform.precompute.preprocess.flag_clear import clear_all_flags
from lerobot.data_platform.precompute.preprocess.flag_fixes import (
    FLAG_FIX_DELETE_ALL_FLAGGED,
    FLAG_FIX_STATE_GRIPPER_TRANSITION_ACTION,
    FLAG_FIX_STUCK_CLOSED_ACTION,
    FLAG_FIX_TRIM_EARLY_GRIPPER,
    load_flagged_episode_ids,
    run_flag_fix,
    trim_v3_episode_inplace,
)
from lerobot.data_platform.precompute.preprocess.prompt_rewrite import (
    DEFAULT_PROMPT_PATTERN,
    DEFAULT_PROMPT_REPLACEMENT,
    fix_prompt_prepositions_text,
    lowercase_prompt_text,
    run_fix_prompt_prepositions,
    run_lowercase_prompts,
    run_rewrite_prompts,
)
from lerobot.data_platform.precompute.preprocess.quality_flags import run_quality_flag_detection
from lerobot.data_platform.precompute.preprocess.runner import get_capabilities, run_preprocess_op
from lerobot.data_platform.precompute.preprocess.smooth_action import run_smooth_action
from lerobot.data_platform.precompute.preprocess.stage_return_alignment import run_stage_return_alignment
from lerobot.data_platform.precompute.preprocess.stage_return_height_alignment import (
    run_stage_return_height_alignment,
)
from lerobot.data_platform.precompute.preprocess.standardize import (
    default_standardize_path,
    run_standardize_dataset,
)

__all__ = [
    "PreprocessResult",
    "default_preprocess_path",
    "default_standardize_path",
    "delete_episodes_inplace",
    "clear_all_flags",
    "FLAG_FIX_DELETE_ALL_FLAGGED",
    "FLAG_FIX_STATE_GRIPPER_TRANSITION_ACTION",
    "FLAG_FIX_STUCK_CLOSED_ACTION",
    "FLAG_FIX_TRIM_EARLY_GRIPPER",
    "get_capabilities",
    "load_flagged_episode_ids",
    "parse_episode_range",
    "DEFAULT_PROMPT_PATTERN",
    "DEFAULT_PROMPT_REPLACEMENT",
    "DEFAULT_DATA_FILE_SIZE_IN_MB",
    "DEFAULT_V3_CONVERT_WORKERS",
    "DEFAULT_VIDEO_FILE_SIZE_IN_MB",
    "IMAGE_VIDEO_MODE_LEROBOT_OFFICIAL",
    "IMAGE_VIDEO_MODE_RGB_LOSSLESS",
    "IMAGE_VIDEO_MODES",
    "default_v3_path",
    "fix_prompt_prepositions_text",
    "lowercase_prompt_text",
    "run_convert_action",
    "run_convert_v3",
    "run_drop_field",
    "run_flag_fix",
    "trim_v3_episode_inplace",
    "run_fix_prompt_prepositions",
    "run_lowercase_prompts",
    "run_merge",
    "run_preprocess_op",
    "run_quality_flag_detection",
    "run_rewrite_prompts",
    "run_smooth_action",
    "run_stage_return_alignment",
    "run_stage_return_height_alignment",
    "run_standardize_dataset",
    "run_split",
    "run_subtract",
    "detect_dataset_version",
    "normalize_image_video_mode",
    "repair_v3_video_timestamps",
    "validate_v3_dataset",
    "V3_BASELINE",
]
