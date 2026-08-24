"""Select inspection targets from the active run configuration."""

from .eval_dataset import evaluation_data_set
from .primitive_dataset import primitive_data_set, primitive_test_data_set
from ..run_config import cfg_mode


# Backward-compatible name for any code that imported this directly.
train_data_set = primitive_data_set


def _select_data_set() -> dict:
    """Return the dataset (and optional single target) requested by cfg_mode."""
    data_set_name = getattr(cfg_mode, "inspection_dataset", "primitive")
    available_data_sets = {
        "primitive": primitive_test_data_set,
        "evaluation": evaluation_data_set,
    }
    if data_set_name not in available_data_sets:
        valid_names = ", ".join(sorted(available_data_sets))
        raise ValueError(
            f"Unknown inspection_dataset {data_set_name!r}. Valid datasets: {valid_names}"
        )

    selected_data_set = available_data_sets[data_set_name]
    target_name = getattr(cfg_mode, "inspection_target", None)
    if target_name is None:
        return selected_data_set
    if target_name not in selected_data_set:
        valid_targets = ", ".join(sorted(selected_data_set))
        raise ValueError(
            f"Inspection target {target_name!r} is not in the {data_set_name!r} dataset. "
            f"Valid targets: {valid_targets}"
        )
    return {target_name: selected_data_set[target_name]}


usd_data_set = _select_data_set()
