"""Select the active inspection dataset here; no CLI argument is required."""

from .eval_dataset import evaluation_data_set
from .primitive_dataset import primitive_data_set, primitive_test_data_set


# Backward-compatible name for any code that imported this directly.
train_data_set = primitive_data_set

# Manually change only this assignment when switching experiments.
usd_data_set = primitive_test_data_set
# usd_data_set = evaluation_data_set
