"""
data-tool: Data Preprocessing Toolkit

Provides:
- handle_missing_values: Handle missing data
- remove_duplicates: Remove duplicate rows
- clip_outliers: Clip extreme values
- one_hot_encode: One-hot encoding for categorical features
- label_encode: Label encoding for categorical features
- minmax_scale: Min-max scaling
- standard_scale: Standardization (z-score)
- robust_scale: Robust scaling
"""

from . import cleaning
from . import encoding
from . import scaling

from .cleaning import (
    handle_missing_values,
    remove_duplicates,
    clip_outliers
)
from .encoding import (
    one_hot_encode,
    label_encode
)
from .scaling import (
    minmax_scale,
    standard_scale,
    robust_scale
)

__version__ = "0.1.0"
__all__ = [
    'handle_missing_values',
    'remove_duplicates',
    'clip_outliers',
    'one_hot_encode',
    'label_encode',
    'minmax_scale',
    'standard_scale',
    'robust_scale'
]
