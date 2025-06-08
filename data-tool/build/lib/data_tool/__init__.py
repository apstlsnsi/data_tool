# data_tool/__init__.py

# Явно импортируем модули
from . import cleaning
from . import encoding
from . import scaling

# Импортируем функции для прямого доступа
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
