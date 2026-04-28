"""Module for feature transformation in TFX pipeline."""

import tensorflow as tf
import tensorflow_transform as tft

NUMERIC_FEATURE_KEYS = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_FEATURE_KEYS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

LABEL_KEY = "target"


def transformed_name(key):
    """Add suffix to transformed feature name.

    Args:
        key: Feature name.

    Returns:
        Feature name with _xf suffix.
    """
    return key + "_xf"


def preprocessing_fn(inputs):
    """Preprocess features for model training.

    Args:
        inputs: Dictionary of input tensors.

    Returns:
        Dictionary of transformed tensors.
    """
    outputs = {}

    for key in NUMERIC_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_0_1(inputs[key])

    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[transformed_name(key)] = inputs[key]

    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        inputs[LABEL_KEY], tf.int64
    )

    return outputs
