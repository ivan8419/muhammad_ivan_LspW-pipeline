"""Module for hyperparameter tuning using Keras Tuner."""

from typing import Any, Dict, NamedTuple, Text

import tensorflow as tf
import tensorflow_transform as tft
from keras_tuner.engine import base_tuner
from keras_tuner.tuners import RandomSearch
from tfx.components.trainer.fn_args_utils import FnArgs

NUMERIC_FEATURE_KEYS = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]
CATEGORICAL_FEATURE_KEYS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]
LABEL_KEY = "target"


def transformed_name(key):
    """Add suffix to transformed feature name."""
    return key + "_xf"


TunerFnResult = NamedTuple(
    'TunerFnResult', [
        ('tuner', base_tuner.BaseTuner),
        ('fit_kwargs', Dict[Text, Any]),
    ]
)


def _input_fn(
    file_pattern,
    tf_transform_output,
    num_epochs=None,
    batch_size=32
):
    """Generate input function for tuning.

    Args:
        file_pattern: File pattern for TFRecord files.
        tf_transform_output: Output from Transform component.
        num_epochs: Number of epochs (None for infinite).
        batch_size: Batch size for training.

    Returns:
        TF Dataset for training.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))

    return dataset


def model_builder(hp):
    """Build model with hyperparameters from tuner.

    Args:
        hp: Hyperparameters object from Keras Tuner.

    Returns:
        Compiled Keras model.
    """
    num_layers = hp.Int(
        'num_layers',
        min_value=1,
        max_value=3,
        default=2
    )
    learning_rate = hp.Choice(
        'learning_rate',
        values=[1e-2, 1e-3, 1e-4]
    )

    inputs = {}
    for key in NUMERIC_FEATURE_KEYS + CATEGORICAL_FEATURE_KEYS:
        inputs[transformed_name(key)] = tf.keras.Input(
            shape=(1,),
            name=transformed_name(key),
            dtype=tf.float32
        )

    x = tf.keras.layers.Concatenate()(list(inputs.values()))

    for i in range(num_layers):
        units = hp.Int(
            f'units_{i}',
            min_value=32,
            max_value=256,
            step=32
        )
        x = tf.keras.layers.Dense(units, activation='relu')(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Tuner function for TFX Tuner component.

    Args:
        fn_args: Arguments for tuner function.

    Returns:
        TunerFnResult with tuner and fit kwargs.
    """
    tf_transform_output = tft.TFTransformOutput(
        fn_args.transform_graph_path
    )

    train_set = _input_fn(
        fn_args.train_files[0],
        tf_transform_output
    )
    val_set = _input_fn(
        fn_args.eval_files[0],
        tf_transform_output
    )

    tuner = RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='heart_disease_tuning'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
