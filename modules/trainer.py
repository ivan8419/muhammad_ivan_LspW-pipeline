"""Module for training the heart disease prediction model."""

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

NUMERIC_FEATURE_KEYS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURE_KEYS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]
LABEL_KEY = "target"


def transformed_name(key):
    """Add suffix to transformed feature name."""
    return key + "_xf"


def _input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=32):
    """Generate input function for training.

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


def model_builder(hp_config=None):
    """Build neural network model.

    Args:
        hp_config: Dictionary of hyperparameters from tuner.

    Returns:
        Compiled Keras model.
    """
    if hp_config is None:
        hp_config = {
            'num_layers': 2,
            'units_0': 64,
            'units_1': 32,
            'learning_rate': 1e-3
        }

    inputs = {}
    for key in NUMERIC_FEATURE_KEYS + CATEGORICAL_FEATURE_KEYS:
        inputs[transformed_name(key)] = tf.keras.Input(
            shape=(1,),
            name=transformed_name(key),
            dtype=tf.float32
        )

    x = tf.keras.layers.Concatenate()(list(inputs.values()))

    for i in range(hp_config['num_layers']):
        units = hp_config.get(f'units_{i}', 64)
        x = tf.keras.layers.Dense(units, activation='relu')(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp_config['learning_rate']
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Create serving function for model.

    Args:
        model: Keras model.
        tf_transform_output: Output from Transform component.

    Returns:
        Serving function for TF Serving.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples,
            feature_spec
        )
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    """Run function for TFX Trainer component.

    Args:
        fn_args: Arguments for trainer function.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_set = _input_fn(
        fn_args.train_files,
        tf_transform_output,
        num_epochs=10
    )
    val_set = _input_fn(
        fn_args.eval_files,
        tf_transform_output,
        num_epochs=10
    )

    hp_config = None
    if fn_args.hyperparameters:
        hp_config = fn_args.hyperparameters.get('values')

    model = model_builder(hp_config)

    model.fit(
        train_set,
        steps_per_epoch=fn_args.train_steps,
        validation_data=val_set,
        validation_steps=fn_args.eval_steps,
        epochs=10
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'
            )
        )
    }

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )
