import os

from absl import logging
import tensorflow as tf

from tfx import components
from tfx.proto import trainer_pb2
from tfx.proto import pusher_pb2
from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Trainer
from tfx.components import Tuner
from tfx.components import Evaluator
from tfx.components import Pusher
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.types.standard_artifacts import HyperParameters

import tensorflow_model_analysis as tfma

PIPELINE_NAME = "muhammad_ivan_LspW-pipeline"

# Use absolute paths to avoid issues on Windows/Docker
abspath = os.path.abspath(os.path.dirname(__file__))

DATA_ROOT = os.path.join(abspath, "data")
TRANSFORM_MODULE_FILE = os.path.join(abspath, "modules", "transform.py")
TRAINER_MODULE_FILE = os.path.join(abspath, "modules", "trainer.py")
TUNER_MODULE_FILE = os.path.join(abspath, "modules", "tuner.py")

SERVING_MODEL_DIR = "C:\\tfx_run\\serving_model\\" + PIPELINE_NAME
PIPELINE_ROOT = "C:\\tfx_run\\tfx_artifacts"
METADATA_PATH = "C:\\tfx_run\\metadata\\metadata.db"

TRAIN_STEPS = 100
EVAL_STEPS = 50

BEAM_PIPELINE_ARGS = [
    '--direct_num_workers=1',
]


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    transform_module_file: str,
    trainer_module_file: str,
    tuner_module_file: str,
    serving_model_dir: str,
    metadata_path: str,
    train_steps: int,
    eval_steps: int,
    beam_pipeline_args: list
):
    """Create TFX pipeline with all required components.

    Args:
        pipeline_name: Name of the pipeline.
        pipeline_root: Root directory for pipeline output.
        data_root: Root directory containing input data.
        transform_module_file: Path to transform module.
        trainer_module_file: Path to trainer module.
        tuner_module_file: Path to tuner module.
        serving_model_dir: Directory to save serving model.
        metadata_path: Path to metadata database.
        train_steps: Number of training steps.
        eval_steps: Number of evaluation steps.
        beam_pipeline_args: Arguments for Beam pipeline.

    Returns:
        TFX Pipeline object.
    """
    example_gen = CsvExampleGen(input_base=data_root)

    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples']
    )

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics']
    )

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module_file
    )

    tuner = Tuner(
        module_file=tuner_module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps)
    )

    trainer = Trainer(
        module_file=trainer_module_file,
        transformed_examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps)
    )

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')

    slicing_specs = [
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=['sex']),
    ]

    metrics_specs = [
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}
                        )
                    )
                ),
                tfma.MetricConfig(class_name='ExampleCount'),
            ]
        )
    ]

    eval_config = tfma.EvalConfig(
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            tuner,
            trainer,
            model_resolver,
            evaluator,
            pusher
        ],
        metadata_connection_config=pipeline.metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        enable_cache=False,
        beam_pipeline_args=beam_pipeline_args
    )


def run_pipeline():
    logging.set_verbosity(logging.INFO)

    tfx_pipeline = create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_root=DATA_ROOT,
        transform_module_file=TRANSFORM_MODULE_FILE,
        trainer_module_file=TRAINER_MODULE_FILE,
        tuner_module_file=TUNER_MODULE_FILE,
        serving_model_dir=SERVING_MODEL_DIR,
        metadata_path=METADATA_PATH,
        train_steps=TRAIN_STEPS,
        eval_steps=EVAL_STEPS,
        beam_pipeline_args=BEAM_PIPELINE_ARGS
    )

    BeamDagRunner().run(pipeline=tfx_pipeline)


if __name__ == '__main__':
    run_pipeline()