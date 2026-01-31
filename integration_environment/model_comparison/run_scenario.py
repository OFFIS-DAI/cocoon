"""
Run a Single COCOON Scenario.

This script allows running individual simulation scenarios with configurable
parameters. It's useful for testing specific configurations or debugging.

Usage:
    python run_scenario.py

    Or with a scenario ID:
    python run_scenario.py --scenario-id "meta_model-five-small-one_min-cbr_broadcast_1_mps-..."

Environment Variables (required for detailed/meta-model simulations):
    INET_INSTALLATION_PATH: Path to INET framework src directory
    SIMU5G_INSTALLATION_PATH: Path to Simu5G src directory
    OMNET_PROJECT_PATH: Path to the cocoon_omnet_project directory

Example:
    export INET_INSTALLATION_PATH=/path/to/inet4.5/src
    export SIMU5G_INSTALLATION_PATH=/path/to/Simu5G/src
    export OMNET_PROJECT_PATH=/path/to/cocoon_omnet_project
    python run_scenario.py

Author: Malin Radtke (OFFIS)
License: MIT
"""

import argparse
import asyncio
import logging
import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from integration_environment.model_comparison.execute_comparison import run_scenario_config
from integration_environment.scenario_configuration import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_scenario_by_id(scenario_id: str) -> None:
    """
    Run a scenario using a scenario ID string.

    Args:
        scenario_id: Full scenario ID string (e.g., "meta_model-five-small-...")
    """
    config = ScenarioConfiguration.from_scenario_id(scenario_id=scenario_id)
    logger.info(f"Running scenario: {scenario_id}")
    await run_scenario_config(scenario_configuration=config, run=config.run, phase=None)


async def run_scenario_by_config(scenario_config: ScenarioConfiguration) -> None:
    """
    Run a scenario using a ScenarioConfiguration object.

    Args:
        scenario_config: Configuration object defining the scenario parameters
    """
    logger.info(f"Running scenario: {scenario_config.scenario_id}")
    await run_scenario_config(scenario_configuration=scenario_config, run=scenario_config.run, phase=None)


def get_default_config() -> ScenarioConfiguration:
    """
    Get a default scenario configuration for testing.

    Returns:
        ScenarioConfiguration with sensible defaults for testing
    """
    return ScenarioConfiguration(
        payload_size=PayloadSizeConfig.medium,
        num_devices=NumDevices.five,
        model_type=ModelType.meta_model,
        scenario_duration=ScenarioDuration.five_min,
        traffic_configuration=TrafficConfig.poisson_broadcast_1_mpm_1,
        network_type=NetworkModelType.evaluation_ethernet,
        # Meta-model specific parameters
        test_train_split=TestTrainSplit.none,
        amount_of_scenarios_in_training_data=AmountOfScenariosTrainingData.all,
        i_pupa=BatchSizeIPupa.hundred_fifty,
        cluster_distance_threshold=ClusterDistanceThreshold.three,
        learning_rate_weighting=LearningRateWeighting.small,
        butterfly_threshold_value=ButterflyThresholdValue.small,
        substitution_priority=SubstitutionPriority.error_level,
        run=0
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run a single COCOON simulation scenario.'
    )
    parser.add_argument(
        '--scenario-id',
        type=str,
        default=None,
        help='Full scenario ID string to run'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['ideal', 'channel', 'static_graph', 'detailed', 'meta_model'],
        default='meta_model',
        help='Type of communication model to use'
    )
    parser.add_argument(
        '--num-devices',
        type=int,
        choices=[5, 10, 20, 50, 100],
        default=5,
        help='Number of devices in the network'
    )
    parser.add_argument(
        '--duration',
        type=str,
        choices=['one_min', 'five_min', 'ten_min', 'thirty_min', 'one_hour'],
        default='five_min',
        help='Scenario duration'
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    if args.scenario_id:
        await run_scenario_by_id(scenario_id=args.scenario_id)
    else:
        # Use default config with any command line overrides
        config = get_default_config()

        # Apply command line overrides
        if args.model_type:
            config.model_type = ModelType[args.model_type]
        if args.num_devices:
            for nd in NumDevices:
                if nd.value == args.num_devices:
                    config.num_devices = nd
                    break
        if args.duration:
            config.scenario_duration = ScenarioDuration[args.duration]

        await run_scenario_by_config(scenario_config=config)


if __name__ == "__main__":
    asyncio.run(main())
