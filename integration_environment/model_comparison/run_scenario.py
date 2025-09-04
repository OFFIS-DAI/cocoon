import asyncio

from integration_environment.model_comparison.execute_comparison import run_scenario_config
from integration_environment.scenario_configuration import *


async def run_scenario_by_id(scenario_id: str):
    config = ScenarioConfiguration.from_scenario_id(scenario_id=scenario_id)
    await run_scenario_config(scenario_configuration=config, run=config.run, phase=None)


async def run_scenario_by_config(scenario_config: ScenarioConfiguration):
    await run_scenario_config(scenario_configuration=scenario_config, run=scenario_config.run, phase=None)


if __name__ == "__main__":
    # meta_model-five-small-one_min-cbr_broadcast_1_mps-simbench_ethernet-half-ten-center-center-none-0
    # s_id = ('static_graph-five-small-one_min-poisson_1_mps_1-simbench_5g-none-three-ten-center-center-none-0')
    # asyncio.run(run_scenario_by_id(scenario_id=s_id))

    # meta_model_training-fifty-medium-thirty_min-central_dsb_1mpm_5s_50p-simbench_ethernet-none-none-none-none-none-none-0.csv

    config = ScenarioConfiguration(payload_size=PayloadSizeConfig.medium,
                                   num_devices=NumDevices.five,
                                   model_type=ModelType.detailed,
                                   scenario_duration=ScenarioDuration.one_min,
                                   traffic_configuration=TrafficConfig.central_dsb_5mpm_30s_75,
                                   network_type=NetworkModelType.simbench_5g,
                                   # specific for meta-model
                                   test_train_split=TestTrainSplit.parametrization_split,
                                   i_pupa=BatchSizeIPupa.hundred_fifty,
                                   cluster_distance_threshold=ClusterDistanceThreshold.three,
                                   learning_rate_weighting=LearningRateWeighting.small,
                                   butterfly_threshold_value=ButterflyThresholdValue.small,
                                   substitution_priority=SubstitutionPriority.error_level,
                                   run=0)
    asyncio.run(run_scenario_by_config(scenario_config=config))
