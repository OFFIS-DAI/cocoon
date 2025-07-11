import asyncio
import logging

from integration_environment.model_comparison.execute_comparison import run_scenario_config
from integration_environment.scenario_configuration import ScenarioConfiguration


async def run_scenario_by_id(scenario_id: str):
    config = ScenarioConfiguration.from_scenario_id(scenario_id=scenario_id)
    await run_scenario_config(scenario_configuration=config, run=config.run, phase=None)


if __name__ == "__main__":
    # meta_model-five-small-one_min-cbr_broadcast_1_mps-simbench_ethernet-half-ten-center-center-none-0
    s_id = ('meta_model-five-small-one_min-cbr_broadcast_1_mps-simbench_5g-three-ten-center-center-none-0')
    asyncio.run(run_scenario_by_id(scenario_id=s_id))
