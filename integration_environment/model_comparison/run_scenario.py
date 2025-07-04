import asyncio
import logging

from integration_environment.model_comparison.execute_comparison import run_scenario_config
from integration_environment.scenario_configuration import ScenarioConfiguration

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


async def run_scenario_by_id(scenario_id: str):
    config = ScenarioConfiguration.from_scenario_id(scenario_id=scenario_id)
    await run_scenario_config(scenario_configuration=config, run=config.run)


if __name__ == "__main__":
    s_id = 'meta_model_training-two-medium-one_day-deer_use_case-simbench_lte450-none-none-none-none-none-0'
    asyncio.run(run_scenario_by_id(scenario_id=s_id))
