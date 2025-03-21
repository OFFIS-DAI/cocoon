# Extract the System State and the Communication technology
from enum import Enum

import pandas as pd
import matplotlib.pyplot as plt

from src.scenario_configuration import SystemState, CommunicationTechnology, ScenarioConfiguration


class LoggingOption(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2


def get_scenario_configuration(df: pd.DataFrame) -> ScenarioConfiguration:
    try:
        system_state = SystemState[
            df.loc[df['Parameter'] == 'System State', 'Value'].values[0].split('SystemState.')[1]]
    except IndexError:
        system_state = SystemState.NORMAL
    network_info = df.loc[df['Parameter'] == 'Network', 'Value'].values[0]

    # Extract the communication technology from the network information
    if 'LTE450' in network_info:
        communication_technology = CommunicationTechnology.Tech_LTE450
    elif 'LTE' in network_info:
        communication_technology = CommunicationTechnology.Tech_LTE
    elif '5G' in network_info:
        communication_technology = CommunicationTechnology.Tech_5G
    elif 'Wifi' in network_info:
        communication_technology = CommunicationTechnology.Tech_Wifi
    elif 'Ethernet' in network_info:
        communication_technology = CommunicationTechnology.Tech_Ethernet
    else:
        raise ValueError('No valid communication technology provided in ', network_info)

    return ScenarioConfiguration(system_state=system_state,
                                 communication_technology=communication_technology)
