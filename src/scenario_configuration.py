from enum import Enum


class SystemState(Enum):
    """
    Enumeration describing the system state used.
    """
    NORMAL = 0
    LIMITED = 1
    FAILED = 2


class CommunicationTechnology(Enum):
    """
    Enumeration describing the communication technology used.
    """
    Tech_LTE = 0
    Tech_LTE450 = 1
    Tech_5G = 2
    Tech_Wifi = 3
    Tech_Ethernet = 4


class ScenarioConfiguration:
    """
    Holds the configuration of the scenario.
    """
    def __init__(self,
                 system_state: SystemState,
                 communication_technology: CommunicationTechnology):
        self.system_state = system_state
        self.communication_technology = communication_technology

        self.scenario_identifier = self.system_state.name + '_' + self.communication_technology.name
