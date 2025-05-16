import logging
import os
import subprocess
import time
from typing import Optional

from mango.container.external_coupling import ExternalAgentMessage

logger = logging.getLogger(__name__)


class OmnetConnection:
    def __init__(self,
                 inet_installation_path: str,
                 config_name: str,
                 omnet_project_path: str,
                 ini_file_name: str = 'omnetpp.ini'):
        self.inet_installation_path = inet_installation_path
        self.config_name = config_name
        self.omnet_project_path = omnet_project_path
        self.ini_file_name = ini_file_name

        # Process management
        self.omnet_process: Optional[subprocess.Popen] = None
        self.running = False

    def initialize(self):
        """Initialize the connection and start OMNeT++ simulation"""
        try:
            # Start OMNeT++ simulation
            self.omnet_process = self.start_omnet_simulation()
        except Exception as e:
            logger.error(f'Error when starting OMNeT++: {e}')

    def start_omnet_simulation(self):
        """Start OMNeT++ simulation process"""
        # Build the command
        command = (f'./run '
                   f'-u Cmdenv '
                   f'-n {self.inet_installation_path} '
                   f'-f {self.ini_file_name} '
                   f'-c {self.config_name}')

        try:
            omnet_process = subprocess.Popen(command,
                                             preexec_fn=os.setsid,
                                             shell=True,
                                             cwd=self.omnet_project_path)

            # Wait a bit to see if process starts successfully
            time.sleep(2)
            if omnet_process.poll() is not None:
                # Process has already terminated
                stdout, stderr = omnet_process.communicate()
                error_msg = f"OMNeT++ process failed to start.\nStdout: {stdout}\nStderr: {stderr}"
                raise Exception(error_msg)

            print(f"OMNeT++ simulation started with PID: {omnet_process.pid}")
            return omnet_process

        except Exception as e:
            print(f"Error starting OMNeT++ simulation: {e}")
            raise


class DetailedNetworkModel:
    def __init__(self,
                 inet_installation_path: str,
                 config_name: str,
                 omnet_project_path: str):
        self.omnet_connection = OmnetConnection(inet_installation_path=inet_installation_path,
                                                config_name=config_name,
                                                omnet_project_path=omnet_project_path)
        self.omnet_connection.initialize()

    def simulate_message_dispatch(self, sender_message_dict: dict[str, list[ExternalAgentMessage]]):
        pass
