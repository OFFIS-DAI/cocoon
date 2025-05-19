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
        """
        Initialize the OMNeT++ connection

        Args:
            inet_installation_path: Path to INET framework
            config_name: Name of the configuration to run
            omnet_project_path: Path to the simulations directory inside the project
            ini_file_name: Name of the ini file (default: omnetpp.ini)
        """
        self.inet_installation_path = inet_installation_path
        self.config_name = config_name
        self.omnet_project_path = omnet_project_path
        self.ini_file_name = ini_file_name

        # Process management
        self.omnet_process = None
        self.running = False

    def initialize(self):
        """Initialize the connection and start OMNeT++ simulation"""
        try:
            # First build the project (from the project root directory)
            self.build_omnet_project()

            # Then start OMNeT++ simulation (from the simulations directory)
            self.omnet_process = self.start_omnet_simulation()
            self.running = True
        except Exception as e:
            logger.error(f'Error during OMNeT++ initialization: {e}')
            raise

    def build_omnet_project(self):
        """Build the OMNeT++ project from the root directory"""
        try:
            print("Building OMNeT++ project...")
            # Print current directory for debugging

            # Now build the project
            build_command = "make"
            build_process = subprocess.run(
                build_command,
                shell=True,
                cwd=self.omnet_project_path,  # Use the parent directory with Makefile
                check=True,  # Raise exception if build fails
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            print("OMNeT++ project built successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error building OMNeT++ project: {e}")
            print(f"Build stdout: {e.stdout.decode('utf-8')}")
            print(f"Build stderr: {e.stderr.decode('utf-8')}")
            raise Exception(f"Failed to build OMNeT++ project: {e}")
        except Exception as e:
            print(f"Unexpected error during build: {e}")
            raise

    def start_omnet_simulation(self):
        """Start OMNeT++ simulation process from the simulations directory"""
        # Build the command
        command = (f'./run '
                   f'-u Cmdenv '
                   f'-n {self.inet_installation_path} '
                   f'-f {self.ini_file_name} '
                   f'-c {self.config_name}')

        try:
            omnet_ini_path = self.omnet_project_path + 'simulations'

            print(f"Starting OMNeT++ simulation with command: {command}")
            print(f"Working directory: {omnet_ini_path}")

            omnet_process = subprocess.Popen(command,
                                             preexec_fn=os.setsid,
                                             shell=True,
                                             cwd=omnet_ini_path)

            # Wait a bit to see if process starts successfully
            time.sleep(2)
            if omnet_process.poll() is not None:
                # Process has already terminated
                stdout, stderr = omnet_process.communicate()
                error_msg = f"OMNeT++ process failed to start.\nStdout: {stdout.decode('utf-8') if stdout else None}\nStderr: {stderr.decode('utf-8') if stderr else None}"
                print(error_msg)
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
