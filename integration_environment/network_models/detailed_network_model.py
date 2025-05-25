import json
import signal
import socket
import threading
import time
import queue
import logging
import os
import subprocess
from typing import Optional, List, Dict, Any

from mango.container.external_coupling import ExternalAgentMessage

logger = logging.getLogger(__name__)


class OmnetConnection:
    def __init__(self,
                 inet_installation_path: str,
                 config_name: str,
                 omnet_project_path: str,
                 ini_file_name: str = 'omnetpp.ini',
                 socket_host: str = "127.0.0.1",
                 socket_port: int = 8345,
                 socket_timeout: int = 30):
        """
        Initialize the OMNeT++ connection

        Args:
            inet_installation_path: Path to INET framework
            config_name: Name of the configuration to run
            omnet_project_path: Path to the simulations directory inside the project
            ini_file_name: Name of the ini file (default: omnetpp.ini)
            socket_host: Host address for TCP socket connection (default: 127.0.0.1)
            socket_port: Port number for TCP socket connection (default: 8345)
            socket_timeout: Socket connection timeout in seconds (default: 30)
        """
        self.inet_installation_path = inet_installation_path
        self.config_name = config_name
        self.omnet_project_path = omnet_project_path
        self.ini_file_name = ini_file_name

        # Process management
        self.omnet_process = None
        self.running = False

        # Socket communication
        self.socket_host = socket_host
        self.socket_port = socket_port
        self.socket_timeout = socket_timeout
        self.socket = None
        self.listener_thread = None
        self.socket_running = False
        self.message_queue = queue.Queue()

        # Remember sent/received message ids
        self.message_ids_sent = []
        self.message_ids_received = []

        # Flag to track if termination was acknowledged
        self.termination_acknowledged = False

    def initialize(self):
        """Initialize the connection and start OMNeT++ simulation"""
        try:
            # First build the project (from the project root directory)
            self.build_omnet_project()

            # Then start OMNeT++ simulation (from the simulations directory)
            self.omnet_process = self.start_omnet_simulation()
            self.running = True

            # Connect to the OMNeT++ simulation via socket
            if not self.connect_socket():
                raise Exception("Failed to establish socket connection with OMNeT++")

        except Exception as e:
            logger.error(f'Error during OMNeT++ initialization: {e}')
            self.cleanup()
            raise

    def build_omnet_project(self):
        """Build the OMNeT++ project from the root directory"""
        try:
            print("Building OMNeT++ project...")

            # Now run the normal build
            build_command = "make MODE=release all"
            build_process = subprocess.run(
                build_command,
                shell=True,
                cwd=self.omnet_project_path,
                check=True,
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
        command = (f'./cocoon_omnet_project -m '
                   f'-u Cmdenv '
                   f'-n {self.inet_installation_path} '
                   f'-f {self.ini_file_name} '
                   f'-c {self.config_name} ')

        try:
            omnet_ini_path = self.omnet_project_path

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
                error_msg = (f"OMNeT++ process failed to start.\nStdout: {stdout.decode('utf-8') if stdout else None}"
                             f"\nStderr: {stderr.decode('utf-8') if stderr else None}")
                print(error_msg)
                raise Exception(error_msg)

            print(f"OMNeT++ simulation started with PID: {omnet_process.pid}")

            # Give the simulator time to initialize and start listening for connections
            time.sleep(5)

            return omnet_process

        except Exception as e:
            print(f"Error starting OMNeT++ simulation: {e}")
            raise

    def connect_socket(self) -> bool:
        """
        Connect to the OMNeT++ simulator via TCP socket

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.socket_timeout)

            logger.info(f"Connecting to OMNeT++ simulator at {self.socket_host}:{self.socket_port}")
            self.socket.connect((self.socket_host, self.socket_port))
            logger.info("Connected to OMNeT++ simulator")

            # Start listener thread
            self.socket_running = True
            self.listener_thread = threading.Thread(target=self._listen_for_messages)
            self.listener_thread.daemon = True
            self.listener_thread.start()

            # Wait for initialization message
            init_msg = self.receive_message(timeout=self.socket_timeout)
            if init_msg != "INIT":
                logger.error(f"Expected INIT message, received: {init_msg}")
                self.disconnect_socket()
                return False

            return True
        except socket.timeout:
            logger.error(f"Connection timed out after {self.socket_timeout} seconds")
            return False
        except ConnectionRefusedError:
            logger.error(
                f"Connection refused. Is the OMNeT++ simulator ready and listening on {self.socket_host}:{self.socket_port}?")
            return False
        except Exception as e:
            logger.error(f"Error connecting to OMNeT++ simulator: {e}")
            return False

    def disconnect_socket(self) -> None:
        """
        Disconnect from the OMNeT++ simulator socket
        """
        self.socket_running = False

        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
            finally:
                self.socket = None

        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=2.0)

        logger.info("Disconnected from OMNeT++ simulator socket")

    def _listen_for_messages(self) -> None:
        """
        Background thread to listen for incoming messages from OMNeT++
        """
        if not self.socket:
            logger.error("Cannot listen for messages: Socket not connected")
            return

        self.socket.settimeout(0.1)
        message_buffer = ""

        while self.socket_running:
            try:
                buffer = bytearray(4096)
                bytes_read = self.socket.recv_into(buffer)

                if bytes_read > 0:
                    message_buffer += buffer[:bytes_read].decode('utf-8')

                    # Split by newlines to get complete messages
                    lines = message_buffer.split('\n')

                    # Process all complete messages (all but the last if it's incomplete)
                    for i in range(len(lines) - 1):
                        message = lines[i].strip()
                        if message:  # Skip empty lines
                            logger.debug(f"Received message: {message}")

                            # Handle special messages
                            if message == "TERM":
                                logger.info("Received termination message from OMNeT++")
                                self.socket_running = False
                                break
                            elif message.startswith("TERM_ACK"):
                                logger.info("Received termination acknowledgment from OMNeT++")
                                self.termination_acknowledged = True

                            # Add to queue
                            self.message_queue.put(message)

                    # Keep the last incomplete line in the buffer
                    message_buffer = lines[-1]

                elif bytes_read == 0:
                    logger.info("OMNeT++ simulator closed the connection")
                    self.socket_running = False
                    break
            except socket.timeout:
                pass
            except Exception as e:
                if self.socket_running:
                    logger.error(f"Error receiving message: {e}")
                    self.socket_running = False
                    break

            time.sleep(0.01)

    def send_message(self, message: str) -> bool:
        if not self.socket:
            logger.error("Cannot send message: Socket not connected")
            return False

        try:
            self.socket.sendall(message.encode('utf-8'))
            logger.debug(f"Sent message: {message}")
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    def receive_message(self, timeout: Optional[float] = None) -> Optional[str]:
        try:
            return self.message_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def send_message_to_omnet(self, payload: Dict, msg_ids: List[str]) -> bool:
        message = f"MESSAGE|{json.dumps(payload)}"
        success = self.send_message(message)

        if not success:
            logger.error(f"Failed to send dispatch message: {message}")
            return False
        else:
            self.message_ids_sent.extend(msg_ids)
            return True

    def send_waiting_message_to_omnet(self, max_advance_ms) -> bool:
        message = f"WAITING|{json.dumps({'max_advance': max_advance_ms})}"
        success = self.send_message(message)

        if not success:
            logger.error(f"Failed to send dispatch message: {message}")
            return False
        else:
            return True

    def send_termination_signal(self) -> bool:
        # Reset flag
        self.termination_acknowledged = False

        # Send signal
        success = self.send_message("TERMINATE|")

        # Poll for acknowledgment with timeout
        max_wait = 10  # seconds
        start_time = time.time()

        # First check any pending messages
        for msg in self.get_all_messages():
            if msg.startswith("TERM_ACK"):
                self.termination_acknowledged = True
                return True

        # Then poll for new messages
        while not self.termination_acknowledged and (time.time() - start_time) < max_wait:
            time.sleep(0.5)
            for msg in self.get_all_messages():
                if msg.startswith("TERM_ACK"):
                    self.termination_acknowledged = True
                    return True

        # Fallback: force termination if needed
        if not self.termination_acknowledged:
            try:
                os.killpg(os.getpgid(self.omnet_process.pid), signal.SIGTERM)
                return True  # We did terminate it one way or another
            except Exception as e:
                logger.error(f"Error forcibly terminating OMNeT++: {e}")

        return self.termination_acknowledged

    def has_messages(self) -> bool:
        """
        Check if there are any messages in the queue

        Returns:
            True if there are messages, False otherwise
        """
        return not self.message_queue.empty()

    def get_all_messages(self) -> List[str]:
        """
        Get all available messages from the queue

        Returns:
            List of message strings
        """
        messages = []
        while self.has_messages():
            messages.append(self.receive_message(timeout=0.01))
        return messages

    def cleanup(self):
        """Clean up resources when shutting down"""
        # First send termination signal if we're still connected
        if self.socket and self.socket_running:
            logger.info("Sending termination signal before cleanup")
            self.send_termination_signal()
            # Give a short delay for the simulator to process the termination
            time.sleep(1)

        # Then disconnect the socket
        self.disconnect_socket()

        # Then terminate the OMNeT++ process if it's running
        if self.omnet_process and self.running:
            try:
                # Send SIGTERM to the process group to kill all child processes
                os.killpg(os.getpgid(self.omnet_process.pid), subprocess.signal.SIGTERM)

                # Wait for process to terminate
                self.omnet_process.wait(timeout=5)
            except:
                # If it doesn't terminate, try SIGKILL
                try:
                    os.killpg(os.getpgid(self.omnet_process.pid), subprocess.signal.SIGKILL)
                except:
                    pass

            self.running = False
            self.omnet_process = None

    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


class DetailedNetworkModel:
    def __init__(self,
                 inet_installation_path: str,
                 config_name: str,
                 omnet_project_path: str):
        self.omnet_connection = OmnetConnection(inet_installation_path=inet_installation_path,
                                                config_name=config_name,
                                                omnet_project_path=omnet_project_path)
        self.msg_id_to_msg = {}
        self.msg_id_counter = 0

        self.waiting_for_omnet = False

        self.omnet_connection.initialize()

    def terminate_simulation(self):
        """
        Send termination signal to OMNeT++ simulation
        """
        if self.omnet_connection:
            return self.omnet_connection.send_termination_signal()
        return False

    async def simulate_message_dispatch(self, sender_message_dict: dict[str, list[ExternalAgentMessage]],
                                        max_advance_ms: int):
        message_list = []
        msg_ids = []
        for sender, messages in sender_message_dict.items():
            for message in messages:
                message_list.append({
                    'sender': sender,
                    'receiver': message.receiver,
                    'size_B': len(message.message),
                    'time_send_ms': round(message.time * 1000),  # convert to ms
                    'msg_id': f'msg_{self.msg_id_counter}'
                })
                self.msg_id_to_msg[self.msg_id_counter] = message
                msg_ids.append(f'msg_{self.msg_id_counter}')
                self.msg_id_counter += 1
        payload = {
            "messages": message_list,
            "max_advance": max_advance_ms
        }
        self.omnet_connection.send_message_to_omnet(payload=payload, msg_ids=msg_ids)

    def waiting_for_messages_from_omnet(self) -> bool:
        messages_sent_but_not_received = [m for m in self.omnet_connection.message_ids_sent
                                          if m not in self.omnet_connection.message_ids_received]
        return len(messages_sent_but_not_received) != 0

    async def get_received_messages_from_omnet_connection(self) -> Dict[int, List[ExternalAgentMessage]]:
        all_messages = self.omnet_connection.get_all_messages()
        time_receive_to_message = {}

        for message in all_messages:
            try:
                # Skip non-data messages (INIT, TERM, etc.)
                if '|' not in message:
                    if 'WAITING' in message:
                        self.waiting_for_omnet = False
                    continue

                # Split message type and payload
                msg_type, payload = message.split('|', 1)

                # Process different message types
                if msg_type == 'SCHEDULED':
                    # These are acknowledgment messages, not delivered messages
                    logger.debug(f'Scheduled messages: {payload}.')

                elif msg_type == 'RECEIVED':
                    # This would be the actual received message from OMNeT++
                    # Parse the delivered message data
                    import json
                    data = json.loads(payload)

                    delivery_time = data.get('time_received', 0) / 1000  # Convert to seconds
                    msg_id = data.get('msg_id')

                    # Track that this message was received
                    if msg_id:
                        if len(msg_id.split('_')) == 2:
                            # Get ExternalAgentMessage
                            external_msg = self.msg_id_to_msg[int(msg_id.split('_')[1])]
                            if delivery_time not in time_receive_to_message.keys():
                                time_receive_to_message[delivery_time] = []
                            time_receive_to_message[delivery_time].append(external_msg)

                        self.omnet_connection.message_ids_received.append(msg_id)

            except Exception as e:
                logger.error(f"Error processing message from OMNeT++: {e}")
                logger.debug(f"Problematic message: {message}")
                continue

        return time_receive_to_message

    async def handle_waiting_with_omnet(self, max_advance_ms):
        logger.info('Handle waiting')
        self.waiting_for_omnet = True
        self.omnet_connection.send_waiting_message_to_omnet(max_advance_ms=max_advance_ms)
        time_receive_to_message = {}
        while self.waiting_for_omnet:
            time_receive_to_message_new = await self.get_received_messages_from_omnet_connection()
            for delivery_time, messages in time_receive_to_message_new.items():
                if delivery_time not in time_receive_to_message:
                    time_receive_to_message[delivery_time] = []
                time_receive_to_message[delivery_time].extend(messages)
        return time_receive_to_message

    def cleanup(self):
        """
        Clean up resources when finished with the simulation
        """
        if self.omnet_connection:
            self.omnet_connection.cleanup()


