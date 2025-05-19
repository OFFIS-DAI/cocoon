import unittest
import logging

from integration_environment.network_models.detailed_network_model import OmnetConnection

logging.basicConfig(level=logging.DEBUG)


class TestOmnetSocketConnection(unittest.TestCase):
    """Test the OMNeT++ socket connection"""

    def test_socket_connection(self):
        """Test establishing a socket connection to OMNeT++"""
        # Create a connection
        connection = OmnetConnection(inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                     config_name='General',
                                     omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/')

        try:
            # Initialize (starts OMNeT++ and connects to socket)
            connection.initialize()

            # Send a test message
            self.assertTrue(connection.send_message("TEST_MESSAGE"),
                            "Failed to send test message")

            # Wait for event messages (simulation events)
            received_event = False
            for _ in range(10):  # Try for up to 10 seconds
                message = connection.receive_message(timeout=1)
                if message and message.startswith("EVENT:"):
                    received_event = True
                    print(f"Received event message: {message}")
                    break

            self.assertTrue(received_event,
                            "Did not receive any EVENT message from OMNeT++")

        finally:
            # Clean up resources
            connection.cleanup()


if __name__ == '__main__':
    unittest.main()
