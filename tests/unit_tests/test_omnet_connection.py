import time
import unittest
import logging

from integration_environment.network_models.detailed_network_model import OmnetConnection

logging.basicConfig(level=logging.DEBUG)


class TestOmnetSocketConnection(unittest.TestCase):
    """Test the OMNeT++ socket connection"""

    def test_message_dispatch(self):
        # Create a connection
        connection = OmnetConnection(inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                     config_name='General',
                                     omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/cocoon_omnet_project/')
        # Start OMNeT++ and connect socket
        connection.initialize()

        time.sleep(1)

        try:
            # Send several messages to simulate
            self.assertTrue(
                connection.send_message_to_omnet(
                    sender="node0",
                    receiver="node1",
                    msg_size_B=1024,
                    time_send_ms=10,
                    msg_id='msg0',
                    max_advance=15
                ))
            time.sleep(10)
            print(connection.get_all_messages())
            self.assertTrue(connection.send_termination_signal())

        finally:
            # Clean up
            connection.cleanup()


if __name__ == '__main__':
    unittest.main()
