from integration_environment.network_models.detailed_network_model import OmnetConnection


def test_initialization_of_omnet_process():
    omnet_connection = OmnetConnection(inet_installation_path='/home/malin/cocoon_omnet_workspace/inet4.5/src',
                                       config_name='General',
                                       omnet_project_path='/home/malin/PycharmProjects/cocoon_DAI/omnet_project/')
    omnet_connection.initialize()
