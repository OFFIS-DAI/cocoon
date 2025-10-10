import math
import random
import warnings
import os
from abc import ABC, abstractmethod
import lxml
from os.path import abspath
from pathlib import Path
from typing import Tuple

from pandapower.networks.mv_oberrhein import mv_oberrhein
import pandapower
import utm
from sklearn.cluster import KMeans
import numpy as np

warnings.filterwarnings('ignore')

ROOT = str(Path(abspath(__file__)).parent.parent.parent)


class CommunicationNode:
    def __init__(self, omnet_name: str, omnet_port: int, coordinates: Tuple[float, float]):
        self.omnet_name = omnet_name
        self.omnet_port = omnet_port
        self.coordinates = coordinates


class CommunicationInfrastructure:
    def __init__(self, class_name: str, identifier: str, position: tuple):
        self.class_name = class_name
        self.identifier = identifier
        self.position = position


class CommunicationConnection:
    def __init__(self, connector_1: tuple,
                 connector_2: tuple, conn_type: str):
        self.connector_1 = connector_1
        self.connector_2 = connector_2
        self.conn_type = conn_type

    def get_connection_string(self) -> str:
        name_1 = name_2 = ''
        if isinstance(self.connector_1[0], CommunicationNode):
            name_1 = self.connector_1[0].omnet_name
        elif isinstance(self.connector_1[0], CommunicationInfrastructure):
            name_1 = self.connector_1[0].identifier
        if isinstance(self.connector_2[0], CommunicationNode):
            name_2 = self.connector_2[0].omnet_name
        elif isinstance(self.connector_2[0], CommunicationInfrastructure):
            name_2 = self.connector_2[0].identifier
        return f'{name_1}.{self.connector_1[1]} <--> {self.conn_type} <--> {name_2}.{self.connector_2[1]}'


def to_xy(row):
    x, y = float(row['x']), float(row['y'])
    # If values look like degrees, convert to UTM; else assume projected meters
    if -180 <= x <= 180 and -90 <= y <= 90:
        e, n, *_ = utm.from_latlon(y, x)
        return (e, n)
    return (x, y)


def delete_old_config_section(technology: str, num_nodes: int):
    # Open the file and read its contents
    with open('cocoon_omnet_project/omnetpp.ini', 'r') as file:
        content = file.read()

    # Create the pattern string to search for
    pattern = f'[EvaluationNetwork{technology}_{num_nodes}]'

    # Check if the pattern exists in the file
    start_index = content.find(pattern)
    if start_index != -1:
        # Find the next occurrence of '[' after the pattern
        end_index = content.find('[Config', start_index + len(pattern))

        # If another '[' is found, delete everything from the pattern to the '['
        if end_index != -1:
            modified_content = content[:start_index] + content[end_index:]
        else:  # If no further '[' is found, delete everything from the pattern to the end of the file
            modified_content = content[:start_index]

        # Write the modified content back to the file
        with open('cocoon_omnet_project/omnetpp.ini', 'w') as file:
            file.write(modified_content)
        print("File modified successfully.")
    else:
        print("Pattern not found in the file.")


class NetworkExtractor(ABC):
    def __init__(self, num_nodes: int = 100, technology: str = 'Ethernet'):
        self.pp_network = None

        self.port = 1000

        # communication infrastructure for OMNeT++ network definition
        self.communication_infrastructure = list()
        # communication connections for OMNeT++ network definition
        self.communication_connections = list()

        self.network_size = (0, 0)

        self.end_point_nodes = list()

        self.control_entity = None

        self.num_nodes = num_nodes
        self.technology = technology

    def initialize_network(self):
        self.pp_network = mv_oberrhein("generation", include_substations=True)

        self.get_nodes_from_pp_network()

        x_min = 3410000
        x_max = 3415000

        y_min = 5360000
        y_max = 5365000

        self.end_point_nodes = random.sample([n for n in self.end_point_nodes
                                              if x_min < n.coordinates[0] < x_max
                                              and y_min < n.coordinates[1] < y_max], self.num_nodes - 1)

        for i, node in enumerate(self.end_point_nodes):
            node.omnet_name = f'node{i + 1}'

        self.rescale_network()

        cx = np.mean([n.coordinates[0] for n in self.end_point_nodes])
        cy = np.mean([n.coordinates[1] for n in self.end_point_nodes])

        self.control_entity = CommunicationNode(
            omnet_name='node0',
            omnet_port=self.get_port(),
            coordinates=(cx, cy)
        )

        self.place_communication_infrastructure()

        os.chdir(ROOT)

        delete_old_config_section(num_nodes=self.num_nodes, technology=self.technology)

        network_description = self.get_omnet_network_description()
        ini_config = self.get_omnet_ini_config()

        with open(f'cocoon_omnet_project/networks/EvaluationNetwork{self.technology}_{self.num_nodes}.ned', 'w') as f:
            f.write(network_description)
            f.close()

        with open(f'cocoon_omnet_project/omnetpp.ini', 'a') as config:
            config.write(ini_config)
            config.close()

    def get_network_area(self):
        min_x = math.inf
        max_x = 0
        min_y = math.inf
        max_y = 0
        for node in self.end_point_nodes:
            if node.coordinates[0] < min_x:
                min_x = node.coordinates[0]
            if node.coordinates[1] < min_y:
                min_y = node.coordinates[1]
            if node.coordinates[0] > max_x:
                max_x = node.coordinates[0]
            if node.coordinates[1] > max_y:
                max_y = node.coordinates[1]
        return (min_x, min_y), (max_x, max_y)

    def rescale_network(self):
        (min_x, min_y), (max_x, max_y) = self.get_network_area()
        self.network_size = (max_x - min_x, max_y - min_y)
        print('Network size: ', self.network_size)
        for node in self.end_point_nodes:
            new_coords = (node.coordinates[0] - min_x + 10, node.coordinates[1] - min_y + 10)
            node.coordinates = new_coords

    def get_port(self):
        port = self.port
        self.port += 1
        return port

    def generate_antenna_positions(self, distance: int) -> list[tuple[int, int]]:
        """
        Generates positions of antennas.
        @param distance: distance between antennas in meters.
        """
        width, height = self.network_size
        positions = []

        # Generate positions along the width and height
        x = 1000 if width > 1000 else int(width / 2)
        while x <= width:
            y = 1000 if height > 1000 else int(height / 2)
            while y <= height:
                positions.append((x, y))
                y += distance
            x += distance

        return positions

    def get_nodes_from_pp_network(self):
        # grid infrastructure agents
        bus_measurements = self.pp_network.measurement[self.pp_network.measurement['element_type'] == 'bus']
        measurement_bus_ids = bus_measurements['element']
        unique_measurement_bus_ids = measurement_bus_ids.unique()
        measurement_bus_coordinates = self.pp_network.bus_geodata.loc[unique_measurement_bus_ids]
        measurements_with_coordinates = (
            self.pp_network.measurement.join(measurement_bus_coordinates, on='element', how='left',
                                             rsuffix='_geo'))

        i = 0

        for index, row in measurements_with_coordinates.iterrows():
            if np.isnan(row['y']) or np.isnan(row['x']):
                continue

            self.end_point_nodes.append(
                CommunicationNode(
                    omnet_name=f'node{i}',
                    omnet_port=self.get_port(),
                    coordinates=to_xy(row)
                )
            )
            i += 1

        # load agents
        load_bus_ids = self.pp_network.load['bus']
        load_bus_coordinates = self.pp_network.bus_geodata.loc[load_bus_ids]
        loads_with_coordinates = (
            self.pp_network.load.join(load_bus_coordinates, on='bus', how='left', rsuffix='_geo'))

        for index, row in loads_with_coordinates.iterrows():
            self.end_point_nodes.append(
                CommunicationNode(
                    omnet_name=f'node{i}',
                    omnet_port=self.get_port(),
                    coordinates=to_xy(row)
                )
            )
            i += 1

        # generation agents
        gen_bus_ids = self.pp_network.sgen['bus']
        gen_bus_coordinates = self.pp_network.bus_geodata.loc[gen_bus_ids]
        gens_with_coordinates = (
            self.pp_network.sgen.join(gen_bus_coordinates, on='bus', how='left', rsuffix='_geo'))

        for index, row in gens_with_coordinates.iterrows():
            self.end_point_nodes.append(
                CommunicationNode(
                    omnet_name=f'node{i}',
                    omnet_port=self.get_port(),
                    coordinates=to_xy(row)
                )
            )
            i += 1

        # storage agents
        storage_bus_ids = self.pp_network.storage['bus']
        storage_bus_coordinates = self.pp_network.bus_geodata.loc[storage_bus_ids]
        storages_with_coordinates = (
            self.pp_network.storage.join(storage_bus_coordinates, on='bus', how='left', rsuffix='_geo'))
        i = 0
        for index, row in storages_with_coordinates.iterrows():
            self.end_point_nodes.append(
                CommunicationNode(
                    omnet_name=f'node{i}',
                    omnet_port=self.get_port(),
                    coordinates=to_xy(row)
                )
            )
            i += 1
        print('Number of nodes: ', len(self.end_point_nodes))

    @abstractmethod
    def place_communication_infrastructure(self):
        """
        Method that fills the following lists: self.communication_infrastructure and self.communication_connections.
        """
        pass

    @abstractmethod
    def get_omnet_network_description(self) -> str:
        """
        Bilds string for the OMNeT++ .ned network description file.
        :return: file as string.
        """
        pass

    def get_omnet_ini_config(self) -> str:
        pass


class EthernetNetworkExtractor(NetworkExtractor):

    def __init__(self, num_nodes: int = 100):
        super().__init__(num_nodes=num_nodes)

    def place_communication_infrastructure(self):
        # -------- link tier presets --------
        HAN_MAX = 100  # 1–100 m  -> HAN/BAN/IAN
        NAN_MAX = 10_000  # 10 m–10 km -> NAN/FAN

        # >= 10 km -> WAN

        def dist(a_xy, b_xy):
            ax, ay = a_xy
            bx, by = b_xy
            return math.hypot(ax - bx, ay - by)

        def choose_channel(distance_m, heavy=False, force_wan=False):
            """
            Returns a channel identifier string that matches a NED channel we will declare.
            Uses your paper-based tiers:
              - WAN:   10 Mbps – 1 Gbps
              - NAN:   100 kbps – 10 Mbps
              - HAN:   10 – 100 kbps
            """
            if force_wan or distance_m >= NAN_MAX:  # WAN
                return 'C1Gbps' if heavy else 'C100Mbps'  # pick within 10 Mbps–1 Gbps
            elif distance_m > HAN_MAX:  # NAN/FAN
                if heavy:
                    return 'C10Mbps'  # 10 Mbps upper NAN range
                # pick mid NAN for general endpoints
                return 'C1Mbps'  # 1 Mbps (within 100 kbps–10 Mbps)
            else:  # HAN/BAN/IAN
                return 'C100kbps' if heavy else 'C10kbps'  # 10–100 kbps

        node_coordinates = np.array([node.coordinates for node in self.end_point_nodes])

        # 1) Central/core router at geometric center of endpoints
        cx = np.mean([n.coordinates[0] for n in self.end_point_nodes])
        cy = np.mean([n.coordinates[1] for n in self.end_point_nodes])
        router_central = CommunicationInfrastructure(
            class_name='Router', identifier='router_central', position=(cx, cy)
        )
        self.communication_infrastructure.append(router_central)

        # Control entity (node0) backhaul to core (force WAN, heavy)
        if getattr(self, 'control_entity', None) is not None:
            ch_name = choose_channel(
                dist(self.control_entity.coordinates, router_central.position),
                heavy=True, force_wan=True
            )
            self.communication_connections.append(
                CommunicationConnection(
                    connector_1=(self.control_entity, 'pppg++'),
                    connector_2=(router_central, 'pppg++'),
                    conn_type=ch_name
                )
            )

        # Network configurator (position irrelevant)
        self.communication_infrastructure.append(
            CommunicationInfrastructure(
                class_name='Ipv4NetworkConfigurator',
                identifier='configurator',
                position=(100, 100)
            )
        )

        # 2) Cluster endpoints geographically (consider making this dynamic to keep ~4–12 per cluster)
        num_clusters = 3
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(node_coordinates)
        labels = kmeans.labels_

        clusters = {i: [] for i in range(num_clusters)}
        for idx, label in enumerate(labels):
            clusters[label].append(self.end_point_nodes[idx])

        # 3) Place a router at the CH (agent nearest to centroid) for each cluster
        cluster_router = {}
        cluster_heads = set()  # remember CH nodes (heavy endpoints)
        for cluster_id, agents_in_cluster in clusters.items():
            coords = np.array([a.coordinates for a in agents_in_cluster])
            centroid = coords.mean(axis=0)

            ch_node = min(
                agents_in_cluster,
                key=lambda a: (a.coordinates[0] - centroid[0]) ** 2 + (a.coordinates[1] - centroid[1]) ** 2
            )
            cluster_heads.add(ch_node)

            router = CommunicationInfrastructure(
                class_name='Router',
                identifier=f'router_{cluster_id}',
                position=tuple(ch_node.coordinates)
            )
            self.communication_infrastructure.append(router)
            cluster_router[cluster_id] = router

            # Backhaul cluster router ↔ central router (force WAN; treat backhaul as heavy)
            ch_name = choose_channel(
                dist(router.position, router_central.position),
                heavy=True, force_wan=True
            )
            self.communication_connections.append(
                CommunicationConnection(
                    connector_1=(router, 'pppg++'),
                    connector_2=(router_central, 'pppg++'),
                    conn_type=ch_name
                )
            )

        # 4) Connect endpoints to their cluster router with distance-tiered channels
        for idx, label in enumerate(labels):
            node = self.end_point_nodes[idx]
            router = cluster_router[label]

            d = dist(node.coordinates, router.position)
            is_heavy = (node in cluster_heads)  # CHs are heavier endpoints
            ch_name = choose_channel(d, heavy=is_heavy, force_wan=False)

            self.communication_connections.append(
                CommunicationConnection(
                    connector_1=(node, 'pppg++'),
                    connector_2=(router, 'pppg++'),
                    conn_type=ch_name
                )
            )

    def get_omnet_network_description(self) -> str:
        imports = ('package networks; \n'
                   'import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;\n'
                   'import inet.networklayer.ipv4.RoutingTableRecorder;\n'
                   'import inet.node.inet.Router;\n'
                   'import inet.node.inet.StandardHost;\n'
                   'import MangoTimeAdvancer;\n')

        network = f'network EvaluationNetworkEthernet_{self.num_nodes} ' + '{\n'

        # add some margin (10m) in network display
        parameters = ('parameters: @display("i=block/network2;bgb=' +
                      f'{self.network_size[0] + 10},{self.network_size[1] + 10}");\n')

        submodules = ('submodules:\n\n'
                      '\ttimeAdvancer: MangoTimeAdvancer {@display("p=337,40");}\n')

        for module in self.communication_infrastructure:
            submodules += (f'\t{module.identifier}: {module.class_name}' + '{' +
                           f'@display("p={int(module.position[0])},{int(module.position[1])}");' + '}\n')
        for node in self.end_point_nodes:
            submodules += (
                    f'\t{node.omnet_name}: StandardHost ' +
                    '{@display("p=' + f'{int(node.coordinates[0])},{int(node.coordinates[1])}' + '");}\n')
        if self.control_entity is not None:
            submodules += (
                f'\t{self.control_entity.omnet_name}: StandardHost '
                f'{{@display("p={int(self.control_entity.coordinates[0])},'
                f'{int(self.control_entity.coordinates[1])}");}}\n'
            )

        connections = 'connections:\n'
        for connection in self.communication_connections:
            connections += '\t' + connection.get_connection_string() + ';\n'

        return imports + network + parameters + submodules + connections + '}'

    def get_omnet_ini_config(self):
        config_string = (f'[EvaluationNetworkEthernet_{self.num_nodes}]\n'
                         f'network = networks.EvaluationNetworkEthernet_{self.num_nodes}\n'
                         f'extends = Ethernet\n')

        if self.control_entity is not None:
            config_string += f'**.{self.control_entity.omnet_name}.app[0].localPort = {self.control_entity.omnet_port}\n'

        for i, node in enumerate(self.end_point_nodes):
            config_string += f'**.{node.omnet_name}.app[0].localPort = {node.omnet_port}\n'
        config_string += '*.server.numApps=0\n'
        return config_string


for n in [5, 10, 20, 50, 100]:
    ex = EthernetNetworkExtractor(num_nodes=n)
    ex.initialize_network()
