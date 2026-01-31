"""
Network Models for COCOON.

This package contains various network model implementations:

- CocoonMetaModel: The main COCOON meta-model for delay prediction
- DetailedNetworkModel: OMNeT++ integration for detailed simulation
- ChannelNetworkModel: Simplified NetworkX-based channel model
- StaticGraphModel: Pre-configured static delay graph model
"""

from .cocoon_meta_model import CocoonMetaModel
from .detailed_network_model import DetailedNetworkModel

__all__ = ['CocoonMetaModel', 'DetailedNetworkModel']
