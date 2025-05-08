from dataclasses import dataclass

from mango import json_serializable


@json_serializable
@dataclass
class TrafficMessage:
    pass