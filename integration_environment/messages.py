from dataclasses import dataclass

from mango import json_serializable


@json_serializable
@dataclass
class TrafficMessage:
    msg_id: str
    payload: str

@json_serializable
@dataclass
class PlanningDataMessage:
    msg_id: str
    baseline: int
    min_p: int
    max_p: int
