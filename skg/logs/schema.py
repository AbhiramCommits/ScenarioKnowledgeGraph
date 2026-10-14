"""Typed record schemas for driving logs."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ScenarioMeta:
    """Scenario metadata."""

    scenario_id: str
    behavior_label: str
    description: str
    n_agents: int
    duration_steps: int
    min_ttc: float
    max_speed: float


@dataclass
class AgentTrack:
    """Agent trajectory point."""

    scenario_id: str
    timestamp_ms: int
    agent_id: str
    agent_type: Literal["VEHICLE", "PEDESTRIAN", "CYCLIST", "EGO"]
    x: float
    y: float
    heading: float
    velocity_x: float
    velocity_y: float
    lane_id: str
    is_ego: bool


@dataclass
class LaneSegment:
    """Lane information."""

    scenario_id: str
    lane_id: str
    centerline_x: list[float]
    centerline_y: list[float]
    successor_id: str | None
    predecessor_id: str | None
    speed_limit: float


@dataclass
class TrafficControl:
    """Traffic control point."""

    scenario_id: str
    control_id: str
    control_type: Literal["STOP", "SIGNAL", "YIELD"]
    lane_id: str
    x: float
    y: float
