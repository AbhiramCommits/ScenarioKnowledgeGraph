"""Entity extraction from raw logs."""

from datetime import datetime
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, first, lit

from skg.config import get_settings


def extract_scenarios(df: DataFrame) -> DataFrame:
    """Extract Scenario nodes."""
    now = datetime.utcnow().isoformat()

    scenarios = df.select("scenario_id", "behavior_label", "description").distinct().select(
        col("scenario_id"),
        lit("Scenario").alias("node_label"),
        col("behavior_label"),
        col("description"),
        lit(now).alias("ingested_at"),
    )

    return scenarios


def extract_agents(df: DataFrame) -> DataFrame:
    """Extract Agent nodes."""
    now = datetime.utcnow().isoformat()

    agents = df.select("scenario_id", "agent_id", "agent_type", "is_ego").distinct().select(
        col("scenario_id"),
        col("agent_id"),
        lit("Agent").alias("node_label"),
        col("agent_type"),
        col("is_ego"),
        lit(now).alias("ingested_at"),
    )

    return agents


def extract_lanes(df: DataFrame) -> DataFrame:
    """Extract Lane nodes."""
    now = datetime.utcnow().isoformat()

    lanes = df.select("scenario_id", "lane_id").distinct().select(
        col("scenario_id"),
        col("lane_id"),
        lit("Lane").alias("node_label"),
        lit(now).alias("ingested_at"),
    )

    return lanes


def extract_traffic_controls(df: DataFrame) -> DataFrame:
    """Extract TrafficControl nodes."""
    now = datetime.utcnow().isoformat()

    controls = df.select("scenario_id").distinct().select(
        col("scenario_id"),
        lit("control_0").alias("control_id"),
        lit("TrafficControl").alias("node_label"),
        lit("STOP").alias("control_type"),
        lit(now).alias("ingested_at"),
    )

    return controls
