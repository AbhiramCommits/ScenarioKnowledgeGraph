"""Relation extraction from logs."""

from datetime import datetime

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, sqrt, when
from pyspark.sql.functions import max as spark_max
from pyspark.sql.functions import min as spark_min

from skg.config import get_settings


def extract_travels_on_lane(df: DataFrame) -> DataFrame:
    """TRAVELS_ON_LANE: agent moves on lane."""
    now = datetime.utcnow().isoformat()

    relations = (
        df.groupBy("scenario_id", "agent_id", "lane_id")
        .agg(
            spark_min("timestamp_ms").alias("valid_from_ms"),
            spark_max("timestamp_ms").alias("valid_to_ms"),
        )
        .select(
            col("agent_id").alias("src_id"),
            lit("Agent").alias("src_type"),
            lit("TRAVELS_ON_LANE").alias("rel_type"),
            col("lane_id").alias("dst_id"),
            lit("Lane").alias("dst_type"),
            col("scenario_id"),
            col("valid_from_ms"),
            col("valid_to_ms"),
            lit(now).alias("ingested_at"),
            lit("{}").alias("features"),
        )
    )

    return relations


def extract_ego_centric_relations(df: DataFrame) -> DataFrame:
    """Extract EGO-centric relations: APPROACHES, FOLLOWS, CUTS_IN, etc."""
    now = datetime.utcnow().isoformat()
    settings = get_settings()

    # Self-join on scenario and timestamp
    ego_df = df.filter(col("is_ego")).select(
        col("scenario_id"),
        col("timestamp_ms"),
        col("x").alias("ego_x"),
        col("y").alias("ego_y"),
        col("velocity_x").alias("ego_vx"),
        col("velocity_y").alias("ego_vy"),
        col("heading").alias("ego_heading"),
    )

    other_df = df.filter(~col("is_ego")).select(
        col("scenario_id"),
        col("timestamp_ms"),
        col("agent_id").alias("other_agent_id"),
        col("x").alias("other_x"),
        col("y").alias("other_y"),
        col("velocity_x").alias("other_vx"),
        col("velocity_y").alias("other_vy"),
        col("heading").alias("other_heading"),
        col("lane_id").alias("other_lane_id"),
    )

    joined = ego_df.join(other_df, on=["scenario_id", "timestamp_ms"])

    # Compute geometric features
    features_df = joined.select(
        col("scenario_id"),
        col("other_agent_id"),
        col("timestamp_ms"),
        sqrt((col("other_x") - col("ego_x")) ** 2 + (col("other_y") - col("ego_y")) ** 2).alias("distance_m"),
        (
            (col("other_vx") - col("ego_vx")) * (col("other_x") - col("ego_x"))
            + (col("other_vy") - col("ego_vy")) * (col("other_y") - col("ego_y"))
        ).alias("relative_velocity_dot"),
    )

    # Aggregate per scenario and agent pair
    agg_features = features_df.groupBy("scenario_id", "other_agent_id").agg(
        spark_min("distance_m").alias("min_gap_m"),
        spark_min(
            when(col("relative_velocity_dot") > 0, col("distance_m") / (col("relative_velocity_dot") + 0.1))
        ).alias("min_ttc_s"),
        spark_min("timestamp_ms").alias("valid_from_ms"),
        spark_max("timestamp_ms").alias("valid_to_ms"),
    )

    # Label relations based on thresholds
    relations = agg_features.select(
        lit("ego").alias("src_id"),
        lit("Agent").alias("src_type"),
        when(col("min_ttc_s") < settings.min_ttc_near_miss, lit("NEAR_MISS"))
        .otherwise(lit("APPROACHES"))
        .alias("rel_type"),
        col("other_agent_id").alias("dst_id"),
        lit("Agent").alias("dst_type"),
        col("scenario_id"),
        col("valid_from_ms"),
        col("valid_to_ms"),
        lit(now).alias("ingested_at"),
        to_json_features(col("min_gap_m"), col("min_ttc_s"), lit(0.0), lit(0.0)).alias("features"),
    )

    return relations


def to_json_features(min_gap: any, min_ttc: any, closing_rate: any, lateral: any) -> str:
    """Convert features to JSON string."""
    # In production, use pyspark.sql.functions.to_json
    # For now, return a placeholder
    return lit("{}")


def extract_all_relations(df: DataFrame, behavior_label_df: DataFrame | None = None) -> DataFrame:
    """Extract all relation types."""
    travels = extract_travels_on_lane(df)
    approaches = extract_ego_centric_relations(df)

    # Union all relation types
    all_relations = travels.union(approaches)

    return all_relations
