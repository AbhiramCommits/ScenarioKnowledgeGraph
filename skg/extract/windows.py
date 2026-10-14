"""Temporal windowing and event segmentation helpers."""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, window, approx_count_distinct
from pyspark.sql.window import Window


def segment_by_time_window(df: DataFrame, window_duration_ms: int = 1000) -> DataFrame:
    """Segment log into time windows."""
    windowed = df.withColumn(
        "time_window",
        window(col("timestamp_ms") / 1000, f"{window_duration_ms // 1000}s")
    )
    return windowed


def segment_by_event(df: DataFrame, event_column: str) -> DataFrame:
    """Segment log by event changes (e.g., lane changes, TTC transitions)."""
    w = Window.partitionBy("scenario_id").orderBy("timestamp_ms")
    from pyspark.sql.functions import lag, when

    # Detect changes in event column
    df_with_change = df.withColumn(
        "event_changed",
        when(col(event_column) != lag(event_column).over(w), 1).otherwise(0)
    )

    # Group by cumulative sum of changes (creates segments)
    from pyspark.sql.functions import sum as spark_sum
    df_with_segment = df_with_change.withColumn(
        "event_segment",
        spark_sum("event_changed").over(w)
    )

    return df_with_segment
